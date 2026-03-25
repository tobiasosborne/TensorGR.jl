@testset "ToBasis for Tetrad Frames" begin
    using TensorGR: to_frame, from_frame,
                    define_frame_bundle!, frame_up, frame_down,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    Tensor, TProduct, TSum, TScalar,
                    up, down, with_registry, register_tensor!,
                    current_registry, has_tensor,
                    free_indices, simplify, indices

    function _make_tetrad_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_frame_bundle!(reg; manifold=:M4)
            register_tensor!(reg, TensorProperties(
                name=:e, manifold=:M4, rank=(1, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle_mixed => (:Tangent, :Lorentz))))
        end
        return reg
    end

    # ---- to_frame basic tests ------------------------------------------------

    @testset "to_frame: vector V^a → e^I_a V^a" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(1, 0), symmetries=SymmetrySpec[]))
            V = Tensor(:V, [up(:a)])
            result = to_frame(V, :e)

            @test result isa TProduct
            # Should have two factors: e^I_a and V^a
            @test length(result.factors) == 2

            # Free indices: one Up Lorentz
            fi = free_indices(result)
            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test isempty(tangent_fi)
            @test length(lorentz_fi) == 1
            @test lorentz_fi[1].position === Up
        end
    end

    @testset "to_frame: covector W_a → e^a_I W_a" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:W, manifold=:M4, rank=(0, 1), symmetries=SymmetrySpec[]))
            W = Tensor(:W, [down(:a)])
            result = to_frame(W, :e)

            @test result isa TProduct
            fi = free_indices(result)
            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test isempty(tangent_fi)
            @test length(lorentz_fi) == 1
            @test lorentz_fi[1].position === Down
        end
    end

    @testset "to_frame: rank-2 tensor T^a_b" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:T, manifold=:M4, rank=(1, 1), symmetries=SymmetrySpec[]))
            T = Tensor(:T, [up(:a), down(:b)])
            result = to_frame(T, :e)

            @test result isa TProduct
            # Should have 3 factors: two tetrads + T
            @test length(result.factors) == 3

            fi = free_indices(result)
            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test isempty(tangent_fi)
            @test length(lorentz_fi) == 2

            # One Up and one Down Lorentz
            positions = Set(idx.position for idx in lorentz_fi)
            @test Up in positions
            @test Down in positions
        end
    end

    @testset "to_frame: scalar expression unchanged" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            s = TScalar(42 // 1)
            result = to_frame(s, :e)
            @test result === s
        end
    end

    @testset "to_frame: already frame-indexed expression unchanged" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:F, manifold=:M4, rank=(1, 0), symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle => :Lorentz)))
            F = Tensor(:F, [frame_up(:I)])
            result = to_frame(F, :e)
            @test result === F  # no Tangent indices to convert
        end
    end

    @testset "to_frame: error on unregistered tetrad" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            V = Tensor(:g, [up(:a), up(:b)])
            @test_throws ErrorException to_frame(V, :nonexistent)
        end
    end

    @testset "to_frame: sum distributes" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(1, 0), symmetries=SymmetrySpec[]))
            register_tensor!(reg, TensorProperties(
                name=:W, manifold=:M4, rank=(1, 0), symmetries=SymmetrySpec[]))
            V = Tensor(:V, [up(:a)])
            W = Tensor(:W, [up(:a)])
            s = TSum([V, W])
            result = to_frame(s, :e)

            # Should produce e^I_a (V^a + W^a) as a TProduct
            @test result isa TProduct
            fi = free_indices(result)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(lorentz_fi) == 1
            @test lorentz_fi[1].position === Up
        end
    end

    @testset "to_frame: metric g_{ab}" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            g = Tensor(:g, [down(:a), down(:b)])
            result = to_frame(g, :e)

            # e^a_I e^b_J g_{ab} should give eta_{IJ} after simplification
            fi = free_indices(result)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(lorentz_fi) == 2
            @test all(idx -> idx.position === Down, lorentz_fi)
        end
    end

    # ---- from_frame tests ---------------------------------------------------

    @testset "from_frame: frame vector V^I → e^a_I V^I" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(1, 0), symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle => :Lorentz)))
            V = Tensor(:V, [frame_up(:I)])
            result = from_frame(V, :e)

            @test result isa TProduct
            fi = free_indices(result)
            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(tangent_fi) == 1
            @test tangent_fi[1].position === Up
            @test isempty(lorentz_fi)
        end
    end

    @testset "from_frame: frame covector W_I → e^I_a W_I" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:W, manifold=:M4, rank=(0, 1), symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle => :Lorentz)))
            W = Tensor(:W, [frame_down(:I)])
            result = from_frame(W, :e)

            @test result isa TProduct
            fi = free_indices(result)
            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(tangent_fi) == 1
            @test tangent_fi[1].position === Down
            @test isempty(lorentz_fi)
        end
    end

    @testset "from_frame: coordinate expression unchanged" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(1, 0), symmetries=SymmetrySpec[]))
            V = Tensor(:V, [up(:a)])
            result = from_frame(V, :e)
            @test result === V  # no Lorentz indices to convert
        end
    end

    @testset "from_frame: error on unregistered tetrad" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            V = Tensor(:g, [frame_up(:I)])
            @test_throws ErrorException from_frame(V, :nonexistent)
        end
    end

    # ---- Round-trip tests ---------------------------------------------------

    @testset "round-trip: to_frame then from_frame preserves rank" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(1, 0), symmetries=SymmetrySpec[]))
            V = Tensor(:V, [up(:a)])

            # to_frame: V^a → e^I_a V^a
            V_frame = to_frame(V, :e)
            # from_frame: e^I_a V^a → e^b_I e^I_a V^a
            V_back = from_frame(V_frame, :e)

            # After simplification, e^b_I e^I_a = δ^b_a (completeness)
            # So V_back simplifies to V^b (renamed dummy)
            fi = free_indices(V_back)
            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            @test length(tangent_fi) == 1
            @test tangent_fi[1].position === Up
        end
    end
end
