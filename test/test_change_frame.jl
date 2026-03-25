@testset "ChangeBasis Between Frame Choices" begin
    using TensorGR: define_frame_transformation!, frame_transformation_expr,
                    change_frame,
                    to_frame, from_frame,
                    define_frame_bundle!, frame_up, frame_down,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    Tensor, TProduct, TSum, TScalar,
                    up, down, with_registry, register_tensor!,
                    current_registry, has_tensor, get_tensor,
                    free_indices, simplify, indices

    function _make_two_tetrad_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_frame_bundle!(reg; manifold=:M4)
            # First tetrad
            register_tensor!(reg, TensorProperties(
                name=:e, manifold=:M4, rank=(1, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle_mixed => (:Tangent, :Lorentz))))
            # Second tetrad
            register_tensor!(reg, TensorProperties(
                name=:f, manifold=:M4, rank=(1, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle_mixed => (:Tangent, :Lorentz))))
        end
        return reg
    end

    # ---- define_frame_transformation! ----------------------------------------

    @testset "define_frame_transformation! registration" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            name = define_frame_transformation!(reg, :e, :f)
            @test name === :Lambda_e_to_f
            @test has_tensor(reg, :Lambda_e_to_f)
            props = get_tensor(reg, :Lambda_e_to_f)
            @test props.rank == (1, 1)
            @test get(props.options, :is_frame_transform, false)
            @test get(props.options, :from_tetrad, nothing) === :e
            @test get(props.options, :to_tetrad, nothing) === :f
        end
    end

    @testset "custom transform name" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            name = define_frame_transformation!(reg, :e, :f;
                transform_name=:L)
            @test name === :L
            @test has_tensor(reg, :L)
        end
    end

    @testset "idempotent registration" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            define_frame_transformation!(reg, :e, :f)
            define_frame_transformation!(reg, :e, :f)  # no error
            @test has_tensor(reg, :Lambda_e_to_f)
        end
    end

    @testset "error: missing Lorentz VBundle" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            register_tensor!(reg, TensorProperties(
                name=:e, manifold=:M4, rank=(1, 1), symmetries=SymmetrySpec[]))
            register_tensor!(reg, TensorProperties(
                name=:f, manifold=:M4, rank=(1, 1), symmetries=SymmetrySpec[]))
        end
        @test_throws ErrorException define_frame_transformation!(reg, :e, :f)
    end

    @testset "error: missing tetrad" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            @test_throws ErrorException define_frame_transformation!(reg, :e, :missing)
            @test_throws ErrorException define_frame_transformation!(reg, :missing, :f)
        end
    end

    # ---- frame_transformation_expr -------------------------------------------

    @testset "frame_transformation_expr structure" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            expr = frame_transformation_expr(:e, :f, frame_up(:I), frame_down(:J))
            @test expr isa TProduct
            @test length(expr.factors) == 2

            # Should contain both tetrad names
            names = Set(f.name for f in expr.factors)
            @test :e in names && :f in names
        end
    end

    @testset "frame_transformation_expr free indices" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            expr = frame_transformation_expr(:e, :f, frame_up(:I), frame_down(:J))
            fi = free_indices(expr)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(lorentz_fi) == 2
            names = Set(idx.name for idx in lorentz_fi)
            @test :I in names
            @test :J in names
        end
    end

    @testset "frame_transformation_expr index validation" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            @test_throws ErrorException frame_transformation_expr(
                :e, :f, up(:a), frame_down(:J))
            @test_throws ErrorException frame_transformation_expr(
                :e, :f, frame_up(:I), frame_up(:J))
        end
    end

    # ---- change_frame --------------------------------------------------------

    @testset "change_frame: Up Lorentz index" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(1, 0), symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle => :Lorentz)))
            V = Tensor(:V, [frame_up(:I)])
            result = change_frame(V, :e, :f)

            @test result isa TProduct
            # Should have 3 factors: f^{I'}_a, e^a_I, V^I
            @test length(result.factors) == 3

            fi = free_indices(result)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(lorentz_fi) == 1
            @test lorentz_fi[1].position === Up
            # The free index should NOT be :I (that's contracted)
            @test lorentz_fi[1].name !== :I
        end
    end

    @testset "change_frame: Down Lorentz index" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:W, manifold=:M4, rank=(0, 1), symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle => :Lorentz)))
            W = Tensor(:W, [frame_down(:I)])
            result = change_frame(W, :e, :f)

            @test result isa TProduct
            @test length(result.factors) == 3

            fi = free_indices(result)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(lorentz_fi) == 1
            @test lorentz_fi[1].position === Down
            @test lorentz_fi[1].name !== :I
        end
    end

    @testset "change_frame: rank-2 Lorentz tensor" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:T, manifold=:M4, rank=(1, 1), symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle => :Lorentz)))
            T = Tensor(:T, [frame_up(:I), frame_down(:J)])
            result = change_frame(T, :e, :f)

            @test result isa TProduct
            # 4 transform factors + 1 original = 5
            @test length(result.factors) == 5

            fi = free_indices(result)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(lorentz_fi) == 2
            # Original I,J should be contracted away
            names = Set(idx.name for idx in lorentz_fi)
            @test !(:I in names)
            @test !(:J in names)
        end
    end

    @testset "change_frame: no Lorentz indices unchanged" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(1, 0), symmetries=SymmetrySpec[]))
            V = Tensor(:V, [up(:a)])
            result = change_frame(V, :e, :f)
            @test result === V
        end
    end

    @testset "change_frame: scalar unchanged" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            s = TScalar(7 // 1)
            result = change_frame(s, :e, :f)
            @test result === s
        end
    end

    @testset "change_frame: error on unregistered tetrad" begin
        reg = _make_two_tetrad_registry()
        with_registry(reg) do
            V = Tensor(:g, [frame_up(:I)])
            @test_throws ErrorException change_frame(V, :e, :nonexistent)
            @test_throws ErrorException change_frame(V, :nonexistent, :f)
        end
    end
end
