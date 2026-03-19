@testset "Frame Bundle (Lorentz VBundle for tetrads)" begin

    @testset "define_frame_bundle! registers Lorentz VBundle" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            @test has_vbundle(reg, :Lorentz)
            lor = get_vbundle(reg, :Lorentz)
            @test lor.manifold == :M4
            @test lor.dim == 4
            @test lor.indices == [:I, :J, :K, :L, :M, :N]
        end
    end

    @testset "Frame metric and delta registered" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            @test has_tensor(reg, :eta)
            @test has_tensor(reg, :delta_frame)

            eta_props = get_tensor(reg, :eta)
            @test eta_props.is_metric == true
            @test eta_props.rank == (0, 2)
            @test eta_props.options[:vbundle] == :Lorentz
            @test eta_props.options[:vbundle_dim] == 4

            df_props = get_tensor(reg, :delta_frame)
            @test df_props.is_delta == true
            @test df_props.rank == (1, 1)
            @test df_props.options[:vbundle] == :Lorentz
        end
    end

    @testset "Caches: frame metric/delta keyed by :Lorentz" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            @test reg.metric_cache[:Lorentz] == :eta
            @test reg.delta_cache[:Lorentz] == :delta_frame
        end
    end

    @testset "Spacetime metric cache preserved" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            @test reg.metric_cache[:M4] == :g
        end
    end

    @testset "Convenience constructors" begin
        @test frame_up(:I) == TIndex(:I, Up, :Lorentz)
        @test frame_down(:J) == TIndex(:J, Down, :Lorentz)
        @test frame_up(:K).vbundle == :Lorentz
        @test frame_down(:L).position == Down
    end

    @testset "is_frame_index" begin
        @test is_frame_index(frame_up(:I)) == true
        @test is_frame_index(frame_down(:J)) == true
        @test is_frame_index(up(:a)) == false
        @test is_frame_index(down(:b)) == false
    end

    @testset "Frame indices distinct from Tangent" begin
        idx_frame = TIndex(:a, Up, :Lorentz)
        idx_tangent = TIndex(:a, Up, :Tangent)
        @test idx_frame != idx_tangent
        @test hash(idx_frame) != hash(idx_tangent)
    end

    @testset "Mixed tensor: tetrad e^a_I" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            e = Tensor(:e, [up(:a), frame_down(:I)])
            @test e.indices[1].vbundle == :Tangent
            @test e.indices[2].vbundle == :Lorentz
        end
    end

    @testset "Cross-bundle contraction refused" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            T1 = Tensor(:V, [frame_up(:a)])
            T2 = Tensor(:W, [down(:a)])  # Tangent
            prod = T1 * T2

            dp = dummy_pairs(prod)
            a_pairs = filter(p -> p[1].name == :a || p[2].name == :a, dp)
            @test isempty(a_pairs)
        end
    end

    @testset "Same-bundle frame contraction works" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            T1 = Tensor(:V, [frame_up(:I)])
            T2 = Tensor(:W, [frame_down(:I)])
            prod = T1 * T2

            dp = dummy_pairs(prod)
            i_pairs = filter(p -> p[1].name == :I || p[2].name == :I, dp)
            @test length(i_pairs) == 1
        end
    end

    @testset "fresh_index dispatches for :Lorentz vbundle" begin
        used = Set{Symbol}()
        @test fresh_index(used; vbundle=:Lorentz) == :I

        used = Set([:I, :J])
        @test fresh_index(used; vbundle=:Lorentz) == :K

        used = Set([:I, :J, :K, :L, :M, :N])
        @test fresh_index(used; vbundle=:Lorentz) == :I1
    end

    @testset "Manifold must exist" begin
        reg = TensorRegistry()
        @test_throws ErrorException define_frame_bundle!(reg; manifold=:M4)
    end

    @testset "Idempotent: calling twice does not error" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            # Second call should be idempotent (guards skip re-registration)
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            @test has_vbundle(reg, :Lorentz)
            @test has_tensor(reg, :eta)
        end
    end

    @testset "Frame metric symmetry" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            eta_props = get_tensor(reg, :eta)
            @test length(eta_props.symmetries) == 1
            @test eta_props.symmetries[1] isa Symmetric
        end
    end

    @testset "Coexists with spinor bundles" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            @test has_vbundle(reg, :SL2C)
            @test has_vbundle(reg, :SL2C_dot)
            @test has_vbundle(reg, :Lorentz)

            @test is_frame_index(frame_up(:I))
            @test is_spinor_index(spin_up(:A))
            @test !is_frame_index(spin_up(:A))
            @test !is_spinor_index(frame_up(:I))
        end
    end

    @testset "fresh_index still works for other vbundles" begin
        used = Set{Symbol}()
        @test fresh_index(used; vbundle=:SL2C) == :A
        @test fresh_index(used; vbundle=:SL2C_dot) == :Ap
        @test fresh_index(used; vbundle=:Tangent) == :a
    end

    @testset "eta contraction: eta^{IJ} eta_{IJ} = dim" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            eta_up = Tensor(:eta, [frame_up(:I), frame_up(:J)])
            eta_dn = Tensor(:eta, [frame_down(:I), frame_down(:J)])
            result = simplify(eta_up * eta_dn; registry=reg)
            @test result == TScalar(4 // 1)
        end
    end

end
