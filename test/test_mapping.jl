@testset "Smooth Mappings" begin
    @testset "define_mapping! and registry" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d, :e, :f]))
        register_manifold!(reg, ManifoldProperties(:N, 3, :h, :∂, [:i, :j, :k, :l]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))
        register_tensor!(reg, TensorProperties(name=:δ_M, manifold=:M, rank=(1, 1), is_delta=true))
        register_tensor!(reg, TensorProperties(name=:h, manifold=:N, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))

        with_registry(reg) do
            mp = define_mapping!(reg, :φ; domain=:M, codomain=:N)
            @test mp isa MappingProperties
            @test mp.name == :φ
            @test mp.domain == :M
            @test mp.codomain == :N
            @test mp.jacobian == :dφ
            @test mp.inv_jacobian === nothing

            @test has_mapping(reg, :φ)
            @test !has_mapping(reg, :ψ)
            @test get_mapping(reg, :φ) === mp

            # Jacobian tensor was registered
            @test has_tensor(reg, :dφ)
            jp = get_tensor(reg, :dφ)
            @test jp.rank == (1, 1)
        end
    end

    @testset "define_mapping! with inverse" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d, :e, :f]))
        register_manifold!(reg, ManifoldProperties(:N, 4, :h, :∂, [:i, :j, :k, :l, :m, :n]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))
        register_tensor!(reg, TensorProperties(name=:h, manifold=:N, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))

        with_registry(reg) do
            mp = define_mapping!(reg, :φ; domain=:M, codomain=:N,
                                 inv_jacobian_name=:dφ_inv)
            @test mp.inv_jacobian == :dφ_inv
            @test has_tensor(reg, :dφ_inv)
        end
    end

    @testset "define_mapping! errors" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d]))

        with_registry(reg) do
            # Codomain not registered
            @test_throws ErrorException define_mapping!(reg, :φ; domain=:M, codomain=:N)

            register_manifold!(reg, ManifoldProperties(:N, 3, :h, :∂, [:i, :j, :k]))
            define_mapping!(reg, :φ; domain=:M, codomain=:N)

            # Duplicate mapping
            @test_throws ErrorException define_mapping!(reg, :φ; domain=:M, codomain=:N)
        end
    end

    @testset "pullback of rank-(0,2) tensor" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d, :e, :f]))
        register_manifold!(reg, ManifoldProperties(:N, 3, :h, :∂, [:i, :j, :k, :l, :m, :n]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))
        register_tensor!(reg, TensorProperties(name=:h, manifold=:N, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))

        with_registry(reg) do
            define_mapping!(reg, :φ; domain=:M, codomain=:N)

            # Pullback of h_{ij} should produce dφ^i_a dφ^j_b h_{ij}
            h_ij = Tensor(:h, [down(:i), down(:j)])
            pb = pullback(h_ij, :φ; registry=reg)

            # Result should be a product with h and two Jacobian factors
            @test pb isa TProduct
            factor_names = [f.name for f in pb.factors if f isa Tensor]
            @test :h in factor_names
            @test count(==(:dφ), factor_names) == 2

            # The Jacobian factors should have up indices matching h's indices
            jac_factors = [f for f in pb.factors if f isa Tensor && f.name == :dφ]
            jac_up_names = Set(f.indices[1].name for f in jac_factors)
            @test :i in jac_up_names
            @test :j in jac_up_names
        end
    end

    @testset "pullback of rank-(0,1) tensor" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d, :e, :f]))
        register_manifold!(reg, ManifoldProperties(:N, 3, :h, :∂, [:i, :j, :k, :l]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))
        register_tensor!(reg, TensorProperties(name=:ω, manifold=:N, rank=(0, 1)))

        with_registry(reg) do
            define_mapping!(reg, :φ; domain=:M, codomain=:N)

            # Pullback of ω_i should produce dφ^i_a ω_i
            ω = Tensor(:ω, [down(:i)])
            pb = pullback(ω, :φ; registry=reg)

            @test pb isa TProduct
            factor_names = [f.name for f in pb.factors if f isa Tensor]
            @test :ω in factor_names
            @test :dφ in factor_names
        end
    end

    @testset "pullback of scalar = identity" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d]))
        register_manifold!(reg, ManifoldProperties(:N, 3, :h, :∂, [:i, :j, :k]))

        with_registry(reg) do
            define_mapping!(reg, :φ; domain=:M, codomain=:N)
            s = TScalar(42 // 1)
            @test pullback(s, :φ; registry=reg) == s
        end
    end

    @testset "pullback of contravariant tensor = identity" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d]))
        register_manifold!(reg, ManifoldProperties(:N, 3, :h, :∂, [:i, :j, :k]))
        register_tensor!(reg, TensorProperties(name=:V, manifold=:N, rank=(1, 0)))

        with_registry(reg) do
            define_mapping!(reg, :φ; domain=:M, codomain=:N)
            V = Tensor(:V, [up(:i)])
            @test pullback(V, :φ; registry=reg) == V  # no Down indices to pull back
        end
    end

    @testset "pushforward of rank-(1,0) tensor" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d, :e, :f]))
        register_manifold!(reg, ManifoldProperties(:N, 3, :h, :∂, [:i, :j, :k, :l]))
        register_tensor!(reg, TensorProperties(name=:V, manifold=:M, rank=(1, 0)))

        with_registry(reg) do
            define_mapping!(reg, :φ; domain=:M, codomain=:N,
                            inv_jacobian_name=:dφ_inv)

            V = Tensor(:V, [up(:a)])
            pf = pushforward(V, :φ; registry=reg)

            @test pf isa TProduct
            factor_names = [f.name for f in pf.factors if f isa Tensor]
            @test :V in factor_names
            @test :dφ_inv in factor_names
        end
    end

    @testset "pushforward requires inverse Jacobian" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d]))
        register_manifold!(reg, ManifoldProperties(:N, 3, :h, :∂, [:i, :j, :k]))
        register_tensor!(reg, TensorProperties(name=:V, manifold=:M, rank=(1, 0)))

        with_registry(reg) do
            define_mapping!(reg, :φ; domain=:M, codomain=:N)  # no inverse
            V = Tensor(:V, [up(:a)])
            @test_throws ErrorException pushforward(V, :φ; registry=reg)
        end
    end

    @testset "pullback_metric" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d, :e, :f]))
        register_manifold!(reg, ManifoldProperties(:N, 3, :h, :∂, [:i, :j, :k, :l, :m, :n]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))
        register_tensor!(reg, TensorProperties(name=:h, manifold=:N, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))

        with_registry(reg) do
            define_mapping!(reg, :φ; domain=:M, codomain=:N)

            pm = pullback_metric(:φ, :h; registry=reg)
            @test pm isa TProduct

            # Should have exactly: h_{ij} * dφ^i_a * dφ^j_b
            factor_names = [f.name for f in pm.factors if f isa Tensor]
            @test :h in factor_names
            @test count(==(:dφ), factor_names) == 2
        end
    end

    @testset "pullback distributes over sum" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d, :e, :f]))
        register_manifold!(reg, ManifoldProperties(:N, 3, :h, :∂, [:i, :j, :k, :l]))
        register_tensor!(reg, TensorProperties(name=:A, manifold=:N, rank=(0, 1)))
        register_tensor!(reg, TensorProperties(name=:B, manifold=:N, rank=(0, 1)))

        with_registry(reg) do
            define_mapping!(reg, :φ; domain=:M, codomain=:N)

            sum_expr = Tensor(:A, [down(:i)]) + Tensor(:B, [down(:i)])
            pb = pullback(sum_expr, :φ; registry=reg)

            @test pb isa TSum
            @test length(pb.terms) == 2
        end
    end

    @testset "same-manifold diffeomorphism" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))
        register_tensor!(reg, TensorProperties(name=:δ, manifold=:M, rank=(1, 1), is_delta=true))

        with_registry(reg) do
            # Diffeomorphism φ: M → M
            define_mapping!(reg, :φ; domain=:M, codomain=:M,
                            inv_jacobian_name=:dφ_inv)

            mp = get_mapping(reg, :φ)
            @test mp.domain == mp.codomain == :M

            # Pullback of metric
            g_ab = Tensor(:g, [down(:a), down(:b)])
            pb = pullback(g_ab, :φ; registry=reg)
            @test pb isa TProduct
            @test count(f -> f isa Tensor && f.name == :dφ, pb.factors) == 2
        end
    end

    @testset "pullback with pattern rules" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))
        register_tensor!(reg, TensorProperties(name=:δ, manifold=:M, rank=(1, 1), is_delta=true))
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)]))
        register_tensor!(reg, TensorProperties(name=:S, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)]))

        with_registry(reg) do
            # Pattern rule: T_{a_,b_} → S_{a_,b_}
            rule = RewriteRule(
                Tensor(:T, [down(:a_), down(:b_)]),
                Tensor(:S, [down(:a_), down(:b_)])
            )

            # Apply rule
            T_cd = Tensor(:T, [down(:c), down(:d)])
            result = apply_rules(T_cd, [rule])
            @test result == Tensor(:S, [down(:c), down(:d)])
        end
    end
end
