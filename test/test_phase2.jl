@testset "Phase 2: Covariant Derivatives" begin

    function gr_registry_with_covd()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂,
            [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M4, rank=(1,1),
            symmetries=Any[],
            options=Dict{Symbol,Any}(:is_delta => true)))
        define_curvature_tensors!(reg, :M4, :g)
        register_tensor!(reg, TensorProperties(
            name=:V, manifold=:M4, rank=(1,0),
            symmetries=Any[]))
        register_tensor!(reg, TensorProperties(
            name=:ω, manifold=:M4, rank=(0,1),
            symmetries=Any[]))
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)]))
        define_covd!(reg, :∇; manifold=:M4, metric=:g)
        reg
    end

    @testset "2.1: DefCovD" begin
        reg = gr_registry_with_covd()
        with_registry(reg) do
            # Christoffel symbol should be registered
            @test has_tensor(reg, :Γ∇)
            props = get_tensor(reg, :Γ∇)
            @test props.rank == (1, 2)
            # Torsion-free → symmetric in lower pair
            @test length(props.symmetries) == 1
            @test props.symmetries[1] isa Symmetric

            # CovD properties should be retrievable
            covd = get_covd(reg, :∇)
            @test covd.manifold == :M4
            @test covd.metric == :g
            @test covd.christoffel == :Γ∇
            @test covd.torsion_free
            @test covd.metric_compatible
        end
    end

    @testset "2.1: Metric compatibility ∇g = 0" begin
        reg = gr_registry_with_covd()
        with_registry(reg) do
            # ∇_a g_{bc} should simplify to 0
            expr = TDeriv(down(:a), Tensor(:g, [down(:b), down(:c)]))
            result = simplify(expr)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "2.1: covd_to_christoffel on vector" begin
        reg = gr_registry_with_covd()
        with_registry(reg) do
            # ∇_a V^b = ∂_a V^b + Γ^b_{ac} V^c
            expr = TDeriv(down(:a), Tensor(:V, [up(:b)]))
            result = covd_to_christoffel(expr, :∇)
            # Should be a sum of two terms
            @test result isa TSum
            @test length(result.terms) == 2
            # One term is ∂_a V^b (the derivative)
            # Other is Γ^b_{a?} V^? (Christoffel × V)
        end
    end

    @testset "2.1: covd_to_christoffel on covector" begin
        reg = gr_registry_with_covd()
        with_registry(reg) do
            # ∇_a ω_b = ∂_a ω_b - Γ^c_{ab} ω_c
            expr = TDeriv(down(:a), Tensor(:ω, [down(:b)]))
            result = covd_to_christoffel(expr, :∇)
            @test result isa TSum
            @test length(result.terms) == 2
        end
    end

    @testset "2.1: covd_to_christoffel on rank-2 tensor" begin
        reg = gr_registry_with_covd()
        with_registry(reg) do
            # ∇_a T_{bc} = ∂_a T_{bc} - Γ^d_{ab} T_{dc} - Γ^d_{ac} T_{bd}
            expr = TDeriv(down(:a), Tensor(:T, [down(:b), down(:c)]))
            result = covd_to_christoffel(expr, :∇)
            # Should have 3 terms: partial + 2 Christoffel terms
            @test result isa TSum
            @test length(result.terms) == 3
        end
    end

    @testset "2.3: Commute covds [∇_a, ∇_b]V^c = R^c_{dab}V^d" begin
        reg = gr_registry_with_covd()
        with_registry(reg) do
            # ∇_b(∇_a(V^c)) with b > a alphabetically, should reorder
            # ∇_b(∇_a(V^c)) = ∇_a(∇_b(V^c)) + R^c_{dba} V^d
            inner = Tensor(:V, [up(:c)])
            expr = TDeriv(down(:b), TDeriv(down(:a), inner))
            result = commute_covds(expr, :∇)

            # Should be a sum: ∇_a(∇_b(V^c)) + Riemann terms
            @test result isa TSum
            terms = result.terms
            @test length(terms) >= 2

            # One term should be the sorted derivatives
            has_sorted = any(terms) do t
                t isa TDeriv && t.index == down(:a) &&
                t.arg isa TDeriv && t.arg.index == down(:b)
            end
            @test has_sorted
        end
    end

    @testset "2.4: Lie derivative of vector" begin
        reg = gr_registry_with_covd()
        register_tensor!(reg, TensorProperties(
            name=:ξ, manifold=:M4, rank=(1,0), symmetries=Any[]))
        with_registry(reg) do
            ξ = Tensor(:ξ, [up(:a)])
            V = Tensor(:V, [up(:b)])
            result = lie_derivative(ξ, V)

            # £_ξ V^b = ξ^c ∂_c V^b - (∂_c ξ^b) V^c
            # Should be a sum of 2 terms
            @test result isa TSum
            @test length(result.terms) == 2
        end
    end

    @testset "2.4: Lie derivative of scalar" begin
        reg = gr_registry_with_covd()
        register_tensor!(reg, TensorProperties(
            name=:ξ, manifold=:M4, rank=(1,0), symmetries=Any[]))
        register_tensor!(reg, TensorProperties(
            name=:Φ, manifold=:M4, rank=(0,0), symmetries=Any[]))
        with_registry(reg) do
            ξ = Tensor(:ξ, [up(:a)])
            Φ = Tensor(:Φ, TIndex[])
            result = lie_derivative(ξ, Φ)

            # £_ξ Φ = ξ^c ∂_c Φ
            @test result isa TProduct || result isa TDeriv
        end
    end

    @testset "2.4: Lie derivative of covector" begin
        reg = gr_registry_with_covd()
        register_tensor!(reg, TensorProperties(
            name=:ξ, manifold=:M4, rank=(1,0), symmetries=Any[]))
        with_registry(reg) do
            ξ = Tensor(:ξ, [up(:a)])
            ω = Tensor(:ω, [down(:b)])
            result = lie_derivative(ξ, ω)

            # £_ξ ω_b = ξ^c ∂_c ω_b + (∂_b ξ^c) ω_c
            # Should be a sum of 2 terms
            @test result isa TSum
            @test length(result.terms) == 2
        end
    end
end
