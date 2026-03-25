using TensorGR: contract_metrics, contract_metrics_with_derivatives, free_indices, indices

@testset "Metric contraction: g^{ab} g_{bc} = δ^a_c" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:δ, manifold=:M4, rank=(1,1), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_delta => true)))

    with_registry(reg) do
        g_up = Tensor(:g, [up(:a), up(:b)])
        g_dn = Tensor(:g, [down(:b), down(:c)])
        expr = g_up * g_dn  # g^{ab} g_{bc}, dummy: b

        result = contract_metrics(expr)
        # Should yield δ^a_c
        @test result isa Tensor
        @test result.name == :δ
        @test result.indices == [up(:a), down(:c)]
    end
end

@testset "Metric contraction: g^{ab} R_{abcd} = R^{}_{cd} (trace)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:R, manifold=:M4, rank=(0,4), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        g_up = Tensor(:g, [up(:a), up(:b)])
        R = Tensor(:R, [down(:a), down(:b), down(:c), down(:d)])
        expr = g_up * R

        result = contract_metrics(expr)
        fi = free_indices(result)
        # After contracting g^{ab} with R_{abcd}, free indices should be c, d only
        @test length(fi) == 2
        free_names = Set(i.name for i in fi)
        @test :c in free_names
        @test :d in free_names
    end
end

@testset "Metric contraction: g^{ab} g_{ab} = dim" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:δ, manifold=:M4, rank=(1,1), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_delta => true)))

    with_registry(reg) do
        g_up = Tensor(:g, [up(:a), up(:b)])
        g_dn = Tensor(:g, [down(:a), down(:b)])
        expr = g_up * g_dn  # g^{ab} g_{ab}: trace of delta = dim

        result = contract_metrics(expr)
        # g^{ab} g_{ab} → δ^a_a → dim = 4
        # First contraction: g^{ab} g_{ab} → δ^a_a (where a is now a dummy)
        # Second step: trace of delta = dimension
        @test result isa TScalar
        @test result.val == 4
    end
end

@testset "Metric raises index on tensor" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        g = Tensor(:g, [up(:a), up(:b)])
        T = Tensor(:T, [down(:b), down(:c)])
        expr = g * T  # g^{ab} T_{bc}

        result = contract_metrics(expr)
        # Should be T^a_c: the metric raised the b index
        @test result isa Tensor
        @test result.name == :T
        @test Set(result.indices) == Set([up(:a), down(:c)])
    end
end

@testset "Delta contraction: δ^a_b T^b_c = T^a_c" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:δ, manifold=:M4, rank=(1,1), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_delta => true)))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(1,1), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        δ = Tensor(:δ, [up(:a), down(:b)])
        T = Tensor(:T, [up(:b), down(:c)])
        expr = δ * T

        result = contract_metrics(expr)
        @test result isa Tensor
        @test result.name == :T
        @test Set(result.indices) == Set([up(:a), down(:c)])
    end
end

@testset "Metric self-trace: g^a_a (bare) = dim" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))

    with_registry(reg) do
        g_traced = Tensor(:g, [up(:a), down(:a)])
        result = contract_metrics(g_traced)
        @test result isa TScalar
        @test result.val == 4
    end
end

@testset "Metric self-trace in product: g^a_a * T_{bc} = 4 * T_{bc}" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        g_traced = Tensor(:g, [up(:a), down(:a)])
        T = Tensor(:T, [down(:b), down(:c)])
        expr = g_traced * T

        result = contract_metrics(expr)
        @test result isa TProduct
        @test result.scalar == 4
        @test length(result.factors) == 1
        @test result.factors[1].name == :T
    end
end

@testset "No metrics: expression unchanged" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:R, manifold=:M4, rank=(0,4), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        R = Tensor(:R, [down(:a), down(:b), down(:c), down(:d)])
        @test contract_metrics(R) == R
    end
end

@testset "Same-position delta → metric: δ_{ab} = g_{ab}" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[Symmetric(1,2)],
        options=Dict{Symbol,Any}(:is_metric => true)))
    register_tensor!(reg, TensorProperties(
        name=:δ, manifold=:M4, rank=(1,1), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_delta => true)))

    with_registry(reg) do
        # Standalone δ_{db} → g_{db}
        d = Tensor(:δ, [down(:d), down(:b)])
        result = contract_metrics(d)
        @test result isa Tensor
        @test result.name == :g
        @test result.indices == [down(:d), down(:b)]

        # δ^{ab} → g^{ab}
        d2 = Tensor(:δ, [up(:a), up(:b)])
        result2 = contract_metrics(d2)
        @test result2.name == :g
        @test result2.indices == [up(:a), up(:b)]

        # In a product: R * δ_{db} → R * g_{db}
        register_tensor!(reg, TensorProperties(
            name=:RicScalar, manifold=:M4, rank=(0,0), symmetries=Any[],
            options=Dict{Symbol,Any}()))
        rs = Tensor(:RicScalar, TIndex[])
        d3 = Tensor(:δ, [down(:d), down(:b)])
        prod = rs * d3
        result3 = contract_metrics(prod)
        @test result3 isa TProduct
        has_g = any(f -> f isa Tensor && f.name == :g, result3.factors)
        @test has_g
    end
end

@testset "Contraction through sums" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:R, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}()))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        g = Tensor(:g, [up(:a), up(:b)])
        R = Tensor(:R, [down(:b), down(:c)])
        T = Tensor(:T, [down(:b), down(:c)])

        expr = g * (R + T)  # g^{ab} (R_{bc} + T_{bc})
        result = contract_metrics(expr)

        @test result isa TSum
        @test length(result.terms) == 2
    end
end

# ── TDeriv contraction tests (fix for C1/C4: contraction engine skipping TDeriv) ──
# contract_metrics_with_derivatives enables derivative-index contraction (opt-in).
# Default contract_metrics does NOT contract with TDeriv indices (matches xAct default).

@testset "Default contract_metrics does NOT contract with TDeriv" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(0,1), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        g = Tensor(:g, [up(:a), up(:b)])
        dT = TDeriv(down(:a), Tensor(:T, [down(:c)]), :partial)
        expr = g * dT  # g^{ab} ∂_a(T_c)

        result = contract_metrics(expr)
        # Default: metric should NOT contract with derivative index
        @test result isa TProduct
        @test any(f -> f isa Tensor && f.name == :g, result.factors)
        @test any(f -> f isa TDeriv, result.factors)
    end
end

@testset "Metric raises derivative index: g^{ab} ∂_a(T_c) → ∂^b(T_c)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(0,1), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        g = Tensor(:g, [up(:a), up(:b)])
        dT = TDeriv(down(:a), Tensor(:T, [down(:c)]), :partial)
        expr = g * dT  # g^{ab} ∂_a(T_c)

        result = contract_metrics_with_derivatives(expr)
        # Should be ∂^b(T_c): metric raised the derivative index
        @test result isa TDeriv
        @test result.index == up(:b)
        @test result.covd == :partial
        @test result.arg isa Tensor
        @test result.arg.name == :T
        @test result.arg.indices == [down(:c)]
    end
end

@testset "Metric raises CovD index: g^{ab} D_a(T_c) → D^b(T_c)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(0,1), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        g = Tensor(:g, [up(:a), up(:b)])
        DT = TDeriv(down(:a), Tensor(:T, [down(:c)]), :D)
        expr = g * DT  # g^{ab} D_a(T_c)

        result = contract_metrics_with_derivatives(expr)
        @test result isa TDeriv
        @test result.index == up(:b)
        @test result.covd == :D  # covd preserved
    end
end

@testset "Delta contracts with derivative index: δ^a_b ∂^b(T_c) → ∂^a(T_c)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:δ, manifold=:M4, rank=(1,1), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_delta => true)))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(0,1), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        δ = Tensor(:δ, [up(:a), down(:b)])
        dT = TDeriv(up(:b), Tensor(:T, [down(:c)]), :partial)
        expr = δ * dT  # δ^a_b ∂^b(T_c)

        result = contract_metrics_with_derivatives(expr)
        @test result isa TDeriv
        @test result.index.name == :a
        @test result.index.position == Up
        @test result.covd == :partial
    end
end

@testset "Metric with two TDeriv factors: g^{ab} ∂_a(S) ∂_b(T)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:S, manifold=:M4, rank=(0,0), symmetries=Any[],
        options=Dict{Symbol,Any}()))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(0,0), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        g = Tensor(:g, [up(:a), up(:b)])
        dS = TDeriv(down(:a), Tensor(:S, TIndex[]), :partial)
        dT = TDeriv(down(:b), Tensor(:T, TIndex[]), :partial)
        expr = tproduct(1 // 1, TensorExpr[g, dS, dT])

        result = contract_metrics_with_derivatives(expr)
        # First contraction: g^{ab} ∂_a(S) → ∂^b(S), leaving ∂^b(S) * ∂_b(T)
        @test result isa TProduct
        derivs = filter(f -> f isa TDeriv, result.factors)
        @test length(derivs) == 2
        positions = Set(d.index.position for d in derivs)
        @test Up in positions
        @test Down in positions
    end
end

@testset "Nested TDeriv: only outermost index contracts" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(0,1), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        g = Tensor(:g, [up(:a), up(:b)])
        inner = TDeriv(down(:b), Tensor(:T, [down(:c)]), :partial)
        outer = TDeriv(down(:a), inner, :partial)
        expr = g * outer  # g^{ab} ∂_a(∂_b(T_c))

        result = contract_metrics_with_derivatives(expr)
        @test result isa TDeriv
        @test result.index == up(:b)
        @test result.arg isa TDeriv
        @test result.arg.index == down(:b)
    end
end

@testset "No contraction when indices don't match (even with derivatives enabled)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M4, rank=(0,2), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        g = Tensor(:g, [up(:a), up(:b)])
        dT = TDeriv(down(:c), Tensor(:T, [down(:d), down(:e)]), :partial)
        expr = g * dT  # g^{ab} ∂_c(T_{de}): no dummy pair

        result = contract_metrics_with_derivatives(expr)
        @test result isa TProduct
        @test any(f -> f isa Tensor && f.name == :g, result.factors)
        @test any(f -> f isa TDeriv, result.factors)
    end
end
