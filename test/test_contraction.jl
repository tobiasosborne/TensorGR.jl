using TensorGR: contract_metrics, free_indices, indices

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
