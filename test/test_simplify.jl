@testset "Simplify Pipeline" begin
    @testset "Basic simplify" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M4, rank=(1,1),
            symmetries=Any[],
            options=Dict{Symbol,Any}(:is_delta => true)))
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)]))

        with_registry(reg) do
            # g^{ab} g_{bc} should simplify to δ^a_c
            g_up = Tensor(:g, [up(:a), up(:b)])
            g_dn = Tensor(:g, [down(:b), down(:c)])
            expr = g_up * g_dn
            result = simplify(expr)
            # Should be a delta
            @test result isa Tensor
            @test result.name == :δ
        end
    end

    @testset "Simplify with rules" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))

        A = Tensor(:A, TIndex[])
        B = Tensor(:B, TIndex[])
        register_rule!(reg, RewriteRule(A, B))

        with_registry(reg) do
            result = simplify(A)
            @test result == B
        end
    end

    @testset "Simplify to fixed point" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M4, rank=(1,1),
            symmetries=Any[],
            options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            # δ^a_a → 4 (dimension)
            delta = Tensor(:δ, [up(:a), down(:a)])
            result = simplify(delta)
            # contract_metrics on a standalone delta should work via canonicalize path
            # Actually contract_metrics on a single Tensor delta checks self-trace
            @test result == TScalar(4 // 1)
        end
    end

    @testset "Simplify sum cancellation" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)]))

        with_registry(reg) do
            T_ab = Tensor(:T, [down(:a), down(:b)])
            # T_{ab} - T_{ab} → 0
            expr = T_ab - T_ab
            result = simplify(expr)
            @test result == TScalar(0 // 1)
        end
    end
end
