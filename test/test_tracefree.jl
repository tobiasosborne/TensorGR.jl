using TensorGR: enforce_tracefree

@testset "Trace-free enforcement" begin
    @testset "Direct self-trace of trace-free tensor -> zero" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2), symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))
        # Weyl tensor C_{abcd}: trace-free on (1,3) and (2,4)
        register_tensor!(reg, TensorProperties(
            name=:Weyl, manifold=:M4, rank=(0,4),
            symmetries=SymmetrySpec[RiemannSymmetry()],
            tracefree_pairs=[(1,3), (1,4), (2,3), (2,4)],
            options=Dict{Symbol,Any}()))

        with_registry(reg) do
            # C^a_{ba}_{d} — trace on slots 1 and 3 -> zero
            C_traced = Tensor(:Weyl, [up(:a), down(:b), down(:a), down(:d)])
            result = enforce_tracefree(C_traced)
            @test result isa TScalar
            @test result.val == 0
        end
    end

    @testset "Metric contraction reveals trace-free -> zero via simplify" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2), symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))
        register_tensor!(reg, TensorProperties(
            name=:Weyl, manifold=:M4, rank=(0,4),
            symmetries=SymmetrySpec[RiemannSymmetry()],
            tracefree_pairs=[(1,3), (1,4), (2,3), (2,4)],
            options=Dict{Symbol,Any}()))

        with_registry(reg) do
            # g^{ac} C_{abcd} -> contract_metrics raises index -> C^c_{bcd} -> tracefree -> 0
            g_up = Tensor(:g, [up(:a), up(:c)])
            C = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            expr = g_up * C
            result = simplify(expr; registry=reg)
            @test result isa TScalar
            @test result.val == 0
        end
    end

    @testset "Non-tracefree tensor is NOT zeroed" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2), symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))
        # Ricci is NOT trace-free
        register_tensor!(reg, TensorProperties(
            name=:Ric, manifold=:M4, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            options=Dict{Symbol,Any}()))

        with_registry(reg) do
            # g^{ab} Ric_{ab} should NOT be zero
            g_up = Tensor(:g, [up(:a), up(:b)])
            Ric = Tensor(:Ric, [down(:a), down(:b)])
            expr = g_up * Ric
            result = simplify(expr; registry=reg)
            # Should become Ric^a_a, not zero
            @test !(result isa TScalar && result.val == 0)
        end
    end

    @testset "set_tracefree! with auto-detection" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        # rank-(1,1) tensor: auto-detect should produce pair (1,2)
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(1,1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}()))
        set_tracefree!(reg, :T)

        tp = get_tensor(reg, :T)
        @test tp.tracefree_pairs == [(1, 2)]

        with_registry(reg) do
            # T^a_a -> 0
            T_traced = Tensor(:T, [up(:a), down(:a)])
            result = enforce_tracefree(T_traced)
            @test result isa TScalar
            @test result.val == 0

            # T^a_b (no trace) -> unchanged
            T_free = Tensor(:T, [up(:a), down(:b)])
            result2 = enforce_tracefree(T_free)
            @test result2 isa Tensor
            @test result2.name == :T
        end
    end

    @testset "set_tracefree! with explicit pairs" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:W, manifold=:M4, rank=(0,4),
            symmetries=SymmetrySpec[RiemannSymmetry()],
            options=Dict{Symbol,Any}()))
        set_tracefree!(reg, :W; pairs=[(1,3), (2,4)])

        tp = get_tensor(reg, :W)
        @test tp.tracefree_pairs == [(1, 3), (2, 4)]

        with_registry(reg) do
            # W^a_{b a d} — trace on slots 1,3
            W_traced = Tensor(:W, [up(:a), down(:b), down(:a), down(:d)])
            result = enforce_tracefree(W_traced)
            @test result isa TScalar
            @test result.val == 0

            # W_{a b c d} — no trace -> unchanged
            W_free = Tensor(:W, [down(:a), down(:b), down(:c), down(:d)])
            result2 = enforce_tracefree(W_free)
            @test result2 isa Tensor
            @test result2.name == :W
        end
    end

    @testset "Trace-free in product zeroes entire product" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            tracefree_pairs=[(1,2)],
            options=Dict{Symbol,Any}()))
        register_tensor!(reg, TensorProperties(
            name=:V, manifold=:M4, rank=(0,1), symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}()))

        with_registry(reg) do
            # T^a_a * V_b -> 0
            T_traced = Tensor(:T, [up(:a), down(:a)])
            V = Tensor(:V, [down(:b)])
            expr = T_traced * V
            result = enforce_tracefree(expr)
            @test result isa TScalar
            @test result.val == 0
        end
    end

    @testset "Trace-free through sums" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            tracefree_pairs=[(1,2)],
            options=Dict{Symbol,Any}()))
        register_tensor!(reg, TensorProperties(
            name=:S, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}()))

        with_registry(reg) do
            # T^a_a + S^a_a -> 0 + S^a_a = S^a_a
            T_traced = Tensor(:T, [up(:a), down(:a)])
            S_traced = Tensor(:S, [up(:a), down(:a)])
            expr = T_traced + S_traced
            result = enforce_tracefree(expr)
            # T^a_a becomes 0, S^a_a stays; tsum collapses
            @test !(result isa TScalar && result.val == 0)
        end
    end
end
