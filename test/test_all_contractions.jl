using TensorGR: all_contractions, free_indices

@testset "all_contractions(expr, metric): scalar input" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        s = TScalar(42 // 1)
        results = all_contractions(s, :g; registry=reg)
        @test length(results) == 1
        @test results[1] == s
    end
end

@testset "all_contractions(expr, metric): rank-2 symmetric tensor" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,2),
        symmetries=Any[Symmetric(1,2)]))

    with_registry(reg) do
        T = Tensor(:T, [down(:a), down(:b)])
        results = all_contractions(T, :g; registry=reg)
        # Rank-2 symmetric: only 1 pairing (a,b) -> gives the trace g^{ab} T_{ab}
        @test length(results) == 1
        # Result should be a scalar (no free indices)
        @test isempty(free_indices(results[1]))
    end
end

@testset "all_contractions(expr, metric): rank-4 non-symmetric tensor" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,4),
        symmetries=Any[]))

    with_registry(reg) do
        T = Tensor(:T, [down(:a), down(:b), down(:c), down(:d)])
        results = all_contractions(T, :g; registry=reg)
        # 4 indices -> 3!! = 3 pairings: (ab)(cd), (ac)(bd), (ad)(bc)
        # Non-symmetric tensor: all 3 should be distinct
        @test length(results) == 3
        for r in results
            @test isempty(free_indices(r))
            @test r != TScalar(0 // 1)
        end
    end
end

@testset "all_contractions(expr, metric): odd-rank error" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)
    register_tensor!(reg, TensorProperties(name=:V, manifold=:M4, rank=(0,1),
        symmetries=Any[]))

    with_registry(reg) do
        V = Tensor(:V, [down(:a)])
        @test_throws ArgumentError all_contractions(V, :g; registry=reg)
    end
end

@testset "all_contractions(expr, metric): rank-4 Riemann symmetry tensor" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        # Weyl has RiemannSymmetry: antisym on (1,2), (3,4), sym on pair-swap
        # 3 pairings, but RiemannSymmetry identifies some -> fewer distinct contractions
        C = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
        results = all_contractions(C, :g; registry=reg)
        @test length(results) >= 1
        @test length(results) <= 3
        for r in results
            @test isempty(free_indices(r))
        end
    end
end

@testset "all_contractions(expr, metric): rank-2 product" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)
    register_tensor!(reg, TensorProperties(name=:V, manifold=:M4, rank=(0,1),
        symmetries=Any[]))

    with_registry(reg) do
        Va = Tensor(:V, [down(:a)])
        Vb = Tensor(:V, [down(:b)])
        expr = Va * Vb  # V_a V_b: rank-2 expression
        results = all_contractions(expr, :g; registry=reg)
        # 1 pairing: (a,b) -> g^{ab} V_a V_b = V^a V_a
        @test length(results) == 1
        @test isempty(free_indices(results[1]))
    end
end
