using TensorGR: all_contractions, filter_independent_contractions, free_indices

@testset "Contraction Filtering" begin

    @testset "filter_independent_contractions: empty input" begin
        @test filter_independent_contractions(TensorExpr[]) == TensorExpr[]
    end

    @testset "filter_independent_contractions: symmetric rank-2 tensor" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
        define_metric!(reg, :g; manifold=:M4)
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)]))

        with_registry(reg) do
            T = Tensor(:T, [down(:a), down(:b)])
            # Only 1 contraction for rank-2 -> filtering should keep it
            unfiltered = all_contractions(T, :g; registry=reg, filter=false)
            @test length(unfiltered) == 1
            filtered = all_contractions(T, :g; registry=reg, filter=true)
            @test length(filtered) == 1
        end
    end

    @testset "filter_independent_contractions: Riemann tensor contractions" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
        define_metric!(reg, :g; manifold=:M4)

        with_registry(reg) do
            R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])

            # Without filtering: get all pairings that survive basic dedup
            unfiltered = all_contractions(R, :g; registry=reg, filter=false)
            # With filtering: Riemann symmetries reduce independent contractions
            filtered = all_contractions(R, :g; registry=reg, filter=true)

            # The Riemann tensor has 3 pairings: (ab)(cd), (ac)(bd), (ad)(bc)
            # Due to antisymmetry R_{abcd} = -R_{bacd} = -R_{abdc}:
            #   g^{ab}g^{cd}R_{abcd} = 0 (trace over antisymmetric pair)
            #   g^{ad}g^{bc}R_{abcd} = 0 (trace over antisymmetric pair)
            #   g^{ac}g^{bd}R_{abcd} = R (Ricci scalar)
            # So the basic dedup already removes zeros, filtering should keep <= unfiltered
            @test length(filtered) <= length(unfiltered)

            # All results should be fully contracted (scalar)
            for r in filtered
                @test isempty(free_indices(r))
            end
        end
    end

    @testset "filter_independent_contractions: non-symmetric rank-4 tensor" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
        define_metric!(reg, :g; manifold=:M4)
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,4),
            symmetries=Any[]))

        with_registry(reg) do
            T = Tensor(:T, [down(:a), down(:b), down(:c), down(:d)])

            # Non-symmetric rank-4: all 3 pairings should be independent
            unfiltered = all_contractions(T, :g; registry=reg, filter=false)
            filtered = all_contractions(T, :g; registry=reg, filter=true)
            @test length(unfiltered) == 3
            @test length(filtered) == 3
        end
    end

    @testset "filter_independent_contractions: Weyl tensor (RiemannSymmetry)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
        define_metric!(reg, :g; manifold=:M4)

        with_registry(reg) do
            C = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])

            # Weyl has RiemannSymmetry: antisym on (1,2) and (3,4), sym on pair-swap.
            # Of 3 pairings, some vanish by antisymmetry (trace over antisymmetric pair).
            # Filtering should not increase the count vs unfiltered.
            unfiltered = all_contractions(C, :g; registry=reg, filter=false)
            filtered = all_contractions(C, :g; registry=reg, filter=true)
            @test length(filtered) <= length(unfiltered)
            for r in filtered
                @test isempty(free_indices(r))
            end
        end
    end

    @testset "filter_independent_contractions: standalone function" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
        define_metric!(reg, :g; manifold=:M4)

        with_registry(reg) do
            # Build some contractions manually and verify the filter works standalone
            R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            raw = all_contractions(R, :g; registry=reg, filter=false)

            # Calling filter_independent_contractions directly
            filtered = filter_independent_contractions(raw; registry=reg)
            @test length(filtered) <= length(raw)

            # Every result should be scalar
            for r in filtered
                @test isempty(free_indices(r))
            end
        end
    end

    @testset "filter=true is default for all_contractions" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
        define_metric!(reg, :g; manifold=:M4)
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)]))

        with_registry(reg) do
            T = Tensor(:T, [down(:a), down(:b)])
            # Default call (filter=true) should match explicit filter=true
            default_result = all_contractions(T, :g; registry=reg)
            explicit_result = all_contractions(T, :g; registry=reg, filter=true)
            @test length(default_result) == length(explicit_result)
        end
    end

    @testset "contraction_ansatz uses filtered contractions" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
        define_metric!(reg, :g; manifold=:M4)

        with_registry(reg) do
            # Ric x Ric: 2 independent invariants (R_{ab}R^{ab} and R^2)
            result = contraction_ansatz([:Ric, :Ric], :g; registry=reg)
            @test result isa TSum
            @test length(result.terms) == 2
        end
    end

end
