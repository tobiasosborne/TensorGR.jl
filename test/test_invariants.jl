@testset "Curvature Invariant Catalog" begin
    # Helper: build a registry with a 4D manifold and curvature tensors
    function invariant_registry(; dim=4)
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=dim metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # ─── InvariantEntry struct ─────────────────────────────────────────
    @testset "InvariantEntry struct" begin
        entry = INVARIANT_CATALOG[:R]
        @test entry isa InvariantEntry
        @test entry.name == :R
        @test entry.order == 1
        @test entry.min_dim == 2
        @test entry.description isa String
        @test entry.expression_fn isa Function
    end

    # ─── Catalog completeness ──────────────────────────────────────────
    @testset "Catalog completeness" begin
        # Total count: 1 (order 1) + 4 (order 2) + 6 (order 3) = 11
        @test length(INVARIANT_CATALOG) == 11

        # Order-1 invariants
        order1_names = [:R]
        for n in order1_names
            @test haskey(INVARIANT_CATALOG, n)
            @test INVARIANT_CATALOG[n].order == 1
        end

        # Order-2 invariants
        order2_names = [:R_sq, :Ric_sq, :Kretschmann, :Weyl_sq]
        for n in order2_names
            @test haskey(INVARIANT_CATALOG, n)
            @test INVARIANT_CATALOG[n].order == 2
        end

        # Order-3 invariants (I1-I6)
        order3_names = [:R_cubed, :R_Ric_sq, :Ric_cubed,
                        :R_Kretschmann, :Ric_Riem_sq, :Riem_cubed]
        for n in order3_names
            @test haskey(INVARIANT_CATALOG, n)
            @test INVARIANT_CATALOG[n].order == 3
        end
    end

    # ─── list_invariants ───────────────────────────────────────────────
    @testset "list_invariants" begin
        all_inv = list_invariants()
        @test length(all_inv) == 11
        @test all(e -> e isa NamedTuple, all_inv)
        # Each entry has the expected fields
        @test all(e -> haskey(e, :name) && haskey(e, :order) && haskey(e, :description), all_inv)

        # Filter by order
        inv1 = list_invariants(order=1)
        @test length(inv1) == 1
        @test inv1[1].name == :R

        inv2 = list_invariants(order=2)
        @test length(inv2) == 4
        @test all(e -> e.order == 2, inv2)

        inv3 = list_invariants(order=3)
        @test length(inv3) == 6
        @test all(e -> e.order == 3, inv3)

        # Filtering with a non-existent order returns empty
        inv99 = list_invariants(order=99)
        @test isempty(inv99)
    end

    # ─── curvature_invariant: order-1 ─────────────────────────────────
    @testset "Order-1: Ricci scalar R" begin
        reg = invariant_registry()
        with_registry(reg) do
            R = curvature_invariant(:R; registry=reg, manifold=:M4, metric=:g)
            @test R isa Tensor
            @test R.name == :RicScalar
            @test isempty(R.indices)   # scalar = no indices
            @test isempty(free_indices(R))
        end
    end

    # ─── curvature_invariant: order-2 ─────────────────────────────────
    @testset "Order-2 invariants" begin
        reg = invariant_registry()
        with_registry(reg) do
            # R^2: product of two RicScalar
            R2 = curvature_invariant(:R_sq; registry=reg, manifold=:M4, metric=:g)
            @test R2 isa TProduct
            @test isempty(free_indices(R2))
            scalars = filter(f -> f isa Tensor && f.name == :RicScalar, R2.factors)
            @test length(scalars) == 2

            # Ric^2: Ric_{ab} Ric^{ab}
            Ric2 = curvature_invariant(:Ric_sq; registry=reg, manifold=:M4, metric=:g)
            @test Ric2 isa TProduct
            @test isempty(free_indices(Ric2))
            rics = filter(f -> f isa Tensor && f.name == :Ric, Ric2.factors)
            @test length(rics) == 2

            # Kretschmann: Riem_{abcd} Riem^{abcd}
            K = curvature_invariant(:Kretschmann; registry=reg, manifold=:M4, metric=:g)
            @test K isa TProduct
            @test isempty(free_indices(K))
            riems = filter(f -> f isa Tensor && f.name == :Riem, K.factors)
            @test length(riems) == 2

            # Weyl^2: Weyl_{abcd} Weyl^{abcd}
            W2 = curvature_invariant(:Weyl_sq; registry=reg, manifold=:M4, metric=:g)
            @test W2 isa TProduct
            @test isempty(free_indices(W2))
            weyls = filter(f -> f isa Tensor && f.name == :Weyl, W2.factors)
            @test length(weyls) == 2
        end
    end

    # ─── curvature_invariant: order-3 ─────────────────────────────────
    @testset "Order-3 invariants" begin
        reg = invariant_registry()
        with_registry(reg) do
            # All order-3 invariants should be scalar (no free indices)
            order3_names = [:R_cubed, :R_Ric_sq, :Ric_cubed,
                            :R_Kretschmann, :Ric_Riem_sq, :Riem_cubed]
            for name in order3_names
                expr = curvature_invariant(name; registry=reg, manifold=:M4, metric=:g)
                @test expr isa TProduct
                @test isempty(free_indices(expr))
            end

            # R^3: should have three RicScalar factors
            R3 = curvature_invariant(:R_cubed; registry=reg, manifold=:M4, metric=:g)
            scalars = filter(f -> f isa Tensor && f.name == :RicScalar, R3.factors)
            @test length(scalars) == 3

            # R * Ric^2: should contain RicScalar and Ric and metric
            R_Ric2 = curvature_invariant(:R_Ric_sq; registry=reg, manifold=:M4, metric=:g)
            has_scalar = any(f -> f isa Tensor && f.name == :RicScalar, R_Ric2.factors)
            has_ric = any(f -> f isa Tensor && f.name == :Ric, R_Ric2.factors)
            @test has_scalar
            @test has_ric

            # Ric^3: should contain three Ric factors
            Ric3 = curvature_invariant(:Ric_cubed; registry=reg, manifold=:M4, metric=:g)
            rics = filter(f -> f isa Tensor && f.name == :Ric, Ric3.factors)
            @test length(rics) == 3

            # Riem^3 (Goroff-Sagnotti): should contain three Riem factors
            Riem3 = curvature_invariant(:Riem_cubed; registry=reg, manifold=:M4, metric=:g)
            riems = filter(f -> f isa Tensor && f.name == :Riem, Riem3.factors)
            @test length(riems) == 3
        end
    end

    # ─── Dimension checks ─────────────────────────────────────────────
    @testset "Dimension constraints" begin
        # 2D manifold: should accept R, R_sq, Ric_sq, but reject Weyl_sq (min_dim=4)
        reg2 = invariant_registry(dim=2)
        with_registry(reg2) do
            # These should work in 2D
            @test curvature_invariant(:R; registry=reg2, manifold=:M4, metric=:g) isa Tensor
            @test curvature_invariant(:R_sq; registry=reg2, manifold=:M4, metric=:g) isa TProduct
            @test curvature_invariant(:Kretschmann; registry=reg2, manifold=:M4, metric=:g) isa TProduct

            # These require dim >= 4
            @test_throws ErrorException curvature_invariant(:Weyl_sq; registry=reg2, manifold=:M4, metric=:g)
            @test_throws ErrorException curvature_invariant(:R_Kretschmann; registry=reg2, manifold=:M4, metric=:g)
            @test_throws ErrorException curvature_invariant(:Ric_Riem_sq; registry=reg2, manifold=:M4, metric=:g)
            @test_throws ErrorException curvature_invariant(:Riem_cubed; registry=reg2, manifold=:M4, metric=:g)
        end
    end

    # ─── Unknown invariant error ───────────────────────────────────────
    @testset "Unknown invariant error" begin
        reg = invariant_registry()
        with_registry(reg) do
            @test_throws ErrorException curvature_invariant(:NonExistent; registry=reg)
        end
    end

    # ─── Gauss-Bonnet syzygy applied to catalog invariants ─────────────
    @testset "Gauss-Bonnet applied to Kretschmann" begin
        reg = invariant_registry()
        for r in gauss_bonnet_rule(; metric=:g)
            register_rule!(reg, r)
        end
        with_registry(reg) do
            K = curvature_invariant(:Kretschmann; registry=reg, manifold=:M4, metric=:g)
            result = simplify(K; registry=reg)

            # After Gauss-Bonnet, Kretschmann (Riem^2) should be
            # replaced by 4 Ric^2 - R^2, so no Riem should remain
            has_riem = false
            walk(result) do node
                if node isa Tensor && node.name == :Riem
                    has_riem = true
                end
                node
            end
            @test !has_riem

            # Should still be a scalar expression
            @test isempty(free_indices(result))
        end
    end

    # ─── Invariant min_dim metadata ────────────────────────────────────
    @testset "min_dim metadata" begin
        # Order-1 and simple order-2 need only dim >= 2
        @test INVARIANT_CATALOG[:R].min_dim == 2
        @test INVARIANT_CATALOG[:R_sq].min_dim == 2
        @test INVARIANT_CATALOG[:Ric_sq].min_dim == 2
        @test INVARIANT_CATALOG[:Kretschmann].min_dim == 2

        # Weyl requires dim >= 4
        @test INVARIANT_CATALOG[:Weyl_sq].min_dim == 4

        # Invariants involving Riem with many indices need dim >= 4
        @test INVARIANT_CATALOG[:R_Kretschmann].min_dim >= 2
        @test INVARIANT_CATALOG[:Riem_cubed].min_dim >= 4
    end

    # ─── Descriptions are non-empty ────────────────────────────────────
    @testset "Descriptions are non-empty" begin
        for (name, entry) in INVARIANT_CATALOG
            @test !isempty(entry.description)
        end
    end
end
