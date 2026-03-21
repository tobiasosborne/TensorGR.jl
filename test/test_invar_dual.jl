#= Test the dual invariant database for the Invar pipeline.
#
# Dual curvature invariants involve the Hodge dual of the Riemann tensor
# (Levi-Civita contractions).  At degree 2 in d=4, the only independent
# dual invariant is the Pontryagin density P = R *R.
#
# Ground truth: Garcia-Parrado & Martin-Garcia (2007), Sec 4, Level 6;
#               Zakhary & McIntosh (1997) GRG 29, 539.
=#

@testset "Invar Dual Database" begin
    # ---- Shared registry ----
    function dual_db_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # ---- Pontryagin density is in the database ----
    @testset "Pontryagin density in dual database" begin
        cr = get_invar_relations(2, "dual_0_0", 6; dim=4)
        @test cr.degree == 2
        @test cr.case_key == "dual_0_0"
        @test cr.step == 6
        @test cr.dim == 4
    end

    # ---- n_independent == 1 for dual degree 2 at Level 6, d=4 ----
    @testset "Degree-2 dual: 1 independent (Pontryagin)" begin
        cr = get_invar_relations(2, "dual_0_0", 6; dim=4)
        @test cr.n_independent == 1
        @test cr.n_dependent == 2
    end

    # ---- Double-dual Kretschmann reduces ----
    @testset "Double-dual Kretschmann relation stored" begin
        cr = get_invar_relations(2, "dual_0_0", 6; dim=4)
        @test length(cr.relations) >= 1

        # The relation: double-dual Kretschmann -> Kretschmann (coefficient 1)
        rel = cr.relations[1]
        @test rel.lhs == [5, 6, 7, 8, 1, 2, 3, 4]  # Kretschmann contraction
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1 // 1  # coefficient 1
        @test rel.rhs[1][2] == [5, 6, 7, 8, 1, 2, 3, 4]  # Kretschmann
    end

    # ---- list_invar_cases includes dual cases ----
    @testset "list_invar_cases includes dual cases" begin
        cases = list_invar_cases()

        # Should include the dual degree-2 case
        @test any(c -> c.case_key == "dual_0_0" && c.degree == 2 && c.step == 6, cases)

        # Should include higher-degree dual cases
        @test any(c -> c.case_key == "dual_0_0_0" && c.degree == 3, cases)
        @test any(c -> c.case_key == "dual_0_0_0_0" && c.degree == 4, cases)
        @test any(c -> c.case_key == "dual_0_0_0_0_0" && c.degree == 5, cases)

        # All dual cases should have dim=4
        dual_cases = filter(c -> startswith(c.case_key, "dual_"), cases)
        @test !isempty(dual_cases)
        @test all(c -> c.dim == 4, dual_cases)
    end

    # ---- pontryagin_rinv() matches database entry ----
    @testset "pontryagin_rinv matches database" begin
        p = pontryagin_rinv()
        @test p isa DualRInv
        @test p.base.degree == 2

        # Verify the base contraction is Kretschmann-type
        @test p.base.contraction == [5, 6, 7, 8, 1, 2, 3, 4]

        # Verify it has a right dual on factor 2
        @test length(p.dual_positions) == 1
        @test p.dual_positions[1] == (2, :right)

        # pontryagin_rinv_canonical should return the same thing
        pc = pontryagin_rinv_canonical()
        @test pc == p
    end

    # ---- Pontryagin is truly independent (not equal to any non-dual invariant) ----
    @testset "Pontryagin is independent of non-dual invariants" begin
        reg = dual_db_registry()
        with_registry(reg) do
            # Build Pontryagin as TensorExpr
            p = pontryagin_rinv()
            pont_expr = to_tensor_expr(p; registry=reg, metric=:g)
            @test isempty(free_indices(pont_expr))

            # Build the three degree-2 non-dual invariants
            rinvs = degree2_independent_rinvs()
            for rinv in rinvs
                nondual_expr = to_tensor_expr(rinv; registry=reg, metric=:g)
                @test isempty(free_indices(nondual_expr))

                # The Pontryagin density is NOT equal to any non-dual invariant.
                # We verify structurally: Pontryagin contains an epsilon tensor,
                # non-dual invariants do not.
                pont_factors = pont_expr isa TProduct ? pont_expr.factors : [pont_expr]
                nondual_factors = nondual_expr isa TProduct ? nondual_expr.factors : [nondual_expr]

                pont_has_eps = any(f -> f isa Tensor && f.name == :εg, pont_factors)
                nondual_has_eps = any(f -> f isa Tensor && f.name == :εg, nondual_factors)

                @test pont_has_eps == true
                @test nondual_has_eps == false
            end
        end
    end

    # ---- dual_independent_rinvs accessor ----
    @testset "dual_independent_rinvs accessor" begin
        # Degree 2: exactly 1 independent dual invariant (Pontryagin)
        dinvs2 = dual_independent_rinvs(2; dim=4)
        @test length(dinvs2) == 1
        @test dinvs2[1] isa DualRInv
        @test dinvs2[1] == pontryagin_rinv()

        # Degree 3: 2 independent dual invariants
        dinvs3 = dual_independent_rinvs(3; dim=4)
        @test length(dinvs3) == 2
        @test all(d -> d isa DualRInv, dinvs3)
        @test all(d -> d.base.degree == 3, dinvs3)

        # Degree 4: at least 1 representative returned (stubbed)
        dinvs4 = dual_independent_rinvs(4; dim=4)
        @test !isempty(dinvs4)
        @test all(d -> d isa DualRInv, dinvs4)

        # Unsupported dimension
        @test_throws ErrorException dual_independent_rinvs(2; dim=3)
    end

    # ---- Higher-degree database entries have correct counts ----
    @testset "Higher-degree dual counts" begin
        cr3 = get_invar_relations(3, "dual_0_0_0", 6; dim=4)
        @test cr3.n_independent == 2

        cr4 = get_invar_relations(4, "dual_0_0_0_0", 6; dim=4)
        @test cr4.n_independent == 5

        cr5 = get_invar_relations(5, "dual_0_0_0_0_0", 6; dim=4)
        @test cr5.n_independent == 12
    end

    # ---- DualRInv to_tensor_expr for database entries ----
    @testset "Database DualRInv entries produce valid expressions" begin
        reg = dual_db_registry()
        with_registry(reg) do
            # Pontryagin from database
            dinvs = dual_independent_rinvs(2; dim=4)
            pont = dinvs[1]
            expr = to_tensor_expr(pont; registry=reg, metric=:g)

            @test expr isa TProduct
            @test isempty(free_indices(expr))

            # Should contain epsilon
            epsilons = filter(f -> f isa Tensor && f.name == :εg, expr.factors)
            @test length(epsilons) == 1

            # Should contain 2 Riemann tensors
            riems = filter(f -> f isa Tensor && f.name == :Riem, expr.factors)
            @test length(riems) == 2
        end
    end

    # ---- Degree-3 dual invariants produce valid expressions ----
    @testset "Degree-3 dual invariants are valid" begin
        reg = dual_db_registry()
        with_registry(reg) do
            dinvs3 = dual_independent_rinvs(3; dim=4)
            for (i, drinv) in enumerate(dinvs3)
                expr = to_tensor_expr(drinv; registry=reg, metric=:g)
                @test isempty(free_indices(expr))
                @test expr isa TProduct

                # Each should have exactly 1 epsilon (left dual on 1 factor)
                epsilons = filter(f -> f isa Tensor && f.name == :εg, expr.factors)
                @test length(epsilons) == 1

                # Each should have exactly 3 Riemann tensors
                riems = filter(f -> f isa Tensor && f.name == :Riem, expr.factors)
                @test length(riems) == 3
            end
        end
    end
end
