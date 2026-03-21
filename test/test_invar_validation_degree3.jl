#= Validation test: degree-3 independence of cubic Riemann invariants.
#
# Verifies that exactly 8 independent cubic Riemann invariants exist in d=4,
# matching Fulling, King, Wybourne & Cummins (1992), CQG 9:1151, Table 2.
#
# Tests:
#   1. Count: degree3_independent_rinvs() returns exactly 8
#   2. Distinctness: all 8 are pairwise non-equivalent
#   3. Independence: none of the 8 appears as the LHS of a Bianchi relation
#   4. Completeness: 8 independent + 5 dependent = 13 canonical forms
#   5. Scalar check: each independent RInv has no free indices
#   6. Relation scalarity: LHS and RHS of each Bianchi relation are scalar
#   7. Catalog cross-check: Goroff-Sagnotti matches independent set
#   8. Fulling et al ground truth: n_independent == 8
=#

@testset "Invar Validation: Degree-3 Independence (Fulling et al. 1992)" begin
    # ---- Shared registry ----
    function val3_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # ---- Test 1: Count ----
    @testset "Count: exactly 8 independent degree-3 RInvs" begin
        indep = degree3_independent_rinvs()
        @test length(indep) == 8
    end

    # ---- Test 2: Distinctness ----
    @testset "Distinctness: all 8 are pairwise non-equivalent" begin
        reg = val3_registry()
        indep = degree3_independent_rinvs()
        with_registry(reg) do
            for i in 1:8, j in i+1:8
                @test !are_equivalent(indep[i], indep[j], reg)
            end
        end
    end

    # ---- Test 3: Independence ----
    @testset "Independence: no independent RInv is a dependent LHS" begin
        cr2 = get_invar_relations(3, "0_0_0", 2)
        dep_lhs_set = Set(rel.lhs for rel in cr2.relations)
        indep = degree3_independent_rinvs()

        for (k, rinv) in enumerate(indep)
            @test !(rinv.contraction in dep_lhs_set)
        end
    end

    # ---- Test 4: Completeness ----
    @testset "Completeness: 8 independent + 5 dependent = 13 total" begin
        cr2 = get_invar_relations(3, "0_0_0", 2)
        @test cr2.n_independent + cr2.n_dependent == 13

        # Every canonical form is either independent or dependent
        canonical = degree3_canonical_rinvs()
        indep = degree3_independent_rinvs()
        dep_lhs_set = Set(rel.lhs for rel in cr2.relations)
        indep_set = Set(r.contraction for r in indep)

        for r in canonical
            in_indep = r.contraction in indep_set
            in_dep = r.contraction in dep_lhs_set
            @test in_indep || in_dep
            # Should be in exactly one category
            @test in_indep != in_dep
        end
    end

    # ---- Test 5: Scalar check ----
    @testset "Scalar check: all 8 independent RInvs have no free indices" begin
        reg = val3_registry()
        with_registry(reg) do
            for rinv in degree3_independent_rinvs()
                expr = to_tensor_expr(rinv; registry=reg, metric=:g)
                fi = free_indices(expr)
                @test isempty(fi)
            end
        end
    end

    # ---- Test 6: Relation scalarity ----
    @testset "Relation scalarity: LHS and RHS of each Bianchi relation are scalar" begin
        reg = val3_registry()
        cr2 = get_invar_relations(3, "0_0_0", 2)
        with_registry(reg) do
            for rel in cr2.relations
                # LHS
                lhs_rinv = RInv(3, rel.lhs, true)
                lhs_expr = to_tensor_expr(lhs_rinv; registry=reg, metric=:g)
                @test isempty(free_indices(lhs_expr))

                # RHS: each term in the linear combination
                for (coeff, contraction) in rel.rhs
                    rhs_rinv = RInv(3, contraction, true)
                    rhs_expr = to_tensor_expr(rhs_rinv; registry=reg, metric=:g)
                    @test isempty(free_indices(rhs_expr))
                end
            end
        end
    end

    # ---- Test 7: Catalog cross-check ----
    @testset "Catalog cross-check: Goroff-Sagnotti matches I9 in independent set" begin
        reg = val3_registry()
        indep = degree3_independent_rinvs()
        indep_contractions = Set(r.contraction for r in indep)

        with_registry(reg) do
            # The Goroff-Sagnotti invariant from the catalog
            gs_expr = curvature_invariant(:Riem_cubed; registry=reg, manifold=:M4, metric=:g)
            gs_rinv = from_tensor_expr(gs_expr; registry=reg, metric=:g)
            gs_canon = canonicalize(gs_rinv)

            # I9 contraction vector (Goroff-Sagnotti)
            I9_contraction = [5, 6, 9, 10, 1, 2, 11, 12, 3, 4, 7, 8]
            @test gs_canon.contraction == I9_contraction

            # Must be in the independent set
            @test gs_canon.contraction in indep_contractions

            # Cross-check via are_equivalent against I9 from independent list
            # I9 is the 7th entry (index 7) in the independent list
            @test are_equivalent(gs_rinv, indep[7], reg)
        end
    end

    # ---- Test 8: Fulling et al ground truth ----
    @testset "Fulling et al (1992) ground truth: n_independent == 8 in d=4" begin
        cr2 = get_invar_relations(3, "0_0_0", 2)
        # Fulling et al. (1992), Table 2, order p=3: 8 independent invariants
        @test cr2.n_independent == 8

        # The database is dimension-independent (valid for any d >= 4)
        @test cr2.dim === nothing

        # Verify the relations count: 5 Bianchi relations reduce 13 to 8
        @test length(cr2.relations) == 5
        @test cr2.n_dependent == 5
    end
end
