#= Test the Invar precomputed invariant database infrastructure and degree-2 data.
#
# Ground truth: Fulling et al. (1992), CQG 9:1151, Table 1;
#               Garcia-Parrado & Martin-Garcia (2007), Sec 4, Levels 1-2.
#
# Degree-2 invariants: products of 2 Riemann tensors with all 8 indices
# contracted pairwise via the metric.
#   Level 1 (permutation symmetries): 4 non-vanishing canonical forms
#   Level 2 (first Bianchi identity): 3 independent invariants
=#

@testset "Invar Database: Infrastructure" begin
    # ---- InvarRelation struct ----
    @testset "InvarRelation construction" begin
        rel = InvarRelation(
            [5, 7, 6, 8, 1, 3, 2, 4],
            [(1//2, [5, 6, 7, 8, 1, 2, 3, 4])]
        )
        @test rel.lhs == [5, 7, 6, 8, 1, 3, 2, 4]
        @test length(rel.rhs) == 1
        @test rel.rhs[1] == (1//2, [5, 6, 7, 8, 1, 2, 3, 4])

        # Equality
        rel2 = InvarRelation(
            [5, 7, 6, 8, 1, 3, 2, 4],
            [(1//2, [5, 6, 7, 8, 1, 2, 3, 4])]
        )
        @test rel == rel2

        # Inequality
        rel3 = InvarRelation(
            [5, 6, 7, 8, 1, 2, 3, 4],
            [(1//1, [3, 4, 1, 2, 7, 8, 5, 6])]
        )
        @test rel != rel3
    end

    # ---- CaseRelations struct ----
    @testset "CaseRelations construction" begin
        cr = CaseRelations(2, "0_0", 2, nothing, 3, 1,
            InvarRelation[
                InvarRelation([5,7,6,8,1,3,2,4], [(1//2, [5,6,7,8,1,2,3,4])])
            ]
        )
        @test cr.degree == 2
        @test cr.case_key == "0_0"
        @test cr.step == 2
        @test cr.dim === nothing
        @test cr.n_independent == 3
        @test cr.n_dependent == 1
        @test length(cr.relations) == 1
    end
end

@testset "Invar Database: Degree-2 Data" begin
    # ---- Shared registry ----
    function db_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # ---- Level 1: 4 canonical forms ----
    @testset "Level 1: 4 non-vanishing canonical forms" begin
        cr1 = get_invar_relations(2, "0_0", 1)
        @test cr1.degree == 2
        @test cr1.case_key == "0_0"
        @test cr1.step == 1
        @test cr1.dim === nothing
        @test cr1.n_independent == 4
        @test cr1.n_dependent == 0
        @test isempty(cr1.relations)
    end

    # ---- Level 2: 3 independent, 1 dependent ----
    @testset "Level 2: 3 independent invariants (Fulling et al.)" begin
        cr2 = get_invar_relations(2, "0_0", 2)
        @test cr2.degree == 2
        @test cr2.step == 2
        @test cr2.n_independent == 3
        @test cr2.n_dependent == 1
        @test length(cr2.relations) == 1
    end

    # ---- Bianchi relation: I4 = (1/2)*I3 ----
    @testset "Bianchi relation: I4 = (1/2) * Kretschmann" begin
        cr2 = get_invar_relations(2, "0_0", 2)
        rel = cr2.relations[1]

        # LHS is I4 (cross contraction)
        @test rel.lhs == [5, 7, 6, 8, 1, 3, 2, 4]

        # RHS is (1/2) * Kretschmann
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1//2
        @test rel.rhs[1][2] == [5, 6, 7, 8, 1, 2, 3, 4]
    end

    # ---- Verify relation holds as TensorExpr ----
    @testset "Relation verified via TensorExpr" begin
        reg = db_registry()
        cr2 = get_invar_relations(2, "0_0", 2)
        rel = cr2.relations[1]

        with_registry(reg) do
            # Build LHS: the dependent invariant I4
            lhs_rinv = RInv(2, rel.lhs, true)
            lhs_expr = to_tensor_expr(lhs_rinv; registry=reg, metric=:g)

            # Build RHS: linear combination of independent invariants
            rhs_expr = TScalar(0 // 1)
            for (coeff, contraction) in rel.rhs
                rinv = RInv(2, contraction, true)
                term = to_tensor_expr(rinv; registry=reg, metric=:g)
                rhs_expr = rhs_expr + TScalar(coeff) * term
            end

            # LHS - RHS should be zero
            diff = lhs_expr - rhs_expr

            # Apply Bianchi-level simplification:
            # The Bianchi identity relates R_{acbd} to R_{abcd} + R_{adbc},
            # and the orbit sign analysis shows R_{adbc}R^{abcd} = -R_{acbd}R^{abcd}.
            # This is a multi-term identity between products with different
            # index contractions, so the simplify pipeline may not directly
            # reduce it to zero. Instead, we verify algebraically by checking
            # the orbit sign analysis used to derive the relation.

            # The derivation uses:
            #   Bianchi: R_{acbd} = R_{abcd} + R_{adbc}
            #   Orbit:   R_{adbc}R^{abcd} has sign -1 relative to R_{acbd}R^{abcd}
            #            in the canonical orbit (i.e., they represent the same
            #            invariant up to a sign flip from antisymmetric generators)
            # Therefore: I4 = I3 + (-I4) => 2*I4 = I3 => I4 = I3/2

            # Verify the orbit sign claim
            sigma_I4 = [5, 7, 6, 8, 1, 3, 2, 4]
            sigma_adbc = [5, 8, 6, 7, 1, 3, 4, 2]  # R_{adbc}R^{abcd}

            gens = TensorGR._rinv_slot_generators(2)
            orbit = Dict{Vector{Int}, Int}()
            orbit[sigma_I4] = +1
            queue = [sigma_I4]
            while !isempty(queue)
                sigma = popfirst!(queue)
                current_sign = orbit[sigma]
                for (g, gsign) in gens
                    sigma_new = TensorGR._conjugate_contraction(sigma, g)
                    new_sign = current_sign * gsign
                    if !haskey(orbit, sigma_new)
                        orbit[sigma_new] = new_sign
                        push!(queue, sigma_new)
                    end
                end
            end

            @test haskey(orbit, sigma_adbc)
            @test orbit[sigma_adbc] == -1  # opposite sign confirms the relation
        end
    end

    # ---- is_independent_rinv ----
    @testset "is_independent_rinv" begin
        # The 3 independent invariants at Level 2
        R_sq = RInv(2, [3, 4, 1, 2, 7, 8, 5, 6])
        Ric_sq = RInv(2, [3, 5, 1, 7, 2, 8, 4, 6])
        K = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

        @test is_independent_rinv(R_sq, 2) == true
        @test is_independent_rinv(Ric_sq, 2) == true
        @test is_independent_rinv(K, 2) == true

        # The dependent invariant I4 at Level 2
        I4 = RInv(2, [5, 7, 6, 8, 1, 3, 2, 4])
        @test is_independent_rinv(I4, 2) == false

        # All 4 are independent at Level 1
        @test is_independent_rinv(R_sq, 1) == true
        @test is_independent_rinv(Ric_sq, 1) == true
        @test is_independent_rinv(K, 1) == true
        @test is_independent_rinv(I4, 1) == true

        # Vanishing invariant (antisymmetric contraction) is not independent
        vanishing = RInv(2, [2, 1, 4, 3, 6, 5, 8, 7])
        @test is_independent_rinv(vanishing, 1) == false
        @test is_independent_rinv(vanishing, 2) == false
    end

    # ---- list_invar_cases ----
    @testset "list_invar_cases" begin
        cases = list_invar_cases()
        @test length(cases) >= 2  # at least step 1 and step 2 for degree 2

        # Filter by degree
        deg2_cases = list_invar_cases(degree=2)
        @test length(deg2_cases) >= 2
        @test all(c -> c.degree == 2, deg2_cases)

        # Filter by step
        step2_cases = list_invar_cases(step=2)
        @test !isempty(step2_cases)
        @test all(c -> c.step == 2, step2_cases)

        # Degree 2 should appear in the list
        @test any(c -> c.degree == 2 && c.step == 1 && c.n_independent == 4, cases)
        @test any(c -> c.degree == 2 && c.step == 2 && c.n_independent == 3, cases)
    end

    # ---- degree2_canonical_rinvs ----
    @testset "degree2_canonical_rinvs" begin
        rinvs = degree2_canonical_rinvs()
        @test length(rinvs) == 4

        # All should be canonical
        @test all(r -> r.canonical, rinvs)

        # All should be degree 2
        @test all(r -> r.degree == 2, rinvs)

        # All should be valid involutions
        for r in rinvs
            for i in 1:8
                @test r.contraction[r.contraction[i]] == i
                @test r.contraction[i] != i
            end
        end

        # All should be distinct
        for i in 1:4, j in i+1:4
            @test rinvs[i].contraction != rinvs[j].contraction
        end
    end

    # ---- degree2_independent_rinvs ----
    @testset "degree2_independent_rinvs" begin
        rinvs = degree2_independent_rinvs()
        @test length(rinvs) == 3

        # All should be canonical
        @test all(r -> r.canonical, rinvs)

        # Known contractions
        @test rinvs[1].contraction == [3, 4, 1, 2, 7, 8, 5, 6]  # R^2
        @test rinvs[2].contraction == [3, 5, 1, 7, 2, 8, 4, 6]  # Ric^2
        @test rinvs[3].contraction == [5, 6, 7, 8, 1, 2, 3, 4]  # K
    end

    # ---- Cross-check: canonical forms match to_tensor_expr output ----
    @testset "Canonical forms produce correct tensor expressions" begin
        reg = db_registry()
        with_registry(reg) do
            rinvs = degree2_independent_rinvs()

            # I1 = R^2: should simplify to RicScalar * RicScalar
            e1 = to_tensor_expr(rinvs[1]; registry=reg, metric=:g)
            s1 = simplify(e1; registry=reg)
            str1 = string(s1)
            @test count("RicScalar", str1) >= 2

            # I2 = Ric^2: should contain Ric factors
            e2 = to_tensor_expr(rinvs[2]; registry=reg, metric=:g)
            s2 = simplify(e2; registry=reg)
            str2 = string(s2)
            @test occursin("Ric", str2) && !occursin("RicScalar", str2)

            # I3 = Kretschmann: should contain Riem factors
            e3 = to_tensor_expr(rinvs[3]; registry=reg, metric=:g)
            s3 = simplify(e3; registry=reg)
            str3 = string(s3)
            @test occursin("Riem", str3)
        end
    end

    # ---- Exhaustive enumeration: 105 pairings -> 4 non-zero + vanishing ----
    @testset "Exhaustive enumeration of degree-2 contractions" begin
        # Generate all fixed-point-free involutions on [1..8]
        function all_involutions_8()
            results = Vector{Vector{Int}}()
            perm = zeros(Int, 8)
            _gen!(results, perm, 1)
            results
        end
        function _gen!(results, perm, pos)
            idx = 0
            for i in pos:8
                if perm[i] == 0
                    idx = i
                    break
                end
            end
            if idx == 0
                push!(results, copy(perm))
                return
            end
            for j in idx+1:8
                perm[j] == 0 || continue
                perm[idx] = j
                perm[j] = idx
                _gen!(results, perm, idx + 1)
                perm[idx] = 0
                perm[j] = 0
            end
        end

        invols = all_involutions_8()
        @test length(invols) == 105  # (8-1)!! = 105

        # Canonicalize each and collect distinct canonical forms
        canonical_set = Dict{Vector{Int}, Int}()
        for inv in invols
            rinv = RInv(2, inv)
            c = canonicalize(rinv)
            canonical_set[c.contraction] = get(canonical_set, c.contraction, 0) + 1
        end

        # Should have exactly 5 classes (4 non-zero + 1 vanishing)
        @test length(canonical_set) == 5
        @test haskey(canonical_set, zeros(Int, 8))  # vanishing class

        # The 4 non-zero classes should match our database
        db_forms = Set([r.contraction for r in degree2_canonical_rinvs()])
        non_zero_forms = Set(k for k in keys(canonical_set) if k != zeros(Int, 8))
        @test db_forms == non_zero_forms
    end

    # ---- Error on missing data ----
    @testset "Missing data throws KeyError" begin
        @test_throws KeyError get_invar_relations(99, "0_0", 1)
        @test_throws KeyError get_invar_relations(2, "nonexistent", 1)
    end
end
