#= Test the Invar degree-3 algebraic invariant database.
#
# Ground truth: Fulling et al. (1992), CQG 9:1151, Table 2;
#               Garcia-Parrado & Martin-Garcia (2007), Sec 4, Levels 1-2.
#
# Degree-3 invariants: products of 3 Riemann tensors with all 12 indices
# contracted pairwise via the metric.
#   Level 1 (permutation symmetries): 13 non-vanishing canonical forms
#   Level 2 (first Bianchi identity): 8 independent invariants
=#

@testset "Invar Database: Degree-3 Data" begin
    # ---- Shared registry ----
    function db3_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # ---- Level 1: 13 canonical forms ----
    @testset "Level 1: 13 non-vanishing canonical forms" begin
        cr1 = get_invar_relations(3, "0_0_0", 1)
        @test cr1.degree == 3
        @test cr1.case_key == "0_0_0"
        @test cr1.step == 1
        @test cr1.dim === nothing
        @test cr1.n_independent == 13
        @test cr1.n_dependent == 0
        @test isempty(cr1.relations)
    end

    # ---- Level 2: 8 independent, 5 dependent ----
    @testset "Level 2: 8 independent invariants (Fulling et al.)" begin
        cr2 = get_invar_relations(3, "0_0_0", 2)
        @test cr2.degree == 3
        @test cr2.step == 2
        @test cr2.n_independent == 8
        @test cr2.n_dependent == 5
        @test length(cr2.relations) == 5
    end

    # ---- Bianchi relations: verify each ----
    @testset "Bianchi relation: I4 = (1/2)*I3" begin
        cr2 = get_invar_relations(3, "0_0_0", 2)
        rel = cr2.relations[1]
        @test rel.lhs == [3, 4, 1, 2, 9, 11, 10, 12, 5, 7, 6, 8]
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1//2
        @test rel.rhs[1][2] == [3, 4, 1, 2, 9, 10, 11, 12, 5, 6, 7, 8]
    end

    @testset "Bianchi relation: I8 = (1/2)*I7" begin
        cr2 = get_invar_relations(3, "0_0_0", 2)
        rel = cr2.relations[2]
        @test rel.lhs == [3, 5, 1, 9, 2, 11, 10, 12, 4, 7, 6, 8]
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1//2
        @test rel.rhs[1][2] == [3, 5, 1, 9, 2, 10, 11, 12, 4, 6, 7, 8]
    end

    @testset "Bianchi relation: I10 = (1/2)*I9" begin
        cr2 = get_invar_relations(3, "0_0_0", 2)
        rel = cr2.relations[3]
        @test rel.lhs == [5, 6, 9, 11, 1, 2, 10, 12, 3, 7, 4, 8]
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1//2
        @test rel.rhs[1][2] == [5, 6, 9, 10, 1, 2, 11, 12, 3, 4, 7, 8]
    end

    @testset "Bianchi relation: I11 = (1/4)*I9" begin
        cr2 = get_invar_relations(3, "0_0_0", 2)
        rel = cr2.relations[4]
        @test rel.lhs == [5, 7, 9, 11, 1, 10, 2, 12, 3, 6, 4, 8]
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1//4
        @test rel.rhs[1][2] == [5, 6, 9, 10, 1, 2, 11, 12, 3, 4, 7, 8]
    end

    @testset "Bianchi relation: I12 = (1/4)*I9 + I13" begin
        cr2 = get_invar_relations(3, "0_0_0", 2)
        rel = cr2.relations[5]
        @test rel.lhs == [5, 9, 7, 11, 1, 10, 3, 12, 2, 6, 4, 8]
        @test length(rel.rhs) == 2
        @test rel.rhs[1] == (1//4, [5, 6, 9, 10, 1, 2, 11, 12, 3, 4, 7, 8])
        @test rel.rhs[2] == (1//1, [5, 9, 7, 11, 1, 12, 3, 10, 2, 8, 4, 6])
    end

    # ---- degree3_canonical_rinvs ----
    @testset "degree3_canonical_rinvs" begin
        rinvs = degree3_canonical_rinvs()
        @test length(rinvs) == 13

        # All should be canonical
        @test all(r -> r.canonical, rinvs)

        # All should be degree 3
        @test all(r -> r.degree == 3, rinvs)

        # All should be valid involutions
        for r in rinvs
            for i in 1:12
                @test r.contraction[r.contraction[i]] == i
                @test r.contraction[i] != i
            end
        end

        # All should be distinct
        for i in 1:13, j in i+1:13
            @test rinvs[i].contraction != rinvs[j].contraction
        end
    end

    # ---- degree3_independent_rinvs ----
    @testset "degree3_independent_rinvs" begin
        rinvs = degree3_independent_rinvs()
        @test length(rinvs) == 8

        # All should be canonical
        @test all(r -> r.canonical, rinvs)

        # All should be degree 3
        @test all(r -> r.degree == 3, rinvs)

        # All should be valid involutions
        for r in rinvs
            for i in 1:12
                @test r.contraction[r.contraction[i]] == i
                @test r.contraction[i] != i
            end
        end

        # All should be distinct (pairwise non-equivalent)
        for i in 1:8, j in i+1:8
            @test rinvs[i].contraction != rinvs[j].contraction
        end
    end

    # ---- All independent RInvs produce scalar tensor expressions ----
    @testset "Independent RInvs are scalar (no free indices)" begin
        reg = db3_registry()
        with_registry(reg) do
            for rinv in degree3_independent_rinvs()
                expr = to_tensor_expr(rinv; registry=reg, metric=:g)
                s = simplify(expr; registry=reg)
                # A fully contracted expression has no free indices
                free = TensorGR.free_indices(s)
                @test isempty(free)
            end
        end
    end

    # ---- Verify Bianchi relations via orbit sign analysis ----
    @testset "Bianchi relation I4=(1/2)*I3 verified by orbit analysis" begin
        sigma_I4 = [3, 4, 1, 2, 9, 11, 10, 12, 5, 7, 6, 8]
        sigma_I3 = [3, 4, 1, 2, 9, 10, 11, 12, 5, 6, 7, 8]

        # Apply Bianchi cycle to factor 2 (slots 5-8)
        perm_cycle = collect(1:12)
        perm_cycle[6], perm_cycle[7], perm_cycle[8] = 7, 8, 6
        sigma_c1 = TensorGR._conjugate_contraction(sigma_I4, perm_cycle)

        perm_cycle2 = collect(1:12)
        perm_cycle2[6], perm_cycle2[7], perm_cycle2[8] = 8, 6, 7
        sigma_c2 = TensorGR._conjugate_contraction(sigma_I4, perm_cycle2)

        # sigma_c1 should canonicalize to I4
        c1 = canonicalize(RInv(3, sigma_c1))
        @test c1.contraction == sigma_I4

        # sigma_c2 should canonicalize to I3 (with sign -1 from orbit)
        c2 = canonicalize(RInv(3, sigma_c2))
        @test c2.contraction == sigma_I3
    end

    # ---- Verify I11 + I13 - I12 = 0 by orbit analysis ----
    @testset "Bianchi relation I11 + I13 - I12 = 0 verified by orbit analysis" begin
        sigma_I11 = [5, 7, 9, 11, 1, 10, 2, 12, 3, 6, 4, 8]

        # Apply Bianchi cycle to factor 1 (slots 1-4), cycling (2,3,4)
        perm_c1 = collect(1:12)
        perm_c1[2], perm_c1[3], perm_c1[4] = 3, 4, 2
        sigma_c1 = TensorGR._conjugate_contraction(sigma_I11, perm_c1)

        perm_c2 = collect(1:12)
        perm_c2[2], perm_c2[3], perm_c2[4] = 4, 2, 3
        sigma_c2 = TensorGR._conjugate_contraction(sigma_I11, perm_c2)

        # sigma_c1 should canonicalize to I13
        c1 = canonicalize(RInv(3, sigma_c1))
        @test c1.contraction == [5, 9, 7, 11, 1, 12, 3, 10, 2, 8, 4, 6]  # I13

        # sigma_c2 should canonicalize to I12 (with sign -1)
        gens = TensorGR._rinv_slot_generators(3)
        orbit = Dict{Vector{Int}, Int}()
        orbit[sigma_c2] = +1
        queue = [sigma_c2]
        while !isempty(queue)
            s = popfirst!(queue)
            csign = orbit[s]
            for (g, gsign) in gens
                s_new = TensorGR._conjugate_contraction(s, g)
                nsign = csign * gsign
                if !haskey(orbit, s_new)
                    orbit[s_new] = nsign
                    push!(queue, s_new)
                end
            end
        end
        # Find canonical form
        best = sigma_c2
        for (s, _) in orbit
            if s < best
                best = s
            end
        end
        @test best == [5, 9, 7, 11, 1, 10, 3, 12, 2, 6, 4, 8]  # I12
        # Sign should be -1 (the invariant value flips)
        @test orbit[best] == -1
    end

    # ---- is_independent_rinv for degree 3 ----
    @testset "is_independent_rinv degree 3" begin
        # The 8 independent invariants at Level 2
        indep = degree3_independent_rinvs()
        for r in indep
            @test is_independent_rinv(r, 2) == true
        end

        # The 5 dependent invariants at Level 2
        I4 = RInv(3, [3, 4, 1, 2, 9, 11, 10, 12, 5, 7, 6, 8])
        I8 = RInv(3, [3, 5, 1, 9, 2, 11, 10, 12, 4, 7, 6, 8])
        I10 = RInv(3, [5, 6, 9, 11, 1, 2, 10, 12, 3, 7, 4, 8])
        I11 = RInv(3, [5, 7, 9, 11, 1, 10, 2, 12, 3, 6, 4, 8])
        I12 = RInv(3, [5, 9, 7, 11, 1, 10, 3, 12, 2, 6, 4, 8])

        @test is_independent_rinv(I4, 2) == false
        @test is_independent_rinv(I8, 2) == false
        @test is_independent_rinv(I10, 2) == false
        @test is_independent_rinv(I11, 2) == false
        @test is_independent_rinv(I12, 2) == false

        # All 13 are independent at Level 1
        for r in degree3_canonical_rinvs()
            @test is_independent_rinv(r, 1) == true
        end
    end

    # ---- list_invar_cases includes degree-3 ----
    @testset "list_invar_cases includes degree 3" begin
        cases = list_invar_cases(degree=3)
        @test length(cases) >= 2  # at least step 1 and step 2

        @test any(c -> c.degree == 3 && c.step == 1 && c.n_independent == 13, cases)
        @test any(c -> c.degree == 3 && c.step == 2 && c.n_independent == 8, cases)
    end

    # ---- Exhaustive enumeration: 10395 pairings -> 13 non-zero + vanishing ----
    @testset "Exhaustive enumeration of degree-3 contractions" begin
        # Generate all fixed-point-free involutions on [1..12]
        function all_involutions_12()
            results = Vector{Vector{Int}}()
            perm = zeros(Int, 12)
            function gen!(pos)
                idx = 0
                for i in pos:12
                    if perm[i] == 0
                        idx = i
                        break
                    end
                end
                if idx == 0
                    push!(results, copy(perm))
                    return
                end
                for j in idx+1:12
                    perm[j] == 0 || continue
                    perm[idx] = j
                    perm[j] = idx
                    gen!(idx + 1)
                    perm[idx] = 0
                    perm[j] = 0
                end
            end
            gen!(1)
            results
        end

        invols = all_involutions_12()
        @test length(invols) == 10395  # (12-1)!! = 10395

        # Canonicalize each and collect distinct canonical forms
        canonical_set = Dict{Vector{Int}, Int}()
        for inv in invols
            rinv = RInv(3, inv)
            c = canonicalize(rinv)
            canonical_set[c.contraction] = get(canonical_set, c.contraction, 0) + 1
        end

        # Should have exactly 14 classes (13 non-zero + 1 vanishing)
        @test length(canonical_set) == 14
        @test haskey(canonical_set, zeros(Int, 12))  # vanishing class

        # The 13 non-zero classes should match our database
        db_forms = Set([r.contraction for r in degree3_canonical_rinvs()])
        non_zero_forms = Set(k for k in keys(canonical_set) if k != zeros(Int, 12))
        @test db_forms == non_zero_forms
    end

    # ---- Cross-check: known catalog invariants match canonical forms ----
    @testset "Catalog invariants match canonical RInv forms" begin
        reg = db3_registry()
        with_registry(reg) do
            # Goroff-Sagnotti (pure Riemann) should match I9
            expr_gs = curvature_invariant(:Riem_cubed; registry=reg, manifold=:M4, metric=:g)
            rinv_gs = from_tensor_expr(expr_gs; registry=reg, metric=:g)
            c_gs = canonicalize(rinv_gs)
            @test c_gs.contraction == [5, 6, 9, 10, 1, 2, 11, 12, 3, 4, 7, 8]  # I9
        end
    end
end
