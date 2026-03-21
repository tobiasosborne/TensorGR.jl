#= Test the Invar degree-4 algebraic invariant database.
#
# Ground truth: Martin-Garcia, Portugal & Manssur (2007), CPC 177:640,
#               Table 1 (57 canonical) and Table 2 (step A=38, step B=15);
#               Fulling, King, Wybourne & Cummins (1992), CQG 9:1151.
#
# Degree-4 invariants: products of 4 Riemann tensors with all 16 indices
# contracted pairwise via the metric.
#   Level 1 (permutation symmetries): 57 non-vanishing canonical forms
#   Level 2 (first Bianchi identity): 26 independent invariants
=#

@testset "Invar Database: Degree-4 Data" begin
    # ---- Shared registry ----
    function db4_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # ---- Level 1: 57 canonical forms ----
    @testset "Level 1: 57 non-vanishing canonical forms" begin
        cr1 = get_invar_relations(4, "0_0_0_0", 1)
        @test cr1.degree == 4
        @test cr1.case_key == "0_0_0_0"
        @test cr1.step == 1
        @test cr1.dim === nothing
        @test cr1.n_independent == 57
        @test cr1.n_dependent == 0
        @test isempty(cr1.relations)
    end

    # ---- Level 2: 26 independent, 31 dependent ----
    @testset "Level 2: 26 independent invariants" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        @test cr2.degree == 4
        @test cr2.step == 2
        @test cr2.n_independent == 26
        @test cr2.n_dependent == 31
        @test length(cr2.relations) == 31  # 8 product-type + 23 non-product Bianchi
    end

    # ---- Bianchi relations: verify product-type relations ----
    @testset "Bianchi relation: I4 = (1/2)*I3 (R^2 * d2 Bianchi)" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        rel = cr2.relations[1]
        @test rel.lhs == [3, 4, 1, 2, 7, 8, 5, 6, 13, 15, 14, 16, 9, 11, 10, 12]
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1//2
        @test rel.rhs[1][2] == [3, 4, 1, 2, 7, 8, 5, 6, 13, 14, 15, 16, 9, 10, 11, 12]
    end

    @testset "Bianchi relation: I8 = (1/2)*I7 (R * d3 Bianchi)" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        rel = cr2.relations[2]
        @test rel.lhs == [3, 4, 1, 2, 7, 9, 5, 13, 6, 15, 14, 16, 8, 11, 10, 12]
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1//2
    end

    @testset "Bianchi relation: I12 = (1/4)*I9 + I13 (R * d3 Bianchi)" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        rel = cr2.relations[5]
        @test rel.lhs == [3, 4, 1, 2, 9, 13, 11, 15, 5, 14, 7, 16, 6, 10, 8, 12]
        @test length(rel.rhs) == 2
        @test rel.rhs[1] == (1//4, [3, 4, 1, 2, 9, 10, 13, 14, 5, 6, 15, 16, 7, 8, 11, 12])
        @test rel.rhs[2] == (1//1, [3, 4, 1, 2, 9, 13, 11, 15, 5, 16, 7, 14, 6, 12, 8, 10])
    end

    @testset "Bianchi relation: I46 = (1/4)*I35 (I4_d2^2)" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        rel = cr2.relations[8]
        @test rel.lhs == [5, 7, 6, 8, 1, 3, 2, 4, 13, 15, 14, 16, 9, 11, 10, 12]
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1//4
        @test rel.rhs[1][2] == [5, 6, 7, 8, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12]
    end

    # ---- degree4_canonical_rinvs ----
    @testset "degree4_canonical_rinvs" begin
        rinvs = degree4_canonical_rinvs()
        @test length(rinvs) == 57

        # All should be canonical
        @test all(r -> r.canonical, rinvs)

        # All should be degree 4
        @test all(r -> r.degree == 4, rinvs)

        # All should have length-16 contractions
        @test all(r -> length(r.contraction) == 16, rinvs)

        # All should be valid involutions
        for r in rinvs
            for i in 1:16
                @test r.contraction[r.contraction[i]] == i
                @test r.contraction[i] != i
            end
        end

        # All should be distinct
        for i in 1:57, j in i+1:57
            @test rinvs[i].contraction != rinvs[j].contraction
        end

        # All should be sorted lexicographically
        for i in 1:56
            @test rinvs[i].contraction < rinvs[i+1].contraction
        end
    end

    # ---- degree4_independent_rinvs ----
    @testset "degree4_independent_rinvs" begin
        rinvs = degree4_independent_rinvs()
        @test length(rinvs) == 26  # 11 product-type + 15 non-product

        # All should be canonical
        @test all(r -> r.canonical, rinvs)

        # All should be degree 4
        @test all(r -> r.degree == 4, rinvs)

        # All should be valid involutions
        for r in rinvs
            for i in 1:16
                @test r.contraction[r.contraction[i]] == i
                @test r.contraction[i] != i
            end
        end

        # All should be distinct (pairwise non-equivalent)
        for i in 1:26, j in i+1:26
            @test rinvs[i].contraction != rinvs[j].contraction
        end
    end

    # ---- Canonical forms are genuinely canonical ----
    @testset "All canonical forms re-canonicalize to themselves" begin
        for rinv in degree4_canonical_rinvs()
            c = canonicalize(RInv(4, rinv.contraction))
            @test c.contraction == rinv.contraction
        end
    end

    # ---- Independent RInvs produce scalar tensor expressions ----
    @testset "Independent RInvs are scalar (no free indices)" begin
        reg = db4_registry()
        with_registry(reg) do
            for rinv in degree4_independent_rinvs()
                expr = to_tensor_expr(rinv; registry=reg, metric=:g)
                free = TensorGR.free_indices(expr)
                @test isempty(free)
            end
        end
    end

    # ---- is_independent_rinv for degree 4 ----
    @testset "is_independent_rinv degree 4" begin
        # All 57 are independent at Level 1
        for r in degree4_canonical_rinvs()
            @test is_independent_rinv(r, 1) == true
        end

        canonical = degree4_canonical_rinvs()
        independent = degree4_independent_rinvs()
        indep_set = Set([r.contraction for r in independent])

        # All 31 dependent forms should be not independent at Level 2
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        for rel in cr2.relations
            dep = RInv(4, rel.lhs, true)
            @test is_independent_rinv(dep, 2) == false
        end

        # All 26 independent forms should be independent at Level 2
        for r in independent
            @test is_independent_rinv(r, 2) == true
        end
    end

    # ---- list_invar_cases includes degree-4 ----
    @testset "list_invar_cases includes degree 4" begin
        cases = list_invar_cases(degree=4)
        @test length(cases) >= 2  # at least step 1 and step 2

        @test any(c -> c.degree == 4 && c.step == 1 && c.n_independent == 57, cases)
        @test any(c -> c.degree == 4 && c.step == 2 && c.n_independent == 26, cases)
    end

    # ---- Completeness: n_independent + n_dependent == 57 ----
    @testset "Completeness: 26 + 31 = 57" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        @test cr2.n_independent + cr2.n_dependent == 57
    end

    # ---- Key product invariants match expected contraction patterns ----
    @testset "Key product invariants have correct structure" begin
        reg = db4_registry()
        canonical = degree4_canonical_rinvs()

        # I1 = R^4: all factors self-contract (1<->3, 2<->4 in each)
        i1 = canonical[1]
        for f in 0:3
            @test i1.contraction[4*f+1] == 4*f+3  # slot 1 <-> slot 3
            @test i1.contraction[4*f+2] == 4*f+4  # slot 2 <-> slot 4
        end

        # I35 = K^2: factors 1,2 fully cross-contract, factors 3,4 fully cross-contract
        i35 = canonical[35]
        # Factor 1 slots pair with factor 2 slots
        for j in 1:4
            partner = i35.contraction[j]
            @test 5 <= partner <= 8
        end
        # Factor 3 slots pair with factor 4 slots
        for j in 9:12
            partner = i35.contraction[j]
            @test 13 <= partner <= 16
        end
    end

    # ---- Product-form detection ----
    @testset "Product form detection: 19 products, 38 non-products" begin
        canonical = degree4_canonical_rinvs()

        function is_product_rinv(c::Vector{Int})
            factor_of(s) = div(s-1, 4) + 1
            adj = Set{Tuple{Int,Int}}()
            for i in 1:16
                fi = factor_of(i)
                fj = factor_of(c[i])
                if fi != fj
                    push!(adj, (min(fi,fj), max(fi,fj)))
                end
            end
            visited = Set([1])
            queue = [1]
            while !isempty(queue)
                f = popfirst!(queue)
                for (a, b) in adj
                    if a == f && !(b in visited)
                        push!(visited, b); push!(queue, b)
                    elseif b == f && !(a in visited)
                        push!(visited, a); push!(queue, a)
                    end
                end
            end
            return length(visited) < 4
        end

        n_product = count(r -> is_product_rinv(r.contraction), canonical)
        n_nonproduct = count(r -> !is_product_rinv(r.contraction), canonical)

        # Martin-Garcia et al (2007): Table 1 has 57, Table 2 step A has 38
        # So 57 - 38 = 19 products
        @test n_product == 19
        @test n_nonproduct == 38
    end

    # ---- Relation LHS and RHS are valid RInvs ----
    @testset "All Bianchi relation LHS and RHS are valid degree-4 RInvs" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        for rel in cr2.relations
            # LHS is a valid involution
            lhs = RInv(4, rel.lhs, true)
            @test length(lhs.contraction) == 16
            for i in 1:16
                @test lhs.contraction[lhs.contraction[i]] == i
                @test lhs.contraction[i] != i
            end

            # Each RHS term is a valid involution
            for (coeff, contraction) in rel.rhs
                rhs = RInv(4, contraction, true)
                @test length(rhs.contraction) == 16
                for i in 1:16
                    @test rhs.contraction[rhs.contraction[i]] == i
                    @test rhs.contraction[i] != i
                end
            end
        end
    end

    # ---- Relation scalarity: LHS and RHS are scalar ----
    @testset "Relation scalarity: LHS and RHS have no free indices" begin
        reg = db4_registry()
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        with_registry(reg) do
            for rel in cr2.relations
                # LHS
                lhs_rinv = RInv(4, rel.lhs, true)
                lhs_expr = to_tensor_expr(lhs_rinv; registry=reg, metric=:g)
                @test isempty(free_indices(lhs_expr))

                # RHS
                for (coeff, contraction) in rel.rhs
                    rhs_rinv = RInv(4, contraction, true)
                    rhs_expr = to_tensor_expr(rhs_rinv; registry=reg, metric=:g)
                    @test isempty(free_indices(rhs_expr))
                end
            end
        end
    end

    # ---- Non-product Bianchi relations: spot checks ----
    @testset "Non-product Bianchi: I20 = (1/2)*I19" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        canonical = degree4_canonical_rinvs()
        # I20 is the 9th relation (8 product + 1st non-product)
        rel = cr2.relations[9]
        @test rel.lhs == canonical[20].contraction
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1//2
        @test rel.rhs[1][2] == canonical[19].contraction
    end

    @testset "Non-product Bianchi: I44 = (1/4)*I42 + I55 - I57" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        canonical = degree4_canonical_rinvs()
        # I44 is the 22nd relation (8 + 14th non-product)
        rel = cr2.relations[22]
        @test rel.lhs == canonical[44].contraction
        @test length(rel.rhs) == 3
        @test rel.rhs[1] == (1//4, canonical[42].contraction)
        @test rel.rhs[2] == (1//1, canonical[55].contraction)
        @test rel.rhs[3] == (-1//1, canonical[57].contraction)
    end

    @testset "Non-product Bianchi: I49 = (1/8)*I39" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        canonical = degree4_canonical_rinvs()
        # I49 is the 26th relation (8 + 18th non-product)
        rel = cr2.relations[26]
        @test rel.lhs == canonical[49].contraction
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1//8
        @test rel.rhs[1][2] == canonical[39].contraction
    end

    @testset "Non-product Bianchi: I56 = -(1/8)*I42 + (1/2)*I55 + (1/2)*I57" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        canonical = degree4_canonical_rinvs()
        # I56 is the 31st (last) relation
        rel = cr2.relations[31]
        @test rel.lhs == canonical[56].contraction
        @test length(rel.rhs) == 3
        @test rel.rhs[1] == (-1//8, canonical[42].contraction)
        @test rel.rhs[2] == (1//2, canonical[55].contraction)
        @test rel.rhs[3] == (1//2, canonical[57].contraction)
    end

    # ---- All 31 dependent LHS contractions are among the 57 canonical forms ----
    @testset "All 31 dependent LHS are canonical degree-4 forms" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        canonical = degree4_canonical_rinvs()
        canonical_set = Set([r.contraction for r in canonical])
        for rel in cr2.relations
            @test rel.lhs in canonical_set
        end
    end

    # ---- All RHS contractions are independent forms ----
    @testset "All RHS contractions are from independent set" begin
        cr2 = get_invar_relations(4, "0_0_0_0", 2)
        independent = degree4_independent_rinvs()
        indep_set = Set([r.contraction for r in independent])
        for rel in cr2.relations
            for (coeff, contraction) in rel.rhs
                @test contraction in indep_set
            end
        end
    end

    # ---- case_key generation ----
    @testset "_algebraic_case_key(4) == \"0_0_0_0\"" begin
        @test TensorGR._algebraic_case_key(4) == "0_0_0_0"
    end

    # ---- Orbit-based completeness verification ----
    @testset "Orbit-based completeness: 57 forms cover all 2,027,025 involutions" begin
        rinvs = degree4_canonical_rinvs()
        total_orbit = 0
        for rinv in rinvs
            gens = TensorGR._rinv_slot_generators(4)
            orbit = Dict{Vector{Int}, Int}()
            orbit[rinv.contraction] = +1
            queue = [rinv.contraction]
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
            total_orbit += length(orbit)
        end
        expected = prod(1:2:15)  # 15!! = 2,027,025
        @test total_orbit <= expected  # orbit sum must not exceed total
        @test total_orbit == 1095440   # known non-vanishing total
        vanishing = expected - total_orbit
        @test vanishing == 931585      # known vanishing total
    end
end
