@testset "RInv: xperm-based canonicalization" begin
    # ---- Helper: standard 4D registry with curvature tensors ----
    function rinv_canon_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # ---- rinv_symmetry_group ----
    @testset "rinv_symmetry_group generators" begin
        # Degree 1: 3 generators (anti12, anti34, pair) on 6 points
        gens1 = rinv_symmetry_group(1)
        @test length(gens1) == 3
        for g in gens1
            @test length(g.data) == 6  # 4 slots + 2 sign bits
        end

        # Degree 2: 3*2 intra + 1 inter = 7 generators on 10 points
        gens2 = rinv_symmetry_group(2)
        @test length(gens2) == 7
        for g in gens2
            @test length(g.data) == 10
        end

        # Degree 3: 3*3 intra + 2 inter = 11 generators on 14 points
        gens3 = rinv_symmetry_group(3)
        @test length(gens3) == 11
        for g in gens3
            @test length(g.data) == 14
        end
    end

    # ---- Kretschmann is already canonical ----
    @testset "Kretschmann canonicalization" begin
        reg = rinv_canon_registry()
        with_registry(reg) do
            # Kretschmann: R_{abcd}R^{abcd} = sigma (1,5)(2,6)(3,7)(4,8)
            kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            ckr = canonicalize_rinv(kr, reg)
            @test ckr.canonical == true
            # Should be a valid involution
            n = 4 * kr.degree
            for i in 1:n
                @test ckr.contraction[ckr.contraction[i]] == i
                @test ckr.contraction[i] != i
            end
        end
    end

    # ---- Two equivalent RInvs canonicalize to same form ----
    @testset "Equivalent RInvs have same canonical form" begin
        reg = rinv_canon_registry()
        with_registry(reg) do
            # Two equivalent representations of Kretschmann:
            #   sigma1 = (1,5)(2,6)(3,7)(4,8)
            #   sigma2 = (1,6)(2,5)(3,8)(4,7)  -- swapped within pairs
            kr1 = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            kr2 = RInv(2, [6, 5, 8, 7, 2, 1, 4, 3])

            ckr1 = canonicalize_rinv(kr1, reg)
            ckr2 = canonicalize_rinv(kr2, reg)

            @test ckr1.canonical == true
            @test ckr2.canonical == true
            @test ckr1.contraction == ckr2.contraction
        end
    end

    # ---- Non-equivalent invariants remain distinct ----
    @testset "Non-equivalent invariants are distinct" begin
        reg = rinv_canon_registry()
        with_registry(reg) do
            # R^2 (double self-contraction)
            r_sq = RInv(2, [2, 1, 4, 3, 6, 5, 8, 7])

            # Ric^2 (partial cross-contraction)
            ric_sq = RInv(2, [3, 6, 1, 8, 7, 2, 5, 4])

            # Kretschmann (full cross-contraction)
            kretschmann = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

            cr_sq = canonicalize_rinv(r_sq, reg)
            cric_sq = canonicalize_rinv(ric_sq, reg)
            ckr = canonicalize_rinv(kretschmann, reg)

            # All three must be distinct
            @test cr_sq.contraction != cric_sq.contraction
            @test cr_sq.contraction != ckr.contraction
            @test cric_sq.contraction != ckr.contraction
        end
    end

    # ---- Degree-3 canonicalization ----
    @testset "Degree-3 canonicalization" begin
        reg = rinv_canon_registry()
        with_registry(reg) do
            # Goroff-Sagnotti: R_{ab}^{cd} R_{cd}^{ef} R_{ef}^{ab}
            # Contraction: (1,11)(2,12)(3,5)(4,6)(7,9)(8,10)
            gs = RInv(3, [11, 12, 5, 6, 3, 4, 9, 10, 7, 8, 1, 2])
            cgs = canonicalize_rinv(gs, reg)
            @test cgs.canonical == true

            # The canonical form should still be a valid involution
            n = 12
            for i in 1:n
                @test cgs.contraction[cgs.contraction[i]] == i
                @test cgs.contraction[i] != i
            end

            # Permuted version of same invariant: swap factors 1 and 2
            # Original: R1(1,2,3,4) R2(5,6,7,8) R3(9,10,11,12)
            # After swap: R2(1,2,3,4) R1(5,6,7,8) R3(9,10,11,12)
            # Original contraction: (1,11)(2,12)(3,5)(4,6)(7,9)(8,10)
            # After swapping factor indices 1-4 <-> 5-8 in the contraction:
            # (5,11)(6,12)(7,1)(8,2)(3,9)(4,10)
            gs2 = RInv(3, [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6])
            cgs2 = canonicalize_rinv(gs2, reg)
            @test cgs.contraction == cgs2.contraction
        end
    end

    # ---- Degree-1 canonicalization (trivial contractions) ----
    @testset "Degree-1 canonicalization" begin
        reg = rinv_canon_registry()
        with_registry(reg) do
            # Ricci scalar: R = R^{ab}{}_{ab}
            # contraction (1,3)(2,4)
            r1 = RInv(1, [3, 4, 1, 2])
            cr1 = canonicalize_rinv(r1, reg)
            @test cr1.canonical == true

            # Alternative representation (1,4)(2,3)
            r2 = RInv(1, [4, 3, 2, 1])
            cr2 = canonicalize_rinv(r2, reg)
            @test cr2.canonical == true

            # Both should canonicalize to the same form
            @test cr1.contraction == cr2.contraction
        end
    end

    # ---- are_equivalent ----
    @testset "are_equivalent" begin
        reg = rinv_canon_registry()
        with_registry(reg) do
            kr1 = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            kr2 = RInv(2, [6, 5, 8, 7, 2, 1, 4, 3])

            @test are_equivalent(kr1, kr2, reg) == true

            # Different invariant
            ric_sq = RInv(2, [3, 6, 1, 8, 7, 2, 5, 4])
            @test are_equivalent(kr1, ric_sq, reg) == false

            # Degree mismatch
            r1 = RInv(1, [3, 4, 1, 2])
            @test are_equivalent(r1, kr1, reg) == false
        end
    end

    # ---- Idempotent xperm canonicalization ----
    @testset "Idempotent canonicalization" begin
        reg = rinv_canon_registry()
        with_registry(reg) do
            kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            c1 = canonicalize_rinv(kr, reg)
            c2 = canonicalize_rinv(c1, reg)
            @test c1.contraction == c2.contraction
            @test c2.canonical == true
        end
    end

    # ---- Consistency with BFS canonicalize ----
    @testset "Consistency with BFS canonicalize" begin
        reg = rinv_canon_registry()
        with_registry(reg) do
            # For degree 1 and 2, xperm and BFS should identify the same
            # equivalence classes (canonical forms may differ in representation
            # but equivalences must agree)

            # Degree 2: all three classes
            r_sq = RInv(2, [2, 1, 4, 3, 6, 5, 8, 7])
            ric_sq = RInv(2, [3, 6, 1, 8, 7, 2, 5, 4])
            kretschmann = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

            # Two equivalent Kretschmanns
            kr2 = RInv(2, [6, 5, 8, 7, 2, 1, 4, 3])

            # BFS says kr1 == kr2 and kr1 != ric_sq
            @test kretschmann == kr2  # BFS
            @test kretschmann != ric_sq  # BFS

            # xperm should agree
            @test are_equivalent(kretschmann, kr2, reg) == true
            @test are_equivalent(kretschmann, ric_sq, reg) == false
            @test are_equivalent(r_sq, ric_sq, reg) == false
            @test are_equivalent(r_sq, kretschmann, reg) == false
        end
    end

    # ---- Degree-3 equivalent representations ----
    @testset "Degree-3 equivalences" begin
        reg = rinv_canon_registry()
        with_registry(reg) do
            # R^3 = (R^{ab}_{ab})^3 : all self-contractions
            # Factor 1: (1,2)(3,4), Factor 2: (5,6)(7,8), Factor 3: (9,10)(11,12)
            r_cubed = RInv(3, [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11])

            # Goroff-Sagnotti cycle
            gs = RInv(3, [11, 12, 5, 6, 3, 4, 9, 10, 7, 8, 1, 2])

            # These are different invariants
            @test are_equivalent(r_cubed, gs, reg) == false

            # R^3 with factors permuted (swap factors 1 and 3):
            # Factor 3 in slots 1-4, Factor 2 in slots 5-8, Factor 1 in slots 9-12
            # Self-contractions remain within each factor
            r_cubed_permuted = RInv(3, [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11])
            @test are_equivalent(r_cubed, r_cubed_permuted, reg) == true
        end
    end

    # ---- Validation: xperm output is valid involution ----
    @testset "Output validity" begin
        reg = rinv_canon_registry()
        with_registry(reg) do
            test_cases = [
                RInv(1, [3, 4, 1, 2]),
                RInv(2, [5, 6, 7, 8, 1, 2, 3, 4]),
                RInv(2, [2, 1, 4, 3, 6, 5, 8, 7]),
                RInv(2, [3, 6, 1, 8, 7, 2, 5, 4]),
                RInv(3, [11, 12, 5, 6, 3, 4, 9, 10, 7, 8, 1, 2]),
            ]

            for tc in test_cases
                c = canonicalize_rinv(tc, reg)
                n = 4 * tc.degree
                @test c.canonical == true
                @test length(c.contraction) == n

                # Skip zero contractions (vanishing invariants)
                all(==(0), c.contraction) && continue

                # Valid involution: sigma(sigma(i)) == i, no fixed points
                for i in 1:n
                    @test 1 <= c.contraction[i] <= n
                    @test c.contraction[i] != i
                    @test c.contraction[c.contraction[i]] == i
                end
            end
        end
    end
end
