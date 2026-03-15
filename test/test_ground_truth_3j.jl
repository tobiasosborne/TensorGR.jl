# Ground truth verification of Wigner 3j symbols, Clebsch-Gordan coefficients,
# and Gaunt integrals against independently computed reference values.
#
# Reference sources:
#   [DLMF]  NIST Digital Library of Mathematical Functions, Section 34.3
#           https://dlmf.nist.gov/34.3
#   [Racah] G. Racah, Phys. Rev. 62, 438 (1942) -- Racah formula
#   [CS]    Condon & Shortley, "Theory of Atomic Spectra" (1935) -- CG tables
#   [Sakurai] J.J. Sakurai, "Modern Quantum Mechanics" (2017), Tables 3.5-3.6
#   [Mathworld] Wolfram MathWorld, "Clebsch-Gordan Coefficient"
#               https://mathworld.wolfram.com/Clebsch-GordanCoefficient.html
#
# Method: reference values computed from the exact Racah formula using BigInt
# arithmetic (independent re-implementation), then string-matched against our
# production wigner3j() output.

@testset "Ground truth: Wigner 3j, CG, and Gaunt" begin

    # ── Helper: independent Racah formula (BigInt exact) ────────────────────
    function _ref_3j(j1, j2, j3, m1, m2, m3)
        m1 + m2 + m3 != 0 && return 0.0
        abs(m1) > j1 && return 0.0
        abs(m2) > j2 && return 0.0
        abs(m3) > j3 && return 0.0
        j3 < abs(j1 - j2) && return 0.0
        j3 > j1 + j2 && return 0.0
        any(j -> j < 0, (j1, j2, j3)) && return 0.0

        a = j1 + j2 - j3; b = j1 - j2 + j3; c = -j1 + j2 + j3
        (a < 0 || b < 0 || c < 0) && return 0.0
        tri = factorial(big(a)) * factorial(big(b)) * factorial(big(c)) //
              factorial(big(j1 + j2 + j3 + 1))
        pf_sq = tri
        for (j, m) in ((j1, m1), (j2, m2), (j3, m3))
            pf_sq *= factorial(big(j + m)) * factorial(big(j - m))
        end
        t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
        t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
        t_min > t_max && return 0.0
        s = Rational{BigInt}(0)
        for t in t_min:t_max
            denom = factorial(big(t)) *
                    factorial(big(j3 - j2 + t + m1)) *
                    factorial(big(j3 - j1 + t - m2)) *
                    factorial(big(j1 + j2 - j3 - t)) *
                    factorial(big(j1 - t - m1)) *
                    factorial(big(j2 - t + m2))
            s += (iseven(t) ? 1 : -1) // denom
        end
        phase = iseven(j1 - j2 + m3) ? 1 : -1
        Float64(phase) * sqrt(Float64(pf_sq)) * Float64(s)
    end

    function _ref_cg(j1, m1, j2, m2, J, M)
        m1 + m2 != M && return 0.0
        phase = iseven(j1 - j2 + M) ? 1.0 : -1.0
        phase * sqrt(2J + 1) * _ref_3j(j1, j2, J, m1, m2, -M)
    end

    # ── 1. DLMF 34.3.1: (j j 0; m -m 0) = (-1)^(j-m) / sqrt(2j+1) ────────
    @testset "DLMF 34.3.1: diagonal j3=0" begin
        for j in 0:6, m in -j:j
            ref = (-1)^(j - m) / sqrt(2j + 1)
            our = wigner3j(j, j, 0, m, -m, 0)
            @test isapprox(our, ref, atol=1e-13)
        end
    end

    # ── 2. DLMF 34.3.4: parity rule (j1+j2+j3 odd, all m=0 => 0) ──────────
    @testset "DLMF 34.3.4: parity selection rule" begin
        for j1 in 0:5, j2 in 0:5, j3 in abs(j1 - j2):(j1 + j2)
            if isodd(j1 + j2 + j3)
                @test wigner3j(j1, j2, j3, 0, 0, 0) == 0.0
            end
        end
    end

    # ── 3. Specific 3j values vs independent Racah (all m=0, even J) ───────
    @testset "3j values vs Racah formula (m=0)" begin
        for j1 in 0:5, j2 in 0:5, j3 in abs(j1 - j2):(j1 + j2)
            iseven(j1 + j2 + j3) || continue
            ref = _ref_3j(j1, j2, j3, 0, 0, 0)
            our = wigner3j(j1, j2, j3, 0, 0, 0)
            @test isapprox(our, ref, atol=1e-13)
        end
    end

    # ── 4. Specific 3j values vs Racah (nonzero m) ─────────────────────────
    @testset "3j values vs Racah formula (nonzero m)" begin
        cases = [
            # (j1, j2, j3, m1, m2, m3) -- selected from standard tables
            (1, 1, 1, 1, 0, -1),
            (1, 1, 1, 0, 1, -1),
            (1, 1, 1, -1, 0, 1),
            (1, 1, 2, 1, 0, -1),
            (1, 1, 2, 1, -1, 0),
            (1, 1, 2, 0, 1, -1),
            (1, 1, 2, 1, 1, -2),
            (1, 1, 0, 1, -1, 0),
            (2, 1, 1, 0, 1, -1),
            (2, 1, 2, 1, 0, -1),
            (2, 1, 3, 1, 0, -1),
            (2, 1, 2, 0, 1, -1),
            (2, 2, 2, 1, 1, -2),
            (2, 2, 4, 1, 1, -2),
            (2, 2, 3, 2, -1, -1),
            (2, 2, 2, 2, 0, -2),
            (2, 2, 4, 2, 0, -2),
            (3, 2, 1, 0, 1, -1),
            (3, 2, 3, 1, 1, -2),
            (3, 2, 5, 1, 1, -2),
            (3, 3, 0, 2, -2, 0),
            (3, 3, 4, 3, -1, -2),
            (4, 3, 1, 0, 1, -1),
            (4, 3, 5, 2, 1, -3),
        ]
        for (j1, j2, j3, m1, m2, m3) in cases
            ref = _ref_3j(j1, j2, j3, m1, m2, m3)
            our = wigner3j(j1, j2, j3, m1, m2, m3)
            @test isapprox(our, ref, atol=1e-13)
        end
    end

    # ── 5. Named closed-form values [DLMF + standard tables] ───────────────
    @testset "Named closed-form 3j values" begin
        # (1 1 0; 0 0 0) = -1/sqrt(3)   [DLMF 34.3.1, j=1]
        @test isapprox(wigner3j(1, 1, 0, 0, 0, 0), -1 / sqrt(3), atol=1e-14)
        # (1 1 2; 0 0 0) = sqrt(2/15)   [DLMF 34.3.4 Racah]
        @test isapprox(wigner3j(1, 1, 2, 0, 0, 0), sqrt(2 / 15), atol=1e-14)
        # (2 2 0; 0 0 0) = 1/sqrt(5)    [DLMF 34.3.1, j=2]
        @test isapprox(wigner3j(2, 2, 0, 0, 0, 0), 1 / sqrt(5), atol=1e-14)
        # (3 3 0; 0 0 0) = -1/sqrt(7)   [DLMF 34.3.1, j=3]
        @test isapprox(wigner3j(3, 3, 0, 0, 0, 0), -1 / sqrt(7), atol=1e-14)
        # (4 4 0; 0 0 0) = 1/sqrt(9) = 1/3  [DLMF 34.3.1, j=4]
        @test isapprox(wigner3j(4, 4, 0, 0, 0, 0), 1 / 3, atol=1e-14)
        # (1 1 0; 1 -1 0) = 1/sqrt(3)   [DLMF 34.3.1, j=1, m=1]
        @test isapprox(wigner3j(1, 1, 0, 1, -1, 0), 1 / sqrt(3), atol=1e-14)
        # (1 1 1; 1 0 -1) = -1/sqrt(6)  [standard tables]
        @test isapprox(wigner3j(1, 1, 1, 1, 0, -1), -1 / sqrt(6), atol=1e-14)
    end

    # ── 6. Selection rules ──────────────────────────────────────────────────
    @testset "Selection rules" begin
        # m1 + m2 + m3 != 0
        @test wigner3j(1, 1, 1, 1, 0, 1) == 0.0
        @test wigner3j(2, 2, 2, 1, 1, 1) == 0.0
        # Triangle inequality
        @test wigner3j(1, 1, 5, 0, 0, 0) == 0.0
        @test wigner3j(3, 1, 0, 0, 0, 0) == 0.0
        # |m| > j
        @test wigner3j(1, 1, 0, 2, 0, 0) == 0.0
        @test wigner3j(1, 1, 2, 0, 3, 0) == 0.0
        # Negative j
        @test wigner3j(-1, 1, 0, 0, 0, 0) == 0.0
    end

    # ── 7. Symmetry properties [DLMF 34.3.8-34.3.10] ────────────────────────
    @testset "3j symmetry: even permutation" begin
        # (j1 j2 j3; m1 m2 m3) = (j2 j3 j1; m2 m3 m1) = (j3 j1 j2; m3 m1 m2)
        for (j1, j2, j3, m1, m2, m3) in [
            (1, 2, 2, 0, 1, -1), (1, 1, 2, 1, 0, -1), (2, 2, 2, 1, 1, -2),
            (2, 3, 3, 1, 1, -2), (3, 2, 5, 1, 1, -2), (4, 3, 5, 2, 1, -3),
        ]
            v1 = wigner3j(j1, j2, j3, m1, m2, m3)
            v2 = wigner3j(j2, j3, j1, m2, m3, m1)
            v3 = wigner3j(j3, j1, j2, m3, m1, m2)
            @test isapprox(v1, v2, atol=1e-14)
            @test isapprox(v1, v3, atol=1e-14)
        end
    end

    @testset "3j symmetry: odd permutation" begin
        # Odd perm: multiply by (-1)^(j1+j2+j3)
        for (j1, j2, j3, m1, m2, m3) in [
            (1, 2, 2, 0, 1, -1), (1, 1, 2, 1, 0, -1), (2, 3, 3, 1, 1, -2),
        ]
            v1 = wigner3j(j1, j2, j3, m1, m2, m3)
            v2 = wigner3j(j2, j1, j3, m2, m1, m3)
            phase = (-1)^(j1 + j2 + j3)
            @test isapprox(v1, phase * v2, atol=1e-14)
        end
    end

    @testset "3j symmetry: sign reversal of m" begin
        # (j1 j2 j3; -m1 -m2 -m3) = (-1)^(j1+j2+j3) * (j1 j2 j3; m1 m2 m3)
        for (j1, j2, j3, m1, m2, m3) in [
            (1, 1, 1, 1, 0, -1), (1, 1, 2, 1, -1, 0), (2, 2, 2, 1, 1, -2),
            (2, 2, 4, 2, 0, -2), (3, 3, 4, 3, -1, -2),
        ]
            v1 = wigner3j(j1, j2, j3, m1, m2, m3)
            v2 = wigner3j(j1, j2, j3, -m1, -m2, -m3)
            phase = (-1)^(j1 + j2 + j3)
            @test isapprox(v1, phase * v2, atol=1e-14)
        end
    end

    # ── 8. Orthogonality [DLMF 34.3.16-34.3.17] ────────────────────────────
    @testset "3j orthogonality" begin
        # Sum over m1,m2 of (j1 j2 j3; m1 m2 m3)*(j1 j2 j3'; m1 m2 m3')
        # = delta_{j3,j3'} delta_{m3,m3'} / (2j3+1)
        for j1 in 1:3, j2 in 1:3
            for j3 in abs(j1 - j2):(j1 + j2), j3p in abs(j1 - j2):(j1 + j2)
                for m3 in -min(j3, 2):min(j3, 2)
                    (abs(m3) > j3 || abs(m3) > j3p) && continue
                    s = 0.0
                    for m1 in -j1:j1
                        m2 = -m3 - m1
                        abs(m2) > j2 && continue
                        s += wigner3j(j1, j2, j3, m1, m2, m3) *
                             wigner3j(j1, j2, j3p, m1, m2, m3)
                    end
                    expected = (j3 == j3p) ? 1.0 / (2j3 + 1) : 0.0
                    @test isapprox(s, expected, atol=1e-12)
                end
            end
        end
    end

    # ── 9. Clebsch-Gordan: known values [Sakurai, Mathworld] ───────────────
    @testset "CG coefficients: j1=1, j2=1 table [Sakurai]" begin
        # J=0 sector
        @test isapprox(clebsch_gordan(1, 1, 1, -1, 0, 0), 1 / sqrt(3), atol=1e-14)
        @test isapprox(clebsch_gordan(1, 0, 1, 0, 0, 0), -1 / sqrt(3), atol=1e-14)
        @test isapprox(clebsch_gordan(1, -1, 1, 1, 0, 0), 1 / sqrt(3), atol=1e-14)

        # J=1 sector
        @test isapprox(clebsch_gordan(1, 1, 1, 0, 1, 1), 1 / sqrt(2), atol=1e-14)
        @test isapprox(clebsch_gordan(1, 0, 1, 1, 1, 1), -1 / sqrt(2), atol=1e-14)
        @test isapprox(clebsch_gordan(1, 1, 1, -1, 1, 0), 1 / sqrt(2), atol=1e-14)
        @test isapprox(clebsch_gordan(1, -1, 1, 1, 1, 0), -1 / sqrt(2), atol=1e-14)
        @test isapprox(clebsch_gordan(1, 0, 1, 0, 1, 0), 0.0, atol=1e-14)

        # J=2 sector
        @test isapprox(clebsch_gordan(1, 1, 1, 1, 2, 2), 1.0, atol=1e-14)
        @test isapprox(clebsch_gordan(1, -1, 1, -1, 2, -2), 1.0, atol=1e-14)
        @test isapprox(clebsch_gordan(1, 1, 1, 0, 2, 1), 1 / sqrt(2), atol=1e-14)
        @test isapprox(clebsch_gordan(1, 0, 1, 1, 2, 1), 1 / sqrt(2), atol=1e-14)
        @test isapprox(clebsch_gordan(1, 0, 1, 0, 2, 0), sqrt(2 / 3), atol=1e-14)
        @test isapprox(clebsch_gordan(1, 1, 1, -1, 2, 0), 1 / sqrt(6), atol=1e-14)
        @test isapprox(clebsch_gordan(1, -1, 1, 1, 2, 0), 1 / sqrt(6), atol=1e-14)
    end

    @testset "CG coefficients: j1=2, j2=1 table" begin
        # Stretched state
        @test isapprox(clebsch_gordan(2, 2, 1, 1, 3, 3), 1.0, atol=1e-14)
        # From lowering: |3,2> = sqrt(1/3)|2,2;1,0> + sqrt(2/3)|2,1;1,1>
        @test isapprox(clebsch_gordan(2, 2, 1, 0, 3, 2), sqrt(1 / 3), atol=1e-14)
        @test isapprox(clebsch_gordan(2, 1, 1, 1, 3, 2), sqrt(2 / 3), atol=1e-14)
        # <2,0;1,0|3,0> = sqrt(3/5)
        @test isapprox(clebsch_gordan(2, 0, 1, 0, 3, 0), sqrt(3 / 5), atol=1e-14)
        # <2,0;1,0|1,0>: from orthogonal complement
        @test isapprox(clebsch_gordan(2, 0, 1, 0, 1, 0), _ref_cg(2, 0, 1, 0, 1, 0), atol=1e-14)
        # <2,0;1,0|2,0> = 0 (by antisymmetry under j1<->j2 exchange for J odd relative to j1+j2)
        @test isapprox(clebsch_gordan(2, 0, 1, 0, 2, 0), 0.0, atol=1e-14)
    end

    # ── 10. CG completeness and orthogonality ──────────────────────────────
    @testset "CG completeness" begin
        # Sum_J |CG(j1,m1;j2,m2|J,M)|^2 = 1 for fixed m1,m2
        for j1 in 1:3, j2 in 1:2
            for m1 in -j1:j1, m2 in -j2:j2
                M = m1 + m2
                s = sum(clebsch_gordan(j1, m1, j2, m2, J, M)^2
                        for J in abs(j1 - j2):(j1 + j2))
                @test isapprox(s, 1.0, atol=1e-13)
            end
        end
    end

    @testset "CG orthogonality" begin
        # Sum_{m1,m2} CG(j1,m1;j2,m2|J,M) * CG(j1,m1;j2,m2|J',M) = delta_{J,J'}
        for j1 in 1:3, j2 in 1:2
            for J in abs(j1 - j2):(j1 + j2), Jp in abs(j1 - j2):(j1 + j2)
                for M in -min(J, Jp):min(J, Jp)
                    s = sum(
                        clebsch_gordan(j1, m1, j2, M - m1, J, M) *
                        clebsch_gordan(j1, m1, j2, M - m1, Jp, M)
                        for m1 in max(-j1, M - j2):min(j1, M + j2)
                    )
                    expected = (J == Jp) ? 1.0 : 0.0
                    @test isapprox(s, expected, atol=1e-13)
                end
            end
        end
    end

    # ── 11. CG-3j relation [Mathworld, DLMF] ──────────────────────────────
    @testset "CG-3j relation" begin
        # CG(j1,m1;j2,m2|J,M) = (-1)^(j1-j2+M) * sqrt(2J+1) * (j1 j2 J; m1 m2 -M)
        for j1 in 0:3, j2 in 0:3
            for J in abs(j1 - j2):(j1 + j2)
                for m1 in -j1:j1, m2 in -j2:j2
                    M = m1 + m2
                    abs(M) > J && continue
                    cg = clebsch_gordan(j1, m1, j2, m2, J, M)
                    w3j = wigner3j(j1, j2, J, m1, m2, -M)
                    phase = (-1)^(j1 - j2 + M)
                    expected = phase * sqrt(2J + 1) * w3j
                    @test isapprox(cg, expected, atol=1e-13)
                end
            end
        end
    end

    # ── 12. Gaunt integral verification ────────────────────────────────────
    # Gaunt(l1,m1;l2,m2;l3,m3) = integral Y_{l1,m1} Y_{l2,m2} Y*_{l3,m3} dOmega
    # = sqrt((2l1+1)(2l2+1)(2l3+1)/(4pi)) * (l1 l2 l3;0 0 0) * (l1 l2 l3;m1 m2 -m3)
    function _gaunt(l1, m1, l2, m2, l3, m3)
        sqrt((2l1 + 1) * (2l2 + 1) * (2l3 + 1) / (4pi)) *
        wigner3j(l1, l2, l3, 0, 0, 0) *
        wigner3j(l1, l2, l3, m1, m2, -m3)
    end

    @testset "Gaunt: Y_{0,0} normalization" begin
        # integral Y_{0,0} Y_{l,m} Y*_{l,m} dOmega = 1/sqrt(4pi) * (-1)^m
        # because Y*_{l,m} has the Condon-Shortley (-1)^m phase
        for l in 0:5, m in -l:l
            g = _gaunt(0, 0, l, m, l, m)
            ref = (-1)^m / sqrt(4pi)
            @test isapprox(g, ref, atol=1e-13)
        end
    end

    @testset "Gaunt: parity selection" begin
        # Gaunt = 0 if l1 + l2 + l3 is odd (from (l1 l2 l3;0 0 0) = 0)
        for l1 in 0:4, l2 in 0:4, l3 in abs(l1 - l2):(l1 + l2)
            if isodd(l1 + l2 + l3)
                @test _gaunt(l1, 0, l2, 0, l3, 0) == 0.0
            end
        end
    end

    @testset "Gaunt: specific values for harmonic products" begin
        # Y_{1,0} * Y_{1,0}: coefficients of Y_{0,0} and Y_{2,0}
        # c_0 = Gaunt(1,0;1,0;0,0) = sqrt(9/(4pi)) * (1/3) = 1/(2*sqrt(pi))
        @test isapprox(_gaunt(1, 0, 1, 0, 0, 0), 1 / (2 * sqrt(pi)), atol=1e-13)
        # c_2 = Gaunt(1,0;1,0;2,0) = sqrt(45/(4pi)) * (2/15)
        @test isapprox(_gaunt(1, 0, 1, 0, 2, 0), sqrt(45 / (4pi)) * (2 / 15), atol=1e-13)

        # Y_{1,1} * Y_{1,-1}: coefficient of Y_{0,0}
        # = sqrt(9/(4pi)) * (-1/sqrt(3)) * (1/sqrt(3)) = -1/sqrt(4pi)
        @test isapprox(_gaunt(1, 1, 1, -1, 0, 0), -1 / sqrt(4pi), atol=1e-13)

        # Y_{2,0} * Y_{2,0}: coefficients of Y_{0,0}, Y_{2,0}, Y_{4,0}
        @test isapprox(_gaunt(2, 0, 2, 0, 0, 0), 1 / (2 * sqrt(pi)), atol=1e-13)
        g_220 = _gaunt(2, 0, 2, 0, 2, 0)
        g_240 = _gaunt(2, 0, 2, 0, 4, 0)
        @test abs(g_220) > 0.1   # nonzero
        @test abs(g_240) > 0.1   # nonzero

        # Cross-check: Y_{1,0} * Y_{2,0} -> Y_{1,0} and Y_{3,0} (parity)
        @test abs(_gaunt(1, 0, 2, 0, 1, 0)) > 0.1
        @test abs(_gaunt(1, 0, 2, 0, 3, 0)) > 0.1
        @test _gaunt(1, 0, 2, 0, 2, 0) == 0.0  # l1+l2+l3 = 5 odd
    end

    # ── 13. harmonic_product output vs Gaunt integral ──────────────────────
    @testset "harmonic_product matches Gaunt integral" begin
        function _extract_coeffs(result)
            coeffs = Dict{Int,Float64}()
            terms = result isa TSum ? result.terms : [result]
            for term in terms
                if term isa TProduct
                    sh = nothing
                    c = Float64(term.scalar)
                    for f in term.factors
                        if f isa ScalarHarmonic
                            sh = f
                        elseif f isa TScalar
                            c *= Float64(f.val)
                        end
                    end
                    sh !== nothing && (coeffs[sh.l] = c)
                end
            end
            coeffs
        end

        # Y_{1,0} * Y_{1,0}
        c10 = _extract_coeffs(harmonic_product(ScalarHarmonic(1, 0), ScalarHarmonic(1, 0)))
        @test isapprox(c10[0], _gaunt(1, 0, 1, 0, 0, 0), atol=1e-13)
        @test isapprox(c10[2], _gaunt(1, 0, 1, 0, 2, 0), atol=1e-13)

        # Y_{1,1} * Y_{1,-1}: m3 = 0, so (-1)^m3 = 1 and coeff = Gaunt directly
        c11 = _extract_coeffs(harmonic_product(ScalarHarmonic(1, 1), ScalarHarmonic(1, -1)))
        @test isapprox(c11[0], _gaunt(1, 1, 1, -1, 0, 0), atol=1e-13)
        @test haskey(c11, 2)

        # Y_{2,0} * Y_{2,0} -> 3 terms: l=0,2,4
        c20 = _extract_coeffs(harmonic_product(ScalarHarmonic(2, 0), ScalarHarmonic(2, 0)))
        @test sort(collect(keys(c20))) == [0, 2, 4]
        @test isapprox(c20[0], _gaunt(2, 0, 2, 0, 0, 0), atol=1e-13)
        @test isapprox(c20[2], _gaunt(2, 0, 2, 0, 2, 0), atol=1e-13)
        @test isapprox(c20[4], _gaunt(2, 0, 2, 0, 4, 0), atol=1e-13)

        # Y_{0,0} * Y_{l,m} = (1/sqrt(4pi)) * Y_{l,m}
        for l in 1:4
            result = harmonic_product(ScalarHarmonic(0, 0), ScalarHarmonic(l, 0))
            # Should be a single term
            if result isa TProduct
                sh = nothing
                c = Float64(result.scalar)
                for f in result.factors
                    if f isa ScalarHarmonic
                        sh = f
                    elseif f isa TScalar
                        c *= Float64(f.val)
                    end
                end
                @test sh == ScalarHarmonic(l, 0)
                @test isapprox(c, 1 / sqrt(4pi), atol=1e-13)
            else
                @test false  # expected single-term product
            end
        end
    end

    # ── 14. Exhaustive brute-force vs reference for small j ────────────────
    @testset "Exhaustive match: wigner3j vs independent Racah, j<=3" begin
        count = 0
        for j1 in 0:3, j2 in 0:3, j3 in abs(j1 - j2):(j1 + j2)
            for m1 in -j1:j1, m2 in -j2:j2
                m3 = -m1 - m2
                abs(m3) > j3 && continue
                ref = _ref_3j(j1, j2, j3, m1, m2, m3)
                our = wigner3j(j1, j2, j3, m1, m2, m3)
                @test isapprox(our, ref, atol=1e-13)
                count += 1
            end
        end
        # Verify we tested a substantial number
        @test count > 500
    end

    @testset "Exhaustive match: clebsch_gordan vs reference, j<=3" begin
        count = 0
        for j1 in 0:3, j2 in 0:3
            for J in abs(j1 - j2):(j1 + j2)
                for m1 in -j1:j1, m2 in -j2:j2
                    M = m1 + m2
                    abs(M) > J && continue
                    ref = _ref_cg(j1, m1, j2, m2, J, M)
                    our = clebsch_gordan(j1, m1, j2, m2, J, M)
                    @test isapprox(our, ref, atol=1e-13)
                    count += 1
                end
            end
        end
        @test count > 500
    end

    # ── 15. Stretched state test [CS convention] ───────────────────────────
    @testset "CG stretched states" begin
        # <j1,j1;j2,j2|j1+j2,j1+j2> = 1 always (Condon-Shortley)
        for j1 in 0:5, j2 in 0:5
            @test isapprox(clebsch_gordan(j1, j1, j2, j2, j1 + j2, j1 + j2), 1.0, atol=1e-14)
        end
    end

end
