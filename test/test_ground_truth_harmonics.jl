# Ground-truth verification of spherical harmonics against
# Martel & Poisson, Phys. Rev. D 71, 104003 (2005), arXiv:gr-qc/0502028.
#
# Reference equations (ar5iv numbering):
#   Unnumbered (Sec III): [Omega^{AB} D_A D_B + l(l+1)] Y^{lm} = 0
#     => Delta_{S2} Y_{lm} = -l(l+1) Y_{lm}
#   Eq 11: Y_A^{lm} := D_A Y^{lm}  (even-parity vector harmonic)
#   Eq 12: X_A^{lm} := -epsilon_A^B D_B Y^{lm}  (odd-parity vector harmonic)
#   Eq 13: integral Y_bar^A_{lm} Y_A^{l'm'} dOmega = l(l+1) delta_{ll'} delta_{mm'}
#   Eq 14: integral X_bar^A_{lm} X_A^{l'm'} dOmega = l(l+1) delta_{ll'} delta_{mm'}
#   Eq 15: integral Y_bar^A_{lm} X_A^{l'm'} dOmega = 0
#   Eq 16: Y_{AB}^{lm} := [D_A D_B + (1/2)l(l+1) Omega_{AB}] Y^{lm}
#   Eq 17: X_{AB}^{lm} := -(1/2)(epsilon_A^C D_B + epsilon_B^C D_A) D_C Y^{lm}
#   Eq 18: integral Y_bar^{AB}_{lm} Y_{AB}^{l'm'} dOmega = (1/2)(l-1)l(l+1)(l+2) delta_{ll'} delta_{mm'}
#   Eq 19: integral X_bar^{AB}_{lm} X_{AB}^{l'm'} dOmega = (1/2)(l-1)l(l+1)(l+2) delta_{ll'} delta_{mm'}
#   Eq 20: integral Y_bar^{AB}_{lm} X_{AB}^{l'm'} dOmega = 0
#   Eq 21: Omega^{AB} Y_{AB}^{lm} = 0 = Omega^{AB} X_{AB}^{lm}  (tracelessness)
#
# Standard identity (Condon-Shortley phase convention):
#   Y*_{lm} = (-1)^m Y_{l,-m}

@testset "Ground truth: Martel & Poisson (2005)" begin

    # ── 1. Angular Laplacian eigenvalue ──────────────────────────────────
    # MP Sec III: Delta_{S2} Y_{lm} = -l(l+1) Y_{lm}
    @testset "Angular Laplacian eigenvalue (MP Sec III)" begin
        for l in 0:10
            Y = ScalarHarmonic(l, 0)
            result = angular_laplacian(Y)
            expected_eigenvalue = Rational{Int}(-l * (l + 1))
            @test result isa TProduct
            @test result.scalar == expected_eigenvalue
            @test result.factors[1] == Y
        end
        # Also check non-zero m values
        for l in 1:5, m in -l:l
            Y = ScalarHarmonic(l, m)
            result = angular_laplacian(Y)
            @test result.scalar == Rational{Int}(-l * (l + 1))
            @test result.factors[1] == Y
        end
    end

    # ── 2. Conjugation (Condon-Shortley) ────────────────────────────────
    # Y*_{lm} = (-1)^m Y_{l,-m}
    @testset "Conjugation: Y*_{lm} = (-1)^m Y_{l,-m}" begin
        for l in 0:5, m in -l:l
            Y = ScalarHarmonic(l, m)
            Yc = conjugate(Y)

            if m == 0
                # Conjugate of m=0 is just Y_{l,0} (real)
                @test Yc isa ScalarHarmonic
                @test Yc == Y
            else
                expected_sign = iseven(m) ? 1 // 1 : -1 // 1
                @test Yc isa TProduct
                @test Yc.scalar == expected_sign
                @test length(Yc.factors) == 1
                @test Yc.factors[1] isa ScalarHarmonic
                @test Yc.factors[1].l == l
                @test Yc.factors[1].m == -m
            end
        end
    end

    # ── 3. Double conjugation: conj(conj(Y)) = Y ────────────────────────
    @testset "Double conjugation: conj(conj(Y)) = Y" begin
        for l in 0:5, m in -l:l
            Y = ScalarHarmonic(l, m)
            Yc = conjugate(Y)

            if m == 0
                # conj(Y_{l,0}) = Y_{l,0}, so conj again = Y_{l,0}
                Ycc = conjugate(Yc)
                @test Ycc == Y
            else
                # Yc = (-1)^m * Y_{l,-m}
                # conj(Yc) = (-1)^m * conj(Y_{l,-m})
                #          = (-1)^m * (-1)^{-m} * Y_{l,m}
                #          = (-1)^{m+(-m)} * Y_{l,m} = Y_{l,m}
                inner_Y = Yc.factors[1]  # ScalarHarmonic(l, -m)
                inner_conj = conjugate(inner_Y)
                # For -m == 0 this is just the harmonic, otherwise TProduct
                if -m == 0
                    # This shouldn't happen since m != 0
                    @test false
                else
                    @test inner_conj isa TProduct
                    combined_sign = Yc.scalar * inner_conj.scalar
                    @test combined_sign == 1 // 1
                    @test inner_conj.factors[1] == Y
                end
            end
        end
    end

    # ── 4. Orthogonality: <Y_{l1,m1}|Y_{l2,m2}> = delta ────────────────
    # Standard normalization on S^2
    @testset "Orthogonality: <Y_{l1,m1}|Y_{l2,m2}> = delta_{ll'} delta_{mm'}" begin
        for l1 in 0:3, l2 in 0:3
            for m1 in -l1:l1, m2 in -l2:l2
                ip = inner_product(ScalarHarmonic(l1, m1), ScalarHarmonic(l2, m2))
                expected = (l1 == l2 && m1 == m2) ? TScalar(1) : TScalar(0)
                @test ip == expected
            end
        end
    end

    # ── 5. Quantum number validation ────────────────────────────────────
    @testset "Quantum number validation" begin
        # l must be non-negative
        @test_throws ArgumentError ScalarHarmonic(-1, 0)
        @test_throws ArgumentError ScalarHarmonic(-5, 0)
        # |m| must be <= l
        @test_throws ArgumentError ScalarHarmonic(2, 3)
        @test_throws ArgumentError ScalarHarmonic(2, -3)
        @test_throws ArgumentError ScalarHarmonic(0, 1)
        @test_throws ArgumentError ScalarHarmonic(0, -1)
        # Valid boundary cases
        @test ScalarHarmonic(0, 0).l == 0
        @test ScalarHarmonic(5, 5).m == 5
        @test ScalarHarmonic(5, -5).m == -5
    end

    # ── 6. Special values ───────────────────────────────────────────────
    @testset "Special angular Laplacian values" begin
        # l=0: eigenvalue = 0 (constant on sphere)
        Y00 = ScalarHarmonic(0, 0)
        @test angular_laplacian(Y00).scalar == 0 // 1

        # l=1: eigenvalue = -2 (dipole)
        Y10 = ScalarHarmonic(1, 0)
        @test angular_laplacian(Y10).scalar == -2 // 1

        # l=2: eigenvalue = -6 (quadrupole)
        Y20 = ScalarHarmonic(2, 0)
        @test angular_laplacian(Y20).scalar == -6 // 1

        # l=3: eigenvalue = -12 (octupole)
        Y30 = ScalarHarmonic(3, 0)
        @test angular_laplacian(Y30).scalar == -12 // 1

        # General check: eigenvalue formula -l(l+1)
        for l in 0:20
            @test angular_laplacian(ScalarHarmonic(l, 0)).scalar == Rational{Int}(-l * (l + 1))
        end
    end

    # ── 7. Harmonic product (Gaunt integral) ────────────────────────────
    # Y_{l1,m1} * Y_{l2,m2} = sum_l3 c_l3 Y_{l3,m3} where m3 = m1+m2
    @testset "Harmonic product: Gaunt coefficients" begin
        # Y_{0,0} * Y_{lm} = (1/sqrt(4pi)) Y_{lm}
        Y00 = ScalarHarmonic(0, 0)
        for l in 0:4, m in -l:l
            Ylm = ScalarHarmonic(l, m)
            result = harmonic_product(Y00, Ylm)
            # Should be a single term: c * Y_{l,m}
            if result isa TProduct
                c = Float64(result.scalar)
                for f in result.factors
                    if f isa TScalar
                        c *= Float64(f.val)
                    end
                end
                @test isapprox(c, 1 / sqrt(4pi), atol = 1e-13)
            elseif result isa TScalar
                # l=0 case: might simplify to scalar
                @test l == 0
            end
        end

        # Y_{1,0} * Y_{1,0}: m3=0, terms at l3=0 and l3=2 (parity selection)
        Y10 = ScalarHarmonic(1, 0)
        result10 = harmonic_product(Y10, Y10)
        @test result10 isa TSum
        @test length(result10.terms) == 2

        # Y_{2,0} * Y_{2,0}: terms at l3=0,2,4
        Y20 = ScalarHarmonic(2, 0)
        result20 = harmonic_product(Y20, Y20)
        @test result20 isa TSum
        @test length(result20.terms) == 3
    end

    # ── 8. Wigner 3j special values ─────────────────────────────────────
    @testset "Wigner 3j: known exact values" begin
        # (j j 0; m -m 0) = (-1)^(j-m) / sqrt(2j+1)
        for j in 0:6, m in -j:j
            expected = (-1)^(j - m) / sqrt(2j + 1)
            @test isapprox(wigner3j(j, j, 0, m, -m, 0), expected, atol = 1e-13)
        end
    end

    # ── 9. Clebsch-Gordan completeness ──────────────────────────────────
    @testset "Clebsch-Gordan completeness" begin
        # sum_J |<j1,m1;j2,m2|J,M>|^2 = 1 for fixed j1,m1,j2,m2 (M=m1+m2)
        for j1 in 0:2, j2 in 0:2
            for m1 in -j1:j1, m2 in -j2:j2
                M = m1 + m2
                s = sum(clebsch_gordan(j1, m1, j2, m2, J, M)^2
                        for J in abs(j1 - j2):(j1 + j2))
                @test isapprox(s, 1.0, atol = 1e-12)
            end
        end
    end

    # ── 10. Dagger == conjugate ─────────────────────────────────────────
    @testset "Dagger delegates to conjugate" begin
        for l in 0:3, m in -l:l
            Y = ScalarHarmonic(l, m)
            @test dagger(Y) == conjugate(Y)
        end
    end

    # ── 11. Vector/tensor harmonics coverage gap ────────────────────────
    # These are defined in Martel & Poisson (2005) Eqs 11-21 but NOT YET
    # implemented in TensorGR.jl. Documenting what would need to be verified:
    #
    # Eq 11: EvenVectorHarmonic Y^a_{lm} = D^a Y_{lm}
    #   - norm_squared = l(l+1) for l >= 1  [Eq 13]
    #   - divergence D_a Y^a = Delta Y = -l(l+1) Y  (follows from definition)
    #
    # Eq 12: OddVectorHarmonic X^a_{lm} = -epsilon^a_b D^b Y_{lm}
    #   - norm_squared = l(l+1) for l >= 1  [Eq 14]
    #   - divergence-free: D_a X^a = 0 (antisymmetry of epsilon)
    #   - cross-orthogonal to even: <Y^a, X_a> = 0  [Eq 15]
    #
    # Eq 16: EvenTensorHarmonic Y_{AB}^{lm} = [D_A D_B + (1/2)l(l+1) Omega_{AB}] Y^{lm}
    #   - norm_squared = (1/2)(l-1)l(l+1)(l+2) for l >= 2  [Eq 18]
    #   - traceless: Omega^{AB} Y_{AB} = 0  [Eq 21]
    #
    # Eq 17: OddTensorHarmonic X_{AB}^{lm} = -(1/2)(eps_A^C D_B + eps_B^C D_A) D_C Y^{lm}
    #   - norm_squared = (1/2)(l-1)l(l+1)(l+2) for l >= 2  [Eq 19]
    #   - traceless: Omega^{AB} X_{AB} = 0  [Eq 21]
    #   - cross-orthogonal: <Y_{AB}, X^{AB}> = 0  [Eq 20]
    #
    # Low multipoles (MP Sec VIII):
    #   l=0: Y_A = X_A = Y_{AB} = X_{AB} = 0
    #   l=1: Y_{AB} = X_{AB} = 0 (tensor harmonics vanish)

    @testset "Vector harmonics implemented, tensor harmonics not yet" begin
        # Vector harmonics (MP Eqs 11-15) are now implemented
        @test isdefined(TensorGR, :EvenVectorHarmonic)
        @test isdefined(TensorGR, :OddVectorHarmonic)
        @test isdefined(TensorGR, :norm_squared)
        @test isdefined(TensorGR, :divergence_eigenvalue)
        @test isdefined(TensorGR, :curl_eigenvalue)
        @test isdefined(TensorGR, :vector_inner_product)
        # Tensor harmonics (MP Eqs 16-21) not yet implemented
        @test !isdefined(TensorGR, :EvenTensorHarmonic)
        @test !isdefined(TensorGR, :OddTensorHarmonic)
    end
end
