# Tests for vector harmonic orthogonality relations.
#
# Ground truth: Martel & Poisson, Phys. Rev. D 71, 104003 (2005),
#               arXiv:gr-qc/0502028, Section III.

@testset "Vector harmonic orthogonality" begin

    # ── Even-even orthogonality (MP Eq 3.3) ─────────────────────────────
    # integral Y_bar^A_{lm} Y_A^{l'm'} dOmega = l(l+1) delta_{ll'} delta_{mm'}
    @testset "Even-even: MP Eq 3.3" begin
        # Specific values
        # l=2, m=1: l(l+1) = 6
        @test inner_product(EvenVectorHarmonic(2, 1, up(:a)),
                            EvenVectorHarmonic(2, 1, up(:a))) == TScalar(6)
        # Different l => 0
        @test inner_product(EvenVectorHarmonic(2, 1, up(:a)),
                            EvenVectorHarmonic(3, 1, up(:a))) == TScalar(0)
        # Different m => 0
        @test inner_product(EvenVectorHarmonic(2, 1, up(:a)),
                            EvenVectorHarmonic(2, 0, up(:a))) == TScalar(0)
        # l=1: l(l+1) = 2
        @test inner_product(EvenVectorHarmonic(1, 0, up(:a)),
                            EvenVectorHarmonic(1, 0, up(:a))) == TScalar(2)
        # l=3, m=2: l(l+1) = 12
        @test inner_product(EvenVectorHarmonic(3, 2, up(:a)),
                            EvenVectorHarmonic(3, 2, up(:a))) == TScalar(12)
    end

    # ── Odd-odd orthogonality (MP Eq 3.4) ───────────────────────────────
    # integral X_bar^A_{lm} X_A^{l'm'} dOmega = l(l+1) delta_{ll'} delta_{mm'}
    @testset "Odd-odd: MP Eq 3.4" begin
        # l=2, m=1: l(l+1) = 6
        @test inner_product(OddVectorHarmonic(2, 1, up(:a)),
                            OddVectorHarmonic(2, 1, up(:a))) == TScalar(6)
        # l=3, m=2: l(l+1) = 12
        @test inner_product(OddVectorHarmonic(3, 2, up(:a)),
                            OddVectorHarmonic(3, 2, up(:a))) == TScalar(12)
        # Different l => 0
        @test inner_product(OddVectorHarmonic(1, 0, up(:a)),
                            OddVectorHarmonic(2, 0, up(:a))) == TScalar(0)
        # Different m => 0
        @test inner_product(OddVectorHarmonic(2, 1, up(:a)),
                            OddVectorHarmonic(2, -1, up(:a))) == TScalar(0)
    end

    # ── Even-odd cross-orthogonality (MP Eq 3.5) ────────────────────────
    # integral Y_bar^A_{lm} X_A^{l'm'} dOmega = 0 for all l,m,l',m'
    @testset "Even-odd cross: MP Eq 3.5" begin
        @test inner_product(EvenVectorHarmonic(2, 1, up(:a)),
                            OddVectorHarmonic(2, 1, up(:a))) == TScalar(0)
        @test inner_product(OddVectorHarmonic(2, 1, up(:a)),
                            EvenVectorHarmonic(2, 1, up(:a))) == TScalar(0)
        # Same (l,m) still zero
        @test inner_product(EvenVectorHarmonic(3, -1, up(:a)),
                            OddVectorHarmonic(3, -1, up(:a))) == TScalar(0)
    end

    # ── vector_inner_product matches inner_product ───────────────────────
    @testset "vector_inner_product == inner_product" begin
        Y1 = EvenVectorHarmonic(2, 1, up(:a))
        Y2 = EvenVectorHarmonic(2, 1, up(:a))
        X1 = OddVectorHarmonic(2, 1, up(:a))
        X2 = OddVectorHarmonic(3, 0, up(:a))
        @test vector_inner_product(Y1, Y2) == inner_product(Y1, Y2)
        @test vector_inner_product(X1, X1) == inner_product(X1, X1)
        @test vector_inner_product(Y1, X1) == inner_product(Y1, X1)
        @test vector_inner_product(X2, Y1) == inner_product(X2, Y1)
    end

    # ── norm_squared consistency ─────────────────────────────────────────
    @testset "norm_squared == inner_product(Y,Y).val" begin
        for l in 1:5, m in -l:l
            Y = EvenVectorHarmonic(l, m, up(:a))
            X = OddVectorHarmonic(l, m, up(:a))
            @test norm_squared(Y) == inner_product(Y, Y).val
            @test norm_squared(X) == inner_product(X, X).val
        end
    end

    # ── Exhaustive check l=1..5 (MP Eqs 3.3-3.5) ──────────────────────
    @testset "Exhaustive l=1..5: MP Eqs 3.3-3.5" begin
        for l1 in 1:5, l2 in 1:5
            for m1 in -l1:l1, m2 in -l2:l2
                Y1 = EvenVectorHarmonic(l1, m1, up(:a))
                Y2 = EvenVectorHarmonic(l2, m2, up(:a))
                X1 = OddVectorHarmonic(l1, m1, up(:a))
                X2 = OddVectorHarmonic(l2, m2, up(:a))

                same = (l1 == l2 && m1 == m2)
                norm_val = l1 * (l1 + 1)

                # Eq 3.3: even-even
                @test inner_product(Y1, Y2) == (same ? TScalar(norm_val) : TScalar(0))
                # Eq 3.4: odd-odd
                @test inner_product(X1, X2) == (same ? TScalar(norm_val) : TScalar(0))
                # Eq 3.5: even-odd
                @test inner_product(Y1, X2) == TScalar(0)
                @test inner_product(X1, Y2) == TScalar(0)
            end
        end
    end

    # ── Scalar harmonic orthogonality (MP (standard, not numbered in paper), verify existing) ───────
    @testset "Scalar: MP (standard, not numbered in paper)" begin
        for l1 in 0:4, l2 in 0:4
            for m1 in -l1:l1, m2 in -l2:l2
                expected = (l1 == l2 && m1 == m2) ? TScalar(1) : TScalar(0)
                @test inner_product(ScalarHarmonic(l1, m1),
                                    ScalarHarmonic(l2, m2)) == expected
            end
        end
    end

end
