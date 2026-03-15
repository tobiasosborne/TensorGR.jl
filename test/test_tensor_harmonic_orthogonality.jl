# Tests for tensor harmonic orthogonality and normalization.
#
# Ground truth: Martel & Poisson, Phys. Rev. D 71, 104003 (2005),
#               arXiv:gr-qc/0502028, Section II.D, Eqs 2.18-2.21.

@testset "Tensor harmonic orthogonality" begin

    # ── Y-Y orthogonality (MP Eq 2.18) ────────────────────────────────────
    # integral Y^{ab}_{lm} Y*_{ab,l'm'} dOmega = 2 delta_{ll'} delta_{mm'}
    @testset "Y-Y: MP Eq 2.18" begin
        # l=2, m=1: norm = 2
        @test inner_product(EvenTensorHarmonicY(2, 1, up(:a), up(:b)),
                            EvenTensorHarmonicY(2, 1, up(:a), up(:b))) == TScalar(2)
        # l=0, m=0: norm = 2
        @test inner_product(EvenTensorHarmonicY(0, 0, up(:a), up(:b)),
                            EvenTensorHarmonicY(0, 0, up(:a), up(:b))) == TScalar(2)
        # Different l => 0
        @test inner_product(EvenTensorHarmonicY(2, 1, up(:a), up(:b)),
                            EvenTensorHarmonicY(3, 1, up(:a), up(:b))) == TScalar(0)
        # Different m => 0
        @test inner_product(EvenTensorHarmonicY(2, 1, up(:a), up(:b)),
                            EvenTensorHarmonicY(2, 0, up(:a), up(:b))) == TScalar(0)
        # l=5, m=-3: norm = 2
        @test inner_product(EvenTensorHarmonicY(5, -3, up(:a), up(:b)),
                            EvenTensorHarmonicY(5, -3, up(:a), up(:b))) == TScalar(2)
    end

    # ── Z-Z orthogonality (MP Eq 2.19) ────────────────────────────────────
    # integral Z^{ab}_{lm} Z*_{ab,l'm'} dOmega = 1/2 (l-1)l(l+1)(l+2) delta_{ll'} delta_{mm'}
    @testset "Z-Z: MP Eq 2.19" begin
        # l=3, m=0: 1/2 * 2 * 3 * 4 * 5 = 60
        @test inner_product(EvenTensorHarmonicZ(3, 0, up(:a), up(:b)),
                            EvenTensorHarmonicZ(3, 0, up(:a), up(:b))) == TScalar(60)
        # l=2, m=1: 1/2 * 1 * 2 * 3 * 4 = 12
        @test inner_product(EvenTensorHarmonicZ(2, 1, up(:a), up(:b)),
                            EvenTensorHarmonicZ(2, 1, up(:a), up(:b))) == TScalar(12)
        # l=4, m=-2: 1/2 * 3 * 4 * 5 * 6 = 180
        @test inner_product(EvenTensorHarmonicZ(4, -2, up(:a), up(:b)),
                            EvenTensorHarmonicZ(4, -2, up(:a), up(:b))) == TScalar(180)
        # Different l => 0
        @test inner_product(EvenTensorHarmonicZ(2, 1, up(:a), up(:b)),
                            EvenTensorHarmonicZ(3, 1, up(:a), up(:b))) == TScalar(0)
        # Different m => 0
        @test inner_product(EvenTensorHarmonicZ(3, 1, up(:a), up(:b)),
                            EvenTensorHarmonicZ(3, 0, up(:a), up(:b))) == TScalar(0)
    end

    # ── X-X orthogonality (MP Eq 2.20) ────────────────────────────────────
    # integral X^{ab}_{lm} X*_{ab,l'm'} dOmega = 1/2 (l-1)l(l+1)(l+2) delta_{ll'} delta_{mm'}
    @testset "X-X: MP Eq 2.20" begin
        # l=3, m=0: same norm as Z-Z = 60
        @test inner_product(OddTensorHarmonic(3, 0, up(:a), up(:b)),
                            OddTensorHarmonic(3, 0, up(:a), up(:b))) == TScalar(60)
        # l=2, m=1: 12
        @test inner_product(OddTensorHarmonic(2, 1, up(:a), up(:b)),
                            OddTensorHarmonic(2, 1, up(:a), up(:b))) == TScalar(12)
        # l=5, m=-3: 1/2 * 4 * 5 * 6 * 7 = 420
        @test inner_product(OddTensorHarmonic(5, -3, up(:a), up(:b)),
                            OddTensorHarmonic(5, -3, up(:a), up(:b))) == TScalar(420)
        # Different l => 0
        @test inner_product(OddTensorHarmonic(2, 0, up(:a), up(:b)),
                            OddTensorHarmonic(3, 0, up(:a), up(:b))) == TScalar(0)
        # Different m => 0
        @test inner_product(OddTensorHarmonic(3, 2, up(:a), up(:b)),
                            OddTensorHarmonic(3, -2, up(:a), up(:b))) == TScalar(0)
    end

    # ── Cross-type orthogonality (MP Eq 2.21) ─────────────────────────────
    # All cross-type inner products vanish identically
    @testset "Cross-type: MP Eq 2.21" begin
        Y = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        Z = EvenTensorHarmonicZ(2, 1, up(:a), up(:b))
        X = OddTensorHarmonic(2, 1, up(:a), up(:b))

        # Y-Z
        @test inner_product(Y, Z) == TScalar(0)
        @test inner_product(Z, Y) == TScalar(0)
        # Y-X
        @test inner_product(Y, X) == TScalar(0)
        @test inner_product(X, Y) == TScalar(0)
        # Z-X
        @test inner_product(Z, X) == TScalar(0)
        @test inner_product(X, Z) == TScalar(0)

        # Same (l,m) still zero for cross-type
        Y2 = EvenTensorHarmonicY(3, -1, up(:a), up(:b))
        Z2 = EvenTensorHarmonicZ(3, -1, up(:a), up(:b))
        X2 = OddTensorHarmonic(3, -1, up(:a), up(:b))
        @test inner_product(Y2, Z2) == TScalar(0)
        @test inner_product(Y2, X2) == TScalar(0)
        @test inner_product(Z2, X2) == TScalar(0)
    end

    # ── tensor_inner_product matches inner_product ────────────────────────
    @testset "tensor_inner_product == inner_product" begin
        Y = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        Z = EvenTensorHarmonicZ(3, 0, up(:a), up(:b))
        X = OddTensorHarmonic(2, 1, up(:a), up(:b))

        # Same-type
        @test tensor_inner_product(Y, Y) == inner_product(Y, Y)
        @test tensor_inner_product(Z, Z) == inner_product(Z, Z)
        @test tensor_inner_product(X, X) == inner_product(X, X)

        # Cross-type
        Y2 = EvenTensorHarmonicY(3, 0, up(:a), up(:b))
        @test tensor_inner_product(Y2, Z) == inner_product(Y2, Z)
        @test tensor_inner_product(Z, Y2) == inner_product(Z, Y2)
        @test tensor_inner_product(Y, X) == inner_product(Y, X)
        @test tensor_inner_product(X, Y) == inner_product(X, Y)
        Z2 = EvenTensorHarmonicZ(2, 1, up(:a), up(:b))
        @test tensor_inner_product(Z2, X) == inner_product(Z2, X)
        @test tensor_inner_product(X, Z2) == inner_product(X, Z2)
    end

    # ── norm_squared consistency ──────────────────────────────────────────
    @testset "norm_squared == inner_product(T,T).val" begin
        for l in 0:5
            for m in -l:l
                Y = EvenTensorHarmonicY(l, m, up(:a), up(:b))
                @test norm_squared(Y) == inner_product(Y, Y).val
            end
        end
        for l in 2:5
            for m in -l:l
                Z = EvenTensorHarmonicZ(l, m, up(:a), up(:b))
                X = OddTensorHarmonic(l, m, up(:a), up(:b))
                @test norm_squared(Z) == inner_product(Z, Z).val
                @test norm_squared(X) == inner_product(X, X).val
            end
        end
    end

    # ── Exhaustive check l=2..5, all m, all 9 type combinations ──────────
    # (MP Eqs 2.18-2.21)
    @testset "Exhaustive l=2..5: MP Eqs 2.18-2.21" begin
        for l1 in 2:5, l2 in 2:5
            for m1 in -l1:l1, m2 in -l2:l2
                Y1 = EvenTensorHarmonicY(l1, m1, up(:a), up(:b))
                Y2 = EvenTensorHarmonicY(l2, m2, up(:a), up(:b))
                Z1 = EvenTensorHarmonicZ(l1, m1, up(:a), up(:b))
                Z2 = EvenTensorHarmonicZ(l2, m2, up(:a), up(:b))
                X1 = OddTensorHarmonic(l1, m1, up(:a), up(:b))
                X2 = OddTensorHarmonic(l2, m2, up(:a), up(:b))

                same = (l1 == l2 && m1 == m2)

                # Eq 2.18: Y-Y norm = 2
                @test inner_product(Y1, Y2) == (same ? TScalar(2) : TScalar(0))

                # Eq 2.19: Z-Z norm = 1/2 (l-1)l(l+1)(l+2)
                zz_norm = (l1 - 1) * l1 * (l1 + 1) * (l1 + 2) // 2
                @test inner_product(Z1, Z2) == (same ? TScalar(zz_norm) : TScalar(0))

                # Eq 2.20: X-X norm = same as Z-Z
                xx_norm = (l1 - 1) * l1 * (l1 + 1) * (l1 + 2) // 2
                @test inner_product(X1, X2) == (same ? TScalar(xx_norm) : TScalar(0))

                # Eq 2.21: all cross-type vanish
                @test inner_product(Y1, Z2) == TScalar(0)
                @test inner_product(Z1, Y2) == TScalar(0)
                @test inner_product(Y1, X2) == TScalar(0)
                @test inner_product(X1, Y2) == TScalar(0)
                @test inner_product(Z1, X2) == TScalar(0)
                @test inner_product(X1, Z2) == TScalar(0)
            end
        end
    end

end
