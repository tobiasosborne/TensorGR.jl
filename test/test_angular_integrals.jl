@testset "Angular integrals" begin

    @testset "gaunt_integral: selection rules" begin
        # m1 + m2 != m3 => 0
        @test gaunt_integral(1, 0, 1, 0, 1, 1) == 0.0
        # Triangle inequality violated
        @test gaunt_integral(1, 0, 1, 0, 5, 0) == 0.0
        # Parity: l1 + l2 + l3 odd => 0
        @test gaunt_integral(1, 0, 1, 0, 1, 0) == 0.0
    end

    @testset "gaunt_integral: known values" begin
        # integral Y_{0,0} Y_{l,m} Y*_{l,m} dOmega = 1/sqrt(4pi)
        for l in 0:3, m in -l:l
            val = gaunt_integral(0, 0, l, m, l, m)
            @test isapprox(val, 1 / sqrt(4pi), atol=1e-13)
        end

        # integral Y_{1,0} Y_{1,0} Y*_{0,0} dOmega
        # = sqrt(3*3*1/(4pi)) * (1 1 0;0 0 0) * (1 1 0;0 0 0)
        expected = sqrt(9 / (4pi)) * wigner3j(1, 1, 0, 0, 0, 0)^2
        @test isapprox(gaunt_integral(1, 0, 1, 0, 0, 0), expected, atol=1e-13)

        # integral Y_{1,0} Y_{1,0} Y*_{2,0} dOmega
        expected2 = sqrt(9 * 5 / (4pi)) * wigner3j(1, 1, 2, 0, 0, 0)^2
        @test isapprox(gaunt_integral(1, 0, 1, 0, 2, 0), expected2, atol=1e-13)
    end

    @testset "gaunt_integral: consistency with harmonic_product" begin
        # The Gaunt coefficient used in harmonic_product should match gaunt_integral
        # harmonic_product gives: c_{l3} = (-1)^m3 * prefactor * w3j_zero * w3j_m
        # gaunt_integral gives the same thing
        Y10 = ScalarHarmonic(1, 0)
        Y11 = ScalarHarmonic(1, 1)
        Y1m1 = ScalarHarmonic(1, -1)

        # Check Y_{1,0}*Y_{1,0} projected onto Y_{0,0}
        g = gaunt_integral(1, 0, 1, 0, 0, 0)
        # This should equal the coefficient of Y_{0,0} in harmonic_product(Y10, Y10)
        result = harmonic_product(Y10, Y10)
        # Extract coefficient for l=0
        coeff_l0 = 0.0
        for term in result.terms
            if term isa TProduct
                for f in term.factors
                    if f isa ScalarHarmonic && f.l == 0
                        c = Float64(term.scalar)
                        for g_f in term.factors
                            g_f isa TScalar && (c *= Float64(g_f.val))
                        end
                        coeff_l0 = c
                    end
                end
            end
        end
        @test isapprox(g, coeff_l0, atol=1e-13)
    end

    @testset "angular_selection_rule" begin
        @test angular_selection_rule(1, 1, 0) == true   # 1+1+0=2 even, triangle OK
        @test angular_selection_rule(1, 1, 2) == true   # 1+1+2=4 even, triangle OK
        @test angular_selection_rule(1, 1, 1) == false  # 1+1+1=3 odd
        @test angular_selection_rule(1, 1, 5) == false  # triangle violated
        @test angular_selection_rule(2, 2, 0) == true   # 2+2+0=4 even
        @test angular_selection_rule(2, 2, 4) == true   # 2+2+4=8 even
        @test angular_selection_rule(2, 2, 3) == false  # 2+2+3=7 odd
    end

    @testset "vector_gaunt: basic properties" begin
        # l < 1 returns 0
        @test vector_gaunt(0, 0, 1, 0, 1, 0) == 0.0
        @test vector_gaunt(1, 0, 0, 0, 1, 0) == 0.0

        # vector_gaunt(l1,m1,l2,m2,l3,m3) = 1/2 [L1+L2-L3] * gaunt(...)
        # For l1=l2=1, l3=0: coupling = 1/2[2+2-0] = 2
        g = gaunt_integral(1, 0, 1, 0, 0, 0)
        @test isapprox(vector_gaunt(1, 0, 1, 0, 0, 0), 2.0 * g, atol=1e-13)

        # For l1=l2=1, l3=2: coupling = 1/2[2+2-6] = -1
        g2 = gaunt_integral(1, 0, 1, 0, 2, 0)
        @test isapprox(vector_gaunt(1, 0, 1, 0, 2, 0), -1.0 * g2, atol=1e-13)
    end

    @testset "vector_gaunt: reduces to Gaunt for divergence-free" begin
        # For divergence-free vector harmonics, the coupling coefficient
        # 1/2[L1+L2-L3] gives a nontrivial factor. Verify numerical consistency
        # by checking vector_gaunt = coupling * gaunt for several cases.
        for (l1, l2, l3) in [(1,1,0), (1,1,2), (2,1,1), (2,1,3), (2,2,0), (2,2,2), (2,2,4)]
            angular_selection_rule(l1, l2, l3) || continue
            L1 = l1 * (l1 + 1)
            L2 = l2 * (l2 + 1)
            L3 = l3 * (l3 + 1)
            coupling = (L1 + L2 - L3) / 2
            g = gaunt_integral(l1, 0, l2, 0, l3, 0)
            vg = vector_gaunt(l1, 0, l2, 0, l3, 0)
            @test isapprox(vg, coupling * g, atol=1e-13)
        end
    end

    @testset "tensor_gaunt: Y-Y (metric type)" begin
        # integral Y^{AB} Y_{AB} Y* dOmega = 2 * gaunt
        for (l1, l2, l3) in [(0,0,0), (1,1,0), (1,1,2), (2,2,0), (2,2,2), (2,2,4)]
            angular_selection_rule(l1, l2, l3) || continue
            g = gaunt_integral(l1, 0, l2, 0, l3, 0)
            tg = tensor_gaunt(l1, 0, l2, 0, l3, 0, :Y, :Y)
            @test isapprox(tg, 2.0 * g, atol=1e-13)
        end
    end

    @testset "tensor_gaunt: cross-type vanish" begin
        @test tensor_gaunt(2, 0, 2, 0, 0, 0, :Y, :Z) == 0.0
        @test tensor_gaunt(2, 0, 2, 0, 0, 0, :Z, :Y) == 0.0
        @test tensor_gaunt(2, 0, 2, 0, 0, 0, :Y, :X) == 0.0
        @test tensor_gaunt(2, 0, 2, 0, 0, 0, :X, :Y) == 0.0
        @test tensor_gaunt(2, 0, 2, 0, 0, 0, :Z, :X) == 0.0
        @test tensor_gaunt(2, 0, 2, 0, 0, 0, :X, :Z) == 0.0
    end

    @testset "tensor_gaunt: Z-Z quadrupole coupling (Gleiser et al. Eq B.12)" begin
        # For l1=l2=2, l3=0 (quadrupole-quadrupole -> monopole):
        # L1 = L2 = 6, L3 = 0
        # Q = (6+6-0)^2/8 + (36-6-6)/4 = 144/8 + 24/4 = 18 + 6 = 24
        g = gaunt_integral(2, 0, 2, 0, 0, 0)
        tg = tensor_gaunt(2, 0, 2, 0, 0, 0, :Z, :Z)
        @test isapprox(tg, 24.0 * g, atol=1e-13)

        # For l1=l2=2, l3=2:
        # L1 = L2 = 6, L3 = 6
        # Q = (6+6-6)^2/8 + (36-12)/4 = 36/8 + 6 = 4.5 + 6 = 10.5
        g2 = gaunt_integral(2, 0, 2, 0, 2, 0)
        tg2 = tensor_gaunt(2, 0, 2, 0, 2, 0, :Z, :Z)
        @test isapprox(tg2, 10.5 * g2, atol=1e-13)

        # For l1=l2=2, l3=4:
        # L1 = L2 = 6, L3 = 20
        # Q = (6+6-20)^2/8 + (36-12)/4 = 64/8 + 6 = 8 + 6 = 14
        g4 = gaunt_integral(2, 0, 2, 0, 4, 0)
        tg4 = tensor_gaunt(2, 0, 2, 0, 4, 0, :Z, :Z)
        @test isapprox(tg4, 14.0 * g4, atol=1e-13)
    end

    @testset "tensor_gaunt: X-X equals Z-Z by parity" begin
        # Odd STF harmonics X^{AB} have same norm structure as even STF Z^{AB}
        for (l1, l2, l3) in [(2,2,0), (2,2,2), (2,2,4), (2,3,3), (3,3,0), (3,3,2)]
            angular_selection_rule(l1, l2, l3) || continue
            l1 < 2 && continue
            l2 < 2 && continue
            zz = tensor_gaunt(l1, 0, l2, 0, l3, 0, :Z, :Z)
            xx = tensor_gaunt(l1, 0, l2, 0, l3, 0, :X, :X)
            @test isapprox(xx, zz, atol=1e-13)
        end
    end

    @testset "tensor_gaunt: l < 2 for Z/X returns 0" begin
        @test tensor_gaunt(1, 0, 2, 0, 1, 0, :Z, :Z) == 0.0
        @test tensor_gaunt(2, 0, 1, 0, 1, 0, :Z, :Z) == 0.0
        @test tensor_gaunt(1, 0, 2, 0, 1, 0, :X, :X) == 0.0
    end

    @testset "angular_integral: two-harmonic dispatch" begin
        # Scalar inner product
        Y10 = ScalarHarmonic(1, 0)
        Y20 = ScalarHarmonic(2, 0)
        @test angular_integral(Y10, Y10) == TScalar(1)
        @test angular_integral(Y10, Y20) == TScalar(0)

        # Vector inner product
        idx = up(:A, :S2)
        Ya = EvenVectorHarmonic(1, 0, idx)
        Yb = EvenVectorHarmonic(1, 0, idx)
        Yc = EvenVectorHarmonic(2, 0, idx)
        @test angular_integral(Ya, Yb) == TScalar(2)    # l(l+1) = 1*2 = 2
        @test angular_integral(Ya, Yc) == TScalar(0)

        # Cross-parity vector
        Xa = OddVectorHarmonic(1, 0, idx)
        @test angular_integral(Ya, Xa) == TScalar(0)
        @test angular_integral(Xa, Ya) == TScalar(0)
    end

    @testset "angular_integral: three-harmonic scalar triple" begin
        Y10 = ScalarHarmonic(1, 0)
        Y00 = ScalarHarmonic(0, 0)
        Y20 = ScalarHarmonic(2, 0)

        # Same as gaunt_integral
        @test isapprox(angular_integral(Y10, Y10, Y00),
                       gaunt_integral(1, 0, 1, 0, 0, 0), atol=1e-15)
        @test isapprox(angular_integral(Y10, Y10, Y20),
                       gaunt_integral(1, 0, 1, 0, 2, 0), atol=1e-15)
    end

    @testset "angular_integral: cross-parity vector pair vanishes" begin
        idx = up(:A, :S2)
        Ya = EvenVectorHarmonic(1, 0, idx)
        Xa = OddVectorHarmonic(1, 0, idx)
        Y00 = ScalarHarmonic(0, 0)

        @test angular_integral(Ya, Xa, Y00) == 0.0
        @test angular_integral(Xa, Ya, Y00) == 0.0
    end

    @testset "angular_integral: cross-type tensor pair vanishes" begin
        i1 = up(:A, :S2)
        i2 = up(:B, :S2)
        Yab = EvenTensorHarmonicY(2, 0, i1, i2)
        Zab = EvenTensorHarmonicZ(2, 0, i1, i2)
        Xab = OddTensorHarmonic(2, 0, i1, i2)
        Y00 = ScalarHarmonic(0, 0)

        @test angular_integral(Yab, Zab, Y00) == 0.0
        @test angular_integral(Zab, Yab, Y00) == 0.0
        @test angular_integral(Yab, Xab, Y00) == 0.0
        @test angular_integral(Xab, Yab, Y00) == 0.0
        @test angular_integral(Zab, Xab, Y00) == 0.0
        @test angular_integral(Xab, Zab, Y00) == 0.0
    end

    @testset "angular_integral: vector-vector-scalar" begin
        idx = up(:A, :S2)
        Ya1 = EvenVectorHarmonic(1, 0, idx)
        Ya2 = EvenVectorHarmonic(1, 0, idx)
        Y00 = ScalarHarmonic(0, 0)

        val = angular_integral(Ya1, Ya2, Y00)
        @test isapprox(val, vector_gaunt(1, 0, 1, 0, 0, 0), atol=1e-15)
    end

    @testset "angular_integral: tensor-tensor-scalar" begin
        i1 = up(:A, :S2)
        i2 = up(:B, :S2)
        Z1 = EvenTensorHarmonicZ(2, 0, i1, i2)
        Z2 = EvenTensorHarmonicZ(2, 0, i1, i2)
        Y00 = ScalarHarmonic(0, 0)

        val = angular_integral(Z1, Z2, Y00)
        @test isapprox(val, tensor_gaunt(2, 0, 2, 0, 0, 0, :Z, :Z), atol=1e-15)
    end

    @testset "angular_integral: scalar-vector vanishes (no contraction)" begin
        Y10 = ScalarHarmonic(1, 0)
        idx = up(:A, :S2)
        Ya = EvenVectorHarmonic(1, 0, idx)
        Y00 = ScalarHarmonic(0, 0)

        @test angular_integral(Y10, Ya, Y00) == 0.0
        @test angular_integral(Ya, Y10, Y00) == 0.0
    end

    @testset "gaunt_integral: nonzero m values" begin
        # integral Y_{1,1} Y_{1,-1} Y*_{2,0} dOmega
        # m1+m2 = 0 = m3, so selection rule passes
        val = gaunt_integral(1, 1, 1, -1, 2, 0)
        expected = sqrt(9 * 5 / (4pi)) * wigner3j(1, 1, 2, 0, 0, 0) *
                   wigner3j(1, 1, 2, 1, -1, 0)
        @test isapprox(val, expected, atol=1e-13)

        # integral Y_{2,1} Y_{1,-1} Y*_{1,0} dOmega
        # m3 = 1 + (-1) = 0
        val2 = gaunt_integral(2, 1, 1, -1, 1, 0)
        expected2 = sqrt(5 * 3 * 3 / (4pi)) * wigner3j(2, 1, 1, 0, 0, 0) *
                    wigner3j(2, 1, 1, 1, -1, 0)
        @test isapprox(val2, expected2, atol=1e-13)
    end
end
