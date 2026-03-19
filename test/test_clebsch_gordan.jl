@testset "Clebsch-Gordan and harmonic product" begin

    @testset "Wigner 3j: selection rules" begin
        # m1 + m2 + m3 != 0 => 0
        @test wigner3j(1, 1, 1, 1, 0, 1) == 0.0
        # Triangle inequality violated
        @test wigner3j(1, 1, 5, 0, 0, 0) == 0.0
        # |m| > j violated
        @test wigner3j(1, 1, 0, 2, 0, 0) == 0.0
        # Negative j
        @test wigner3j(-1, 1, 0, 0, 0, 0) == 0.0
    end

    @testset "Wigner 3j: known values" begin
        # (1 1 0; 0 0 0) = (-1)^1 / sqrt(3)
        val = wigner3j(1, 1, 0, 0, 0, 0)
        @test isapprox(val, -1 / sqrt(3), atol=1e-14)

        # (1 1 2; 0 0 0) = sqrt(2/15)
        val2 = wigner3j(1, 1, 2, 0, 0, 0)
        @test isapprox(val2, sqrt(2 / 15), atol=1e-14)

        # (1 1 1; 1 0 -1) = -1/sqrt(6)
        val3 = wigner3j(1, 1, 1, 1, 0, -1)
        @test isapprox(val3, -1 / sqrt(6), atol=1e-14)

        # (j j 0; m -m 0) = (-1)^(j-m) / sqrt(2j+1) for any valid j,m
        for j in 0:4, m in -j:j
            expected = (-1)^(j - m) / sqrt(2j + 1)
            @test isapprox(wigner3j(j, j, 0, m, -m, 0), expected, atol=1e-13)
        end
    end

    @testset "Wigner 3j: parity" begin
        # l1 + l2 + l3 odd with all m=0 => 0
        @test wigner3j(1, 1, 1, 0, 0, 0) == 0.0
        @test wigner3j(1, 2, 2, 0, 0, 0) == 0.0
        @test wigner3j(2, 3, 4, 0, 0, 0) == 0.0
    end

    @testset "Clebsch-Gordan coefficients" begin
        # <1,0;1,0|0,0> = -1/sqrt(3)
        cg1 = clebsch_gordan(1, 0, 1, 0, 0, 0)
        @test isapprox(cg1, -1 / sqrt(3), atol=1e-14)

        # <1,0;1,0|2,0> = sqrt(2/3)
        cg2 = clebsch_gordan(1, 0, 1, 0, 2, 0)
        @test isapprox(cg2, sqrt(2 / 3), atol=1e-14)

        # <1,1;1,-1|0,0> = 1/sqrt(3)
        cg3 = clebsch_gordan(1, 1, 1, -1, 0, 0)
        @test isapprox(cg3, 1 / sqrt(3), atol=1e-14)

        # M != m1 + m2 => 0
        @test clebsch_gordan(1, 0, 1, 0, 1, 1) == 0.0

        # Completeness: sum_J |CG|^2 = 1 for fixed j1,m1,j2,m2
        for m1 in -1:1, m2 in -1:1
            M = m1 + m2
            s = sum(clebsch_gordan(1, m1, 1, m2, J, M)^2 for J in 0:2)
            @test isapprox(s, 1.0, atol=1e-13)
        end
    end

    @testset "Harmonic product: Y_{1,0} * Y_{1,0}" begin
        Y10 = ScalarHarmonic(1, 0)
        result = harmonic_product(Y10, Y10)

        # Should produce terms with Y_{0,0} and Y_{2,0}
        @test result isa TSum
        @test length(result.terms) == 2

        # Extract coefficients for each harmonic
        coeffs = Dict{Int, Float64}()
        for term in result.terms
            # Each term is tproduct(1//1, [TScalar(coeff), ScalarHarmonic(l,0)])
            # which collapses to TProduct with TScalar(coeff) factor and ScalarHarmonic factor
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
                if sh !== nothing
                    coeffs[sh.l] = c
                end
            end
        end

        # Y_{1,0}*Y_{1,0} = (1/(2*sqrt(pi)))*Y_{0,0} + (1/sqrt(5*pi))*Y_{2,0}
        # Gaunt(1,1,0,0,0) = sqrt(3*3*1/(4pi)) * (1 1 0;0 0 0)^2
        #   = sqrt(9/(4pi)) * (1/3) = 1/(2*sqrt(pi))
        @test isapprox(coeffs[0], 1 / (2 * sqrt(pi)), atol=1e-13)

        # Gaunt(1,1,2,0,0) = sqrt(3*3*5/(4pi)) * (1 1 2;0 0 0) * (1 1 2;0 0 0)
        #   = sqrt(45/(4pi)) * (2/15) = (2/15)*sqrt(45/(4pi))
        expected_2 = sqrt(45 / (4pi)) * (2 / 15)
        @test isapprox(coeffs[2], expected_2, atol=1e-13)
    end

    @testset "Harmonic product: Y_{1,1} * Y_{1,-1}" begin
        Y11 = ScalarHarmonic(1, 1)
        Y1m1 = ScalarHarmonic(1, -1)
        result = harmonic_product(Y11, Y1m1)

        # m3 = 1 + (-1) = 0, so terms are Y_{0,0} and Y_{2,0}
        @test result isa TSum
        @test length(result.terms) == 2

        coeffs = Dict{Int, Float64}()
        for term in result.terms
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
                if sh !== nothing
                    coeffs[sh.l] = c
                end
            end
        end

        # Y_{1,1}*Y_{1,-1}: m3=0, phase=(-1)^0=1
        # coefficient of Y_{0,0}
        # = sqrt(9/(4pi)) * (1 1 0;0 0 0) * (1 1 0;1 -1 0)
        @test haskey(coeffs, 0)
        expected_0 = sqrt(9 / (4pi)) * wigner3j(1, 1, 0, 0, 0, 0) * wigner3j(1, 1, 0, 1, -1, 0)
        @test isapprox(coeffs[0], expected_0, atol=1e-13)
        @test haskey(coeffs, 2)
    end

    @testset "Harmonic product: selection rules" begin
        # Y_{0,0} * Y_{l,m} = (1/sqrt(4pi)) * Y_{l,m}
        Y00 = ScalarHarmonic(0, 0)
        Y21 = ScalarHarmonic(2, 1)
        result = harmonic_product(Y00, Y21)

        # Only one term: coeff * Y_{2,1}
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
            @test sh == ScalarHarmonic(2, 1)
            @test isapprox(c, 1 / sqrt(4pi), atol=1e-13)
        else
            # Could be a TSum with one term, but tsum should collapse
            error("Expected single-term result for Y00 * Ylm")
        end
    end

    @testset "Harmonic product: parity" begin
        # l1 + l2 + l3 odd terms have zero Gaunt coefficient
        # Y_{1,0} * Y_{2,0}: l3 ranges 1..3, only l3=1,3 survive parity (1+2+l3 even)
        Y10 = ScalarHarmonic(1, 0)
        Y20 = ScalarHarmonic(2, 0)
        result = harmonic_product(Y10, Y20)

        # l3=0 violates triangle, l3=2 violates parity (1+2+2=5 odd)
        # valid: l3=1 (1+2+1=4 even), l3=3 (1+2+3=6 even)
        if result isa TSum
            for term in result.terms
                if term isa TProduct
                    for f in term.factors
                        if f isa ScalarHarmonic
                            @test iseven(1 + 2 + f.l)
                        end
                    end
                end
            end
        elseif result isa TProduct
            for f in result.factors
                if f isa ScalarHarmonic
                    @test iseven(1 + 2 + f.l)
                end
            end
        end
    end

    @testset "Harmonic product: higher l" begin
        # Y_{2,0} * Y_{2,0}: l3 in 0,2,4 (all even parity sums)
        Y20 = ScalarHarmonic(2, 0)
        result = harmonic_product(Y20, Y20)
        @test result isa TSum
        @test length(result.terms) == 3  # l3 = 0, 2, 4

        ls = Int[]
        for term in result.terms
            if term isa TProduct
                for f in term.factors
                    if f isa ScalarHarmonic
                        push!(ls, f.l)
                    end
                end
            end
        end
        @test sort(ls) == [0, 2, 4]
    end

    @testset "Wigner 3j: symmetry" begin
        # Even permutation: invariant
        # (j1 j2 j3; m1 m2 m3) = (j2 j3 j1; m2 m3 m1)
        for (j1, j2, j3, m1, m2, m3) in [(1,2,2,0,1,-1), (1,1,2,1,0,-1), (2,2,2,1,1,-2)]
            v1 = wigner3j(j1, j2, j3, m1, m2, m3)
            v2 = wigner3j(j2, j3, j1, m2, m3, m1)
            @test isapprox(v1, v2, atol=1e-14)
        end

        # Odd permutation: phase (-1)^(j1+j2+j3)
        for (j1, j2, j3, m1, m2, m3) in [(1,2,2,0,1,-1), (1,1,2,1,0,-1)]
            v1 = wigner3j(j1, j2, j3, m1, m2, m3)
            v2 = wigner3j(j2, j1, j3, m2, m1, m3)
            phase = (-1)^(j1 + j2 + j3)
            @test isapprox(v1, phase * v2, atol=1e-14)
        end
    end
end
