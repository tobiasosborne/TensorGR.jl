@testset "Scalar harmonic decomposition" begin
    # Ground truth: Martel & Poisson (2005) Sec III (standard harmonic decomposition)
    # f(t,r,theta,phi) = sum_{l,m} f_{lm}(t,r) Y_{lm}(theta,phi)

    @testset "decompose_scalar: lmax=0 -> 1 mode" begin
        decomp = decompose_scalar(:Phi, 0)
        @test decomp.field == :Phi
        @test decomp.lmax == 0
        @test mode_count(decomp) == 1
        @test length(decomp.modes) == 1

        mode = decomp.modes[1]
        @test mode.l == 0
        @test mode.m == 0
        @test mode.harmonic == ScalarHarmonic(0, 0)
        @test mode.coeff == :Phi_0_0
    end

    @testset "decompose_scalar: lmax=1 -> 4 modes" begin
        decomp = decompose_scalar(:Phi, 1)
        @test decomp.field == :Phi
        @test decomp.lmax == 1
        @test mode_count(decomp) == 4  # (1+1)^2 = 4

        # Check all (l,m) pairs present
        @test decomp.modes[1].l == 0 && decomp.modes[1].m == 0
        @test decomp.modes[2].l == 1 && decomp.modes[2].m == -1
        @test decomp.modes[3].l == 1 && decomp.modes[3].m == 0
        @test decomp.modes[4].l == 1 && decomp.modes[4].m == 1
    end

    @testset "decompose_scalar: lmax=2 -> 9 modes (lmax+1)^2" begin
        decomp = decompose_scalar(:Phi, 2)
        @test decomp.field == :Phi
        @test decomp.lmax == 2
        @test mode_count(decomp) == 9  # (2+1)^2 = 9

        # Verify all l values present
        ls = [m.l for m in decomp.modes]
        @test count(==(0), ls) == 1
        @test count(==(1), ls) == 3
        @test count(==(2), ls) == 5
    end

    @testset "to_expr produces TSum with correct terms" begin
        decomp = decompose_scalar(:Phi, 2)
        expr = to_expr(decomp)
        @test expr isa TSum
        @test length(expr.terms) == 9

        # Each term should be a TProduct of TScalar(coeff) * Y_{lm}
        for (i, term) in enumerate(expr.terms)
            @test term isa TProduct
            @test term.scalar == 1//1
            @test length(term.factors) == 2
            @test term.factors[1] isa TScalar
            @test term.factors[2] isa ScalarHarmonic
        end
    end

    @testset "to_expr single mode returns TProduct" begin
        decomp = decompose_scalar(:Phi, 0)
        expr = to_expr(decomp)
        @test expr isa TProduct
        @test expr.scalar == 1//1
        @test length(expr.factors) == 2
        @test expr.factors[1] == TScalar(:Phi_0_0)
        @test expr.factors[2] == ScalarHarmonic(0, 0)
    end

    @testset "get_mode retrieves correct mode" begin
        decomp = decompose_scalar(:Phi, 2)

        mode21 = get_mode(decomp, 2, 1)
        @test mode21.l == 2
        @test mode21.m == 1
        @test mode21.harmonic == ScalarHarmonic(2, 1)

        mode00 = get_mode(decomp, 0, 0)
        @test mode00.l == 0
        @test mode00.m == 0

        mode1neg1 = get_mode(decomp, 1, -1)
        @test mode1neg1.l == 1
        @test mode1neg1.m == -1
    end

    @testset "get_mode throws for missing mode" begin
        decomp = decompose_scalar(:Phi, 1)
        @test_throws ArgumentError get_mode(decomp, 2, 0)
        @test_throws ArgumentError get_mode(decomp, 1, 2)
    end

    @testset "Custom coeff_prefix" begin
        decomp = decompose_scalar(:Phi, 1; coeff_prefix=:h)
        @test decomp.field == :Phi
        @test decomp.lmax == 1
        @test mode_count(decomp) == 4

        # Coefficient names should use prefix :h
        mode00 = get_mode(decomp, 0, 0)
        @test mode00.coeff == :h_0_0

        mode10 = get_mode(decomp, 1, 0)
        @test mode10.coeff == :h_1_0

        mode11 = get_mode(decomp, 1, 1)
        @test mode11.coeff == :h_1_1

        mode1neg1 = get_mode(decomp, 1, -1)
        @test mode1neg1.coeff == :h_1_neg1
    end

    @testset "Negative m coefficient naming" begin
        decomp = decompose_scalar(:f, 2)
        mode2neg1 = get_mode(decomp, 2, -1)
        @test mode2neg1.coeff == :f_2_neg1

        mode2neg2 = get_mode(decomp, 2, -2)
        @test mode2neg2.coeff == :f_2_neg2
    end

    @testset "Validation" begin
        @test_throws ArgumentError decompose_scalar(:Phi, -1)
    end

    @testset "Equality and hashing" begin
        d1 = decompose_scalar(:Phi, 2)
        d2 = decompose_scalar(:Phi, 2)
        @test d1 == d2
        @test hash(d1) == hash(d2)

        d3 = decompose_scalar(:Phi, 1)
        @test d1 != d3

        # ScalarMode equality
        m1 = ScalarMode(2, 1, :f_2_1, ScalarHarmonic(2, 1))
        m2 = ScalarMode(2, 1, :f_2_1, ScalarHarmonic(2, 1))
        @test m1 == m2
        @test hash(m1) == hash(m2)
    end

    @testset "Pretty printing" begin
        decomp = decompose_scalar(:Phi, 2)
        s = sprint(show, decomp)
        @test occursin("Phi", s)
        @test occursin("9 modes", s)

        mode = get_mode(decomp, 2, 1)
        ms = sprint(show, mode)
        @test occursin("Y_{2,1}", ms)
    end

    @testset "Mode count formula: (lmax+1)^2" begin
        for lmax in 0:5
            decomp = decompose_scalar(:f, lmax)
            @test mode_count(decomp) == (lmax + 1)^2
        end
    end

    @testset "to_expr coefficient-harmonic correspondence" begin
        decomp = decompose_scalar(:Phi, 2)
        expr = to_expr(decomp)

        # Verify each term's coefficient matches its harmonic
        for (i, mode) in enumerate(decomp.modes)
            term = expr.terms[i]
            @test term.factors[1] == TScalar(mode.coeff)
            @test term.factors[2] == mode.harmonic
        end
    end
end
