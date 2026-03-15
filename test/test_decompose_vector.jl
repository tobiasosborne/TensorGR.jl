@testset "Vector field harmonic decomposition" begin
    # Ground truth: Martel & Poisson (2005) Sec III, Eqs 3.1-3.2
    # v_a = sum_{l,m} [ v^{lm} r_a Y_{lm} + v^{lm}_even Y^A_{lm} + v^{lm}_odd X^A_{lm} ]

    @testset "decompose_vector: lmax=0 -> 1 mode, radial only" begin
        decomp = decompose_vector(:v, 0)
        @test decomp.field == :v
        @test decomp.lmax == 0
        @test mode_count(decomp) == 1

        mode = decomp.modes[1]
        @test mode.l == 0
        @test mode.m == 0
        @test mode.coeff_radial == :v_rad_0_0
        # l=0: no vector harmonics
        @test mode.coeff_even === nothing
        @test mode.coeff_odd === nothing
    end

    @testset "decompose_vector: lmax=1 -> 4 modes" begin
        decomp = decompose_vector(:v, 1)
        @test decomp.lmax == 1
        @test mode_count(decomp) == 4  # (1+1)^2 = 4

        # l=0 mode: radial only
        mode00 = get_mode(decomp, 0, 0)
        @test mode00.coeff_radial == :v_rad_0_0
        @test mode00.coeff_even === nothing
        @test mode00.coeff_odd === nothing

        # l=1 modes: all three sectors
        mode10 = get_mode(decomp, 1, 0)
        @test mode10.coeff_radial == :v_rad_1_0
        @test mode10.coeff_even == :v_even_1_0
        @test mode10.coeff_odd == :v_odd_1_0

        mode11 = get_mode(decomp, 1, 1)
        @test mode11.coeff_radial == :v_rad_1_1
        @test mode11.coeff_even == :v_even_1_1
        @test mode11.coeff_odd == :v_odd_1_1

        mode1neg1 = get_mode(decomp, 1, -1)
        @test mode1neg1.coeff_radial == :v_rad_1_neg1
        @test mode1neg1.coeff_even == :v_even_1_neg1
        @test mode1neg1.coeff_odd == :v_odd_1_neg1
    end

    @testset "decompose_vector: lmax=2 -> 9 modes" begin
        decomp = decompose_vector(:v, 2)
        @test decomp.lmax == 2
        @test mode_count(decomp) == 9  # (2+1)^2 = 9

        # Verify l distribution
        ls = [m.l for m in decomp.modes]
        @test count(==(0), ls) == 1
        @test count(==(1), ls) == 3
        @test count(==(2), ls) == 5
    end

    @testset "Mode count formula: (lmax+1)^2" begin
        for lmax in 0:5
            decomp = decompose_vector(:v, lmax)
            @test mode_count(decomp) == (lmax + 1)^2
        end
    end

    @testset "to_expr: lmax=0 returns single TProduct (radial only)" begin
        decomp = decompose_vector(:v, 0)
        expr = to_expr(decomp)
        @test expr isa TProduct
        @test expr.scalar == 1//1
        @test length(expr.factors) == 2
        @test expr.factors[1] == TScalar(:v_rad_0_0)
        @test expr.factors[2] == ScalarHarmonic(0, 0)
    end

    @testset "to_expr: lmax=1 produces TSum with correct term count" begin
        decomp = decompose_vector(:v, 1)
        expr = to_expr(decomp)
        @test expr isa TSum
        # l=0: 1 radial term; l=1: 3*(radial+even+odd) = 9 terms; total = 10
        @test length(expr.terms) == 10
    end

    @testset "to_expr: lmax=2 term count" begin
        decomp = decompose_vector(:v, 2)
        expr = to_expr(decomp)
        @test expr isa TSum
        # l=0: 1 radial; l=1: 3*3=9; l=2: 5*3=15; total = 25
        @test length(expr.terms) == 25
    end

    @testset "to_expr term structure" begin
        decomp = decompose_vector(:v, 1)
        expr = to_expr(decomp)

        # First term is radial for l=0: TScalar * ScalarHarmonic
        t0 = expr.terms[1]
        @test t0 isa TProduct
        @test t0.factors[1] == TScalar(:v_rad_0_0)
        @test t0.factors[2] isa ScalarHarmonic

        # l=1, m=-1: radial, even, odd (terms 2,3,4)
        t_rad = expr.terms[2]
        @test t_rad.factors[1] == TScalar(:v_rad_1_neg1)
        @test t_rad.factors[2] isa ScalarHarmonic
        @test t_rad.factors[2] == ScalarHarmonic(1, -1)

        t_even = expr.terms[3]
        @test t_even.factors[1] == TScalar(:v_even_1_neg1)
        @test t_even.factors[2] isa EvenVectorHarmonic

        t_odd = expr.terms[4]
        @test t_odd.factors[1] == TScalar(:v_odd_1_neg1)
        @test t_odd.factors[2] isa OddVectorHarmonic
    end

    @testset "to_expr angular index propagation" begin
        idx = down(:B, :S2)
        decomp = decompose_vector(:v, 1)
        expr = to_expr(decomp; angular_index=idx)

        # Even harmonic term for l=1,m=0 should carry the custom index
        # l=0 radial (1 term), l=1 m=-1 (3 terms), l=1 m=0: radial at [5], even at [6]
        t_even = expr.terms[6]
        @test t_even.factors[2] isa EvenVectorHarmonic
        @test t_even.factors[2].index == idx

        t_odd = expr.terms[7]
        @test t_odd.factors[2] isa OddVectorHarmonic
        @test t_odd.factors[2].index == idx
    end

    @testset "get_mode retrieves correct mode" begin
        decomp = decompose_vector(:v, 2)

        mode21 = get_mode(decomp, 2, 1)
        @test mode21.l == 2
        @test mode21.m == 1
        @test mode21.coeff_radial == :v_rad_2_1
        @test mode21.coeff_even == :v_even_2_1
        @test mode21.coeff_odd == :v_odd_2_1

        mode00 = get_mode(decomp, 0, 0)
        @test mode00.l == 0
        @test mode00.m == 0
    end

    @testset "get_mode throws for missing mode" begin
        decomp = decompose_vector(:v, 1)
        @test_throws ArgumentError get_mode(decomp, 2, 0)
        @test_throws ArgumentError get_mode(decomp, 1, 2)
    end

    @testset "Custom coeff_prefix" begin
        decomp = decompose_vector(:v, 1; coeff_prefix=:w)
        @test decomp.field == :v

        mode10 = get_mode(decomp, 1, 0)
        @test mode10.coeff_radial == :w_rad_1_0
        @test mode10.coeff_even == :w_even_1_0
        @test mode10.coeff_odd == :w_odd_1_0
    end

    @testset "Negative m coefficient naming" begin
        decomp = decompose_vector(:v, 2)
        mode2neg2 = get_mode(decomp, 2, -2)
        @test mode2neg2.coeff_radial == :v_rad_2_neg2
        @test mode2neg2.coeff_even == :v_even_2_neg2
        @test mode2neg2.coeff_odd == :v_odd_2_neg2
    end

    @testset "Validation" begin
        @test_throws ArgumentError decompose_vector(:v, -1)
    end

    @testset "Equality and hashing" begin
        d1 = decompose_vector(:v, 2)
        d2 = decompose_vector(:v, 2)
        @test d1 == d2
        @test hash(d1) == hash(d2)

        d3 = decompose_vector(:v, 1)
        @test d1 != d3

        # VectorMode equality
        m1 = VectorMode(2, 1, :v_rad_2_1, :v_even_2_1, :v_odd_2_1)
        m2 = VectorMode(2, 1, :v_rad_2_1, :v_even_2_1, :v_odd_2_1)
        @test m1 == m2
        @test hash(m1) == hash(m2)
    end

    @testset "Pretty printing" begin
        decomp = decompose_vector(:v, 2)
        s = sprint(show, decomp)
        @test occursin("v_a", s)
        @test occursin("9 modes", s)

        mode = get_mode(decomp, 2, 1)
        ms = sprint(show, mode)
        @test occursin("Y_{2,1}", ms)
        @test occursin("Y^A_{2,1}", ms)
        @test occursin("X^A_{2,1}", ms)
    end

    @testset "Even/odd parity separation (Martel & Poisson Sec III)" begin
        # For each l>=1 mode, verify even and odd sectors exist and are distinct
        decomp = decompose_vector(:v, 3)
        for l in 1:3
            for m in -l:l
                mode = get_mode(decomp, l, m)
                @test mode.coeff_even !== nothing
                @test mode.coeff_odd !== nothing
                @test mode.coeff_even != mode.coeff_odd
                @test mode.coeff_radial != mode.coeff_even
                @test mode.coeff_radial != mode.coeff_odd
            end
        end
    end
end
