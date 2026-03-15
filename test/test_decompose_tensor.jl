# Tests for symmetric tensor harmonic decomposition.
# Ground truth: Martel & Poisson (2005) Sec III.A, Eqs 3.1-3.3

@testset "Symmetric tensor harmonic decomposition" begin

    @testset "TensorMode construction" begin
        coeffs = Dict(:tt => :h_tt_2_1, :tr => :h_tr_2_1)
        mode = TensorMode(2, 1, EVEN, coeffs)
        @test mode.l == 2
        @test mode.m == 1
        @test mode.parity == EVEN
        @test mode.coeffs[:tt] == :h_tt_2_1
    end

    @testset "TensorMode equality and hashing" begin
        c1 = Dict(:tt => :h_tt_2_1)
        c2 = Dict(:tt => :h_tt_2_1)
        m1 = TensorMode(2, 1, EVEN, c1)
        m2 = TensorMode(2, 1, EVEN, c2)
        @test m1 == m2
        @test hash(m1) == hash(m2)

        m3 = TensorMode(2, 1, ODD, Dict(:ht => :h_ht_2_1))
        @test m1 != m3
    end

    @testset "decompose_symmetric_tensor: lmax=0 -> even only, no odd" begin
        decomp = decompose_symmetric_tensor(:h, 0)
        @test decomp.field == :h
        @test decomp.lmax == 0
        # l=0 even: tt, tr, rr, K (4 sectors); l=0 odd: empty (no sectors)
        @test mode_count(decomp) == 1  # 1 even mode, 0 odd modes
        mode = decomp.modes[1]
        @test mode.l == 0
        @test mode.m == 0
        @test mode.parity == EVEN
        @test length(mode.coeffs) == 4  # tt, tr, rr, K
        @test haskey(mode.coeffs, :tt)
        @test haskey(mode.coeffs, :tr)
        @test haskey(mode.coeffs, :rr)
        @test haskey(mode.coeffs, :K)
        @test !haskey(mode.coeffs, :jt)  # no vector harmonics at l=0
        @test !haskey(mode.coeffs, :G)   # no STF tensor harmonic at l=0
    end

    @testset "decompose_symmetric_tensor: lmax=1 -> even + odd" begin
        decomp = decompose_symmetric_tensor(:h, 1)
        @test decomp.lmax == 1

        # l=0: 1 even (4 sectors), 0 odd
        # l=1 (m=-1,0,1): 3 even (6 sectors each) + 3 odd (2 sectors each)
        # Total: 1 + 3 + 3 = 7 modes
        @test mode_count(decomp) == 7

        # Check l=1 even has 6 sectors: tt, tr, rr, jt, jr, K
        even_1_0 = get_mode(decomp, 1, 0, EVEN)
        @test length(even_1_0.coeffs) == 6
        @test haskey(even_1_0.coeffs, :jt)
        @test haskey(even_1_0.coeffs, :jr)
        @test !haskey(even_1_0.coeffs, :G)  # l=1, no Z^{AB}

        # Check l=1 odd has 2 sectors: ht, hr
        odd_1_0 = get_mode(decomp, 1, 0, ODD)
        @test length(odd_1_0.coeffs) == 2
        @test haskey(odd_1_0.coeffs, :ht)
        @test haskey(odd_1_0.coeffs, :hr)
        @test !haskey(odd_1_0.coeffs, :h2)  # l=1, no X^{AB}
    end

    @testset "decompose_symmetric_tensor: lmax=2 -> full 10 sectors" begin
        decomp = decompose_symmetric_tensor(:h, 2)
        @test decomp.lmax == 2

        # l=2 even has all 7 sectors: tt, tr, rr, jt, jr, K, G
        even_2_0 = get_mode(decomp, 2, 0, EVEN)
        @test length(even_2_0.coeffs) == 7
        @test haskey(even_2_0.coeffs, :G)

        # l=2 odd has all 3 sectors: ht, hr, h2
        odd_2_0 = get_mode(decomp, 2, 0, ODD)
        @test length(odd_2_0.coeffs) == 3
        @test haskey(odd_2_0.coeffs, :h2)

        # Ground truth: 10 independent components at l >= 2
        @test length(even_2_0.coeffs) + length(odd_2_0.coeffs) == 10
    end

    @testset "Component count: 7 even + 3 odd at l >= 2 (MP Sec III.A)" begin
        decomp = decompose_symmetric_tensor(:h, 5)
        for l in 2:5
            even = get_mode(decomp, l, 0, EVEN)
            odd = get_mode(decomp, l, 0, ODD)
            @test length(even.coeffs) == 7
            @test length(odd.coeffs) == 3
        end
    end

    @testset "Component count: low multipoles" begin
        decomp = decompose_symmetric_tensor(:h, 2)

        # l=0 even: 4 (tt, tr, rr, K); l=0 odd: absent
        even_0 = get_mode(decomp, 0, 0, EVEN)
        @test length(even_0.coeffs) == 4
        @test_throws ArgumentError get_mode(decomp, 0, 0, ODD)

        # l=1 even: 6 (tt, tr, rr, jt, jr, K); l=1 odd: 2 (ht, hr)
        even_1 = get_mode(decomp, 1, 0, EVEN)
        odd_1 = get_mode(decomp, 1, 0, ODD)
        @test length(even_1.coeffs) == 6
        @test length(odd_1.coeffs) == 2
    end

    @testset "Total mode count formula" begin
        # l=0: 1 even (no odd)
        # l>=1: even + odd per (l,m)
        # Total modes = 1 + sum_{l=1}^{lmax} (2l+1)*2
        for lmax in 0:5
            decomp = decompose_symmetric_tensor(:h, lmax)
            expected = 1  # l=0 even only
            for l in 1:lmax
                expected += 2 * (2l + 1)  # even + odd for each (l,m)
            end
            @test mode_count(decomp) == expected
        end
    end

    @testset "Coefficient naming" begin
        decomp = decompose_symmetric_tensor(:h, 2)
        even_2_1 = get_mode(decomp, 2, 1, EVEN)
        @test even_2_1.coeffs[:tt] == :h_tt_2_1
        @test even_2_1.coeffs[:G] == :h_G_2_1

        odd_2_1 = get_mode(decomp, 2, 1, ODD)
        @test odd_2_1.coeffs[:h2] == :h_h2_2_1
    end

    @testset "Negative m coefficient naming" begin
        decomp = decompose_symmetric_tensor(:h, 2)
        even_2_neg1 = get_mode(decomp, 2, -1, EVEN)
        @test even_2_neg1.coeffs[:tt] == :h_tt_2_neg1
        @test even_2_neg1.coeffs[:G] == :h_G_2_neg1

        odd_2_neg2 = get_mode(decomp, 2, -2, ODD)
        @test odd_2_neg2.coeffs[:h2] == :h_h2_2_neg2
    end

    @testset "Custom coeff_prefix" begin
        decomp = decompose_symmetric_tensor(:h, 2; coeff_prefix=:p)
        even = get_mode(decomp, 2, 0, EVEN)
        @test even.coeffs[:tt] == :p_tt_2_0
        @test even.coeffs[:G] == :p_G_2_0
        @test decomp.field == :h
    end

    @testset "Validation" begin
        @test_throws ArgumentError decompose_symmetric_tensor(:h, -1)
    end

    @testset "get_mode: missing mode throws" begin
        decomp = decompose_symmetric_tensor(:h, 1)
        @test_throws ArgumentError get_mode(decomp, 2, 0, EVEN)
        @test_throws ArgumentError get_mode(decomp, 0, 0, ODD)
    end

    @testset "to_expr produces TSum" begin
        decomp = decompose_symmetric_tensor(:h, 2)
        expr = to_expr(decomp)
        @test expr isa TSum

        # Each term should be TProduct of TScalar(coeff) * harmonic
        for term in expr.terms
            @test term isa TProduct
            @test term.scalar == 1//1
            @test length(term.factors) == 2
            @test term.factors[1] isa TScalar
            @test term.factors[2] isa Union{ScalarHarmonic, EvenVectorHarmonic,
                                            OddVectorHarmonic, EvenTensorHarmonicY,
                                            EvenTensorHarmonicZ, OddTensorHarmonic}
        end
    end

    @testset "to_expr term count matches sector count" begin
        decomp = decompose_symmetric_tensor(:h, 2)
        expr = to_expr(decomp)
        total_sectors = sum(length(m.coeffs) for m in decomp.modes)
        @test length(expr.terms) == total_sectors
    end

    @testset "to_expr: single mode (lmax=0) -> TProduct" begin
        # lmax=0 has 1 even mode with 4 sectors, so to_expr gives TSum not TProduct
        # Actually 4 terms means TSum. Check:
        decomp = decompose_symmetric_tensor(:h, 0)
        expr = to_expr(decomp)
        @test expr isa TSum
        @test length(expr.terms) == 4  # tt, tr, rr, K
    end

    @testset "to_expr: harmonic types correct per sector" begin
        decomp = decompose_symmetric_tensor(:h, 2)
        expr = to_expr(decomp)

        scalar_coeffs = Symbol[]
        even_vec_coeffs = Symbol[]
        odd_vec_coeffs = Symbol[]
        even_Y_coeffs = Symbol[]
        even_Z_coeffs = Symbol[]
        odd_X_coeffs = Symbol[]

        for term in expr.terms
            coeff = term.factors[1].val
            harm = term.factors[2]
            if harm isa ScalarHarmonic
                push!(scalar_coeffs, coeff)
            elseif harm isa EvenVectorHarmonic
                push!(even_vec_coeffs, coeff)
            elseif harm isa OddVectorHarmonic
                push!(odd_vec_coeffs, coeff)
            elseif harm isa EvenTensorHarmonicY
                push!(even_Y_coeffs, coeff)
            elseif harm isa EvenTensorHarmonicZ
                push!(even_Z_coeffs, coeff)
            elseif harm isa OddTensorHarmonic
                push!(odd_X_coeffs, coeff)
            end
        end

        # Scalar harmonics: tt, tr, rr for each even mode
        # l=0: 3; l=1: 3*3=9; l=2: 3*5=15; total = 27
        @test length(scalar_coeffs) == 3 * (0 + 1) + 3 * (2 + 1) + 3 * (4 + 1)  # 3 + 9 + 15 = 27

        # EvenVectorHarmonic: jt, jr for l >= 1 even modes
        # l=1: 2*3=6; l=2: 2*5=10; total = 16
        @test length(even_vec_coeffs) == 2 * 3 + 2 * 5  # 16

        # OddVectorHarmonic: ht, hr for l >= 1 odd modes
        # l=1: 2*3=6; l=2: 2*5=10; total = 16
        @test length(odd_vec_coeffs) == 2 * 3 + 2 * 5  # 16

        # EvenTensorHarmonicY: K for each even mode
        # l=0: 1; l=1: 3; l=2: 5; total = 9
        @test length(even_Y_coeffs) == 1 + 3 + 5  # 9

        # EvenTensorHarmonicZ: G for l >= 2 even modes
        # l=2: 5; total = 5
        @test length(even_Z_coeffs) == 5

        # OddTensorHarmonic: h2 for l >= 2 odd modes
        # l=2: 5; total = 5
        @test length(odd_X_coeffs) == 5
    end

    @testset "to_expr: angular indices propagated" begin
        decomp = decompose_symmetric_tensor(:h, 2)
        custom_a = up(:C)
        custom_b = up(:D)
        expr = to_expr(decomp; angular_idx1=custom_a, angular_idx2=custom_b)

        # Check that vector/tensor harmonics carry the custom indices
        for term in expr.terms
            harm = term.factors[2]
            if harm isa EvenVectorHarmonic || harm isa OddVectorHarmonic
                @test harm.index == custom_a
            elseif harm isa EvenTensorHarmonicY || harm isa EvenTensorHarmonicZ ||
                   harm isa OddTensorHarmonic
                @test harm.index1 == custom_a
                @test harm.index2 == custom_b
            end
        end
    end

    @testset "Equality and hashing" begin
        d1 = decompose_symmetric_tensor(:h, 2)
        d2 = decompose_symmetric_tensor(:h, 2)
        @test d1 == d2
        @test hash(d1) == hash(d2)

        d3 = decompose_symmetric_tensor(:h, 1)
        @test d1 != d3

        d4 = decompose_symmetric_tensor(:g, 2)
        @test d1 != d4
    end

    @testset "Pretty printing" begin
        decomp = decompose_symmetric_tensor(:h, 2)
        s = sprint(show, decomp)
        @test occursin("h", s)
        @test occursin("even", s)
        @test occursin("odd", s)

        mode = get_mode(decomp, 2, 1, EVEN)
        ms = sprint(show, mode)
        @test occursin("l=2", ms)
        @test occursin("m=1", ms)
        @test occursin("even", ms)
    end

    @testset "Parity enum" begin
        @test EVEN != ODD
        @test EVEN isa Parity
        @test ODD isa Parity
    end
end
