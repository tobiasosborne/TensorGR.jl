@testset "LaplacianS2" begin
    @testset "Scalar harmonic eigenvalues (Martel & Poisson 2005, Eq 2.1)" begin
        # Delta_Omega Y_{lm} = -l(l+1) Y_{lm}

        # l=0: eigenvalue = 0 -> result is TScalar(0)
        res0 = laplacian_S2(ScalarHarmonic(0, 0))
        @test res0 == TScalar(0//1)

        # l=1: eigenvalue = -2
        Y10 = ScalarHarmonic(1, 0)
        res1 = laplacian_S2(Y10)
        @test res1 isa TProduct
        @test res1.scalar == -2//1
        @test res1.factors[1] == Y10

        # l=2: eigenvalue = -6
        Y21 = ScalarHarmonic(2, 1)
        res2 = laplacian_S2(Y21)
        @test res2 isa TProduct
        @test res2.scalar == -6//1
        @test res2.factors[1] == Y21

        # l=3: eigenvalue = -12 (ground truth: Martel & Poisson Eq 2.1)
        Y31 = ScalarHarmonic(3, 1)
        res3 = laplacian_S2(Y31)
        @test res3 isa TProduct
        @test res3.scalar == -12//1
        @test res3.factors[1] == Y31

        # l=4: eigenvalue = -20
        Y42 = ScalarHarmonic(4, 2)
        res4 = laplacian_S2(Y42)
        @test res4.scalar == -20//1
        @test res4.factors[1] == Y42
    end

    @testset "Scalar eigenvalue sweep l=0..10 (NIST DLMF 14.30)" begin
        for l in 0:10
            Y = ScalarHarmonic(l, 0)
            result = laplacian_S2(Y)
            expected_ev = -l * (l + 1)
            if expected_ev == 0
                @test result == TScalar(0//1)
            else
                @test result isa TProduct
                @test result.scalar == Rational{Int}(expected_ev)
                @test result.factors[1] == Y
            end
        end
    end

    @testset "Vector harmonic eigenvalues" begin
        # Even vector harmonic: Delta_Omega Y^a_{lm} = -(l(l+1)-1) Y^a_{lm}
        # l=1: -(2-1) = -1
        Ya1 = EvenVectorHarmonic(1, 0, up(:a))
        res_e1 = laplacian_S2(Ya1)
        @test res_e1 isa TProduct
        @test res_e1.scalar == -1//1
        @test res_e1.factors[1] == Ya1

        # l=2: -(6-1) = -5
        Ya2 = EvenVectorHarmonic(2, 1, up(:a))
        res_e2 = laplacian_S2(Ya2)
        @test res_e2.scalar == -5//1
        @test res_e2.factors[1] == Ya2

        # l=3: -(12-1) = -11
        Ya3 = EvenVectorHarmonic(3, 0, up(:b))
        res_e3 = laplacian_S2(Ya3)
        @test res_e3.scalar == -11//1
        @test res_e3.factors[1] == Ya3

        # Odd vector harmonic: same eigenvalue as even
        Xa1 = OddVectorHarmonic(1, 0, up(:a))
        res_o1 = laplacian_S2(Xa1)
        @test res_o1 isa TProduct
        @test res_o1.scalar == -1//1
        @test res_o1.factors[1] == Xa1

        Xa2 = OddVectorHarmonic(2, 1, up(:a))
        res_o2 = laplacian_S2(Xa2)
        @test res_o2.scalar == -5//1
        @test res_o2.factors[1] == Xa2

        Xa3 = OddVectorHarmonic(3, -2, up(:c))
        res_o3 = laplacian_S2(Xa3)
        @test res_o3.scalar == -11//1
        @test res_o3.factors[1] == Xa3
    end

    @testset "Vector eigenvalue sweep l=1..10" begin
        for l in 1:10
            ev = -(l * (l + 1) - 1)
            Ya = EvenVectorHarmonic(l, 0, up(:a))
            Xa = OddVectorHarmonic(l, 0, up(:a))
            res_e = laplacian_S2(Ya)
            res_o = laplacian_S2(Xa)
            @test res_e isa TProduct
            @test res_e.scalar == Rational{Int}(ev)
            @test res_o isa TProduct
            @test res_o.scalar == Rational{Int}(ev)
        end
    end

    @testset "Non-harmonic wrapping" begin
        T = Tensor(:T, [up(:a), down(:b)])
        result = laplacian_S2(T)
        @test result isa LaplacianS2
        @test result.arg == T

        s = TScalar(42)
        result2 = laplacian_S2(s)
        @test result2 isa LaplacianS2
        @test result2.arg == s
    end

    @testset "simplify_laplacian" begin
        # Harmonic argument: resolves
        Y = ScalarHarmonic(3, 1)
        L = LaplacianS2(Y)
        resolved = simplify_laplacian(L)
        @test resolved isa TProduct
        @test resolved.scalar == -12//1
        @test resolved.factors[1] == Y

        # Non-harmonic argument: returns unchanged
        T = Tensor(:T, [up(:a)])
        L2 = LaplacianS2(T)
        @test simplify_laplacian(L2) === L2
    end

    @testset "Equality and hashing" begin
        Y = ScalarHarmonic(2, 1)
        @test LaplacianS2(Y) == LaplacianS2(Y)
        @test LaplacianS2(Y) != LaplacianS2(ScalarHarmonic(3, 1))
        @test hash(LaplacianS2(Y)) == hash(LaplacianS2(Y))

        s = Set([LaplacianS2(Y), LaplacianS2(Y), LaplacianS2(ScalarHarmonic(1, 0))])
        @test length(s) == 2
    end

    @testset "AST integration" begin
        T = Tensor(:T, [up(:a), down(:b)])
        L = LaplacianS2(T)

        # indices propagates from arg
        @test indices(L) == indices(T)
        @test free_indices(L) == free_indices(T)

        # children
        @test children(L) == TensorExpr[T]

        # derivative_order adds 2
        @test derivative_order(L) == derivative_order(T) + 2

        # is_constant
        @test is_constant(LaplacianS2(ScalarHarmonic(2, 1))) == true

        # is_sorted_covds
        @test is_sorted_covds(L) == true

        # is_well_formed
        @test is_well_formed(L) == true
    end

    @testset "Walk" begin
        Y = ScalarHarmonic(2, 1)
        L = LaplacianS2(Y)
        result = walk(identity, L)
        @test result == L

        # Walk applies function to children and self
        result2 = walk(x -> x isa LaplacianS2 ? TScalar(99) : x, L)
        @test result2 == TScalar(99)
    end

    @testset "Rename dummy" begin
        Ya = EvenVectorHarmonic(2, 1, up(:a))
        L = LaplacianS2(Ya)

        L2 = rename_dummy(L, :a, :b)
        @test L2 isa LaplacianS2
        @test L2.arg == EvenVectorHarmonic(2, 1, up(:b))

        L3 = rename_dummies(L, Dict(:a => :x))
        @test L3.arg == EvenVectorHarmonic(2, 1, up(:x))

        L4 = TensorGR._replace_index_name(L, :a, :z)
        @test L4.arg == EvenVectorHarmonic(2, 1, up(:z))

        # No match: unchanged
        L5 = rename_dummy(L, :c, :d)
        @test L5.arg == Ya
    end

    @testset "Escape hatch" begin
        Y = ScalarHarmonic(3, -1)
        L = LaplacianS2(Y)
        expr = to_expr(L)
        @test expr.head == :call
        @test expr.args[1] == :LaplacianS2
    end

    @testset "Display" begin
        Y = ScalarHarmonic(2, 1)
        L = LaplacianS2(Y)

        s = sprint(show, L)
        @test occursin("LaplacianS2", s)
        @test occursin("Y", s)

        tex = to_latex(L)
        @test occursin("\\Delta", tex)
        @test occursin("\\Omega", tex)
        @test occursin("Y", tex)

        u = to_unicode(L)
        @test occursin("Delta_Omega", u)
        @test occursin("Y", u)
    end
end
