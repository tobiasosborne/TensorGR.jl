@testset "EvenVectorHarmonic" begin
    @testset "Construction" begin
        Y = EvenVectorHarmonic(2, 1, up(:a))
        @test Y.l == 2
        @test Y.m == 1
        @test Y.index == up(:a)

        Y10 = EvenVectorHarmonic(1, 0, down(:b))
        @test Y10.l == 1
        @test Y10.m == 0
        @test Y10.index == down(:b)

        Y3m3 = EvenVectorHarmonic(3, -3, up(:c))
        @test Y3m3.l == 3
        @test Y3m3.m == -3
    end

    @testset "Validation" begin
        @test_throws ArgumentError EvenVectorHarmonic(0, 0, up(:a))
        @test_throws ArgumentError EvenVectorHarmonic(-1, 0, up(:a))
        @test_throws ArgumentError EvenVectorHarmonic(2, 3, up(:a))
        @test_throws ArgumentError EvenVectorHarmonic(2, -3, up(:a))
        @test_throws ArgumentError EvenVectorHarmonic(1, 2, up(:a))
    end

    @testset "Equality and hashing" begin
        @test EvenVectorHarmonic(2, 1, up(:a)) == EvenVectorHarmonic(2, 1, up(:a))
        @test EvenVectorHarmonic(2, 1, up(:a)) != EvenVectorHarmonic(2, 0, up(:a))
        @test EvenVectorHarmonic(2, 1, up(:a)) != EvenVectorHarmonic(3, 1, up(:a))
        @test EvenVectorHarmonic(2, 1, up(:a)) != EvenVectorHarmonic(2, 1, down(:a))
        @test hash(EvenVectorHarmonic(2, 1, up(:a))) == hash(EvenVectorHarmonic(2, 1, up(:a)))
        @test hash(EvenVectorHarmonic(2, 1, up(:a))) != hash(EvenVectorHarmonic(3, 1, up(:a)))

        s = Set([EvenVectorHarmonic(1, 0, up(:a)), EvenVectorHarmonic(1, 1, up(:a)),
                 EvenVectorHarmonic(1, 0, up(:a))])
        @test length(s) == 2
    end

    @testset "Subtype hierarchy" begin
        @test EvenVectorHarmonic <: TensorExpr
        @test EvenVectorHarmonic(1, 0, up(:a)) isa TensorExpr
    end

    @testset "Free indices" begin
        Y = EvenVectorHarmonic(2, 1, up(:a))
        @test free_indices(Y) == [up(:a)]
        @test indices(Y) == [up(:a)]

        Y_down = EvenVectorHarmonic(2, 1, down(:b))
        @test free_indices(Y_down) == [down(:b)]
    end

    @testset "Conjugation" begin
        # m == 0: conjugate returns EvenVectorHarmonic directly
        Y20 = EvenVectorHarmonic(2, 0, up(:a))
        conj20 = conjugate(Y20)
        @test conj20 isa EvenVectorHarmonic
        @test conj20 == EvenVectorHarmonic(2, 0, up(:a))

        # m > 0 odd: (-1)^1 = -1
        Y21 = EvenVectorHarmonic(2, 1, up(:a))
        conj21 = conjugate(Y21)
        @test conj21 isa TProduct
        @test conj21.scalar == -1//1
        @test length(conj21.factors) == 1
        @test conj21.factors[1] == EvenVectorHarmonic(2, -1, up(:a))

        # m even: positive sign
        Y22 = EvenVectorHarmonic(2, 2, up(:a))
        conj22 = conjugate(Y22)
        @test conj22 isa TProduct
        @test conj22.scalar == 1//1
        @test conj22.factors[1] == EvenVectorHarmonic(2, -2, up(:a))

        # Negative m
        Y2m1 = EvenVectorHarmonic(2, -1, up(:a))
        conj2m1 = conjugate(Y2m1)
        @test conj2m1 isa TProduct
        @test conj2m1.scalar == -1//1
        @test conj2m1.factors[1] == EvenVectorHarmonic(2, 1, up(:a))
    end

    @testset "Divergence eigenvalue" begin
        # D_a Y^a_{lm} = -l(l+1) Y_{lm}
        @test divergence_eigenvalue(EvenVectorHarmonic(1, 0, up(:a))) == -2
        @test divergence_eigenvalue(EvenVectorHarmonic(2, 1, up(:a))) == -6
        @test divergence_eigenvalue(EvenVectorHarmonic(3, 0, up(:a))) == -12
    end

    @testset "Norm squared" begin
        # ||Y^a_{lm}||^2 = l(l+1)
        @test norm_squared(EvenVectorHarmonic(1, 0, up(:a))) == 2
        @test norm_squared(EvenVectorHarmonic(2, 1, up(:a))) == 6
        @test norm_squared(EvenVectorHarmonic(3, 2, up(:a))) == 12
    end

    @testset "Display" begin
        Y = EvenVectorHarmonic(2, 1, up(:a))
        s = sprint(show, Y)
        @test occursin("Y", s)
        @test occursin("2", s)
        @test occursin("1", s)
        @test occursin("a", s)
    end

    @testset "LaTeX" begin
        Y_up = EvenVectorHarmonic(3, -2, up(:a))
        tex_up = to_latex(Y_up)
        @test occursin("Y", tex_up)
        @test occursin("3", tex_up)
        @test occursin("-2", tex_up)
        @test occursin("a", tex_up)

        Y_dn = EvenVectorHarmonic(2, 1, down(:b))
        tex_dn = to_latex(Y_dn)
        @test occursin("Y", tex_dn)
        @test occursin("b", tex_dn)
    end

    @testset "Unicode" begin
        Y = EvenVectorHarmonic(2, 1, up(:a))
        u = to_unicode(Y)
        @test occursin("Y", u)
    end

    @testset "Walk infrastructure" begin
        Y = EvenVectorHarmonic(2, 1, up(:a))
        result = walk(identity, Y)
        @test result == Y

        result2 = walk(x -> x isa EvenVectorHarmonic ? TScalar(42) : x, Y)
        @test result2 == TScalar(42)

        @test isempty(children(Y))
    end

    @testset "AST utility methods" begin
        Y = EvenVectorHarmonic(2, 1, up(:a))
        @test derivative_order(Y) == 0
        @test is_constant(Y) == true
        @test is_sorted_covds(Y) == true
        @test is_well_formed(Y) == true
    end

    @testset "Escape hatch" begin
        Y = EvenVectorHarmonic(3, -1, up(:a))
        expr = to_expr(Y)
        @test expr.head == :call
        @test expr.args[1] == :EvenVectorHarmonic
        @test expr.args[2] == 3
        @test expr.args[3] == -1
    end

    @testset "Rename dummy" begin
        Y = EvenVectorHarmonic(2, 1, up(:a))
        Y2 = rename_dummy(Y, :a, :b)
        @test Y2 == EvenVectorHarmonic(2, 1, up(:b))

        # No match -- unchanged
        Y3 = rename_dummy(Y, :c, :d)
        @test Y3 == Y

        # rename_dummies with mapping
        Y4 = rename_dummies(Y, Dict(:a => :x))
        @test Y4 == EvenVectorHarmonic(2, 1, up(:x))

        # _replace_index_name
        Y5 = TensorGR._replace_index_name(Y, :a, :z)
        @test Y5 == EvenVectorHarmonic(2, 1, up(:z))
    end

    @testset "Arithmetic integration" begin
        Y1 = EvenVectorHarmonic(1, 0, up(:a))
        Y2 = EvenVectorHarmonic(2, 1, up(:b))

        prod = TProduct(1//1, TensorExpr[Y1, Y2])
        @test prod isa TProduct
        @test length(prod.factors) == 2

        s = TSum(TensorExpr[Y1, Y2])
        @test s isa TSum
        @test length(s.terms) == 2
    end

    @testset "Dagger" begin
        Y = EvenVectorHarmonic(2, 1, up(:a))
        d = dagger(Y)
        @test d isa TProduct
        @test d.scalar == -1//1
        @test d.factors[1] == EvenVectorHarmonic(2, -1, up(:a))
    end
end

@testset "OddVectorHarmonic" begin
    @testset "Construction" begin
        X = OddVectorHarmonic(2, 1, up(:a))
        @test X.l == 2
        @test X.m == 1
        @test X.index == up(:a)

        X10 = OddVectorHarmonic(1, 0, down(:b))
        @test X10.l == 1
        @test X10.m == 0
    end

    @testset "Validation" begin
        @test_throws ArgumentError OddVectorHarmonic(0, 0, up(:a))
        @test_throws ArgumentError OddVectorHarmonic(-1, 0, up(:a))
        @test_throws ArgumentError OddVectorHarmonic(2, 3, up(:a))
        @test_throws ArgumentError OddVectorHarmonic(2, -3, up(:a))
    end

    @testset "Equality and hashing" begin
        @test OddVectorHarmonic(2, 1, up(:a)) == OddVectorHarmonic(2, 1, up(:a))
        @test OddVectorHarmonic(2, 1, up(:a)) != OddVectorHarmonic(2, 0, up(:a))
        @test OddVectorHarmonic(2, 1, up(:a)) != OddVectorHarmonic(2, 1, down(:a))
        @test hash(OddVectorHarmonic(2, 1, up(:a))) == hash(OddVectorHarmonic(2, 1, up(:a)))
    end

    @testset "Subtype hierarchy" begin
        @test OddVectorHarmonic <: TensorExpr
        @test OddVectorHarmonic(1, 0, up(:a)) isa TensorExpr
    end

    @testset "Free indices" begin
        X = OddVectorHarmonic(2, 1, up(:a))
        @test free_indices(X) == [up(:a)]
        @test indices(X) == [up(:a)]
    end

    @testset "Conjugation" begin
        X20 = OddVectorHarmonic(2, 0, up(:a))
        conj20 = conjugate(X20)
        @test conj20 isa OddVectorHarmonic
        @test conj20 == OddVectorHarmonic(2, 0, up(:a))

        X21 = OddVectorHarmonic(2, 1, up(:a))
        conj21 = conjugate(X21)
        @test conj21 isa TProduct
        @test conj21.scalar == -1//1
        @test conj21.factors[1] == OddVectorHarmonic(2, -1, up(:a))
    end

    @testset "Curl eigenvalue" begin
        @test curl_eigenvalue(OddVectorHarmonic(1, 0, up(:a))) == 2
        @test curl_eigenvalue(OddVectorHarmonic(2, 1, up(:a))) == 6
        @test curl_eigenvalue(OddVectorHarmonic(3, 0, up(:a))) == 12
    end

    @testset "Norm squared" begin
        @test norm_squared(OddVectorHarmonic(1, 0, up(:a))) == 2
        @test norm_squared(OddVectorHarmonic(2, 1, up(:a))) == 6
        @test norm_squared(OddVectorHarmonic(3, 2, up(:a))) == 12
    end

    @testset "Display" begin
        X = OddVectorHarmonic(2, 1, up(:a))
        s = sprint(show, X)
        @test occursin("X", s)
        @test occursin("2", s)
        @test occursin("1", s)
    end

    @testset "LaTeX" begin
        X = OddVectorHarmonic(3, -2, up(:a))
        tex = to_latex(X)
        @test occursin("X", tex)
        @test occursin("3", tex)
        @test occursin("-2", tex)
    end

    @testset "Unicode" begin
        X = OddVectorHarmonic(2, 1, up(:a))
        u = to_unicode(X)
        @test occursin("X", u)
    end

    @testset "Walk infrastructure" begin
        X = OddVectorHarmonic(2, 1, up(:a))
        result = walk(identity, X)
        @test result == X
        @test isempty(children(X))
    end

    @testset "AST utility methods" begin
        X = OddVectorHarmonic(2, 1, up(:a))
        @test derivative_order(X) == 0
        @test is_constant(X) == true
        @test is_sorted_covds(X) == true
        @test is_well_formed(X) == true
    end

    @testset "Escape hatch" begin
        X = OddVectorHarmonic(3, -1, up(:a))
        expr = to_expr(X)
        @test expr.head == :call
        @test expr.args[1] == :OddVectorHarmonic
        @test expr.args[2] == 3
        @test expr.args[3] == -1
    end

    @testset "Rename dummy" begin
        X = OddVectorHarmonic(2, 1, up(:a))
        X2 = rename_dummy(X, :a, :b)
        @test X2 == OddVectorHarmonic(2, 1, up(:b))

        X3 = rename_dummy(X, :c, :d)
        @test X3 == X

        X4 = rename_dummies(X, Dict(:a => :x))
        @test X4 == OddVectorHarmonic(2, 1, up(:x))

        X5 = TensorGR._replace_index_name(X, :a, :z)
        @test X5 == OddVectorHarmonic(2, 1, up(:z))
    end

    @testset "Dagger" begin
        X = OddVectorHarmonic(2, 1, up(:a))
        d = dagger(X)
        @test d isa TProduct
        @test d.scalar == -1//1
        @test d.factors[1] == OddVectorHarmonic(2, -1, up(:a))
    end
end
