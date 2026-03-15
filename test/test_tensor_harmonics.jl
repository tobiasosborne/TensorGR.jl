# Tests for rank-2 tensor spherical harmonics on S^2.
# Ground truth: Martel & Poisson, Phys. Rev. D 71, 104003 (2005).

@testset "EvenTensorHarmonicY" begin
    @testset "Construction" begin
        Y = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        @test Y.l == 2
        @test Y.m == 1
        @test Y.index1 == up(:a)
        @test Y.index2 == up(:b)

        Y00 = EvenTensorHarmonicY(0, 0, down(:a), down(:b))
        @test Y00.l == 0
        @test Y00.m == 0
    end

    @testset "Validation" begin
        @test_throws ArgumentError EvenTensorHarmonicY(-1, 0, up(:a), up(:b))
        @test_throws ArgumentError EvenTensorHarmonicY(2, 3, up(:a), up(:b))
        @test_throws ArgumentError EvenTensorHarmonicY(2, -3, up(:a), up(:b))
    end

    @testset "Equality and hashing" begin
        @test EvenTensorHarmonicY(2, 1, up(:a), up(:b)) == EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        @test EvenTensorHarmonicY(2, 1, up(:a), up(:b)) != EvenTensorHarmonicY(2, 0, up(:a), up(:b))
        @test EvenTensorHarmonicY(2, 1, up(:a), up(:b)) != EvenTensorHarmonicY(3, 1, up(:a), up(:b))
        @test EvenTensorHarmonicY(2, 1, up(:a), up(:b)) != EvenTensorHarmonicY(2, 1, down(:a), up(:b))
        @test hash(EvenTensorHarmonicY(2, 1, up(:a), up(:b))) == hash(EvenTensorHarmonicY(2, 1, up(:a), up(:b)))
        @test hash(EvenTensorHarmonicY(2, 1, up(:a), up(:b))) != hash(EvenTensorHarmonicY(3, 1, up(:a), up(:b)))

        s = Set([EvenTensorHarmonicY(2, 0, up(:a), up(:b)),
                 EvenTensorHarmonicY(2, 1, up(:a), up(:b)),
                 EvenTensorHarmonicY(2, 0, up(:a), up(:b))])
        @test length(s) == 2
    end

    @testset "Subtype hierarchy" begin
        @test EvenTensorHarmonicY <: TensorExpr
    end

    @testset "Free indices" begin
        Y = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        @test free_indices(Y) == [up(:a), up(:b)]
        @test indices(Y) == [up(:a), up(:b)]
    end

    # MP Eq 2.14: trace = 2 Y_{lm}
    @testset "Trace (MP Eq 2.14)" begin
        t = TensorGR.trace(EvenTensorHarmonicY(3, 1, up(:a), up(:b)))
        @test t isa TProduct
        @test t.scalar == 2//1
        @test length(t.factors) == 1
        @test t.factors[1] == ScalarHarmonic(3, 1)
    end

    # MP Eq 2.18: ||Y^{ab}_{lm}||^2 = 2
    @testset "Norm squared (MP Eq 2.18)" begin
        for l in 0:5
            @test norm_squared(EvenTensorHarmonicY(l, 0, up(:a), up(:b))) == 2
        end
    end

    @testset "Conjugation" begin
        Y20 = EvenTensorHarmonicY(2, 0, up(:a), up(:b))
        @test conjugate(Y20) isa EvenTensorHarmonicY
        @test conjugate(Y20) == EvenTensorHarmonicY(2, 0, up(:a), up(:b))

        Y21 = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        c21 = conjugate(Y21)
        @test c21 isa TProduct
        @test c21.scalar == -1//1
        @test c21.factors[1] == EvenTensorHarmonicY(2, -1, up(:a), up(:b))

        Y22 = EvenTensorHarmonicY(2, 2, up(:a), up(:b))
        c22 = conjugate(Y22)
        @test c22.scalar == 1//1
        @test c22.factors[1] == EvenTensorHarmonicY(2, -2, up(:a), up(:b))
    end

    @testset "Dagger" begin
        Y = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        @test dagger(Y) == conjugate(Y)
    end

    @testset "Display" begin
        Y = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        s = sprint(show, Y)
        @test occursin("Y", s)
        @test occursin("2", s)
        @test occursin("1", s)
    end

    @testset "LaTeX" begin
        Y_up = EvenTensorHarmonicY(3, -2, up(:a), up(:b))
        tex = to_latex(Y_up)
        @test occursin("Y", tex)
        @test occursin("3", tex)
        @test occursin("-2", tex)

        Y_dn = EvenTensorHarmonicY(2, 1, down(:a), down(:b))
        tex_dn = to_latex(Y_dn)
        @test occursin("Y", tex_dn)
    end

    @testset "Unicode" begin
        Y = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        u = to_unicode(Y)
        @test occursin("Y", u)
    end

    @testset "Walk infrastructure" begin
        Y = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        @test walk(identity, Y) == Y
        @test walk(x -> x isa EvenTensorHarmonicY ? TScalar(42) : x, Y) == TScalar(42)
        @test isempty(children(Y))
    end

    @testset "AST utility methods" begin
        Y = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        @test derivative_order(Y) == 0
        @test is_constant(Y) == true
        @test is_sorted_covds(Y) == true
        @test is_well_formed(Y) == true
    end

    @testset "Escape hatch" begin
        Y = EvenTensorHarmonicY(3, -1, up(:a), up(:b))
        expr = to_expr(Y)
        @test expr.head == :call
        @test expr.args[1] == :EvenTensorHarmonicY
        @test expr.args[2] == 3
        @test expr.args[3] == -1
    end

    @testset "Rename dummy" begin
        Y = EvenTensorHarmonicY(2, 1, up(:a), up(:b))
        Y2 = rename_dummy(Y, :a, :c)
        @test Y2 == EvenTensorHarmonicY(2, 1, up(:c), up(:b))

        Y3 = rename_dummy(Y, :b, :d)
        @test Y3 == EvenTensorHarmonicY(2, 1, up(:a), up(:d))

        # No match
        @test rename_dummy(Y, :x, :y) == Y

        # rename_dummies
        Y4 = rename_dummies(Y, Dict(:a => :p, :b => :q))
        @test Y4 == EvenTensorHarmonicY(2, 1, up(:p), up(:q))

        # _replace_index_name
        Y5 = TensorGR._replace_index_name(Y, :a, :z)
        @test Y5 == EvenTensorHarmonicY(2, 1, up(:z), up(:b))
    end
end

@testset "EvenTensorHarmonicZ" begin
    @testset "Construction" begin
        Z = EvenTensorHarmonicZ(2, 1, up(:a), up(:b))
        @test Z.l == 2
        @test Z.m == 1
    end

    # MP: Z vanishes for l=0,1
    @testset "Validation (l >= 2)" begin
        @test_throws ArgumentError EvenTensorHarmonicZ(0, 0, up(:a), up(:b))
        @test_throws ArgumentError EvenTensorHarmonicZ(1, 0, up(:a), up(:b))
        @test_throws ArgumentError EvenTensorHarmonicZ(1, 1, up(:a), up(:b))
        @test_throws ArgumentError EvenTensorHarmonicZ(-1, 0, up(:a), up(:b))
        @test_throws ArgumentError EvenTensorHarmonicZ(2, 3, up(:a), up(:b))
    end

    @testset "Equality and hashing" begin
        @test EvenTensorHarmonicZ(2, 1, up(:a), up(:b)) == EvenTensorHarmonicZ(2, 1, up(:a), up(:b))
        @test EvenTensorHarmonicZ(2, 1, up(:a), up(:b)) != EvenTensorHarmonicZ(3, 1, up(:a), up(:b))
        @test hash(EvenTensorHarmonicZ(2, 1, up(:a), up(:b))) == hash(EvenTensorHarmonicZ(2, 1, up(:a), up(:b)))
    end

    @testset "Subtype hierarchy" begin
        @test EvenTensorHarmonicZ <: TensorExpr
    end

    @testset "Free indices" begin
        Z = EvenTensorHarmonicZ(2, 1, up(:a), up(:b))
        @test free_indices(Z) == [up(:a), up(:b)]
        @test indices(Z) == [up(:a), up(:b)]
    end

    # MP Eq 2.21: trace-free
    @testset "Trace = 0 (MP Eq 2.21)" begin
        @test TensorGR.trace(EvenTensorHarmonicZ(2, 0, up(:a), up(:b))) == TScalar(0//1)
        @test TensorGR.trace(EvenTensorHarmonicZ(3, 1, up(:a), up(:b))) == TScalar(0//1)
        @test TensorGR.trace(EvenTensorHarmonicZ(5, -3, up(:a), up(:b))) == TScalar(0//1)
    end

    # MP Eq 2.19: ||Z||^2 = 1/2 (l-1)l(l+1)(l+2)
    @testset "Norm squared (MP Eq 2.19)" begin
        @test norm_squared(EvenTensorHarmonicZ(2, 0, up(:a), up(:b))) == 1 * 2 * 3 * 4 // 2  # 12
        @test norm_squared(EvenTensorHarmonicZ(3, 0, up(:a), up(:b))) == 2 * 3 * 4 * 5 // 2  # 60
        @test norm_squared(EvenTensorHarmonicZ(4, 0, up(:a), up(:b))) == 3 * 4 * 5 * 6 // 2  # 180
        @test norm_squared(EvenTensorHarmonicZ(5, 0, up(:a), up(:b))) == 4 * 5 * 6 * 7 // 2  # 420
    end

    @testset "Conjugation" begin
        Z20 = EvenTensorHarmonicZ(2, 0, up(:a), up(:b))
        @test conjugate(Z20) isa EvenTensorHarmonicZ
        @test conjugate(Z20) == EvenTensorHarmonicZ(2, 0, up(:a), up(:b))

        Z21 = EvenTensorHarmonicZ(2, 1, up(:a), up(:b))
        c21 = conjugate(Z21)
        @test c21 isa TProduct
        @test c21.scalar == -1//1
        @test c21.factors[1] == EvenTensorHarmonicZ(2, -1, up(:a), up(:b))
    end

    @testset "Display" begin
        Z = EvenTensorHarmonicZ(2, 1, up(:a), up(:b))
        @test occursin("Z", sprint(show, Z))
    end

    @testset "LaTeX" begin
        Z = EvenTensorHarmonicZ(3, -2, up(:a), up(:b))
        tex = to_latex(Z)
        @test occursin("Z", tex)
        @test occursin("3", tex)
    end

    @testset "Walk infrastructure" begin
        Z = EvenTensorHarmonicZ(2, 1, up(:a), up(:b))
        @test walk(identity, Z) == Z
        @test isempty(children(Z))
    end

    @testset "AST utility methods" begin
        Z = EvenTensorHarmonicZ(2, 1, up(:a), up(:b))
        @test derivative_order(Z) == 0
        @test is_constant(Z) == true
        @test is_sorted_covds(Z) == true
        @test is_well_formed(Z) == true
    end

    @testset "Escape hatch" begin
        Z = EvenTensorHarmonicZ(3, -1, up(:a), up(:b))
        expr = to_expr(Z)
        @test expr.head == :call
        @test expr.args[1] == :EvenTensorHarmonicZ
        @test expr.args[2] == 3
        @test expr.args[3] == -1
    end

    @testset "Rename dummy" begin
        Z = EvenTensorHarmonicZ(2, 1, up(:a), up(:b))
        Z2 = rename_dummy(Z, :a, :c)
        @test Z2 == EvenTensorHarmonicZ(2, 1, up(:c), up(:b))

        Z3 = rename_dummies(Z, Dict(:b => :d))
        @test Z3 == EvenTensorHarmonicZ(2, 1, up(:a), up(:d))

        Z4 = TensorGR._replace_index_name(Z, :b, :z)
        @test Z4 == EvenTensorHarmonicZ(2, 1, up(:a), up(:z))
    end
end

@testset "OddTensorHarmonic" begin
    @testset "Construction" begin
        X = OddTensorHarmonic(2, 1, up(:a), up(:b))
        @test X.l == 2
        @test X.m == 1
    end

    # MP: X vanishes for l=0,1
    @testset "Validation (l >= 2)" begin
        @test_throws ArgumentError OddTensorHarmonic(0, 0, up(:a), up(:b))
        @test_throws ArgumentError OddTensorHarmonic(1, 0, up(:a), up(:b))
        @test_throws ArgumentError OddTensorHarmonic(1, 1, up(:a), up(:b))
        @test_throws ArgumentError OddTensorHarmonic(-1, 0, up(:a), up(:b))
        @test_throws ArgumentError OddTensorHarmonic(2, 3, up(:a), up(:b))
    end

    @testset "Equality and hashing" begin
        @test OddTensorHarmonic(2, 1, up(:a), up(:b)) == OddTensorHarmonic(2, 1, up(:a), up(:b))
        @test OddTensorHarmonic(2, 1, up(:a), up(:b)) != OddTensorHarmonic(3, 1, up(:a), up(:b))
        @test OddTensorHarmonic(2, 1, up(:a), up(:b)) != OddTensorHarmonic(2, 1, down(:a), up(:b))
        @test hash(OddTensorHarmonic(2, 1, up(:a), up(:b))) == hash(OddTensorHarmonic(2, 1, up(:a), up(:b)))
    end

    @testset "Subtype hierarchy" begin
        @test OddTensorHarmonic <: TensorExpr
    end

    @testset "Free indices" begin
        X = OddTensorHarmonic(2, 1, up(:a), up(:b))
        @test free_indices(X) == [up(:a), up(:b)]
        @test indices(X) == [up(:a), up(:b)]
    end

    # MP Eq 2.21: trace-free
    @testset "Trace = 0 (MP Eq 2.21)" begin
        @test TensorGR.trace(OddTensorHarmonic(2, 0, up(:a), up(:b))) == TScalar(0//1)
        @test TensorGR.trace(OddTensorHarmonic(3, 1, up(:a), up(:b))) == TScalar(0//1)
    end

    # MP Eq 2.20: ||X||^2 = 1/2 (l-1)l(l+1)(l+2)
    @testset "Norm squared (MP Eq 2.20)" begin
        @test norm_squared(OddTensorHarmonic(2, 0, up(:a), up(:b))) == 12  # 1*2*3*4/2
        @test norm_squared(OddTensorHarmonic(3, 0, up(:a), up(:b))) == 60  # 2*3*4*5/2
        @test norm_squared(OddTensorHarmonic(4, 0, up(:a), up(:b))) == 180 # 3*4*5*6/2
    end

    @testset "Conjugation" begin
        X20 = OddTensorHarmonic(2, 0, up(:a), up(:b))
        @test conjugate(X20) isa OddTensorHarmonic
        @test conjugate(X20) == OddTensorHarmonic(2, 0, up(:a), up(:b))

        X21 = OddTensorHarmonic(2, 1, up(:a), up(:b))
        c21 = conjugate(X21)
        @test c21 isa TProduct
        @test c21.scalar == -1//1
        @test c21.factors[1] == OddTensorHarmonic(2, -1, up(:a), up(:b))
    end

    @testset "Dagger" begin
        X = OddTensorHarmonic(2, 1, up(:a), up(:b))
        @test dagger(X) == conjugate(X)
    end

    @testset "Display" begin
        X = OddTensorHarmonic(2, 1, up(:a), up(:b))
        @test occursin("X", sprint(show, X))
    end

    @testset "LaTeX" begin
        X = OddTensorHarmonic(3, -2, up(:a), up(:b))
        tex = to_latex(X)
        @test occursin("X", tex)
        @test occursin("3", tex)
    end

    @testset "Walk infrastructure" begin
        X = OddTensorHarmonic(2, 1, up(:a), up(:b))
        @test walk(identity, X) == X
        @test isempty(children(X))
    end

    @testset "AST utility methods" begin
        X = OddTensorHarmonic(2, 1, up(:a), up(:b))
        @test derivative_order(X) == 0
        @test is_constant(X) == true
        @test is_sorted_covds(X) == true
        @test is_well_formed(X) == true
    end

    @testset "Escape hatch" begin
        X = OddTensorHarmonic(3, -1, up(:a), up(:b))
        expr = to_expr(X)
        @test expr.head == :call
        @test expr.args[1] == :OddTensorHarmonic
        @test expr.args[2] == 3
        @test expr.args[3] == -1
    end

    @testset "Rename dummy" begin
        X = OddTensorHarmonic(2, 1, up(:a), up(:b))
        X2 = rename_dummy(X, :a, :c)
        @test X2 == OddTensorHarmonic(2, 1, up(:c), up(:b))

        X3 = rename_dummies(X, Dict(:a => :p, :b => :q))
        @test X3 == OddTensorHarmonic(2, 1, up(:p), up(:q))

        X4 = TensorGR._replace_index_name(X, :b, :z)
        @test X4 == OddTensorHarmonic(2, 1, up(:a), up(:z))
    end
end

@testset "Tensor harmonic orthogonality (MP Eqs 2.18-2.20)" begin
    # Same-type inner product: delta_{ll'} delta_{mm'} * norm
    @testset "Same-type inner product" begin
        for l in 2:4, m in -l:l
            Y = EvenTensorHarmonicY(l, m, up(:a), up(:b))
            @test inner_product(Y, Y) == TScalar(2)

            Z = EvenTensorHarmonicZ(l, m, up(:a), up(:b))
            n_Z = (l - 1) * l * (l + 1) * (l + 2) // 2
            @test inner_product(Z, Z) == TScalar(n_Z)

            X = OddTensorHarmonic(l, m, up(:a), up(:b))
            @test inner_product(X, X) == TScalar(n_Z)
        end
    end

    # Different quantum numbers: zero
    @testset "Orthogonality in (l,m)" begin
        @test inner_product(
            EvenTensorHarmonicZ(2, 0, up(:a), up(:b)),
            EvenTensorHarmonicZ(3, 0, up(:a), up(:b))) == TScalar(0)
        @test inner_product(
            OddTensorHarmonic(2, 0, up(:a), up(:b)),
            OddTensorHarmonic(2, 1, up(:a), up(:b))) == TScalar(0)
    end

    # Cross-type: zero (MP Eq 2.20)
    @testset "Cross-type orthogonality (MP Eq 2.20)" begin
        Y = EvenTensorHarmonicY(2, 0, up(:a), up(:b))
        Z = EvenTensorHarmonicZ(2, 0, up(:a), up(:b))
        X = OddTensorHarmonic(2, 0, up(:a), up(:b))

        @test inner_product(Y, Z) == TScalar(0)
        @test inner_product(Z, Y) == TScalar(0)
        @test inner_product(Y, X) == TScalar(0)
        @test inner_product(X, Y) == TScalar(0)
        @test inner_product(Z, X) == TScalar(0)
        @test inner_product(X, Z) == TScalar(0)
    end
end

@testset "Arithmetic integration" begin
    Y = EvenTensorHarmonicY(2, 0, up(:a), up(:b))
    Z = EvenTensorHarmonicZ(2, 0, up(:c), up(:d))
    prod = TProduct(1//1, TensorExpr[Y, Z])
    @test prod isa TProduct
    @test length(prod.factors) == 2

    s = TSum(TensorExpr[Y])
    @test s isa TSum
end
