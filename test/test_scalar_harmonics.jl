@testset "ScalarHarmonic" begin
    @testset "Construction" begin
        Y = ScalarHarmonic(2, 1)
        @test Y.l == 2
        @test Y.m == 1

        Y00 = ScalarHarmonic(0, 0)
        @test Y00.l == 0
        @test Y00.m == 0

        Y33 = ScalarHarmonic(3, -3)
        @test Y33.l == 3
        @test Y33.m == -3
    end

    @testset "Validation" begin
        @test_throws ArgumentError ScalarHarmonic(-1, 0)
        @test_throws ArgumentError ScalarHarmonic(2, 3)
        @test_throws ArgumentError ScalarHarmonic(2, -3)
        @test_throws ArgumentError ScalarHarmonic(0, 1)
    end

    @testset "Equality and hashing" begin
        @test ScalarHarmonic(2, 1) == ScalarHarmonic(2, 1)
        @test ScalarHarmonic(2, 1) != ScalarHarmonic(2, 0)
        @test ScalarHarmonic(2, 1) != ScalarHarmonic(3, 1)
        @test hash(ScalarHarmonic(2, 1)) == hash(ScalarHarmonic(2, 1))
        @test hash(ScalarHarmonic(2, 1)) != hash(ScalarHarmonic(3, 1))

        # Usable in Sets
        s = Set([ScalarHarmonic(1, 0), ScalarHarmonic(1, 1), ScalarHarmonic(1, 0)])
        @test length(s) == 2
    end

    @testset "Subtype hierarchy" begin
        @test ScalarHarmonic <: TensorExpr
        Y = ScalarHarmonic(1, 0)
        @test Y isa TensorExpr
    end

    @testset "Free indices (scalar)" begin
        Y = ScalarHarmonic(2, 1)
        @test isempty(free_indices(Y))
        @test isempty(indices(Y))
    end

    @testset "Conjugation" begin
        # m == 0: conjugate returns ScalarHarmonic directly
        Y20 = ScalarHarmonic(2, 0)
        conj20 = conjugate(Y20)
        @test conj20 isa ScalarHarmonic
        @test conj20 == ScalarHarmonic(2, 0)

        # m > 0: conjugate returns (-1)^m * Y_{l,-m}
        Y21 = ScalarHarmonic(2, 1)
        conj21 = conjugate(Y21)
        @test conj21 isa TProduct
        @test conj21.scalar == -1//1
        @test length(conj21.factors) == 1
        @test conj21.factors[1] == ScalarHarmonic(2, -1)

        # m even: positive sign
        Y22 = ScalarHarmonic(2, 2)
        conj22 = conjugate(Y22)
        @test conj22 isa TProduct
        @test conj22.scalar == 1//1
        @test conj22.factors[1] == ScalarHarmonic(2, -2)

        # Negative m
        Y2m1 = ScalarHarmonic(2, -1)
        conj2m1 = conjugate(Y2m1)
        @test conj2m1 isa TProduct
        @test conj2m1.scalar == -1//1
        @test conj2m1.factors[1] == ScalarHarmonic(2, 1)
    end

    @testset "Angular Laplacian" begin
        # l=0: eigenvalue = 0
        Y00 = ScalarHarmonic(0, 0)
        lap00 = angular_laplacian(Y00)
        @test lap00 isa TProduct
        @test lap00.scalar == 0//1

        # l=1: eigenvalue = -2
        Y10 = ScalarHarmonic(1, 0)
        lap10 = angular_laplacian(Y10)
        @test lap10.scalar == -2//1
        @test lap10.factors[1] == Y10

        # l=3, m=1: eigenvalue = -12
        Y31 = ScalarHarmonic(3, 1)
        lap31 = angular_laplacian(Y31)
        @test lap31.scalar == -12//1
        @test lap31.factors[1] == Y31

        # l=2: eigenvalue = -6
        Y21 = ScalarHarmonic(2, 1)
        lap21 = angular_laplacian(Y21)
        @test lap21.scalar == -6//1
        @test lap21.factors[1] == Y21
    end

    @testset "Inner product (orthogonality)" begin
        # Same quantum numbers: 1
        @test inner_product(ScalarHarmonic(2, 1), ScalarHarmonic(2, 1)) == TScalar(1)
        @test inner_product(ScalarHarmonic(0, 0), ScalarHarmonic(0, 0)) == TScalar(1)

        # Different l: 0
        @test inner_product(ScalarHarmonic(2, 1), ScalarHarmonic(3, 1)) == TScalar(0)

        # Different m: 0
        @test inner_product(ScalarHarmonic(2, 1), ScalarHarmonic(2, 0)) == TScalar(0)

        # Both different: 0
        @test inner_product(ScalarHarmonic(1, 0), ScalarHarmonic(3, 2)) == TScalar(0)
    end

    @testset "Display" begin
        Y = ScalarHarmonic(2, 1)
        s = sprint(show, Y)
        @test occursin("Y", s)
        @test occursin("2", s)
        @test occursin("1", s)
    end

    @testset "LaTeX" begin
        Y = ScalarHarmonic(3, -2)
        tex = to_latex(Y)
        @test occursin("Y", tex)
        @test occursin("3", tex)
        @test occursin("-2", tex)
    end

    @testset "Unicode" begin
        Y = ScalarHarmonic(2, 1)
        u = to_unicode(Y)
        @test occursin("Y", u)
    end

    @testset "Walk infrastructure" begin
        Y = ScalarHarmonic(2, 1)
        # walk is identity for leaf
        result = walk(identity, Y)
        @test result == Y

        # walk applies function
        result2 = walk(x -> x isa ScalarHarmonic ? TScalar(42) : x, Y)
        @test result2 == TScalar(42)

        # children is empty
        @test isempty(children(Y))
    end

    @testset "AST utility methods" begin
        Y = ScalarHarmonic(2, 1)
        @test derivative_order(Y) == 0
        @test is_constant(Y) == true
        @test is_sorted_covds(Y) == true
        @test is_well_formed(Y) == true
    end

    @testset "Escape hatch" begin
        Y = ScalarHarmonic(3, -1)
        expr = to_expr(Y)
        @test expr.head == :call
        @test expr.args[1] == :ScalarHarmonic
        @test expr.args[2] == 3
        @test expr.args[3] == -1
    end

    @testset "Arithmetic integration" begin
        Y1 = ScalarHarmonic(1, 0)
        Y2 = ScalarHarmonic(2, 1)

        # Product with tensor: ScalarHarmonic in TProduct
        prod = TProduct(1//1, TensorExpr[Y1, Y2])
        @test prod isa TProduct
        @test length(prod.factors) == 2

        # Sum: ScalarHarmonic in TSum
        s = TSum(TensorExpr[Y1, Y2])
        @test s isa TSum
        @test length(s.terms) == 2
    end

    @testset "Dagger (complex conjugation)" begin
        Y = ScalarHarmonic(2, 1)
        d = dagger(Y)
        # dagger delegates to conjugate
        @test d isa TProduct
        @test d.scalar == -1//1
        @test d.factors[1] == ScalarHarmonic(2, -1)
    end

    @testset "Rename dummy (no-op for scalar)" begin
        Y = ScalarHarmonic(2, 1)
        @test rename_dummy(Y, :a, :b) === Y
        @test rename_dummies(Y, Dict(:a => :b)) === Y
    end
end
