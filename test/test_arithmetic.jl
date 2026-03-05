using TensorGR: indices, free_indices, dummy_pairs

@testset "Scalar * TensorExpr" begin
    R = Tensor(:R, [down(:a), down(:b)])

    # Integer * Tensor
    p = 2 * R
    @test p isa TProduct
    @test p.scalar == 2//1
    @test p.factors == TensorExpr[R]

    # Rational * Tensor
    p2 = (1//2) * R
    @test p2.scalar == 1//2

    # Tensor * scalar
    p3 = R * 3
    @test p3.scalar == 3//1

    # 1 * Tensor collapses
    @test (1 * R) == R
    @test (1//1 * R) == R

    # 0 * Tensor → zero
    z = 0 * R
    @test z == TScalar(0//1)

    # -Tensor
    neg = -R
    @test neg isa TProduct
    @test neg.scalar == -1//1
    @test neg.factors == TensorExpr[R]
end

@testset "Tensor * Tensor" begin
    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:c), down(:d)])

    p = g * R
    @test p isa TProduct
    @test p.scalar == 1//1
    @test length(p.factors) == 2
    @test p.factors[1] == g
    @test p.factors[2] == R
end

@testset "TProduct * TProduct flattening" begin
    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:c), down(:d)])
    T = Tensor(:T, [down(:e)])

    p1 = TProduct(2//1, TensorExpr[g])
    p2 = TProduct(3//1, TensorExpr[R, T])
    p = p1 * p2
    @test p.scalar == 6//1
    @test length(p.factors) == 3
    @test p.factors == TensorExpr[g, R, T]
end

@testset "Scalar * TProduct absorption" begin
    R = Tensor(:R, [down(:a), down(:b)])
    p = TProduct(3//1, TensorExpr[R])

    p2 = 2 * p
    @test p2.scalar == 6//1
    @test p2.factors == TensorExpr[R]

    # (1//3) * 3R = 1*R, which normalizes to just R
    p3 = (1//3) * p
    @test p3 == R
end

@testset "Dummy clash in multiplication" begin
    # Both use dummy :a — should be auto-renamed in the product
    g1 = Tensor(:g, [up(:a), up(:b)])
    T1 = Tensor(:T, [down(:a), down(:c)])
    p1 = g1 * T1  # dummy: :a

    g2 = Tensor(:g, [up(:a), up(:d)])
    T2 = Tensor(:S, [down(:a), down(:e)])
    p2 = g2 * T2  # dummy: :a (clash!)

    product = p1 * p2
    dp = dummy_pairs(product)
    dummy_names = [pair[1].name for pair in dp]
    # The two original :a dummies should now have different names
    @test length(unique(dummy_names)) == length(dummy_names)
end

@testset "Tensor + Tensor" begin
    R = Tensor(:R, [down(:a), down(:b)])
    g = Tensor(:g, [down(:a), down(:b)])

    s = R + g
    @test s isa TSum
    @test length(s.terms) == 2
end

@testset "TSum + TensorExpr flattening" begin
    R = Tensor(:R, [down(:a), down(:b)])
    g = Tensor(:g, [down(:a), down(:b)])
    T = Tensor(:T, [down(:a), down(:b)])

    s1 = R + g
    s2 = s1 + T
    # Should flatten: 3 terms, not nested TSum
    @test s2 isa TSum
    @test length(s2.terms) == 3

    # TensorExpr + TSum
    s3 = T + s1
    @test s3 isa TSum
    @test length(s3.terms) == 3

    # TSum + TSum
    s4 = TSum(TensorExpr[R]) + TSum(TensorExpr[g, T])
    @test length(s4.terms) == 3
end

@testset "Subtraction" begin
    R = Tensor(:R, [down(:a), down(:b)])
    g = Tensor(:g, [down(:a), down(:b)])

    s = R - g
    @test s isa TSum
    @test length(s.terms) == 2
    @test s.terms[2] isa TProduct
    @test s.terms[2].scalar == -1//1
end

@testset "Algebraic identities" begin
    R = Tensor(:R, [down(:a), down(:b)])

    # a + 0 = a  (via TScalar(0))
    @test (R + TScalar(0//1)) == R  # TSum collapses single non-zero term

    # 0 * a = 0
    @test (0 * R) == TScalar(0//1)

    # 1 * a = a
    @test (1 * R) == R

    # (-1) * ((-1) * R) = R
    @test (-1) * ((-1) * R) == R

    # 2 * (3 * R) = 6 * R
    p = 2 * (3 * R)
    @test p isa TProduct
    @test p.scalar == 6//1
end
