@testset "TIndex show" begin
    @test sprint(show, up(:a)) == "a"
    @test sprint(show, down(:b)) == "-b"
end

@testset "Tensor show" begin
    t = Tensor(:R, [down(:a), down(:b), down(:c), down(:d)])
    @test sprint(show, t) == "R[-a, -b, -c, -d]"

    g = Tensor(:g, [up(:a), up(:b)])
    @test sprint(show, g) == "g[a, b]"

    s = Tensor(:RicciScalar, TIndex[])
    @test sprint(show, s) == "RicciScalar"

    # Mixed indices
    t2 = Tensor(:R, [up(:a), down(:b), up(:c), down(:d)])
    @test sprint(show, t2) == "R[a, -b, c, -d]"
end

@testset "TScalar show" begin
    @test sprint(show, TScalar(3//2)) == "3//2"
    @test sprint(show, TScalar(:β)) == "β"
    @test sprint(show, TScalar(1)) == "1"
    @test sprint(show, TScalar(0)) == "0"
end

@testset "TProduct show" begin
    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:c), down(:d)])

    # Unit scalar
    p1 = TProduct(1//1, TensorExpr[g, R])
    @test sprint(show, p1) == "g[a, b] * R[-c, -d]"

    # Negative unit scalar
    p2 = TProduct(-1//1, TensorExpr[R])
    @test sprint(show, p2) == "-R[-c, -d]"

    # Fractional scalar
    p3 = TProduct(1//2, TensorExpr[R])
    @test sprint(show, p3) == "(1//2) * R[-c, -d]"

    p4 = TProduct(-3//2, TensorExpr[g, R])
    @test sprint(show, p4) == "(-3//2) * g[a, b] * R[-c, -d]"

    # Single factor, scalar = 1
    p5 = TProduct(1//1, TensorExpr[g])
    @test sprint(show, p5) == "g[a, b]"
end

@testset "TSum show" begin
    t1 = Tensor(:R, [down(:a), down(:b)])
    t2 = Tensor(:g, [down(:a), down(:b)])
    s = TSum(TensorExpr[t1, t2])
    @test sprint(show, s) == "R[-a, -b] + g[-a, -b]"

    # Sum with negative product term
    t3 = TProduct(-1//2, TensorExpr[Tensor(:g, [down(:a), down(:b)]), Tensor(:RicciScalar, TIndex[])])
    s2 = TSum(TensorExpr[t1, t3])
    @test sprint(show, s2) == "R[-a, -b] + (-1//2) * g[-a, -b] * RicciScalar"
end

@testset "TDeriv show" begin
    t = Tensor(:T, [up(:b), down(:c)])
    d = TDeriv(down(:a), t)
    @test sprint(show, d) == "∂[-a](T[b, -c])"

    # Nested derivative
    d2 = TDeriv(down(:d), d)
    @test sprint(show, d2) == "∂[-d](∂[-a](T[b, -c]))"
end
