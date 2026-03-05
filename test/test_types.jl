@testset "IndexPosition enum" begin
    @test Up isa IndexPosition
    @test Down isa IndexPosition
    @test Up != Down
end

@testset "TIndex construction" begin
    i = TIndex(:a, Up)
    @test i.name == :a
    @test i.position == Up

    j = TIndex(:b, Down)
    @test j.name == :b
    @test j.position == Down
end

@testset "TIndex convenience constructors" begin
    i = up(:a)
    @test i == TIndex(:a, Up)
    @test i.position == Up

    j = down(:b)
    @test j == TIndex(:b, Down)
    @test j.position == Down
end

@testset "TIndex equality and hashing" begin
    @test up(:a) == up(:a)
    @test up(:a) != down(:a)
    @test up(:a) != up(:b)
    @test hash(up(:a)) == hash(up(:a))
    @test hash(up(:a)) != hash(down(:a))

    # Usable in Sets and Dicts
    s = Set([up(:a), down(:b), up(:a)])
    @test length(s) == 2
end

@testset "TensorExpr subtype hierarchy" begin
    @test Tensor <: TensorExpr
    @test TProduct <: TensorExpr
    @test TSum <: TensorExpr
    @test TDeriv <: TensorExpr
    @test TScalar <: TensorExpr
end

@testset "Tensor construction" begin
    t = Tensor(:R, [down(:a), down(:b), down(:c), down(:d)])
    @test t.name == :R
    @test length(t.indices) == 4
    @test t.indices[1] == down(:a)
    @test t.indices[4] == down(:d)

    # Empty indices (scalar-like tensor, e.g. Ricci scalar)
    s = Tensor(:RicciScalar, TIndex[])
    @test s.name == :RicciScalar
    @test isempty(s.indices)
end

@testset "Tensor equality and hashing" begin
    t1 = Tensor(:R, [down(:a), down(:b)])
    t2 = Tensor(:R, [down(:a), down(:b)])
    t3 = Tensor(:R, [down(:a), down(:c)])
    t4 = Tensor(:g, [down(:a), down(:b)])

    @test t1 == t2
    @test t1 != t3
    @test t1 != t4
    @test hash(t1) == hash(t2)
end

@testset "TProduct construction" begin
    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:a), down(:c), down(:b), down(:d)])
    p = TProduct(1//1, TensorExpr[g, R])

    @test p.scalar == 1//1
    @test length(p.factors) == 2
    @test p.factors[1] == g
    @test p.factors[2] == R

    # Negative scalar
    pn = TProduct(-1//2, TensorExpr[R])
    @test pn.scalar == -1//2
end

@testset "TProduct equality" begin
    g = Tensor(:g, [up(:a), up(:b)])
    p1 = TProduct(1//1, TensorExpr[g])
    p2 = TProduct(1//1, TensorExpr[g])
    p3 = TProduct(2//1, TensorExpr[g])

    @test p1 == p2
    @test p1 != p3
    @test hash(p1) == hash(p2)
end

@testset "TSum construction" begin
    t1 = Tensor(:R, [down(:a), down(:b)])
    t2 = Tensor(:g, [down(:a), down(:b)])
    s = TSum(TensorExpr[t1, t2])

    @test length(s.terms) == 2
    @test s.terms[1] == t1
    @test s.terms[2] == t2
end

@testset "TSum equality" begin
    t1 = Tensor(:R, [down(:a), down(:b)])
    t2 = Tensor(:g, [down(:a), down(:b)])
    s1 = TSum(TensorExpr[t1, t2])
    s2 = TSum(TensorExpr[t1, t2])
    s3 = TSum(TensorExpr[t2, t1])

    @test s1 == s2
    @test s1 != s3  # order matters in the raw representation
    @test hash(s1) == hash(s2)
end

@testset "TDeriv construction" begin
    t = Tensor(:T, [up(:b), down(:c)])
    d = TDeriv(down(:a), t)

    @test d.index == down(:a)
    @test d.arg == t
end

@testset "TDeriv equality" begin
    t = Tensor(:T, [up(:b), down(:c)])
    d1 = TDeriv(down(:a), t)
    d2 = TDeriv(down(:a), t)
    d3 = TDeriv(up(:a), t)

    @test d1 == d2
    @test d1 != d3
    @test hash(d1) == hash(d2)
end

@testset "TScalar construction" begin
    s1 = TScalar(3//2)
    @test s1.val == 3//2

    s2 = TScalar(:β)
    @test s2.val == :β

    s3 = TScalar(0)
    @test s3.val == 0
end

@testset "TScalar equality" begin
    @test TScalar(3//2) == TScalar(3//2)
    @test TScalar(:β) == TScalar(:β)
    @test TScalar(3//2) != TScalar(1//2)
    @test hash(TScalar(:β)) == hash(TScalar(:β))
end

@testset "Nested expressions" begin
    # ∂_a(g^{bc} R_{bcde})
    g = Tensor(:g, [up(:b), up(:c)])
    R = Tensor(:R, [down(:b), down(:c), down(:d), down(:e)])
    prod = TProduct(1//1, TensorExpr[g, R])
    deriv = TDeriv(down(:a), prod)

    @test deriv isa TensorExpr
    @test deriv.arg isa TProduct
    @test deriv.arg.factors[1].name == :g

    # Sum of products
    t1 = TProduct(1//1, TensorExpr[Tensor(:R, [down(:a), down(:b)])])
    t2 = TProduct(-1//2, TensorExpr[Tensor(:g, [down(:a), down(:b)]), Tensor(:RicciScalar, TIndex[])])
    einstein = TSum(TensorExpr[t1, t2])

    @test einstein isa TensorExpr
    @test length(einstein.terms) == 2
end
