using TensorGR: to_expr, from_expr, is_well_formed

@testset "to_expr / from_expr round-trip" begin
    # Tensor
    t = Tensor(:R, [down(:a), down(:b), up(:c), down(:d)])
    @test from_expr(to_expr(t)) == t

    # TScalar with rational
    s1 = TScalar(3//2)
    @test from_expr(to_expr(s1)) == s1

    # TScalar with symbol
    s2 = TScalar(:β)
    @test from_expr(to_expr(s2)) == s2

    # TProduct
    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:c), down(:d)])
    p = TProduct(-1//2, TensorExpr[g, R])
    @test from_expr(to_expr(p)) == p

    # TSum
    sm = TSum(TensorExpr[g, R])
    @test from_expr(to_expr(sm)) == sm

    # TDeriv
    d = TDeriv(down(:e), R)
    @test from_expr(to_expr(d)) == d

    # Nested: ∂_e(g^{ab} R_{cd})
    nested = TDeriv(down(:e), TProduct(1//1, TensorExpr[g, R]))
    @test from_expr(to_expr(nested)) == nested

    # Empty indices
    scalar_tensor = Tensor(:Phi, TIndex[])
    @test from_expr(to_expr(scalar_tensor)) == scalar_tensor
end

@testset "is_well_formed" begin
    # Well-formed expressions
    @test is_well_formed(Tensor(:R, [down(:a), down(:b)]))
    @test is_well_formed(TScalar(42))
    @test is_well_formed(TDeriv(down(:a), Tensor(:T, [up(:b)])))

    # Well-formed product with proper dummy pairing
    g = Tensor(:g, [up(:a), up(:b)])
    T = Tensor(:T, [down(:a), down(:b)])
    @test is_well_formed(TProduct(1//1, TensorExpr[g, T]))

    # Malformed: same name appears twice as Up
    t1 = Tensor(:A, [up(:a)])
    t2 = Tensor(:B, [up(:a)])
    @test !is_well_formed(TProduct(1//1, TensorExpr[t1, t2]))

    # Well-formed sum
    @test is_well_formed(TSum(TensorExpr[
        Tensor(:R, [down(:a), down(:b)]),
        Tensor(:g, [down(:a), down(:b)])
    ]))
end
