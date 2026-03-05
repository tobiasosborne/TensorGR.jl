using TensorGR: walk, substitute, children

@testset "children" begin
    t = Tensor(:R, [down(:a), down(:b)])
    @test isempty(children(t))

    s = TScalar(3//2)
    @test isempty(children(s))

    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:a), down(:b)])
    p = TProduct(1//1, TensorExpr[g, R])
    @test children(p) == TensorExpr[g, R]

    sm = TSum(TensorExpr[g, R])
    @test children(sm) == TensorExpr[g, R]

    d = TDeriv(down(:c), R)
    @test children(d) == TensorExpr[R]
end

@testset "walk identity" begin
    t = Tensor(:R, [down(:a), down(:b), down(:c), down(:d)])
    @test walk(identity, t) == t

    p = TProduct(1//1, TensorExpr[t])
    @test walk(identity, p) == p

    s = TSum(TensorExpr[t])
    @test walk(identity, s) == s

    d = TDeriv(down(:e), t)
    @test walk(identity, d) == d
end

@testset "walk transformation" begin
    # Replace all Tensor names :R with :S
    function rename_R_to_S(expr::TensorExpr)
        if expr isa Tensor && expr.name == :R
            return Tensor(:S, expr.indices)
        end
        return expr
    end

    t = Tensor(:R, [down(:a), down(:b)])
    @test walk(rename_R_to_S, t) == Tensor(:S, [down(:a), down(:b)])

    # In a product
    g = Tensor(:g, [up(:a), up(:b)])
    p = TProduct(1//1, TensorExpr[g, t])
    p2 = walk(rename_R_to_S, p)
    @test p2.factors[1] == g  # g unchanged
    @test p2.factors[2] == Tensor(:S, [down(:a), down(:b)])

    # In a sum
    s = TSum(TensorExpr[t, g])
    s2 = walk(rename_R_to_S, s)
    @test s2.terms[1] == Tensor(:S, [down(:a), down(:b)])
    @test s2.terms[2] == g

    # In a derivative
    d = TDeriv(down(:c), t)
    d2 = walk(rename_R_to_S, d)
    @test d2.arg == Tensor(:S, [down(:a), down(:b)])
end

@testset "substitute" begin
    R = Tensor(:R, [down(:a), down(:b)])
    g = Tensor(:g, [down(:a), down(:b)])
    S = Tensor(:S, [down(:a), down(:b)])

    # Direct substitution
    result = substitute(R, R => S)
    @test result == S

    # Substitution in product
    p = TProduct(1//1, TensorExpr[g, R])
    p2 = substitute(p, R => S)
    @test p2.factors[2] == S
    @test p2.factors[1] == g

    # Substitution in sum
    sm = TSum(TensorExpr[R, g])
    sm2 = substitute(sm, R => S)
    @test sm2.terms[1] == S
    @test sm2.terms[2] == g

    # Substitution in derivative
    d = TDeriv(down(:c), R)
    d2 = substitute(d, R => S)
    @test d2.arg == S

    # No match: expression unchanged
    @test substitute(g, R => S) == g
end
