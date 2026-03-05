using TensorGR: expand_derivatives

@testset "Leibniz rule on products" begin
    # ∂_a(T^b S^c) = (∂_a T^b) S^c + T^b (∂_a S^c)
    T = Tensor(:T, [up(:b)])
    S = Tensor(:S, [up(:c)])
    prod = T * S
    deriv = TDeriv(down(:a), prod)

    result = expand_derivatives(deriv)
    @test result isa TSum
    @test length(result.terms) == 2
end

@testset "Leibniz rule on triple product" begin
    # ∂_a(A * B * C) = (∂_a A) B C + A (∂_a B) C + A B (∂_a C)
    A = Tensor(:A, [up(:b)])
    B = Tensor(:B, [up(:c)])
    C = Tensor(:C, [up(:d)])
    prod = A * B * C
    deriv = TDeriv(down(:a), prod)

    result = expand_derivatives(deriv)
    @test result isa TSum
    @test length(result.terms) == 3
end

@testset "Derivative of single tensor: no expansion" begin
    T = Tensor(:T, [up(:b), down(:c)])
    deriv = TDeriv(down(:a), T)
    result = expand_derivatives(deriv)
    @test result == deriv  # unchanged
end

@testset "Derivative of scalar: zero" begin
    s = TScalar(42)
    deriv = TDeriv(down(:a), s)
    result = expand_derivatives(deriv)
    @test result == TScalar(0//1)
end

@testset "Derivative of sum distributes" begin
    # ∂_a(R + S) = ∂_a R + ∂_a S
    R = Tensor(:R, [down(:b)])
    S = Tensor(:S, [down(:b)])
    deriv = TDeriv(down(:a), R + S)

    result = expand_derivatives(deriv)
    @test result isa TSum
    @test length(result.terms) == 2
    @test all(t -> t isa TDeriv, result.terms)
end

@testset "Scalar coefficient passes through derivative" begin
    T = Tensor(:T, [up(:b)])
    expr = TProduct(3//2, TensorExpr[T])
    deriv = TDeriv(down(:a), expr)

    result = expand_derivatives(deriv)
    # ∂_a(3/2 T^b) = 3/2 ∂_a T^b
    @test result isa TProduct
    @test result.scalar == 3//2
end

@testset "Nested derivatives" begin
    T = Tensor(:T, [up(:c)])
    inner = TDeriv(down(:b), T)
    outer = TDeriv(down(:a), inner)

    result = expand_derivatives(outer)
    # ∂_a(∂_b T^c) — already expanded, no product to apply Leibniz to
    @test result == outer
end

@testset "Derivative commutator structure" begin
    # [∂_a, ∂_b] V^c = ∂_a(∂_b V^c) - ∂_b(∂_a V^c)
    V = Tensor(:V, [up(:c)])
    comm = TDeriv(down(:a), TDeriv(down(:b), V)) - TDeriv(down(:b), TDeriv(down(:a), V))
    @test comm isa TSum
    @test length(comm.terms) == 2
end
