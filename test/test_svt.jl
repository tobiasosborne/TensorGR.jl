using TensorGR: to_fourier, FourierConvention, transverse_projector, tt_projector,
                SVTFields, svt_substitute

@testset "to_fourier: single derivative → momentum" begin
    T = Tensor(:T, [down(:b)])
    deriv = TDeriv(down(:a), T)
    result = to_fourier(deriv)

    # ∂_a T_b → k_a T_b (product of momentum and field)
    @test result isa TProduct
    @test length(result.factors) == 2
    @test result.factors[1] == Tensor(:k, [down(:a)])
    @test result.factors[2] == Tensor(:T, [down(:b)])
end

@testset "to_fourier: double derivative → k k" begin
    T = Tensor(:T, TIndex[])
    d1 = TDeriv(down(:b), T)
    d2 = TDeriv(down(:a), d1)

    result = to_fourier(d2)
    # ∂_a ∂_b T → k_a k_b T
    @test result isa TProduct
    # Should contain k_a, k_b, and T as factors
    all_tensors = result.factors
    k_count = count(f -> f isa Tensor && f.name == :k, all_tensors)
    @test k_count == 2
end

@testset "to_fourier: sum distributes" begin
    A = TDeriv(down(:a), Tensor(:A, TIndex[]))
    B = TDeriv(down(:a), Tensor(:B, TIndex[]))
    expr = A + B

    result = to_fourier(expr)
    @test result isa TSum
    @test length(result.terms) == 2
end

@testset "to_fourier: product with derivative" begin
    # ∂_a(T * S) already expanded by expand_derivatives
    # Here test that to_fourier handles products correctly
    k = Tensor(:k, [down(:a)])
    T = Tensor(:T, TIndex[])
    expr = TProduct(1//1, TensorExpr[TDeriv(down(:a), T)])

    result = to_fourier(expr)
    @test result isa TProduct
end

@testset "Transverse projector structure" begin
    P = transverse_projector(down(:i), down(:j))
    @test P isa TSum
    @test length(P.terms) == 2  # δ_{ij} - k_i k_j / k²
end

@testset "TT projector structure" begin
    Π = tt_projector(down(:i), down(:j), down(:k), down(:l))
    @test Π isa TProduct  # 1//2 * (sum of 3 terms)
end

@testset "SVT field substitution" begin
    h = Tensor(:h, [down(:a), down(:b)])
    result = svt_substitute(h, :h)
    @test result isa Tensor
    @test result.name == :svt_h
end
