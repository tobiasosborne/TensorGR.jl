@testset "@tensor single tensor" begin
    t = @tensor R[-a, -b, -c, -d]
    @test t isa Tensor
    @test t.name == :R
    @test t.indices == [down(:a), down(:b), down(:c), down(:d)]

    g = @tensor g[a, b]
    @test g isa Tensor
    @test g.indices == [up(:a), up(:b)]

    # Mixed indices
    t2 = @tensor R[a, -b, c, -d]
    @test t2.indices == [up(:a), down(:b), up(:c), down(:d)]
end

@testset "@tensor scalar-like" begin
    # Tensor with no indices
    s = @tensor Φ
    @test s isa Tensor
    @test s.name == :Φ
    @test isempty(s.indices)
end

@testset "@tensor products" begin
    p = @tensor g[a, b] * R[-a, -c, -b, -d]
    @test p isa TProduct
    @test length(p.factors) == 2
    @test p.factors[1].name == :g
    @test p.factors[2].name == :R

    # Triple product
    p2 = @tensor g[a, b] * g[c, d] * R[-a, -b, -c, -d]
    @test p2 isa TProduct
    @test length(p2.factors) == 3
end

@testset "@tensor with scalar coefficients" begin
    p = @tensor (1 // 2) * R[-a, -b]
    @test p isa TProduct
    @test p.scalar == 1//2

    p2 = @tensor (-1) * g[-a, -b]
    @test p2 isa TProduct
    @test p2.scalar == -1//1

    # Negation
    p3 = @tensor -R[-a, -b]
    @test p3 isa TProduct
    @test p3.scalar == -1//1
end

@testset "@tensor sums" begin
    s = @tensor R[-a, -b] + g[-a, -b]
    @test s isa TSum
    @test length(s.terms) == 2

    # Sum with scalar product
    s2 = @tensor R[-a, -b] - (1 // 2) * g[-a, -b]
    @test s2 isa TSum
    @test length(s2.terms) == 2
end

@testset "@tensor derivatives" begin
    d = @tensor ∂[-a](T[b, -c])
    @test d isa TDeriv
    @test d.index == down(:a)
    @test d.arg isa Tensor
    @test d.arg.name == :T

    # Nested derivative
    d2 = @tensor ∂[-a](∂[-b](T[-c]))
    @test d2 isa TDeriv
    @test d2.arg isa TDeriv
end

@testset "@tensor complex expressions" begin
    # Einstein tensor: R_{ab} - 1/2 g_{ab} R
    e = @tensor Ric[-a, -b] - (1 // 2) * g[-a, -b] * RicciScalar
    @test e isa TSum
    @test length(e.terms) == 2
    @test e.terms[1] isa Tensor
    @test e.terms[2] isa TProduct
end
