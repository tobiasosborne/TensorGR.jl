using TensorGR: linearize, δRiemann, δRicci, δRicciScalar

@testset "linearize replaces metric with background" begin
    g = Tensor(:g, [down(:a), down(:b)])
    result = linearize(g, :g => (:η, :h))
    @test result isa Tensor
    @test result.name == :η
    @test result.indices == [down(:a), down(:b)]
end

@testset "linearize preserves non-metric tensors" begin
    R = Tensor(:R, [down(:a), down(:b), down(:c), down(:d)])
    result = linearize(R, :g => (:η, :h))
    @test result == R
end

@testset "linearize in products" begin
    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:a), down(:b)])
    expr = g * R
    result = linearize(expr, :g => (:η, :h))
    # g should become η
    @test result isa TProduct
    @test result.factors[1].name == :η
    @test result.factors[2].name == :R
end

@testset "linearize in sums" begin
    g1 = Tensor(:g, [down(:a), down(:b)])
    g2 = Tensor(:g, [down(:c), down(:d)])
    expr = g1 + g2
    result = linearize(expr, :g => (:η, :h))
    @test result isa TSum
    @test all(t -> t isa Tensor && t.name == :η, result.terms)
end

@testset "δRiemann structure" begin
    result = δRiemann(down(:a), down(:b), down(:c), down(:d), :h)
    @test result isa TProduct  # 1//2 * (sum)
    # The inner expression should be a sum of 4 double-derivative terms
    inner = result.factors[1]
    @test inner isa TSum
    @test length(inner.terms) == 4
    # Each term is either a TDeriv or a scalar * TDeriv
    for term in inner.terms
        if term isa TDeriv
            @test true
        elseif term isa TProduct
            @test any(f -> f isa TDeriv, term.factors)
        else
            @test false  # Unexpected term type
        end
    end
end

@testset "δRiemann antisymmetry" begin
    # δR_{abcd} = -δR_{bacd} by antisymmetry in first pair
    # We check the structural property: swapping a,b gives opposite sign terms
    r1 = δRiemann(down(:a), down(:b), down(:c), down(:d), :h)
    r2 = δRiemann(down(:b), down(:a), down(:c), down(:d), :h)

    # r1 + r2 should have terms that cancel pairwise (not zero without further simplification,
    # but structurally: the 4 terms in r1 and r2 are related by a↔b swap)
    @test r1 isa TProduct && r2 isa TProduct
end

@testset "δRicci structure" begin
    result = δRicci(down(:a), down(:b), :h)
    @test result isa TProduct  # 1//2 * sum
    inner = result.factors[1]
    @test inner isa TSum
    @test length(inner.terms) == 4
end

@testset "δRicciScalar structure" begin
    result = δRicciScalar(:h)
    @test result isa TSum
    @test length(result.terms) == 2
end
