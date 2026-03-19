# Ground truth: Peskin & Schroeder (1995) Eq A.30; Wald, GR, Appendix B.
#
# γ⁵ properties:
#   (γ⁵)² = I
#   {γ⁵, γ^a} = 0
#   Tr(γ⁵) = 0
#   Tr(γ⁵ γ^a γ^b) = 0
#   Tr(γ⁵ γ^a γ^b γ^c γ^d) = -4i ε^{abcd}

@testset "Gamma^5 chirality matrix" begin

    @testset "Gamma5 type" begin
        g5 = Gamma5()
        @test g5 isa TensorExpr
        @test g5 == Gamma5()
        @test isempty(indices(g5))
        @test isempty(free_indices(g5))
        @test isempty(children(g5))
    end

    @testset "(γ⁵)² = I" begin
        @test gamma5_squared() == TScalar(1)
    end

    @testset "{γ⁵, γ^a} = 0" begin
        @test gamma5_anticommutator() == TScalar(0)
    end

    @testset "Tr(γ⁵) = 0" begin
        @test gamma5_trace(0) == 0
    end

    @testset "Tr(γ⁵ γ^a) = 0 (odd)" begin
        @test gamma5_trace(1) == 0
    end

    @testset "Tr(γ⁵ γ^a γ^b) = 0" begin
        @test gamma5_trace(2) == 0
    end

    @testset "Tr(γ⁵ γ^a γ^b γ^c) = 0 (odd)" begin
        @test gamma5_trace(3) == 0
    end

    @testset "Tr(γ⁵ γ^a γ^b γ^c γ^d) = -4i ε^{abcd}" begin
        result = gamma5_trace(4)
        @test result == :(-4im)
    end

    @testset "Display" begin
        g5 = Gamma5()
        @test sprint(show, g5) == "γ⁵"
        @test to_latex(g5) == "\\gamma^5"
        @test to_unicode(g5) == "γ⁵"
    end

    @testset "Hermiticity: (γ⁵)† = γ⁵" begin
        g5 = Gamma5()
        @test dagger(g5) == g5
    end

    @testset "AST integration" begin
        g5 = Gamma5()
        @test derivative_order(g5) == 0
        @test is_well_formed(g5)
        @test walk(identity, g5) == g5
        @test rename_dummy(g5, :a, :b) == g5
    end

    @testset "gamma5() product form" begin
        g5_prod = gamma5()
        @test g5_prod isa TProduct
        # Should contain 4 gamma matrices + imaginary unit
        @test length(g5_prod.factors) == 5
    end

    @testset "slash notation" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            # slash(v_a) = γ^a v_a
            v = Tensor(:v, [down(:a)])
            sv = slash(v)
            @test sv isa TProduct
            @test length(sv.factors) == 2

            # One factor is GammaMatrix, other is the vector
            has_gamma = any(f -> f isa GammaMatrix, sv.factors)
            has_v = any(f -> f isa Tensor && f.name == :v, sv.factors)
            @test has_gamma
            @test has_v

            # Result should be scalar (contracted index)
            @test isempty(free_indices(sv))
        end
    end

    @testset "slash requires vector" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            # Scalar: should error
            s = Tensor(:RicScalar, TIndex[])
            @test_throws ErrorException slash(s)
            # Rank-2: should error
            T = Tensor(:g, [down(:a), down(:b)])
            @test_throws ErrorException slash(T)
        end
    end
end
