@testset "Gamma matrices and Clifford algebra" begin

    @testset "GammaMatrix construction" begin
        g = GammaMatrix(up(:a))
        @test g isa GammaMatrix
        @test g.index == up(:a)
        @test g isa TensorExpr
    end

    @testset "Equality and hashing" begin
        g1 = GammaMatrix(up(:a))
        g2 = GammaMatrix(up(:a))
        g3 = GammaMatrix(down(:a))
        @test g1 == g2
        @test hash(g1) == hash(g2)
        @test g1 != g3
    end

    @testset "AST integration" begin
        g = GammaMatrix(up(:a))
        @test indices(g) == [up(:a)]
        @test free_indices(g) == [up(:a)]
        @test children(g) == TensorExpr[]
        @test derivative_order(g) == 0
        @test !is_constant(g)
        @test is_sorted_covds(g)
    end

    @testset "rename_dummy" begin
        g = GammaMatrix(up(:a))
        g2 = rename_dummy(g, :a, :b)
        @test g2 == GammaMatrix(up(:b))
    end

    @testset "rename_dummies" begin
        g = GammaMatrix(down(:a))
        g2 = rename_dummies(g, Dict(:a => :c))
        @test g2 == GammaMatrix(down(:c))
    end

    @testset "_replace_index_name" begin
        g = GammaMatrix(up(:a))
        g2 = TensorGR._replace_index_name(g, :a, :z)
        @test g2 == GammaMatrix(up(:z))
    end

    @testset "walk" begin
        g = GammaMatrix(up(:a))
        result = walk(identity, g)
        @test result == g
    end

    @testset "to_expr" begin
        g = GammaMatrix(up(:a))
        e = to_expr(g)
        @test e isa Expr
    end

    @testset "Display" begin
        g_up = GammaMatrix(up(:a))
        g_down = GammaMatrix(down(:b))

        @test occursin("γ", sprint(show, g_up))
        @test occursin("γ", sprint(show, g_down))
        @test occursin("gamma", to_latex(g_up))
        @test occursin("γ", to_unicode(g_up))
    end

    @testset "dagger" begin
        g = GammaMatrix(up(:a))
        gd = dagger(g)
        @test gd isa GammaMatrix
        @test gd.index.position == Down
    end

    @testset "clifford_relation" begin
        rel = clifford_relation(up(:a), up(:b))
        @test rel isa TensorExpr
        # Should be 2 g^{ab}
        @test rel isa TProduct
        @test rel.scalar == 2 // 1
    end

    @testset "gamma_trace" begin
        @test gamma_trace(0) == 4   # Tr(I) = 4
        @test gamma_trace(1) == 0   # Tr(γ^a) = 0
        @test gamma_trace(2) == 4   # Tr(γ^a γ^b) = 4 g^{ab}
        @test gamma_trace(3) == 0   # Odd traces vanish
    end

    @testset "gamma5 construction" begin
        g5 = gamma5()
        @test g5 isa TensorExpr
    end

    @testset "Product of gamma matrices" begin
        g1 = GammaMatrix(up(:a))
        g2 = GammaMatrix(up(:b))
        prod = g1 * g2
        @test prod isa TProduct
    end

end
