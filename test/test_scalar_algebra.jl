@testset "Scalar Algebra" begin
    @testset "scalar_expand basics" begin
        # Numbers pass through
        @test scalar_expand(42) == 42
        @test scalar_expand(3 // 4) == 3 // 4

        # Symbols pass through
        @test scalar_expand(:x) == :x
    end

    @testset "scalar_expand distribution" begin
        # (a + b) * c → a*c + b*c
        ex = :(($(:a) + $(:b)) * $(:c))
        result = scalar_expand(ex)
        # Should be a sum
        @test result isa Expr
        @test result.head == :call
        @test result.args[1] == :+
    end

    @testset "scalar_expand nested" begin
        # (a + b) * (c + d) should expand to 4 terms
        ex = :(($(:a) + $(:b)) * ($(:c) + $(:d)))
        result = scalar_expand(ex)
        @test result isa Expr
        @test result.args[1] == :+
        @test length(result.args) - 1 == 4  # 4 terms in the sum
    end

    @testset "scalar_expand numeric" begin
        # 2 * (a + b) → 2a + 2b
        ex = :(2 * ($(:a) + $(:b)))
        result = scalar_expand(ex)
        @test result isa Expr
        @test result.args[1] == :+
    end

    @testset "scalar_collect" begin
        # 3x + 2x → {1: 5}... well, collect returns Dict
        # Actually collect groups by power of var
        # 3*x + 2*x: power 1 with coeff 5
        ex = :(3 * x + 2 * x)
        result = scalar_collect(ex, :x)
        @test haskey(result, 1)
        # The coefficient of x^1 should evaluate to 5
        coeff = result[1]
        if coeff isa Number
            @test coeff == 5
        else
            # It's an expression :(3 + 2), we can check it evaluates
            @test eval(coeff) == 5
        end
    end

    @testset "scalar_subst" begin
        # Substitute x → 2 in x + 1
        ex = :(x + 1)
        result = scalar_subst(ex, Dict(:x => 2))
        @test result isa Expr
        @test eval(result) == 3
    end

    @testset "scalar_cancel" begin
        @test scalar_cancel(3 // 6) == 1 // 2
        @test scalar_cancel(:x) == :x
    end
end
