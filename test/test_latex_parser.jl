@testset "LaTeX Parser" begin

    @testset "Basic tensors" begin
        # Single tensor with down indices
        t = tex"R_{abcd}"
        @test t isa Tensor
        @test t.name == :R
        @test t.indices == [down(:a), down(:b), down(:c), down(:d)]

        # Up indices
        t = tex"g^{ab}"
        @test t.name == :g
        @test t.indices == [up(:a), up(:b)]

        # Scalar tensor (no indices)
        t = tex"RicScalar"
        @test t isa Tensor
        @test t.name == :RicScalar
        @test isempty(t.indices)

        # Single-char name
        t = tex"R"
        @test t.name == :R
        @test isempty(t.indices)
    end

    @testset "Mixed indices" begin
        # R^a{}_{bcd} — up a, down b,c,d
        t = tex"R^a{}_{bcd}"
        @test t.indices == [up(:a), down(:b), down(:c), down(:d)]

        # T^{ab}_{cd}
        t = tex"T^{ab}_{cd}"
        @test t.indices == [up(:a), up(:b), down(:c), down(:d)]

        # Down first: T_{ab}^{cd}
        t = tex"T_{ab}^{cd}"
        @test t.indices == [down(:a), down(:b), up(:c), up(:d)]

        # Single indices without braces
        t = tex"T^a_b"
        @test t.indices == [up(:a), down(:b)]

        # Alternating: T_a^b
        t = tex"T_a^b"
        @test t.indices == [down(:a), up(:b)]
    end

    @testset "Products (juxtaposition)" begin
        # g^{ab} R_{abcd}
        p = tex"g^{ab} R_{abcd}"
        @test p isa TProduct
        @test length(p.factors) == 2
        @test p.factors[1].name == :g
        @test p.factors[2].name == :R

        # Triple product
        p = tex"g^{ab} g^{cd} R_{abcd}"
        @test p isa TProduct
        @test length(p.factors) == 3
    end

    @testset "Sums" begin
        s = tex"R_{ab} + g_{ab}"
        @test s isa TSum
        @test length(s.terms) == 2

        s = tex"R_{ab} - g_{ab}"
        @test s isa TSum
        @test length(s.terms) == 2
    end

    @testset "Coefficients" begin
        # Integer coefficient
        p = tex"2 R_{ab}"
        @test p isa TProduct
        @test p.scalar == 2 // 1

        # Fraction with \frac
        p = tex"\frac{1}{2} g_{ab}"
        @test p isa TProduct
        @test p.scalar == 1 // 2

        # Fraction with /
        p = parse_tex("1/2 g_{ab}")
        @test p isa TProduct
        @test p.scalar == 1 // 2

        # Unary minus
        p = tex"-R_{ab}"
        @test p isa TProduct
        @test p.scalar == -1 // 1
    end

    @testset "Einstein tensor expression" begin
        # R_{ab} - \frac{1}{2} g_{ab} R
        e = tex"Ric_{ab} - \frac{1}{2} g_{ab} RicScalar"
        @test e isa TSum
        @test length(e.terms) == 2

        # First term: Ric_{ab}
        @test e.terms[1] isa Tensor
        @test e.terms[1].name == :Ric

        # Second term: -(1/2) g_{ab} RicScalar
        t2 = e.terms[2]
        @test t2 isa TProduct
        @test t2.scalar == -1 // 2
    end

    @testset "Derivatives" begin
        # \partial_a V^b
        d = tex"\partial_a V^b"
        @test d isa TDeriv
        @test d.index == down(:a)
        @test d.arg isa Tensor
        @test d.arg.name == :V
        @test d.arg.indices == [up(:b)]

        # \nabla_a T^{bc}
        d = tex"\nabla_a T^{bc}"
        @test d isa TDeriv
        @test d.index == down(:a)
        @test d.arg.indices == [up(:b), up(:c)]

        # Derivative with explicit superscript: \partial^a V_b
        d = tex"\partial^a V_b"
        @test d.index == up(:a)

        # Nested derivatives: \partial_a \partial_b T_c
        d = tex"\partial_a \partial_b T_c"
        @test d isa TDeriv
        @test d.index == down(:a)
        @test d.arg isa TDeriv
        @test d.arg.index == down(:b)
        @test d.arg.arg isa Tensor

        # Unicode partial: ∂_a V^b
        d = parse_tex("∂_a V^b")
        @test d isa TDeriv
        @test d.index == down(:a)
    end

    @testset "Greek letters" begin
        # \Gamma^a_{bc}
        t = tex"\Gamma^a_{bc}"
        @test t.name == :Γ
        @test t.indices == [up(:a), down(:b), down(:c)]

        # \alpha index
        t = tex"T_{\alpha\beta}"
        @test t.indices == [down(:α), down(:β)]

        # Unicode Greek directly
        t = parse_tex("T_{αβ}")
        @test t.indices == [down(:α), down(:β)]

        # Mixed Greek and Latin
        t = tex"F^A_{\mu\nu}"
        @test t.indices == [up(:A), down(:μ), down(:ν)]
    end

    @testset "Parenthesized expressions" begin
        # (R_{ab} + g_{ab}) V^a
        p = tex"(R_{ab} + g_{ab}) V^a"
        @test p isa TProduct
    end

    @testset "Complex expressions" begin
        # Christoffel in terms of metric gradients
        expr = tex"\frac{1}{2} g^{ad} (\partial_b g_{cd} + \partial_c g_{bd} - \partial_d g_{bc})"
        @test expr isa TProduct

        # Kretschmann structure: R_{abcd} R^{abcd}
        expr = tex"R_{abcd} R^{abcd}"
        @test expr isa TProduct
        @test length(expr.factors) == 2
        @test expr.factors[1].indices == [down(:a), down(:b), down(:c), down(:d)]
        @test expr.factors[2].indices == [up(:a), up(:b), up(:c), up(:d)]
    end

    @testset "parse_tex function" begin
        # Same as tex"..." but called as a function
        t = parse_tex("R_{abcd}")
        @test t isa Tensor
        @test t.name == :R
        @test length(t.indices) == 4
    end

    @testset "Integration with simplify" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)

            # g^{ab} g_{bc} = delta^a_c
            result = simplify(tex"g^{ab} g_{bc}")
            @test result isa Tensor
            @test result.name == :δ
            @test result.indices == [up(:a), down(:c)]

            # Riemann antisymmetry
            result = simplify(tex"Riem_{abcd} + Riem_{bacd}")
            @test result == TScalar(0 // 1)

            # delta trace
            result = simplify(tex"g^{ab} g_{ab}")
            @test result == TScalar(4 // 1)
        end
    end

    @testset "Edge cases" begin
        # Empty braces as separator: R^{ab}{}_{cd}
        t = tex"R^{ab}{}_{cd}"
        @test t.indices == [up(:a), up(:b), down(:c), down(:d)]

        # Multiple empty braces
        t = tex"T^a{}{}_{b}"
        @test t.indices == [up(:a), down(:b)]

        # Bare number
        s = tex"42"
        @test s == TScalar(42 // 1)

        # Negative number
        s = tex"-3"
        @test s == TScalar(-3 // 1)
    end

    @testset "Registry tex aliases" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)

            # R_{abcd} → Riem (rank 4 alias)
            t = tex"R_{abcd}"
            @test t.name == :Riem
            @test t.indices == [down(:a), down(:b), down(:c), down(:d)]

            # R_{ab} → Ric (rank 2 alias)
            t2 = tex"R_{ab}"
            @test t2.name == :Ric

            # R (rank 0) → RicScalar
            t3 = tex"R"
            @test t3.name == :RicScalar
            @test isempty(t3.indices)

            # G_{ab} → Ein
            t4 = tex"G_{ab}"
            @test t4.name == :Ein

            # C_{abcd} → Weyl
            t5 = tex"C_{abcd}"
            @test t5.name == :Weyl

            # Unaliased name passes through
            t6 = tex"T_{ab}"
            @test t6.name == :T
        end

        # Without registry: graceful fallback
        t7 = parse_tex("R_{abcd}")
        @test t7.name == :R
    end
end
