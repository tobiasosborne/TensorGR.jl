@testset "Rules Engine" begin
    @testset "Basic rule application" begin
        # Simple replacement rule
        T_ab = Tensor(:T, [down(:a), down(:b)])
        S_ab = Tensor(:S, [down(:a), down(:b)])
        rule = RewriteRule(T_ab, S_ab)

        # Direct match
        result = apply_rules(T_ab, [rule])
        @test result == S_ab

        # No match
        U_ab = Tensor(:U, [down(:a), down(:b)])
        @test apply_rules(U_ab, [rule]) == U_ab
    end

    @testset "Rule in product" begin
        T_ab = Tensor(:T, [down(:a), down(:b)])
        S_ab = Tensor(:S, [down(:a), down(:b)])
        rule = RewriteRule(T_ab, S_ab)

        V_cd = Tensor(:V, [down(:c), down(:d)])
        product = tproduct(1 // 1, TensorExpr[T_ab, V_cd])
        result = apply_rules(product, [rule])
        @test result isa TProduct
        # T should be replaced by S in the product
        found_S = any(f -> f isa Tensor && f.name == :S, result.factors)
        @test found_S
    end

    @testset "Rule in sum" begin
        T_ab = Tensor(:T, [down(:a), down(:b)])
        S_ab = Tensor(:S, [down(:a), down(:b)])
        U_ab = Tensor(:U, [down(:a), down(:b)])
        rule = RewriteRule(T_ab, S_ab)

        expr = tsum(TensorExpr[T_ab, U_ab])
        result = apply_rules(expr, [rule])
        @test result isa TSum
        has_S = any(t -> t isa Tensor && t.name == :S, result.terms)
        has_U = any(t -> t isa Tensor && t.name == :U, result.terms)
        @test has_S && has_U
    end

    @testset "Rule in derivative" begin
        T_ab = Tensor(:T, [down(:a), down(:b)])
        S_ab = Tensor(:S, [down(:a), down(:b)])
        rule = RewriteRule(T_ab, S_ab)

        d_expr = TDeriv(down(:c), T_ab)
        result = apply_rules(d_expr, [rule])
        @test result isa TDeriv
        @test result.arg == S_ab
    end

    @testset "Functional pattern matching" begin
        # Rule that matches any Tensor named :T
        pattern = (expr) -> expr isa Tensor && expr.name == :T
        replacement = (expr) -> begin
            t = expr::Tensor
            Tensor(:S, t.indices)
        end
        rule = RewriteRule(pattern, replacement)

        T1 = Tensor(:T, [down(:a), down(:b)])
        T2 = Tensor(:T, [up(:c)])
        U = Tensor(:U, [down(:a)])

        @test apply_rules(T1, [rule]) == Tensor(:S, [down(:a), down(:b)])
        @test apply_rules(T2, [rule]) == Tensor(:S, [up(:c)])
        @test apply_rules(U, [rule]) == U
    end

    @testset "Conditional rules" begin
        T_ab = Tensor(:T, [down(:a), down(:b)])
        S_ab = Tensor(:S, [down(:a), down(:b)])

        # Rule only fires if condition holds
        cond_rule = RewriteRule(T_ab, S_ab, _ -> false)
        @test apply_rules(T_ab, [cond_rule]) == T_ab  # condition blocks

        cond_rule2 = RewriteRule(T_ab, S_ab, _ -> true)
        @test apply_rules(T_ab, [cond_rule2]) == S_ab  # condition allows
    end

    @testset "Fixed-point iteration" begin
        # A → B → C via two rules; fixed point should reach C
        A = Tensor(:A, TIndex[])
        B = Tensor(:B, TIndex[])
        C = Tensor(:C, TIndex[])
        rules = [RewriteRule(A, B), RewriteRule(B, C)]

        result = apply_rules_fixpoint(A, rules)
        @test result == C
    end

    @testset "Fixed-point convergence" begin
        # Self-replacing rule: should converge immediately
        A = Tensor(:A, TIndex[])
        rules = [RewriteRule(A, A)]
        result = apply_rules_fixpoint(A, rules; maxiter=10)
        @test result == A
    end

    @testset "Registry rules" begin
        reg = TensorRegistry()
        A = Tensor(:A, TIndex[])
        B = Tensor(:B, TIndex[])
        rule = RewriteRule(A, B)

        register_rule!(reg, rule)
        @test length(get_rules(reg)) == 1
        @test get_rules(reg)[1] === rule
    end

    @testset "@rule macro" begin
        A = Tensor(:A, TIndex[])
        B = Tensor(:B, TIndex[])
        r = @rule A => B
        @test r isa RewriteRule
        @test apply_rules(A, [r]) == B
    end
end
