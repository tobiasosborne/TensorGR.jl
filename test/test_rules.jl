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

    @testset "Pattern variable detection" begin
        @test is_pattern_variable(down(:a_)) == true
        @test is_pattern_variable(up(:a_)) == true
        @test is_pattern_variable(down(:a)) == false
        @test is_pattern_variable(up(:x)) == false
        @test is_pattern_variable(down(:ab_)) == true
    end

    @testset "Pattern index matching: basic tensor" begin
        pat = Tensor(:T, [down(:a_), down(:b_)])
        repl = Tensor(:S, [down(:a_), down(:b_)])
        rule = RewriteRule(pat, repl)

        # Matches any two down indices
        expr1 = Tensor(:T, [down(:a), down(:b)])
        expr2 = Tensor(:T, [down(:c), down(:d)])
        expr3 = Tensor(:T, [down(:x), down(:y)])

        @test apply_rules(expr1, [rule]) == Tensor(:S, [down(:a), down(:b)])
        @test apply_rules(expr2, [rule]) == Tensor(:S, [down(:c), down(:d)])
        @test apply_rules(expr3, [rule]) == Tensor(:S, [down(:x), down(:y)])
    end

    @testset "Pattern index: wrong tensor name → no match" begin
        pat = Tensor(:T, [down(:a_), down(:b_)])
        repl = Tensor(:S, [down(:a_), down(:b_)])
        rule = RewriteRule(pat, repl)

        expr = Tensor(:U, [down(:a), down(:b)])
        @test apply_rules(expr, [rule]) == expr  # no match
    end

    @testset "Pattern index: wrong index count → no match" begin
        pat = Tensor(:T, [down(:a_), down(:b_)])
        repl = Tensor(:S, [down(:a_), down(:b_)])
        rule = RewriteRule(pat, repl)

        expr = Tensor(:T, [down(:a)])
        @test apply_rules(expr, [rule]) == expr  # no match
    end

    @testset "Pattern index: repeated variable" begin
        # T_{a_,a_} matches T_{cc} but not T_{cd}
        pat = Tensor(:T, [down(:a_), down(:a_)])
        repl = TScalar(1 // 1)
        rule = RewriteRule(pat, repl)

        expr_same = Tensor(:T, [down(:c), down(:c)])
        expr_diff = Tensor(:T, [down(:c), down(:d)])

        @test apply_rules(expr_same, [rule]) == TScalar(1 // 1)
        @test apply_rules(expr_diff, [rule]) == expr_diff  # no match
    end

    @testset "Pattern index: mixed pattern and literal" begin
        # T_{a_,:b} matches first any, second must be :b
        pat = Tensor(:T, [down(:a_), down(:b)])
        repl = Tensor(:S, [down(:a_)])
        rule = RewriteRule(pat, repl)

        expr1 = Tensor(:T, [down(:x), down(:b)])  # matches
        expr2 = Tensor(:T, [down(:x), down(:c)])  # no match (c ≠ b)

        @test apply_rules(expr1, [rule]) == Tensor(:S, [down(:x)])
        @test apply_rules(expr2, [rule]) == expr2
    end

    @testset "Pattern index: in product" begin
        pat = Tensor(:T, [down(:a_), down(:b_)])
        repl = Tensor(:S, [down(:a_), down(:b_)])
        rule = RewriteRule(pat, repl)

        V = Tensor(:V, [up(:c)])
        prod = tproduct(1 // 1, TensorExpr[Tensor(:T, [down(:x), down(:y)]), V])
        result = apply_rules(prod, [rule])
        @test result isa TProduct
        found_S = any(f -> f isa Tensor && f.name == :S, result.factors)
        @test found_S
    end

    @testset "Pattern index: in sum" begin
        pat = Tensor(:T, [down(:a_)])
        repl = Tensor(:S, [down(:a_)])
        rule = RewriteRule(pat, repl)

        U = Tensor(:U, [down(:a)])
        expr = tsum(TensorExpr[Tensor(:T, [down(:a)]), U])
        result = apply_rules(expr, [rule])
        @test result isa TSum
        has_S = any(t -> t isa Tensor && t.name == :S, result.terms)
        @test has_S
    end

    @testset "Pattern index: in TDeriv" begin
        # ∂_{a_} T_{b_} → 0
        pat = TDeriv(down(:a_), Tensor(:T, [down(:b_)]))
        rule = RewriteRule(pat, TScalar(0 // 1))

        expr = TDeriv(down(:c), Tensor(:T, [down(:d)]))
        @test apply_rules(expr, [rule]) == TScalar(0 // 1)

        # Wrong covd → no match
        expr2 = TDeriv(down(:c), Tensor(:T, [down(:d)]), :nabla)
        @test apply_rules(expr2, [rule]) == expr2
    end

    @testset "Pattern index: TDeriv with matched covd" begin
        pat = TDeriv(down(:a_), Tensor(:g, [down(:b_), down(:c_)]), :nabla)
        rule = RewriteRule(pat, TScalar(0 // 1))

        expr = TDeriv(down(:x), Tensor(:g, [down(:y), down(:z)]), :nabla)
        @test apply_rules(expr, [rule]) == TScalar(0 // 1)
    end

    @testset "Pattern index: product pattern with cross-binding" begin
        # T_{a_} * U_{b_} → S_{a_,b_} (same position on both sides)
        pat = TProduct(1 // 1, TensorExpr[
            Tensor(:T, [down(:a_)]),
            Tensor(:U, [down(:b_)])
        ])
        repl = Tensor(:S, [down(:a_), down(:b_)])
        rule = RewriteRule(pat, repl)

        expr = TProduct(1 // 1, TensorExpr[
            Tensor(:T, [down(:x)]),
            Tensor(:U, [down(:y)])
        ])
        result = apply_rules(expr, [rule])
        @test result == Tensor(:S, [down(:x), down(:y)])
    end

    @testset "Pattern index: backward compat with functional patterns" begin
        pattern = (expr) -> expr isa Tensor && expr.name == :T
        replacement = (expr) -> Tensor(:S, expr.indices)
        rule = RewriteRule(pattern, replacement)

        T1 = Tensor(:T, [down(:a), down(:b)])
        @test apply_rules(T1, [rule]) == Tensor(:S, [down(:a), down(:b)])
    end

    @testset "Pattern index: scalar replacement" begin
        # Ric_{a_,b_} → Λ * g_{a_,b_}
        pat = Tensor(:Ric, [down(:a_), down(:b_)])
        repl = tproduct(1 // 1, TensorExpr[
            TScalar(:Lambda),
            Tensor(:g, [down(:a_), down(:b_)])
        ])
        rule = RewriteRule(pat, repl)

        expr = Tensor(:Ric, [down(:c), down(:d)])
        result = apply_rules(expr, [rule])
        @test result isa TProduct
        has_g = any(f -> f isa Tensor && f.name == :g &&
                    f.indices == [down(:c), down(:d)], result.factors)
        @test has_g
    end

    @testset "Pattern index: with condition" begin
        pat = Tensor(:T, [down(:a_), down(:b_)])
        repl = TScalar(0 // 1)
        # Only match if tensor has exactly 2 indices (redundant but tests condition path)
        rule = RewriteRule(pat, repl, expr -> length(indices(expr)) == 2)

        expr = Tensor(:T, [down(:a), down(:b)])
        @test apply_rules(expr, [rule]) == TScalar(0 // 1)
    end

    @testset "Pattern index: fixpoint with patterns" begin
        # A_{a_} → B_{a_} → C_{a_}
        r1 = RewriteRule(Tensor(:A, [down(:a_)]), Tensor(:B, [down(:a_)]))
        r2 = RewriteRule(Tensor(:B, [down(:a_)]), Tensor(:C, [down(:a_)]))

        result = apply_rules_fixpoint(Tensor(:A, [down(:x)]), [r1, r2])
        @test result == Tensor(:C, [down(:x)])
    end
end
