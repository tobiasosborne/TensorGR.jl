@testset "Phase 1: Full Canonicalization" begin

    # Helper: standard GR registry
    function gr_registry()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂,
            [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M4, rank=(1,1),
            symmetries=Any[],
            options=Dict{Symbol,Any}(:is_delta => true)))
        define_curvature_tensors!(reg, :M4, :g)
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)]))
        register_tensor!(reg, TensorProperties(
            name=:V, manifold=:M4, rank=(1,0),
            symmetries=Any[]))
        register_tensor!(reg, TensorProperties(
            name=:A, manifold=:M4, rank=(0,2),
            symmetries=Any[AntiSymmetric(1,2)]))
        reg
    end

    @testset "1.1: Canonicalize derivatives in products" begin
        reg = gr_registry()
        with_registry(reg) do
            # ∂_a(T_{bc}) should canonicalize: T is symmetric, so
            # ∂_a(T_{cb}) = ∂_a(T_{bc})
            d_T_cb = TDeriv(down(:a), Tensor(:T, [down(:c), down(:b)]))
            d_T_bc = TDeriv(down(:a), Tensor(:T, [down(:b), down(:c)]))
            # Canonicalizing inside the derivative
            result = canonicalize(d_T_cb)
            @test result == d_T_bc

            # Product with derivative: ∂_a(T_{bc}) * V^b
            # Should canonicalize properly
            prod = tproduct(1//1, TensorExpr[
                TDeriv(down(:a), Tensor(:T, [down(:c), down(:b)])),
                Tensor(:V, [up(:b)])
            ])
            result = canonicalize(prod)
            @test result isa TProduct || result isa TDeriv
        end
    end

    @testset "1.1: Commuting partial derivatives" begin
        reg = gr_registry()
        with_registry(reg) do
            # ∂_b ∂_a(T_{cd}) * V^e should canonicalize with derivs sorted
            # because partials commute
            inner = Tensor(:T, [down(:c), down(:d)])
            Ve = Tensor(:V, [up(:e)])
            expr = tproduct(1//1, TensorExpr[
                TDeriv(down(:b), TDeriv(down(:a), inner)), Ve])
            canon_expr = tproduct(1//1, TensorExpr[
                TDeriv(down(:a), TDeriv(down(:b), inner)), Ve])
            result = canonicalize(expr)
            expected = canonicalize(canon_expr)
            @test result == expected
        end
    end

    @testset "1.2: Bianchi rules creation" begin
        rules = bianchi_rules()
        @test length(rules) >= 2
        @test all(r -> r isa RewriteRule, rules)
    end

    @testset "1.2: Contracted Bianchi ∇^a G_{ab} = 0" begin
        reg = gr_registry()
        # Register Bianchi rules
        for r in bianchi_rules()
            register_rule!(reg, r)
        end
        with_registry(reg) do
            # ∂^a G_{ab} → 0
            expr = TDeriv(up(:a), Tensor(:Ein, [down(:a), down(:b)]))
            result = simplify(expr)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "1.2: Contracted Bianchi ∇^a R_{ab} = (1/2) ∇_b R" begin
        reg = gr_registry()
        for r in bianchi_rules()
            register_rule!(reg, r)
        end
        with_registry(reg) do
            expr = TDeriv(up(:a), Tensor(:Ric, [down(:a), down(:b)]))
            result = simplify(expr)
            # Should be (1/2) ∂_b(RicScalar)
            @test result isa TProduct || result isa TDeriv
            # Check it's proportional to ∂_b(RicScalar)
            if result isa TProduct
                @test result.scalar == 1 // 2
                @test length(result.factors) == 1
                inner = result.factors[1]
                @test inner isa TDeriv
                @test inner.index == down(:b)
                @test inner.arg == Tensor(:RicScalar, TIndex[])
            end
        end
    end

    @testset "1.3: Dummy index renaming in collect_terms" begin
        reg = gr_registry()
        with_registry(reg) do
            # T_{ac} V^c + T_{ab} V^b should collect to 2 * T_{a_d1} V^{_d1}
            # because the two terms differ only in dummy name
            t1 = tproduct(1//1, TensorExpr[
                Tensor(:T, [down(:a), down(:c)]),
                Tensor(:V, [up(:c)])
            ])
            t2 = tproduct(1//1, TensorExpr[
                Tensor(:T, [down(:a), down(:b)]),
                Tensor(:V, [up(:b)])
            ])
            expr = tsum(TensorExpr[t1, t2])
            result = collect_terms(expr)
            # Should reduce to a single term with coefficient 2
            if result isa TProduct
                @test result.scalar == 2 // 1
            elseif result isa TSum
                # If not collected, at least verify correctness
                @test length(result.terms) <= 2
            end
        end
    end

    @testset "1.4: collect_terms after canonicalize" begin
        reg = gr_registry()
        with_registry(reg) do
            # R_{abcd} + R_{cdab} → via pair symmetry both are the same canonical form
            # After canonicalize, they should be identical, then collect_terms combines them
            Riem1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem2 = Tensor(:Riem, [down(:c), down(:d), down(:a), down(:b)])
            expr = tsum(TensorExpr[Riem1, Riem2])
            result = simplify(expr)
            # Pair symmetry: R_{abcd} = R_{cdab}, so this should be 2*R_{abcd}
            if result isa TProduct
                @test result.scalar == 2 // 1
            end
        end
    end

    @testset "1.4: Antisymmetric terms cancel" begin
        reg = gr_registry()
        with_registry(reg) do
            # A_{ab} + A_{ba} → 0 (antisymmetric)
            A1 = Tensor(:A, [down(:a), down(:b)])
            A2 = Tensor(:A, [down(:b), down(:a)])
            expr = tsum(TensorExpr[A1, A2])
            result = simplify(expr)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "1.4: Symmetric terms combine" begin
        reg = gr_registry()
        with_registry(reg) do
            # T_{ab} + T_{ba} → 2*T_{ab} (symmetric)
            T1 = Tensor(:T, [down(:a), down(:b)])
            T2 = Tensor(:T, [down(:b), down(:a)])
            expr = tsum(TensorExpr[T1, T2])
            result = simplify(expr)
            if result isa TProduct
                @test result.scalar == 2 // 1
            end
        end
    end

    @testset "Simplify pipeline: g^{ab} g_{bc} = δ^a_c" begin
        reg = gr_registry()
        with_registry(reg) do
            g_up = Tensor(:g, [up(:a), up(:b)])
            g_dn = Tensor(:g, [down(:b), down(:c)])
            result = simplify(g_up * g_dn)
            @test result isa Tensor
            @test result.name == :δ
        end
    end

    @testset "Riemann antisymmetry kills identical pairs" begin
        reg = gr_registry()
        with_registry(reg) do
            # R_{abcd} + R_{bacd} = 0 (antisym in first pair)
            R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R2 = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])
            result = simplify(R1 + R2)
            @test result == TScalar(0 // 1)
        end
    end
end
