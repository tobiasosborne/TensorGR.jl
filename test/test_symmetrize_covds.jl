@testset "symmetrize_covds" begin
    # Set up a standard 4D manifold with metric, curvature, and CovD
    function _setup_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d,
            [:a, :b, :c, :d, :e, :f, :p, :q, :r, :s]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(name=:d, manifold=:M, rank=(1, 1),
            options=Dict{Symbol,Any}(:is_delta => true)))
        define_curvature_tensors!(reg, :M, :g)
        define_covd!(reg, :D; manifold=:M, metric=:g)
        # Register a scalar field (no indices)
        register_tensor!(reg, TensorProperties(name=:phi, manifold=:M, rank=(0, 0)))
        # Register a covariant vector
        register_tensor!(reg, TensorProperties(name=:V, manifold=:M, rank=(1, 0)))
        # Register a covariant 1-form
        register_tensor!(reg, TensorProperties(name=:W, manifold=:M, rank=(0, 1)))
        # Register a rank-(1,1) tensor
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(1, 1)))
        reg
    end

    @testset "scalar: unchanged" begin
        reg = _setup_reg()
        phi = Tensor(:phi, TIndex[])
        # ∇_a(∇_b(φ)) -- scalar body, already symmetric
        expr = TDeriv(down(:a), TDeriv(down(:b), phi, :D), :D)
        result = symmetrize_covds(expr, :D; registry=reg)
        # Should be returned as-is (not decomposed into a TSum)
        @test result isa TDeriv
        @test result == expr
    end

    @testset "vector V^c: symmetrized + Riemann" begin
        reg = _setup_reg()
        V = Tensor(:V, [up(:c)])
        expr = TDeriv(down(:a), TDeriv(down(:b), V, :D), :D)
        result = with_registry(reg) do
            symmetrize_covds(expr, :D; registry=reg)
        end
        # Result should be a TSum with symmetrized part + commutator
        @test result isa TSum
        # Should have at least 2 terms: the sym part and the Riemann part
        @test length(result.terms) >= 2
    end

    @testset "1-form W_c: symmetrized + Riemann" begin
        reg = _setup_reg()
        W = Tensor(:W, [down(:c)])
        expr = TDeriv(down(:a), TDeriv(down(:b), W, :D), :D)
        result = with_registry(reg) do
            symmetrize_covds(expr, :D; registry=reg)
        end
        @test result isa TSum
        @test length(result.terms) >= 2
    end

    @testset "rank-(1,1) tensor T^c_d: two Riemann terms" begin
        reg = _setup_reg()
        T = Tensor(:T, [up(:c), down(:d)])
        expr = TDeriv(down(:a), TDeriv(down(:b), T, :D), :D)
        result = with_registry(reg) do
            symmetrize_covds(expr, :D; registry=reg)
        end
        @test result isa TSum
        # Symmetrized part + commutator (which itself has two Riemann terms)
        @test length(result.terms) >= 2
        # Verify two Riemann tensors appear (one per index on T)
        riem_count = Ref(0)
        walk(result) do node
            if node isa Tensor && node.name == :Riem
                riem_count[] += 1
            end
            node
        end
        @test riem_count[] == 2
    end

    @testset "no double derivatives: passthrough" begin
        reg = _setup_reg()
        V = Tensor(:V, [up(:c)])
        # Single derivative -- no double deriv to symmetrize
        expr = TDeriv(down(:a), V, :D)
        result = symmetrize_covds(expr, :D; registry=reg)
        @test result == expr

        # Bare tensor -- no derivatives at all
        result2 = symmetrize_covds(V, :D; registry=reg)
        @test result2 == V

        # Scalar value
        s = TScalar(42 // 1)
        result3 = symmetrize_covds(s, :D; registry=reg)
        @test result3 == s
    end

    @testset "sum and product passthrough" begin
        reg = _setup_reg()
        V = Tensor(:V, [up(:c)])
        W = Tensor(:W, [down(:c)])

        # TSum: symmetrize_covds distributes over sums
        term1 = TDeriv(down(:a), V, :D)
        term2 = TDeriv(down(:a), W, :D)
        s = TSum([term1, term2])
        result = symmetrize_covds(s, :D; registry=reg)
        @test result isa TSum

        # TProduct: symmetrize_covds distributes into factors
        p = TProduct(1 // 1, TensorExpr[TDeriv(down(:a), TDeriv(down(:b), V, :D), :D)])
        result_p = with_registry(reg) do
            symmetrize_covds(p, :D; registry=reg)
        end
        # The double-deriv factor should have been expanded
        @test result_p isa TProduct || result_p isa TSum
    end

    @testset "symmetrized part structure" begin
        reg = _setup_reg()
        V = Tensor(:V, [up(:c)])
        expr = TDeriv(down(:a), TDeriv(down(:b), V, :D), :D)
        result = with_registry(reg) do
            symmetrize_covds(expr, :D; registry=reg)
        end
        # The symmetrized part should contain both ∇_a∇_b V and ∇_b∇_a V
        # Check that we can find derivative structures with both orderings
        has_ab = false
        has_ba = false
        walk(result) do node
            if node isa TDeriv && node.arg isa TDeriv
                outer = node.index.name
                inner = node.arg.index.name
                if outer == :a && inner == :b
                    has_ab = true
                elseif outer == :b && inner == :a
                    has_ba = true
                end
            end
            node
        end
        @test has_ab
        @test has_ba
    end

    @testset "commutator produces Riemann" begin
        reg = _setup_reg()
        V = Tensor(:V, [up(:c)])
        expr = TDeriv(down(:a), TDeriv(down(:b), V, :D), :D)
        result = with_registry(reg) do
            symmetrize_covds(expr, :D; registry=reg)
        end
        # The commutator term should contain a Riemann tensor
        has_riem = false
        walk(result) do node
            if node isa Tensor && node.name == :Riem
                has_riem = true
            end
            node
        end
        @test has_riem
    end

    @testset "mixed covd: still symmetrizes" begin
        reg = _setup_reg()
        V = Tensor(:V, [up(:c)])
        # Inner derivative uses :partial, outer uses :D -- still symmetrized
        # (consistent with commute_covds which does not filter on covd tag)
        expr = TDeriv(down(:a), TDeriv(down(:b), V, :partial), :D)
        result = with_registry(reg) do
            symmetrize_covds(expr, :D; registry=reg)
        end
        # Should produce a TSum (symmetrized + commutator)
        @test result isa TSum
    end
end
