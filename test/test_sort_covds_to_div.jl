@testset "sort_covds_to_div" begin
    # ── Setup: 4D manifold with metric g, covariant derivative D ──
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :dM4, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:g, manifold=:M4, rank=(0,2),
        symmetries=SymmetrySpec[Symmetric(1,2)],
        options=Dict{Symbol,Any}(:is_metric => true)))
    register_tensor!(reg, TensorProperties(name=:dM4, manifold=:M4, rank=(1,1),
        options=Dict{Symbol,Any}(:is_delta => true)))
    define_curvature_tensors!(reg, :M4, :g)
    define_covd!(reg, :D; manifold=:M4, metric=:g)

    # Register a scalar field and a vector field for testing
    register_tensor!(reg, TensorProperties(name=:phi, manifold=:M4, rank=(0,0)))
    register_tensor!(reg, TensorProperties(name=:V, manifold=:M4, rank=(1,0)))
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,2)))

    with_registry(reg) do
        phi = Tensor(:phi, TIndex[])
        V_a = Tensor(:V, [up(:a)])

        # ── Test 1: D_a(V^a) is already a divergence — passes through ──
        @testset "D_a(V^a) already divergence" begin
            expr = TDeriv(down(:a), V_a, :D)
            result = sort_covds_to_div(expr, :D; registry=reg)
            # The result should still contain D_a(V^a) structure
            @test result isa TensorExpr
            # Should have a TDeriv acting on V with matching contracted index
            has_div = false
            walk(result) do node
                if node isa TDeriv && node.index.position == Down
                    for idx in indices(node.arg)
                        if idx.name == node.index.name && idx.position == Up
                            has_div = true
                        end
                    end
                end
                node
            end
            @test has_div
        end

        # ── Test 2: D_a(phi * V^a) — Leibniz gives phi * D_a(V^a) + V^a * D_a(phi) ──
        @testset "D_a(phi * V^a) Leibniz expansion" begin
            product = phi * V_a
            expr = TDeriv(down(:a), product, :D)
            result = sort_covds_to_div(expr, :D; registry=reg)
            @test result isa TensorExpr
            # After Leibniz expansion, we should get a sum (or simplified form)
            # with at least one divergence term phi * D_a(V^a)
            has_div = false
            walk(result) do node
                if node isa TDeriv && node.index.position == Down
                    for idx in indices(node.arg)
                        if idx.name == node.index.name && idx.position == Up
                            has_div = true
                        end
                    end
                end
                node
            end
            @test has_div
        end

        # ── Test 3: expression without divergence pattern passes through unchanged ──
        @testset "no divergence pattern unchanged" begin
            # D_a(D_b(phi)) — no Up index matching :a or :b in the argument
            expr = TDeriv(down(:a), TDeriv(down(:b), phi, :D), :D)
            result = sort_covds_to_div(expr, :D; registry=reg)
            @test result isa TensorExpr
            # Should still have the double derivative structure
            has_double_deriv = false
            walk(result) do node
                if node isa TDeriv && node.arg isa TDeriv
                    has_double_deriv = true
                end
                node
            end
            @test has_double_deriv
        end

        # ── Test 4: scalar field passes through without crash ──
        @testset "scalar field passthrough" begin
            result = sort_covds_to_div(phi, :D; registry=reg)
            @test result isa TensorExpr
        end

        # ── Test 5: pure tensor without derivatives passes through ──
        @testset "pure tensor passthrough" begin
            result = sort_covds_to_div(V_a, :D; registry=reg)
            @test result isa TensorExpr
        end

        # ── Test 6: TScalar passes through ──
        @testset "TScalar passthrough" begin
            s = TScalar(42 // 1)
            result = sort_covds_to_div(s, :D; registry=reg)
            @test result isa TensorExpr
        end

        # ── Test 7: backward-compat keyword-only form ──
        @testset "keyword-only backward compat" begin
            # The old API: sort_covds_to_div(expr)
            T_up = Tensor(:T, [up(:a)])
            d = TDeriv(down(:a), T_up, :partial)
            result = sort_covds_to_div(d)
            # Already a divergence; should return unchanged
            @test result == d
        end

        # ── Test 8: sum with mixed divergence/non-divergence terms ──
        @testset "sum with mixed terms" begin
            # D_a(V^a) + D_b(phi)  — first is divergence, second is not
            div_term = TDeriv(down(:a), V_a, :D)
            non_div_term = TDeriv(down(:b), phi, :D)
            expr = div_term + non_div_term
            result = sort_covds_to_div(expr, :D; registry=reg)
            @test result isa TensorExpr
        end

        # ── Test 9: verify partial-derivative form works ──
        @testset "partial derivative divergence" begin
            T_up = Tensor(:V, [up(:a)])
            d = TDeriv(down(:a), T_up, :partial)
            result = sort_covds_to_div(d; metric=:g)
            @test result == d  # already divergence form
        end

        # ── Test 10: product of scalar and vector — Leibniz expansion via keyword form ──
        @testset "keyword form Leibniz expansion" begin
            product = phi * V_a
            expr = TDeriv(down(:a), product, :partial)
            result = sort_covds_to_div(expr; metric=:g)
            # After Leibniz expansion, result should be a sum
            @test result isa TSum || result isa TProduct || result isa TDeriv
            # At least one term should have divergence structure
            has_div = false
            walk(result) do node
                if node isa TDeriv && node.index.position == Down
                    for idx in indices(node.arg)
                        if idx.name == node.index.name && idx.position == Up
                            has_div = true
                        end
                    end
                end
                node
            end
            @test has_div
        end
    end
end
