@testset "sort_covds_to_box" begin
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

    with_registry(reg) do
        phi = Tensor(:phi, TIndex[])
        V_up_a = Tensor(:V, [up(:a)])

        # ── Test 1: g^{ab} D_a D_b phi => contains box(phi) ──
        @testset "g^{ab} D_a D_b phi -> box" begin
            expr = Tensor(:g, [up(:a), up(:b)]) *
                   TDeriv(down(:a), TDeriv(down(:b), phi, :D), :D)
            result = sort_covds_to_box(expr, :D; registry=reg)
            # The result should still contain g^{cd} D_c(D_d(phi)) structure
            # (box form: metric * double derivative)
            @test result isa TProduct
            @test any(f -> f isa Tensor && f.name == :g, result.factors)
            # Should have a nested TDeriv
            has_double_deriv = any(result.factors) do f
                f isa TDeriv && f.arg isa TDeriv
            end
            @test has_double_deriv
        end

        # ── Test 2: double box g^{ab} D_a D_b (g^{cd} D_c D_d phi) ──
        @testset "double box" begin
            inner_box = Tensor(:g, [up(:c), up(:d)]) *
                        TDeriv(down(:c), TDeriv(down(:d), phi, :D), :D)
            expr = Tensor(:g, [up(:a), up(:b)]) *
                   TDeriv(down(:a), TDeriv(down(:b), inner_box, :D), :D)
            result = sort_covds_to_box(expr, :D; registry=reg)
            # Should be simplified; the inner box should also be detected
            @test result isa TensorExpr
            # Count metrics: box(box(phi)) has structure with 2 metrics
            metric_count = 0
            walk(result) do node
                if node isa Tensor && node.name == :g
                    metric_count += 1
                end
                node
            end
            @test metric_count >= 2
        end

        # ── Test 3: expression without box pattern passes through ──
        @testset "no box pattern unchanged" begin
            # D_a(D_b(phi)) with distinct indices is not a box pattern
            expr = TDeriv(down(:a), TDeriv(down(:b), phi, :D), :D)
            result = sort_covds_to_box(expr, :D; registry=reg)
            # Should contain the same derivative structure (no box)
            @test result isa TensorExpr
            # No metric should be introduced
            has_metric = false
            walk(result) do node
                if node isa Tensor && node.name == :g
                    has_metric = true
                end
                node
            end
            @test !has_metric
        end

        # ── Test 4: mixed expression with some box and non-box terms ──
        @testset "mixed expression" begin
            # Box term: g^{ab} D_a D_b phi
            box_term = Tensor(:g, [up(:a), up(:b)]) *
                       TDeriv(down(:a), TDeriv(down(:b), phi, :D), :D)
            # Non-box term: just the scalar field
            non_box_term = phi
            expr = box_term + non_box_term
            result = sort_covds_to_box(expr, :D; registry=reg)
            @test result isa TensorExpr
            # Result should be a sum
            if result isa TSum
                @test length(result.terms) >= 1
            end
        end

        # ── Test 5: backward-compat keyword-only form ──
        @testset "keyword-only backward compat" begin
            # This is the old API: sort_covds_to_box(expr; metric=:g)
            d = TDeriv(down(:b), TDeriv(up(:b), phi, :partial), :partial)
            result = sort_covds_to_box(d; metric=:g)
            @test result isa TProduct
            @test any(f -> f isa Tensor && f.name == :g, result.factors)
        end

        # ── Test 6: already-contracted form D^a(D_a(phi)) ──
        @testset "contracted form D^a D_a" begin
            expr = TDeriv(up(:a), TDeriv(down(:a), phi, :partial), :partial)
            result = sort_covds_to_box(expr; metric=:g)
            @test result isa TProduct
            @test any(f -> f isa Tensor && f.name == :g, result.factors)
        end

        # ── Test 7: grad_squared pattern ──
        @testset "grad_squared detection" begin
            # D_a(phi) D^a(phi) = g^{ab} D_a(phi) D_b(phi)
            expr = TDeriv(down(:a), phi) * TDeriv(up(:a), phi)
            result = sort_covds_to_box(expr; metric=:g)
            # Should introduce explicit metric g^{ab}
            @test result isa TProduct
            @test any(f -> f isa Tensor && f.name == :g, result.factors)
        end
    end
end
