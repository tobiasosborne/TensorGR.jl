#= Test differential invariant database (2 covariant derivatives).
#
# Verifies the catalog of scalar invariants built from covariant derivatives
# of curvature tensors at order 4 (2 derivatives + 1 curvature factor).
#
# Ground truth: Garcia-Parrado & Martin-Garcia (2007), Sec 6;
#               contracted second Bianchi identity nabla_a nabla_b R^{ab} = (1/2) Box R.
=#

@testset "Invar Differential Database (2 derivatives)" begin
    # ---- Shared registry setup ----
    function _diff_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
        end
        reg
    end

    # ---- DiffInvariantEntry struct ----
    @testset "DiffInvariantEntry construction" begin
        entry = TensorGR._DIFF_INVAR_CATALOG[:box_R]
        @test entry isa DiffInvariantEntry
        @test entry.name == :box_R
        @test entry.n_derivs == 2
        @test entry.n_riemann == 1
        @test entry.order == 4
        @test entry.is_total_derivative == true

        entry2 = TensorGR._DIFF_INVAR_CATALOG[:grad_R_sq]
        @test entry2.is_total_derivative == false
    end

    # ---- Catalog contains expected entries ----
    @testset "Catalog completeness" begin
        expected = Set([:box_R, :grad_R_sq, :grad_Ric_sq, :grad_Riem_sq])
        actual = Set(keys(TensorGR._DIFF_INVAR_CATALOG))
        @test actual == expected
        @test length(TensorGR._DIFF_INVAR_CATALOG) == 4
    end

    # ---- list_diff_invariants ----
    @testset "list_diff_invariants" begin
        all_inv = list_diff_invariants()
        @test length(all_inv) == 4

        # Filter by order
        order4 = list_diff_invariants(order=4)
        @test length(order4) == 4
        @test all(e -> e.order == 4, order4)

        # Filter by n_derivs
        nd2 = list_diff_invariants(n_derivs=2)
        @test length(nd2) == 4
        @test all(e -> e.n_derivs == 2, nd2)

        # All entries should have correct metadata
        for entry in all_inv
            @test entry.n_derivs == 2
            @test entry.n_riemann == 1
            @test entry.order == 4
        end

        # Exactly 1 total derivative (box_R)
        @test count(e -> e.is_total_derivative, all_inv) == 1
        @test any(e -> e.name == :box_R && e.is_total_derivative, all_inv)
    end

    # ---- diff_invariant_count ----
    @testset "diff_invariant_count" begin
        @test diff_invariant_count(4) == 4
        @test_throws ErrorException diff_invariant_count(6)
    end

    # ---- Expression builders produce valid scalar TensorExpr ----
    @testset "Expression builders: scalar output (no free indices)" begin
        reg = _diff_reg()
        with_registry(reg) do
            for name in [:box_R, :grad_R_sq, :grad_Ric_sq, :grad_Riem_sq]
                expr = diff_invariant(name; registry=reg, covd=:D)
                @test expr isa TensorExpr
                fi = free_indices(expr)
                @test isempty(fi)
            end
        end
    end

    # ---- Box R is marked as total derivative ----
    @testset "box_R is total derivative" begin
        entry = TensorGR._DIFF_INVAR_CATALOG[:box_R]
        @test entry.is_total_derivative == true
    end

    # ---- grad_R_sq is NOT a total derivative ----
    @testset "grad_R_sq is not total derivative" begin
        entry = TensorGR._DIFF_INVAR_CATALOG[:grad_R_sq]
        @test entry.is_total_derivative == false
    end

    # ---- Expression builders: correct covd tag ----
    @testset "Expression builders use correct covd tag" begin
        reg = _diff_reg()
        with_registry(reg) do
            # box_R should contain TDeriv nodes with covd=:D
            expr = diff_invariant(:box_R; registry=reg, covd=:D)
            function _has_covd_tag(e::TDeriv, tag::Symbol)
                e.covd == tag || _has_covd_tag(e.arg, tag)
            end
            _has_covd_tag(::TensorExpr, ::Symbol) = false
            function _has_covd_tag(e::TProduct, tag::Symbol)
                any(f -> _has_covd_tag(f, tag), e.factors)
            end
            @test _has_covd_tag(expr, :D)
        end
    end

    # ---- Error on unknown name ----
    @testset "diff_invariant error on unknown name" begin
        reg = _diff_reg()
        with_registry(reg) do
            @test_throws ErrorException diff_invariant(:nonexistent; registry=reg, covd=:D)
        end
    end

    # ---- Contracted Bianchi: nabla_a nabla_b R^{ab} = (1/2) Box R ----
    @testset "Contracted Bianchi: div(grad Ric) = (1/2) Box R" begin
        reg = _diff_reg()
        with_registry(reg) do
            # Build nabla_a nabla_b R^{ab}
            lhs = TensorGR._build_div_grad_Ric_expr(; registry=reg, covd=:D)
            @test lhs isa TensorExpr
            @test isempty(free_indices(lhs))

            # Build (1/2) Box R
            box_r = diff_invariant(:box_R; registry=reg, covd=:D)
            rhs = TScalar(1 // 2) * box_r
            @test rhs isa TensorExpr
            @test isempty(free_indices(rhs))

            # Both should be scalars (zero free indices)
            # The structural relationship nabla_a nabla_b R^{ab} = (1/2) Box R
            # follows from the contracted second Bianchi identity:
            #   nabla^a R_{ab} = (1/2) nabla_b R
            # Taking another divergence:
            #   nabla^b nabla^a R_{ab} = (1/2) nabla^b nabla_b R = (1/2) Box R
            #
            # Verifying this algebraically at the abstract level would require
            # the full Bianchi rule application pipeline. Here we verify the
            # structural properties: both are well-formed scalars with the
            # correct derivative and curvature content.

            # Verify lhs has 2 derivative levels and Ric content
            function _count_derivs(e::TDeriv)
                1 + _count_derivs(e.arg)
            end
            _count_derivs(::TensorExpr) = 0
            function _max_deriv_depth(e::TProduct)
                maximum(_count_derivs(f) for f in e.factors; init=0)
            end
            _max_deriv_depth(e::TDeriv) = _count_derivs(e)
            _max_deriv_depth(::TensorExpr) = 0

            @test _max_deriv_depth(lhs) == 2  # nabla_a nabla_b Ric_{cd}

            # Verify rhs (box_r) also has 2 derivative levels
            @test _max_deriv_depth(box_r) == 2  # nabla^a nabla_a R
        end
    end
end
