#= Test differential invariant database (4 covariant derivatives, order 6).
#
# Verifies the catalog of scalar invariants built from covariant derivatives
# of curvature tensors at order 6.
#
# Case {4}: 4 derivatives + 1 curvature factor (box2_R, hessian_R_sq, hessian_Ric_sq)
# Case {0,2}: 2 curvature factors + 2 derivatives (R_box_R, Ric_box_Ric, Riem_box_Riem)
#
# Ground truth: Fulling, King, Wybourne & Cummins (1992);
#               Garcia-Parrado & Martin-Garcia (2007), Sec 6.
=#

@testset "Invar Differential Database (4 derivatives, order 6)" begin
    # ---- Shared registry setup ----
    function _diff4_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
        end
        reg
    end

    # ---- Catalog contains expected order-6 entries ----
    @testset "Catalog completeness (order 6)" begin
        expected_order6 = Set([:box2_R, :hessian_R_sq, :hessian_Ric_sq,
                               :R_box_R, :Ric_box_Ric, :Riem_box_Riem])
        actual = Set(keys(TensorGR._DIFF_INVAR_CATALOG))
        @test expected_order6 ⊆ actual
    end

    # ---- list_diff_invariants(order=6) ----
    @testset "list_diff_invariants(order=6)" begin
        order6 = list_diff_invariants(order=6)
        @test length(order6) == 6
        @test all(e -> e.order == 6, order6)

        # Case {4}: n_derivs=4, n_riemann=1
        case4 = filter(e -> e.n_derivs == 4, order6)
        @test length(case4) == 3
        @test all(e -> e.n_riemann == 1, case4)

        # Case {0,2}: n_derivs=2, n_riemann=2
        case02 = filter(e -> e.n_derivs == 2, order6)
        @test length(case02) == 3
        @test all(e -> e.n_riemann == 2, case02)

        # Exactly 1 total derivative at order 6 (box2_R)
        @test count(e -> e.is_total_derivative, order6) == 1
        @test any(e -> e.name == :box2_R && e.is_total_derivative, order6)
    end

    # ---- diff_invariant_count(6) ----
    @testset "diff_invariant_count(6)" begin
        @test diff_invariant_count(6) == 6
    end

    # ---- DiffInvariantEntry metadata for each order-6 entry ----
    @testset "Entry metadata" begin
        # Case {4} entries
        for name in [:box2_R, :hessian_R_sq, :hessian_Ric_sq]
            entry = TensorGR._DIFF_INVAR_CATALOG[name]
            @test entry isa DiffInvariantEntry
            @test entry.name == name
            @test entry.n_derivs == 4
            @test entry.n_riemann == 1
            @test entry.order == 6
        end

        # Case {0,2} entries
        for name in [:R_box_R, :Ric_box_Ric, :Riem_box_Riem]
            entry = TensorGR._DIFF_INVAR_CATALOG[name]
            @test entry isa DiffInvariantEntry
            @test entry.name == name
            @test entry.n_derivs == 2
            @test entry.n_riemann == 2
            @test entry.order == 6
        end
    end

    # ---- Total derivative flags ----
    @testset "Total derivative flags" begin
        @test TensorGR._DIFF_INVAR_CATALOG[:box2_R].is_total_derivative == true
        @test TensorGR._DIFF_INVAR_CATALOG[:hessian_R_sq].is_total_derivative == false
        @test TensorGR._DIFF_INVAR_CATALOG[:hessian_Ric_sq].is_total_derivative == false
        @test TensorGR._DIFF_INVAR_CATALOG[:R_box_R].is_total_derivative == false
        @test TensorGR._DIFF_INVAR_CATALOG[:Ric_box_Ric].is_total_derivative == false
        @test TensorGR._DIFF_INVAR_CATALOG[:Riem_box_Riem].is_total_derivative == false
    end

    # ---- Expression builders produce valid scalar TensorExpr (no free indices) ----
    @testset "Expression builders: scalar output" begin
        reg = _diff4_reg()
        with_registry(reg) do
            for name in [:box2_R, :hessian_R_sq, :hessian_Ric_sq,
                         :R_box_R, :Ric_box_Ric, :Riem_box_Riem]
                expr = diff_invariant(name; registry=reg, covd=:D)
                @test expr isa TensorExpr
                fi = free_indices(expr)
                @test isempty(fi)
            end
        end
    end

    # ---- Expression builders use correct covd tag ----
    @testset "Expression builders use correct covd tag" begin
        reg = _diff4_reg()
        with_registry(reg) do
            function _has_covd_tag_r(e::TDeriv, tag::Symbol)
                e.covd == tag || _has_covd_tag_r(e.arg, tag)
            end
            _has_covd_tag_r(::TensorExpr, ::Symbol) = false
            function _has_covd_tag_r(e::TProduct, tag::Symbol)
                any(f -> _has_covd_tag_r(f, tag), e.factors)
            end

            for name in [:box2_R, :hessian_R_sq, :hessian_Ric_sq,
                         :R_box_R, :Ric_box_Ric, :Riem_box_Riem]
                expr = diff_invariant(name; registry=reg, covd=:D)
                @test _has_covd_tag_r(expr, :D)
            end
        end
    end

    # ---- Derivative depth checks ----
    @testset "Derivative depth" begin
        reg = _diff4_reg()
        with_registry(reg) do
            function _count_derivs_r(e::TDeriv)
                1 + _count_derivs_r(e.arg)
            end
            _count_derivs_r(::TensorExpr) = 0
            function _max_deriv_depth_r(e::TProduct)
                maximum(_count_derivs_r(f) for f in e.factors; init=0)
            end
            _max_deriv_depth_r(e::TDeriv) = _count_derivs_r(e)
            _max_deriv_depth_r(::TensorExpr) = 0

            # Case {4}: 4 derivatives deep on some factor
            for name in [:box2_R]
                expr = diff_invariant(name; registry=reg, covd=:D)
                # box2_R = g^{ab} D_a D_b (g^{cd} D_c D_d R)
                # The outer D_a D_b acts on a TProduct, so max depth is 2
                # on each nested structure. The overall structure has derivatives
                # at multiple levels.
                @test _max_deriv_depth_r(expr) >= 2
            end

            # Case {4} hessian squared: each factor has depth 2
            for name in [:hessian_R_sq, :hessian_Ric_sq]
                expr = diff_invariant(name; registry=reg, covd=:D)
                @test _max_deriv_depth_r(expr) == 2
            end

            # Case {0,2}: 2 derivatives deep on boxed factor
            for name in [:R_box_R, :Ric_box_Ric, :Riem_box_Riem]
                expr = diff_invariant(name; registry=reg, covd=:D)
                @test _max_deriv_depth_r(expr) == 2
            end
        end
    end

    # ---- diff_invariant(:box2_R) builds valid expression ----
    @testset "diff_invariant(:box2_R) specific" begin
        reg = _diff4_reg()
        with_registry(reg) do
            expr = diff_invariant(:box2_R; registry=reg, covd=:D)
            @test expr isa TensorExpr
            @test isempty(free_indices(expr))
        end
    end

    # ---- Error on unknown name still works ----
    @testset "diff_invariant error on unknown name" begin
        reg = _diff4_reg()
        with_registry(reg) do
            @test_throws ErrorException diff_invariant(:nonexistent_order6; registry=reg, covd=:D)
        end
    end
end
