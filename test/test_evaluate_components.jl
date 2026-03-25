@testset "evaluate_components" begin
    using TensorGR

    # ── Setup: 2D Euclidean ──────────────────────────────────────────────
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M dim=2 metric=g
        chart = define_chart!(reg, :cart; manifold=:M, coords=[:x, :y])

        # Flat metric: g_{ab} = diag(1, 1)
        vals = Dict{Any,Any}(
            (:g, [1, 1]) => 1.0, (:g, [1, 2]) => 0.0,
            (:g, [2, 1]) => 0.0, (:g, [2, 2]) => 1.0,
            # Inverse metric (same for flat Euclidean)
            (:g_inv, [1, 1]) => 1.0, (:g_inv, [1, 2]) => 0.0,
            (:g_inv, [2, 1]) => 0.0, (:g_inv, [2, 2]) => 1.0,
            # Test tensor T_{ab}
            (:T, [1, 1]) => 3.0, (:T, [1, 2]) => 1.0,
            (:T, [2, 1]) => 1.0, (:T, [2, 2]) => 5.0,
            # Vector V^a
            (:V, [1]) => 2.0, (:V, [2]) => 7.0,
            # Kronecker delta
            (:δ, [1, 1]) => 1, (:δ, [1, 2]) => 0,
            (:δ, [2, 1]) => 0, (:δ, [2, 2]) => 1,
        )

        # ── Scalar expression (rank 0, no indices) ──────────────────
        @testset "scalar" begin
            expr = TScalar(42)
            ct = evaluate_components(expr, chart, vals; registry=reg)
            @test ct.data[] == 42
            @test ndims(ct) == 0
        end

        # ── Rank-1 tensor (no contraction) ──────────────────────────
        @testset "rank-1 no contraction" begin
            V_expr = Tensor(:V, [up(:a)])
            ct = evaluate_components(V_expr, chart, vals; registry=reg)
            @test size(ct.data) == (2,)
            @test ct.data[1] == 2.0
            @test ct.data[2] == 7.0
        end

        # ── Rank-2 tensor (no contraction) ──────────────────────────
        @testset "rank-2 no contraction" begin
            T_expr = Tensor(:T, [down(:a), down(:b)])
            ct = evaluate_components(T_expr, chart, vals; registry=reg)
            @test size(ct.data) == (2, 2)
            @test ct.data[1, 1] == 3.0
            @test ct.data[1, 2] == 1.0
            @test ct.data[2, 2] == 5.0
        end

        # ── Single contraction: trace g^{ab} T_{ab} ────────────────
        @testset "trace g^ab T_ab" begin
            # For Euclidean metric, g^{ab} T_{ab} = T_{11} + T_{22} = 3 + 5 = 8
            expr = Tensor(:g, [up(:a), up(:b)]) * Tensor(:T, [down(:a), down(:b)])
            ct = evaluate_components(expr, chart, vals; registry=reg)
            @test ct.data[] ≈ 8.0
        end

        # ── Contraction with free index: g^{ac} T_{cb} ─────────────
        @testset "contraction with free index" begin
            # g^{ac} T_{cb} = T^a_b (index raising for flat metric)
            expr = Tensor(:g, [up(:a), up(:c)]) * Tensor(:T, [down(:c), down(:b)])
            ct = evaluate_components(expr, chart, vals; registry=reg)
            @test size(ct.data) == (2, 2)
            # For Euclidean: T^a_b = T_{ab}
            @test ct.data[1, 1] ≈ 3.0
            @test ct.data[1, 2] ≈ 1.0
            @test ct.data[2, 1] ≈ 1.0
            @test ct.data[2, 2] ≈ 5.0
        end

        # ── Double contraction: δ^a_b T^b_a (trace via delta) ──────
        @testset "delta trace" begin
            expr = Tensor(:δ, [up(:a), down(:b)]) * Tensor(:T, [down(:a), up(:b)])
            # Need T with mixed indices; for flat metric T_{ab} = T^{ab}
            # Store T with up indices too
            vals[(:T, [1, 1])] = 3.0  # already there
            ct = evaluate_components(expr, chart, vals; registry=reg)
            @test ct.data[] ≈ 8.0  # trace = 3 + 5
        end

        # ── TSum evaluation ─────────────────────────────────────────
        @testset "TSum" begin
            T1 = Tensor(:T, [down(:a), down(:b)])
            T2 = tproduct(2 // 1, TensorExpr[Tensor(:g, [down(:a), down(:b)])])
            expr = T1 + T2
            ct = evaluate_components(expr, chart, vals; registry=reg)
            @test ct.data[1, 1] ≈ 5.0   # 3 + 2*1
            @test ct.data[1, 2] ≈ 1.0   # 1 + 2*0
            @test ct.data[2, 2] ≈ 7.0   # 5 + 2*1
        end

        # ── TProduct with scalar coefficient ────────────────────────
        @testset "scalar coefficient" begin
            expr = tproduct(3 // 1, TensorExpr[Tensor(:V, [up(:a)])])
            ct = evaluate_components(expr, chart, vals; registry=reg)
            @test ct.data[1] ≈ 6.0   # 3 * 2
            @test ct.data[2] ≈ 21.0  # 3 * 7
        end

        # ── simplify_fn callback ────────────────────────────────────
        @testset "simplify_fn" begin
            expr = Tensor(:V, [up(:a)])
            ct = evaluate_components(expr, chart, vals; registry=reg,
                                     simplify_fn=x -> round(x; digits=0))
            @test ct.data[1] == 2.0
            @test ct.data[2] == 7.0
        end

        # ── Error on unevaluated covariant derivative ───────────────
        @testset "error on covd" begin
            covd_expr = TDeriv(down(:a), Tensor(:V, [up(:b)]), :D)
            @test_throws ErrorException evaluate_components(covd_expr, chart, vals; registry=reg)
        end

        # ── TSum with per-term internal dummies ─────────────────────
        @testset "TSum per-term dummies" begin
            # T_{ab} + g^{cd} * g_{ca} * g_{db}  (second term has internal dummies c,d)
            # For Euclidean: g^{cd} g_{ca} g_{db} = δ_a^d g_{db} = g_{ab}
            # So result = T_{ab} + g_{ab}
            T_term = Tensor(:T, [down(:a), down(:b)])
            g_inv = Tensor(:g, [up(:c), up(:d)])
            g1 = Tensor(:g, [down(:c), down(:a)])
            g2 = Tensor(:g, [down(:d), down(:b)])
            contracted = g_inv * g1 * g2
            expr = TSum(TensorExpr[T_term, contracted])
            ct = evaluate_components(expr, chart, vals; registry=reg)
            @test size(ct.data) == (2, 2)
            @test ct.data[1, 1] ≈ 4.0   # T_{11} + g_{11} = 3 + 1
            @test ct.data[2, 2] ≈ 6.0   # T_{22} + g_{22} = 5 + 1
            @test ct.data[1, 2] ≈ 1.0   # T_{12} + g_{12} = 1 + 0
        end
    end

    # ── Setup: 2D Minkowski ─────────────────────────────────────────────
    @testset "Minkowski 2D" begin
        reg2 = TensorRegistry()
        with_registry(reg2) do
            @manifold M2 dim=2 metric=g
            chart2 = define_chart!(reg2, :mink; manifold=:M2, coords=[:t, :x])

            # Minkowski: g_{ab} = diag(-1, 1)
            vals2 = Dict{Any,Any}(
                (:g, [1, 1]) => -1.0, (:g, [1, 2]) => 0.0,
                (:g, [2, 1]) => 0.0,  (:g, [2, 2]) => 1.0,
                (:V, [1]) => 3.0, (:V, [2]) => 4.0,
            )

            # V^a V_a with Minkowski = -V^0 V_0 + V^1 V_1
            # But this uses abstract contraction — need metric explicitly
            # g_{ab} V^a V^b = -9 + 16 = 7
            expr = Tensor(:g, [down(:a), down(:b)]) *
                   Tensor(:V, [up(:a)]) * Tensor(:V, [up(:b)])
            ct = evaluate_components(expr, chart2, vals2; registry=reg2)
            @test ct.data[] ≈ 7.0
        end
    end

    # ── prepare_values helper ───────────────────────────────────────────
    @testset "prepare_values" begin
        reg3 = TensorRegistry()
        with_registry(reg3) do
            @manifold M dim=2 metric=g
            chart3 = define_chart!(reg3, :c; manifold=:M, coords=[:x, :y])

            g_mat = [1.0 0.0; 0.0 1.0]
            pv = prepare_values(chart3, g_mat; registry=reg3)

            # Check metric stored
            @test pv[(:g, [1, 1])] == 1.0
            @test pv[(:g, [1, 2])] == 0.0

            # Check inverse stored
            @test pv[(:g_inv, [1, 1])] == 1.0

            # Check delta stored
            @test pv[(:δ, [1, 1])] == 1
            @test pv[(:δ, [1, 2])] == 0
        end
    end

    # ── Pre-computed derivative values ──────────────────────────────────
    @testset "pre-computed derivatives" begin
        reg4 = TensorRegistry()
        with_registry(reg4) do
            @manifold M dim=2 metric=g
            register_tensor!(reg4, TensorProperties(
                name=:F, manifold=:M, rank=(0, 1),
                symmetries=SymmetrySpec[]))
            chart4 = define_chart!(reg4, :c; manifold=:M, coords=[:x, :y])

            # F_a and ∂_b F_a
            vals4 = Dict{Any,Any}(
                (:F, [1]) => 10.0, (:F, [2]) => 20.0,
                # ∂_1 F_1 = 1, ∂_1 F_2 = 2, ∂_2 F_1 = 3, ∂_2 F_2 = 4
                (Symbol("∂F"), [1, 1]) => 1.0,
                (Symbol("∂F"), [1, 2]) => 2.0,
                (Symbol("∂F"), [2, 1]) => 3.0,
                (Symbol("∂F"), [2, 2]) => 4.0,
            )

            # ∂_b F_a  is a rank-2 tensor
            # indices(TDeriv) returns [deriv_idx, arg_indices...] = [b, a]
            # so output axis 1 = b (deriv), axis 2 = a (tensor): ct.data[b, a]
            expr = TDeriv(down(:b), Tensor(:F, [down(:a)]), :partial)
            ct = evaluate_components(expr, chart4, vals4; registry=reg4)
            @test size(ct.data) == (2, 2)
            @test ct.data[1, 1] ≈ 1.0   # ∂_1 F_1
            @test ct.data[1, 2] ≈ 2.0   # ∂_1 F_2
            @test ct.data[2, 1] ≈ 3.0   # ∂_2 F_1
            @test ct.data[2, 2] ≈ 4.0   # ∂_2 F_2
        end
    end

    # ── deriv_fn callback ───────────────────────────────────────────────
    @testset "deriv_fn callback" begin
        reg5 = TensorRegistry()
        with_registry(reg5) do
            @manifold M dim=2 metric=g
            register_tensor!(reg5, TensorProperties(
                name=:phi, manifold=:M, rank=(0, 0),
                symmetries=SymmetrySpec[]))
            chart5 = define_chart!(reg5, :c; manifold=:M, coords=[:x, :y])

            # Scalar field phi(x,y) = x^2 + y^2, ∂_x phi = 2x, ∂_y phi = 2y
            # For a constant evaluation, use symbolic values
            # We'll use a simple polynomial model
            vals5 = Dict{Any,Any}(
                (:phi, Int[]) => 5.0,  # phi at some point
            )

            # A trivial deriv_fn that returns 0 for constants
            trivial_deriv = (val, coord) -> 0.0

            expr = TDeriv(down(:a), TScalar(:phi), :partial)
            # This won't work for TScalar — need Tensor(:phi, [])
            # Use direct scalar derivative
            expr2 = TDeriv(down(:a), Tensor(:phi, TIndex[]), :partial)
            ct = evaluate_components(expr2, chart5, vals5; registry=reg5,
                                     deriv_fn=trivial_deriv)
            @test size(ct.data) == (2,)
            @test ct.data[1] ≈ 0.0
            @test ct.data[2] ≈ 0.0
        end
    end

    # ── Product of three tensors with multiple contractions ─────────────
    @testset "triple product" begin
        reg6 = TensorRegistry()
        with_registry(reg6) do
            @manifold M dim=2 metric=g
            register_tensor!(reg6, TensorProperties(
                name=:A, manifold=:M, rank=(0, 2),
                symmetries=SymmetrySpec[]))
            register_tensor!(reg6, TensorProperties(
                name=:B, manifold=:M, rank=(0, 2),
                symmetries=SymmetrySpec[]))
            chart6 = define_chart!(reg6, :c; manifold=:M, coords=[:x, :y])

            vals6 = Dict{Any,Any}(
                (:g, [1, 1]) => 1.0, (:g, [1, 2]) => 0.0,
                (:g, [2, 1]) => 0.0, (:g, [2, 2]) => 1.0,
                (:A, [1, 1]) => 1.0, (:A, [1, 2]) => 2.0,
                (:A, [2, 1]) => 3.0, (:A, [2, 2]) => 4.0,
                (:B, [1, 1]) => 5.0, (:B, [1, 2]) => 6.0,
                (:B, [2, 1]) => 7.0, (:B, [2, 2]) => 8.0,
            )

            # g^{ac} A_{cd} B^{db} = A_{cd} B^{db} (Euclidean)
            # = sum_c A_{cd} B^{db} = matrix product AB
            # [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19, 22; 43, 50]
            expr = Tensor(:g, [up(:a), up(:c)]) *
                   Tensor(:A, [down(:c), down(:d)]) *
                   Tensor(:B, [up(:d), up(:b)])
            ct = evaluate_components(expr, chart6, vals6; registry=reg6)
            @test size(ct.data) == (2, 2)
            @test ct.data[1, 1] ≈ 19.0
            @test ct.data[1, 2] ≈ 22.0
            @test ct.data[2, 1] ≈ 43.0
            @test ct.data[2, 2] ≈ 50.0
        end
    end
end
