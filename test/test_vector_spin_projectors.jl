@testset "Vector field spin projectors" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g

        μ = down(:μ)
        ν = down(:ν)

        # Test 1: Completeness P^(1) + P^(0) = g  (exact symbolic)
        P1 = vector_spin1_projector(μ, ν)
        P0 = vector_spin0_projector(μ, ν)
        g_mn = Tensor(:g, [μ, ν])
        completeness = simplify(P1 + P0 - g_mn; registry=reg)
        @test completeness == TScalar(0//1)

        # Test 2: Trace P^(1)_{μ}^{μ} = d-1 = 3  (transverse DOF count)
        P1_trace_expr = vector_spin1_projector(down(:α), down(:β))
        tr_P1 = Tensor(:g, [up(:α), up(:β)]) * P1_trace_expr
        tr_P1_simplified = simplify(tr_P1; registry=reg)
        tr_P1_simplified = contract_momenta(tr_P1_simplified)
        val_tr1 = _eval_spin_scalar(tr_P1_simplified, 1.0)
        @test isapprox(val_tr1, 3.0; atol=1e-12)  # d-1 = 3

        # Test 3: Trace P^(0)_{μ}^{μ} = 1  (longitudinal DOF count)
        P0_trace_expr = vector_spin0_projector(down(:α), down(:β))
        tr_P0 = Tensor(:g, [up(:α), up(:β)]) * P0_trace_expr
        tr_P0_simplified = simplify(tr_P0; registry=reg)
        tr_P0_simplified = contract_momenta(tr_P0_simplified)
        val_tr0 = _eval_spin_scalar(tr_P0_simplified, 1.0)
        @test isapprox(val_tr0, 1.0; atol=1e-12)  # 1 DOF

        # Test 4: Idempotency via trace: Tr(P1 * P1) = Tr(P1) = 3
        # P^(1)_{μρ} g^{ρσ} P^(1)_{σν} g^{μν} = 3
        P1_mr = vector_spin1_projector(down(:α), down(:ρ))
        g_rs = Tensor(:g, [up(:ρ), up(:σ)])
        P1_sn = vector_spin1_projector(down(:σ), down(:β))
        g_ab = Tensor(:g, [up(:α), up(:β)])
        tr_P1P1 = P1_mr * g_rs * P1_sn * g_ab
        tr_P1P1_s = simplify(tr_P1P1; registry=reg)
        tr_P1P1_s = contract_momenta(tr_P1P1_s)
        val_P1P1 = _eval_spin_scalar(tr_P1P1_s, 1.0)
        @test isapprox(val_P1P1, 3.0; atol=1e-12)  # idempotent: Tr(P^2) = Tr(P)

        # Test 5: Idempotency via trace: Tr(P0 * P0) = Tr(P0) = 1
        P0_mr = vector_spin0_projector(down(:α), down(:ρ))
        P0_sn = vector_spin0_projector(down(:σ), down(:β))
        tr_P0P0 = P0_mr * g_rs * P0_sn * g_ab
        tr_P0P0_s = simplify(tr_P0P0; registry=reg)
        tr_P0P0_s = contract_momenta(tr_P0P0_s)
        val_P0P0 = _eval_spin_scalar(tr_P0P0_s, 1.0)
        @test isapprox(val_P0P0, 1.0; atol=1e-12)  # idempotent: Tr(P^2) = Tr(P)

        # Test 6: Orthogonality via trace: Tr(P1 * P0) = 0
        P0_sn2 = vector_spin0_projector(down(:σ), down(:β))
        tr_cross = P1_mr * g_rs * P0_sn2 * g_ab
        tr_cross_s = simplify(tr_cross; registry=reg)
        tr_cross_s = contract_momenta(tr_cross_s)
        val_cross = _eval_spin_scalar(tr_cross_s, 1.0)
        @test isapprox(val_cross, 0.0; atol=1e-12)  # orthogonal

        # Test 7: Wrapper delegates correctly
        # vector_spin1_projector wraps theta_projector
        @test vector_spin1_projector(μ, ν) == theta_projector(μ, ν)
        # vector_spin0_projector wraps omega_projector
        @test vector_spin0_projector(μ, ν) == omega_projector(μ, ν)

        # Test 8: Proca theory motivation (physics comment, no computation)
        # L_Proca = -1/4 F_{μν}F^{μν} + 1/2 m² A_μ A^μ
        # Kinetic kernel in momentum space: K_{μν} = (k² η_{μν} - k_μ k_ν) + m² η_{μν}
        # = k² θ_{μν} + m² η_{μν}
        # spin-1 projection: Tr(K P^(1)) = k² (d-1) + m² (d-1) = (d-1)(k² + m²)
        # spin-0 projection: Tr(K P^(0)) = m²
        # In d=4: spin-1 -> 3(k² + m²), spin-0 -> m²
        # For massless (m=0): spin-1 -> 3k², spin-0 -> 0 (gauge invariance!)
    end
end
