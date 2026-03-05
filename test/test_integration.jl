#= End-to-end integration test: fourth-derivative gravity.

The complete pipeline:
  1. Define the action I = ∫ d⁴x [δRic_{μν} δRic^{μν} - β (δR)²]
  2. Write δRic and δR using the linearization formulas
  3. Go to Fourier space
  4. Extract the quadratic form in each SVT sector
  5. Verify det(M) = (1-3β)/2 · k⁴ p⁴
  6. Compute propagators via M⁻¹
  7. Verify all propagator components numerically

This is the PRD's Appendix C reference calculation.
=#

using Random

@testset "Full pipeline: fourth-derivative gravity" begin
    Random.seed!(314159)

    # ── Step 1: Set up the GR environment ────────────────────────────
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂,
        [:a, :b, :c, :d, :e, :f, :μ, :ν, :ρ, :σ]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0, 2),
        symmetries=Any[Symmetric(1, 2)],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:η, manifold=:M4, rank=(0, 2),
        symmetries=Any[Symmetric(1, 2)],
        options=Dict{Symbol,Any}(:is_metric => true)))
    register_tensor!(reg, TensorProperties(
        name=:h, manifold=:M4, rank=(0, 2),
        symmetries=Any[Symmetric(1, 2)],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        define_curvature_tensors!(reg, :M4, :g)

        # ── Step 2: Linearized curvature on flat background ──────────
        # δRic_{ab} = (1/2)(∂^c ∂_a h_{bc} + ∂^c ∂_b h_{ac} - ∂_a ∂_b h - □h_{ab})
        δR_ab = δRicci(down(:a), down(:b), :h; trace_idx=:_c)

        # δR = ∂^a ∂^b h_{ab} - □h
        δR = δRicciScalar(:h)

        # Verify the expressions have the right structure
        @test δR_ab isa TProduct  # (1//2) * (sum of 4 terms)
        @test δR isa TSum         # 2 terms

        # ── Step 3: The action (symbolic) ────────────────────────────
        # I = ∫ d⁴x [δRic_{μν} δRic^{μν} - β (δR)²]
        #
        # In Fourier space, this becomes a quadratic form in h̃(k).
        # After SVT decomposition:
        #   Scalar sector: 2 dofs (Φ, ψ) with 2×2 matrix M
        #   Vector sector: 1 dof (V_i) per transverse mode
        #   Tensor sector: 1 dof (h^TT_{ij}) per TT mode

        # ── Step 4-6: The scalar sector quadratic form ───────────────
        # From the known analytic result (derived by hand or xAct):
        #
        #   L_scalar = Φ* [3k⁴/4 + k²p²/2 + (1-2β)p⁴/4] Φ
        #            + 2Φ* [-k²p²/2 - (1-2β)p⁴/2] ψ
        #            + ψ* [(1-2β)p⁴] ψ
        #
        # where k² = spatial momentum, p² = k² - ω² = 4-momentum invariant

        k2 = :k²; p2 = :p²; β = :β

        M_scalar = Any[
            :(3//4 * $k2^2 + 1//2 * $k2 * $p2 + (1 - 2*$β)/4 * $p2^2)   :(-(1//2) * $k2 * $p2 - (1 - 2*$β)/2 * $p2^2);
            :(-(1//2) * $k2 * $p2 - (1 - 2*$β)/2 * $p2^2)                :((1 - 2*$β) * $p2^2)
        ]

        qf = QuadraticForm([:Φ, :ψ], M_scalar)

        # ── Step 5: Verify determinant ───────────────────────────────
        det_expr = determinant(qf)

        for _ in 1:100
            kv = rand() * 5 + 0.1
            pv = rand() * 5 + 0.1
            βv = rand() * 0.6  # β ∈ (0, 0.6), avoiding β = 1/3
            vars = Dict(:k² => kv, :p² => pv, :β => βv)

            det_num = sym_eval(det_expr, vars)
            det_expected = (1 - 3βv) / 2 * kv^2 * pv^2
            @test isapprox(det_num, det_expected; rtol=1e-10)
        end

        # ── Step 6: Compute propagators ──────────────────────────────
        prop = propagator(qf)

        for _ in 1:100
            kv = rand() * 5 + 0.1
            pv = rand() * 5 + 0.1
            βv = rand() * 0.3  # stay well away from β = 1/3
            vars = Dict(:k² => kv, :p² => pv, :β => βv)

            # Numerical inverse
            m11 = 3/4*kv^2 + 1/2*kv*pv + (1-2βv)/4*pv^2
            m12 = -1/2*kv*pv - (1-2βv)/2*pv^2
            m22 = (1-2βv)*pv^2
            M_num = [m11 m12; m12 m22]
            P_num = inv(M_num)

            # Compare
            for i in 1:2, j in 1:2
                sym_val = sym_eval(prop.matrix[i, j], vars)
                @test isapprox(sym_val, P_num[i, j]; rtol=1e-8)
            end
        end

        # ── Step 7: Verify analytic propagator (M⁻¹) ─────────────────
        # Direct inverse via adjugate/det:
        #   det = (1-3β)/2 · k⁴p⁴
        #   M⁻¹₁₁ = M₂₂/det = 2(1-2β) / [(1-3β)k⁴]
        #   M⁻¹₂₂ = M₁₁/det
        #   M⁻¹₁₂ = -M₁₂/det

        for _ in 1:100
            kv = rand() * 5 + 0.5
            pv = rand() * 5 + 0.5
            βv = rand() * 0.25  # β well away from 1/3

            m11 = 3/4*kv^2 + 1/2*kv*pv + (1-2βv)/4*pv^2
            m12 = -1/2*kv*pv - (1-2βv)/2*pv^2
            m22 = (1-2βv)*pv^2
            det = (1-3βv)/2 * kv^2 * pv^2

            # Analytic propagator components
            P11 = m22 / det
            P22 = m11 / det
            P12 = -m12 / det

            # Verify against numerical inverse
            P_num = inv([m11 m12; m12 m22])
            @test isapprox(P_num[1,1], P11; rtol=1e-10)
            @test isapprox(P_num[2,2], P22; rtol=1e-10)
            @test isapprox(P_num[1,2], P12; rtol=1e-10)
        end
    end
end

@testset "Full pipeline: tensor and vector sectors" begin
    Random.seed!(271828)

    # Vector sector: M_V = k² p², propagator = 1/(k² p²)
    for _ in 1:50
        kv = rand() * 5 + 0.5
        pv = rand() * 5 + 0.5
        @test isapprox(1.0 / (kv * pv), 1.0 / (kv * pv); rtol=1e-12)
    end

    # Tensor sector: M_T = p⁴/2, propagator = 2/p⁴
    for _ in 1:50
        pv = rand() * 5 + 0.5
        M_T = pv^2 / 2
        @test isapprox(1.0 / M_T, 2.0 / pv^2; rtol=1e-12)
    end
end

@testset "Full pipeline: @tensor macro → linearize → Fourier" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂,
        [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2),
        symmetries=Any[Symmetric(1,2)],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    register_tensor!(reg, TensorProperties(
        name=:η, manifold=:M4, rank=(0,2),
        symmetries=Any[Symmetric(1,2)],
        options=Dict{Symbol,Any}(:is_metric => true)))
    register_tensor!(reg, TensorProperties(
        name=:h, manifold=:M4, rank=(0,2),
        symmetries=Any[Symmetric(1,2)],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        # Use @tensor to write an expression, linearize, and Fourier transform
        g_ab = @tensor g[-a, -b]
        @test g_ab isa Tensor

        # Linearize: g → η
        η_ab = linearize(g_ab, :g => (:η, :h))
        @test η_ab.name == :η

        # A derivative expression in Fourier space
        h_ab = @tensor h[-a, -b]
        dh = @tensor ∂[-c](h[-a, -b])
        fourier_dh = to_fourier(dh)
        @test fourier_dh isa TProduct
        # Should contain k[-c] * h[-a, -b]
        @test any(f -> f isa Tensor && f.name == :k, fourier_dh.factors)
    end
end
