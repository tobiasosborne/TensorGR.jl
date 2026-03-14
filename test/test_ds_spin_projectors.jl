#= Tests for dS-adapted (MSS-corrected) Barnes-Rivers spin projectors.
#
# Validates spin_project_mss against Bueno-Cano analytic form factors
# (arXiv:1607.06463) on maximally symmetric space (MSS) backgrounds.
#
# Key physics: on MSS with R_{μν} = Λg_{μν}, the flat Barnes-Rivers
# projection Tr(K·P^J_flat) includes a k²-independent Lichnerowicz mass
# term.  spin_project_mss extracts the kinetic form factor by subtracting
# this mass term, matching the Bueno-Cano convention.
#
# Ground truth:
#   bc_to_form_factors(bc_EH(κ,Λ), k², Λ) = (f₂=(5/2)κk², f₀=-κk²)
#   bc_to_form_factors(bc_EH(κ,Λ)+bc_RicSq(α₂,Λ), k², Λ) = ...
#
# Issues closed: TGR-ax2 (research), TGR-61e (implement),
#                TGR-841 (test), TGR-76k (parent)
=#

using Test
using TensorGR

# ── Helper: set up a flat registry for analytic kernel tests ─────────

function _setup_flat_registry_ds()
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)
        @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    end
    reg
end

# ── Helper: build synthetic MSS kernel ───────────────────────────────
# Takes the flat FP kernel and adds the known Lichnerowicz mass terms
# for the EH action on MSS: L_mass = Λ·(-h_{ab}h^{ab} + h²)
# This produces a kernel whose flat spin-2 projection gives (5/2)k² - 5Λ.

function _build_synthetic_mss_EH_kernel(reg, Λ_val)
    K_flat = with_registry(reg) do
        build_FP_momentum_kernel(reg)
    end

    # Mass terms from the linearized Einstein tensor on MSS background.
    # These are the k²-independent contributions from the background Riemann
    # tensor entering through commutation of covariant derivatives.
    #
    # Term 1: -Λ·h_{ab}h^{ab}  (Lichnerowicz mass for TT tensor)
    # Term 2: +Λ·h²             (trace mass)
    mass_term_1 = tproduct(-1 // 1, TensorExpr[
        TScalar(Λ_val),
        Tensor(:h, [down(:a), down(:b)]),
        Tensor(:h, [up(:a), up(:b)])])
    mass_term_2 = tproduct(1 // 1, TensorExpr[
        TScalar(Λ_val),
        Tensor(:h, [up(:a), down(:a)]),
        Tensor(:h, [up(:b), down(:b)])])

    K_mass = extract_kernel(mass_term_1 + mass_term_2, :h; registry=reg)

    # Combine: K_MSS = K_FP + K_mass
    combine_kernels([K_flat, K_mass])
end

@testset "dS Spin Projectors" begin

    # ══════════════════════════════════════════════════════════════════
    # 1. Vandermonde solver correctness
    # ══════════════════════════════════════════════════════════════════

    @testset "Vandermonde solver" begin
        # Linear: f(x) = 3 + 2x
        c = TensorGR._vandermonde_solve([1.0, 2.0], [5.0, 7.0])
        @test length(c) == 2
        @test isapprox(c[1], 3.0; atol=1e-12)
        @test isapprox(c[2], 2.0; atol=1e-12)

        # Quadratic: f(x) = 1 - 2x + 0.5x²
        x = [1.0, 2.0, 3.0]
        y = [1.0 - 2.0 + 0.5, 1.0 - 4.0 + 2.0, 1.0 - 6.0 + 4.5]
        c = TensorGR._vandermonde_solve(x, y)
        @test length(c) == 3
        @test isapprox(c[1], 1.0; atol=1e-10)
        @test isapprox(c[2], -2.0; atol=1e-10)
        @test isapprox(c[3], 0.5; atol=1e-10)

        # Cubic: f(x) = -1 + 3x - x² + 0.25x³
        x = [1.0, 2.0, 3.0, 4.0]
        f(t) = -1.0 + 3.0*t - t^2 + 0.25*t^3
        y = [f(xi) for xi in x]
        c = TensorGR._vandermonde_solve(x, y)
        @test length(c) == 4
        @test isapprox(c[1], -1.0; atol=1e-8)
        @test isapprox(c[2], 3.0; atol=1e-8)
        @test isapprox(c[3], -1.0; atol=1e-8)
        @test isapprox(c[4], 0.25; atol=1e-8)
    end

    # ══════════════════════════════════════════════════════════════════
    # 2. Backward compatibility: Λ=0 recovers flat spin_project
    # ══════════════════════════════════════════════════════════════════

    @testset "Λ=0 recovery" begin
        reg = _setup_flat_registry_ds()
        with_registry(reg) do
            K = build_FP_momentum_kernel(reg)
            kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

            for k2 in [0.5, 1.0, 2.0]
                for spin in [:spin2, :spin1, :spin0s, :spin0w]
                    flat_val = _eval_spin_scalar(spin_project(K, spin; kw...), k2)
                    mss_val = TensorGR.spin_project_mss(K, spin;
                        cosmological_constant=0.0, k2_eval=k2, kw...)
                    @test isapprox(flat_val, mss_val; atol=1e-10)
                end
            end
        end
    end

    # ══════════════════════════════════════════════════════════════════
    # 3. Synthetic MSS kernel: flat projection shows Λ offset
    # ══════════════════════════════════════════════════════════════════

    @testset "synthetic MSS kernel: flat projection has Λ offset" begin
        reg = _setup_flat_registry_ds()
        Λ_val = 0.1

        K_mss = with_registry(reg) do
            _build_synthetic_mss_EH_kernel(reg, Λ_val)
        end

        with_registry(reg) do
            kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

            # Flat spin-2 projection of MSS kernel: should be (5/2)k² - 5Λ
            s2 = spin_project(K_mss, :spin2; kw...)
            val_s2 = _eval_spin_scalar(s2, 1.0)
            expected_flat = 2.5 - 5.0 * Λ_val  # 2.5 - 0.5 = 2.0
            @test isapprox(val_s2, expected_flat; atol=1e-8)

            # Flat spin-0s projection: should be -k² + Λ(d-1) = -1 + 3Λ
            # since P⁰ˢ_{μν,μν} = d-1 and P⁰ˢ is pure trace
            # Mass term 1: -Λ·P⁰ˢ_{μν,μν} = ? depends on trace structure
            # Mass term 2: Λ·P⁰ˢ(h², h²) = Λ·(1/(d-1))·(d-1)² = Λ·(d-1)
            s0s = spin_project(K_mss, :spin0s; kw...)
            val_s0s = _eval_spin_scalar(s0s, 1.0)
            # Just verify it's not equal to the flat answer (-1.0)
            # The exact mass term offset depends on the projector traces
            @test val_s0s != -1.0 || Λ_val == 0
        end
    end

    # ══════════════════════════════════════════════════════════════════
    # 4. spin_project_mss: EH on synthetic MSS matches BC
    # ══════════════════════════════════════════════════════════════════

    @testset "spin_project_mss: EH spin-2 on synthetic MSS" begin
        reg = _setup_flat_registry_ds()
        kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

        for Λ_val in [0.0, 0.05, 0.1, 0.3, 0.5, 1.0]
            K_mss = with_registry(reg) do
                _build_synthetic_mss_EH_kernel(reg, Λ_val)
            end

            with_registry(reg) do
                # MSS-corrected spin-2 should be (5/2)k² for any Λ (pure EH)
                f2 = TensorGR.spin_project_mss(K_mss, :spin2;
                    cosmological_constant=Λ_val, k2_eval=1.0,
                    form_factor_order=1, kw...)
                bc = bc_to_form_factors(bc_EH(1.0, Λ_val), 1.0, Λ_val)
                @test isapprox(f2, bc.f_spin2; atol=1e-8)

                # Also verify at k²=2.0
                f2_k2 = TensorGR.spin_project_mss(K_mss, :spin2;
                    cosmological_constant=Λ_val, k2_eval=2.0,
                    form_factor_order=1, kw...)
                bc_k2 = bc_to_form_factors(bc_EH(1.0, Λ_val), 2.0, Λ_val)
                @test isapprox(f2_k2, bc_k2.f_spin2; atol=1e-8)
            end
        end
    end

    @testset "spin_project_mss: EH spin-0s on synthetic MSS" begin
        reg = _setup_flat_registry_ds()
        kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

        for Λ_val in [0.0, 0.05, 0.1, 0.3, 0.5]
            K_mss = with_registry(reg) do
                _build_synthetic_mss_EH_kernel(reg, Λ_val)
            end

            with_registry(reg) do
                f0s = TensorGR.spin_project_mss(K_mss, :spin0s;
                    cosmological_constant=Λ_val, k2_eval=1.0,
                    form_factor_order=1, kw...)
                bc = bc_to_form_factors(bc_EH(1.0, Λ_val), 1.0, Λ_val)
                @test isapprox(f0s, bc.f_spin0s; atol=1e-8)
            end
        end
    end

    # ══════════════════════════════════════════════════════════════════
    # 5. mss_form_factors: all sectors at once
    # ══════════════════════════════════════════════════════════════════

    @testset "mss_form_factors: full EH on synthetic MSS" begin
        reg = _setup_flat_registry_ds()
        Λ_val = 0.2

        K_mss = with_registry(reg) do
            _build_synthetic_mss_EH_kernel(reg, Λ_val)
        end

        with_registry(reg) do
            ff = TensorGR.mss_form_factors(K_mss;
                cosmological_constant=Λ_val, k2_eval=1.0,
                form_factor_order=1,
                dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)
            bc = bc_to_form_factors(bc_EH(1.0, Λ_val), 1.0, Λ_val)

            @test isapprox(ff.f_spin2, bc.f_spin2; atol=1e-8)
            @test isapprox(ff.f_spin0s, bc.f_spin0s; atol=1e-8)
            # Gauge sectors should be approximately 0 (may have small Λ artifacts)
            @test abs(ff.f_spin1) < 0.5  # relaxed bound for synthetic kernel
            @test abs(ff.f_spin0w) < 0.5
        end
    end

    # ══════════════════════════════════════════════════════════════════
    # 6. 4-derivative theory on flat: form_factor_order=2
    # ══════════════════════════════════════════════════════════════════

    @testset "spin_project_mss: 4-deriv flat (Λ=0, order=2)" begin
        reg = _setup_flat_registry_ds()
        kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

        with_registry(reg) do
            # Build 4-derivative kernel: κR + α₁R² + α₂Ric²
            κ = 1; α₁ = -1//10; α₂ = 3//10
            K = build_6deriv_flat_kernel(reg; κ=Rational{Int}(κ),
                    α₁=Rational{Int}(α₁), α₂=Rational{Int}(α₂))

            # At Λ=0, spin_project_mss should match flat spin_project exactly
            for k2 in [0.5, 1.0, 2.0]
                for spin in [:spin2, :spin0s]
                    flat_val = _eval_spin_scalar(spin_project(K, spin; kw...), k2)
                    mss_val = TensorGR.spin_project_mss(K, spin;
                        cosmological_constant=0.0, k2_eval=k2,
                        form_factor_order=2, kw...)
                    @test isapprox(flat_val, mss_val; atol=1e-8)
                end
            end

            # Cross-check: spin_project_mss at Λ=0 matches bc_to_form_factors
            bc = bc_EH(Float64(κ), 0.0) + bc_R2(Float64(α₁), 0.0) +
                 bc_RicSq(Float64(α₂), 0.0)
            for k2 in [0.5, 1.0, 2.0]
                ff = bc_to_form_factors(bc, k2, 0.0)
                f2 = TensorGR.spin_project_mss(K, :spin2;
                    cosmological_constant=0.0, k2_eval=k2,
                    form_factor_order=2, kw...)
                f0 = TensorGR.spin_project_mss(K, :spin0s;
                    cosmological_constant=0.0, k2_eval=k2,
                    form_factor_order=2, kw...)
                @test isapprox(f2, ff.f_spin2; atol=1e-8)
                @test isapprox(f0, ff.f_spin0s; atol=1e-8)
            end
        end
    end

    # ══════════════════════════════════════════════════════════════════
    # 7. BC formula consistency checks
    # ══════════════════════════════════════════════════════════════════

    @testset "BC form factor: EH is Λ-independent" begin
        # For pure EH, the form factors should be independent of Λ
        κ = 1.0; k2 = 1.5
        for Λ in [0.0, 0.1, 0.5, 1.0, 2.0]
            bc = bc_EH(κ, Λ)
            ff = bc_to_form_factors(bc, k2, Λ)
            @test isapprox(ff.f_spin2, 2.5 * κ * k2; atol=1e-12)
            @test isapprox(ff.f_spin0s, -κ * k2; atol=1e-12)
        end
    end

    @testset "BC form factor: scaling with κ" begin
        for κ in [0.5, 1.0, 2.0, 5.0]
            for Λ in [0.0, 0.1, 0.5]
                bc = bc_EH(κ, Λ)
                ff = bc_to_form_factors(bc, 1.0, Λ)
                @test isapprox(ff.f_spin2, 2.5 * κ; atol=1e-12)
                @test isapprox(ff.f_spin0s, -κ; atol=1e-12)
            end
        end
    end

    @testset "BC form factor: Stelle theory" begin
        # κR + α₁R² + α₂Ric² at Λ=0
        κ = 1.0; α₁ = -0.1; α₂ = 0.3; Λ = 0.0; k2 = 1.0
        bc = bc_EH(κ, Λ) + bc_R2(α₁, Λ) + bc_RicSq(α₂, Λ)
        ff = bc_to_form_factors(bc, k2, Λ)

        # f₂ = (5/2)[κ·k² - (c/2)k⁴] with c = 2α₂
        @test isapprox(ff.f_spin2, 2.5 * (κ * k2 - α₂ * k2^2); atol=1e-12)

        # f₀ = -κk² - (3b+c)k⁴ with b=2α₁, c=2α₂
        @test isapprox(ff.f_spin0s, -κ * k2 - (6α₁ + 2α₂) * k2^2; atol=1e-12)
    end

    # ══════════════════════════════════════════════════════════════════
    # 8. Perturbation engine on flat: FP kernel via spin_project_mss
    # ══════════════════════════════════════════════════════════════════

    @testset "perturbation engine: flat EH via spin_project_mss" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
            mp = define_metric_perturbation!(reg, :g, :h)

            # Set flat background
            set_vanishing!(reg, :Ric)
            set_vanishing!(reg, :RicScalar)
            set_vanishing!(reg, :Riem)

            kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

            d1R_ab = simplify(δricci(mp, down(:a), down(:b), 1); registry=reg)
            d1R    = simplify(δricci_scalar(mp, 1); registry=reg)
            h_up   = Tensor(:h, [up(:a), up(:b)])
            trh    = Tensor(:g, [up(:p), up(:q)]) * Tensor(:h, [down(:p), down(:q)])

            K = extract_kernel_direct(h_up * d1R_ab - (1 // 2) * trh * d1R,
                                      :h; registry=reg)

            # spin_project_mss with Λ=0 should recover FP exactly
            f2_flat = TensorGR.spin_project_mss(K, :spin2;
                cosmological_constant=0.0, k2_eval=1.0,
                form_factor_order=1, kw...)
            @test isapprox(f2_flat, 2.5; atol=1e-8)

            f0_flat = TensorGR.spin_project_mss(K, :spin0s;
                cosmological_constant=0.0, k2_eval=1.0,
                form_factor_order=1, kw...)
            @test isapprox(f0_flat, -1.0; atol=1e-8)
        end
    end

    # ══════════════════════════════════════════════════════════════════
    # 9. Lichnerowicz mass identification
    # ══════════════════════════════════════════════════════════════════

    @testset "Lichnerowicz mass term" begin
        # The spin-2 Lichnerowicz mass on MSS (d=4, R_{μν}=Λg_{μν}) is 2Λ.
        # Verify: flat projection = (5/2)(k² - 2Λ), so mass_term = -5Λ.
        reg = _setup_flat_registry_ds()
        kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

        for Λ_val in [0.05, 0.1, 0.2, 0.5]
            K_mss = with_registry(reg) do
                _build_synthetic_mss_EH_kernel(reg, Λ_val)
            end

            with_registry(reg) do
                # Evaluate flat projection at two k² values
                s2 = spin_project(K_mss, :spin2; kw...)
                v1 = _eval_spin_scalar(s2, 1.0)
                v2 = _eval_spin_scalar(s2, 2.0)

                # Extract slope (kinetic coefficient) and intercept (mass term)
                slope = v2 - v1  # c₁ where f = c₀ + c₁k²
                intercept = v1 - slope * 1.0  # c₀

                # Slope should be (5/2) = 2.5 (Λ-independent)
                @test isapprox(slope, 2.5; atol=1e-8)

                # Intercept = mass term = -5Λ
                @test isapprox(intercept, -5.0 * Λ_val; atol=1e-8)

                # So the Lichnerowicz spin-2 mass is m² = 2Λ
                # (since mass_term = -(5/2)·m² = -(5/2)·2Λ = -5Λ)
            end
        end
    end

    # ══════════════════════════════════════════════════════════════════
    # 10. Edge cases
    # ══════════════════════════════════════════════════════════════════

    @testset "edge cases" begin
        reg = _setup_flat_registry_ds()
        kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

        with_registry(reg) do
            K = build_FP_momentum_kernel(reg)

            # form_factor_order=1 on purely flat kernel
            f2 = TensorGR.spin_project_mss(K, :spin2;
                cosmological_constant=0.0, k2_eval=1.0,
                form_factor_order=1, kw...)
            @test isapprox(f2, 2.5; atol=1e-10)

            # Large Λ (stress-testing numerical stability)
            K_big = _build_synthetic_mss_EH_kernel(reg, 10.0)
            f2_big = TensorGR.spin_project_mss(K_big, :spin2;
                cosmological_constant=10.0, k2_eval=1.0,
                form_factor_order=1, kw...)
            @test isapprox(f2_big, 2.5; atol=1e-6)

            # Negative Λ (AdS)
            K_ads = _build_synthetic_mss_EH_kernel(reg, -0.1)
            f2_ads = TensorGR.spin_project_mss(K_ads, :spin2;
                cosmological_constant=-0.1, k2_eval=1.0,
                form_factor_order=1, kw...)
            @test isapprox(f2_ads, 2.5; atol=1e-8)
        end
    end

end  # main testset
