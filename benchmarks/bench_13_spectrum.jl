# ============================================================================
# Benchmark 13: 6-Derivative Gravity Particle Spectrum
#
# Tests the full spectrum pipeline:
#   Tier 1: dS spectrum API (BuenoCanoParams, dS_spectrum_6deriv)
#   Tier 2: Barnes-Rivers spin projection on flat background
#   Tier 3: Perturbation engine δ²S → Fourier → kernel → spin projection
#
# Ground truth: Buoninfante et al. 2012.11829 Eq. (2.13) (flat)
#               Bueno & Cano 1607.06463 Eqs. (17)-(19) (dS)
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))

# ── Tier 1: dS spectrum API ────────────────────────────────────────────────

@testset "bench_13: dS spectrum API" begin
    # GR limit
    s = dS_spectrum_6deriv(κ=1.0, Λ=0.1)
    @test s.κ_eff_inv ≈ 4.0
    @test isinf(s.m2_graviton)
    @test isinf(s.m2_scalar)

    # Stelle flat limit: known mass formulas
    κ, α₁, α₂ = 1.0, -0.2, 0.3
    s = dS_spectrum_6deriv(κ=κ, α₁=α₁, α₂=α₂, Λ=0.0)
    @test isapprox(s.m2_graviton, -κ/(2α₂); rtol=1e-10)
    @test isapprox(s.m2_scalar, 2κ/(24α₁+8α₂); rtol=1e-10)

    # Flat form factors (Buoninfante Eq. 2.13)
    β₁, β₂ = 0.05, 0.1
    s = dS_spectrum_6deriv(κ=κ, α₁=α₁, α₂=α₂, β₁=β₁, β₂=β₂, Λ=0.0)
    @test s.flat_f2 == (-α₂/κ, -β₂/κ)
    @test s.flat_f0 == ((6α₁+2α₂)/κ, (6β₁+2β₂)/κ)

    # Cubics shift dS masses
    s1 = dS_spectrum_6deriv(κ=1.0, α₂=0.3, Λ=0.1)
    s2 = dS_spectrum_6deriv(κ=1.0, α₂=0.3, γ₄=0.02, Λ=0.1)
    @test s1.m2_graviton != s2.m2_graviton

    # BuenoCanoParams additive
    p1 = BuenoCanoParams(1.0, 2.0, 3.0, 4.0)
    p2 = BuenoCanoParams(0.5, 1.5, 2.5, 3.5)
    p = p1 + p2
    @test p.a ≈ 1.5 && p.b ≈ 3.5 && p.c ≈ 5.5 && p.e ≈ 7.5

    # Random consistency: API matches manual BC sum
    using Random
    Random.seed!(1313)
    for _ in 1:50
        κ = rand()*3+0.5; α₁ = (rand()-0.5)*0.5; α₂ = rand()*2+0.1
        γs = [(rand()-0.5)*0.1 for _ in 1:6]; Λ = rand()*0.2
        s = dS_spectrum_6deriv(κ=κ, α₁=α₁, α₂=α₂,
                                γ₁=γs[1], γ₂=γs[2], γ₃=γs[3],
                                γ₄=γs[4], γ₅=γs[5], γ₆=γs[6], Λ=Λ)
        p = bc_EH(κ, Λ) + bc_R2(α₁, Λ) + bc_RicSq(α₂, Λ) +
            bc_R3(γs[1], Λ) + bc_RRicSq(γs[2], Λ) + bc_Ric3(γs[3], Λ) +
            bc_RRiem2(γs[4], Λ) + bc_RicRiem2(γs[5], Λ) + bc_Riem3(γs[6], Λ)
        @test s.params.a ≈ p.a rtol=1e-10
        @test s.params.e ≈ p.e rtol=1e-10
    end
end

# ── Tier 2: Barnes-Rivers spin projection ─────────────────────────────────

@testset "bench_13: spin projection (flat)" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)
        @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
        @define_tensor k on=M4 rank=(0,1)

        # Fierz-Pauli kernel
        K_FP = build_FP_momentum_kernel(reg)
        @test length(K_FP.terms) == 4

        r2 = spin_project(K_FP, :spin2; registry=reg)
        r1 = spin_project(K_FP, :spin1; registry=reg)
        r0s = spin_project(K_FP, :spin0s; registry=reg)
        r0w = spin_project(K_FP, :spin0w; registry=reg)

        k2 = 1.7
        v2 = _eval_spin_scalar(r2, k2)
        v1 = _eval_spin_scalar(r1, k2)
        v0s = _eval_spin_scalar(r0s, k2)
        v0w = _eval_spin_scalar(r0w, k2)

        # FP: Tr(K·P²)=(5/2)k², spin-1=0, Tr(K·P⁰ˢ)=-k², spin-0-w=0
        @test abs(v2 - 2.5*k2) < 1e-10
        @test abs(v1) < 1e-10
        @test abs(v0s - (-k2)) < 1e-10
        @test abs(v0w) < 1e-10

        # R² kernel
        K_R2 = build_R2_momentum_kernel(reg)
        r2_R2 = spin_project(K_R2, :spin2; registry=reg)
        r0s_R2 = spin_project(K_R2, :spin0s; registry=reg)

        v2_R2 = _eval_spin_scalar(r2_R2, k2)
        v0s_R2 = _eval_spin_scalar(r0s_R2, k2)

        @test abs(v2_R2) < 1e-10            # R² has no spin-2
        @test abs(v0s_R2 - 3*k2^2) < 1e-10  # R² contributes to spin-0

        # Ric² kernel
        K_Ric2 = build_Ric2_momentum_kernel(reg)
        r2_Ric2 = spin_project(K_Ric2, :spin2; registry=reg)
        r0s_Ric2 = spin_project(K_Ric2, :spin0s; registry=reg)

        v2_Ric2 = _eval_spin_scalar(r2_Ric2, k2)
        v0s_Ric2 = _eval_spin_scalar(r0s_Ric2, k2)

        @test abs(v2_Ric2 - 5*k2^2/4) < 1e-10   # Ric² Tr(K·P²) = 5k⁴/4
        @test abs(v0s_Ric2 - k2^2) < 1e-10     # Ric² Tr(K·P⁰ˢ) = k⁴
    end
end

# ── Tier 3: Perturbation engine → Fourier → kernel ────────────────────────

@testset "bench_13: perturbation engine pipeline" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)
        @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
        @define_tensor k on=M4 rank=(0,1)

        mp = define_metric_perturbation!(reg, :g, :h)

        # δ²R on flat → Fourier → kernel extraction
        tc = timed_compute() do
            δ2R = simplify(δricci_scalar(mp, 2); registry=reg)
            fourier = simplify(to_fourier(δ2R); registry=reg)
            K = extract_kernel(fourier, :h; registry=reg)
            K
        end
        @test length(tc.result.terms) > 0
        println("  δ²R → kernel: $(length(tc.result.terms)) terms, $(round(tc.time; digits=1))s")
    end
end
