# ============================================================================
# Benchmark 13: 6-Derivative Gravity Particle Spectrum
#
# Tests the full spectrum pipeline:
#   Tier 1: dS spectrum API (BuenoCanoParams, dS_spectrum_6deriv)
#   Tier 2: Barnes-Rivers spin projection on flat background
#   Tier 3: Perturbation engine δ²S → Fourier → kernel → spin projection
#   Tier 3: SVT quadratic forms (Path B) + Path A vs B cross-check
#
# Ground truth: Buoninfante et al. 2012.11829 Eq. (2.13) (flat)
#               Bueno & Cano 1607.06463 Eqs. (17)-(19) (dS)
# ============================================================================

using TensorGR, Test, Random
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

# ── Tier 3: SVT quadratic forms (Path B) ─────────────────────────────────

@testset "bench_13: SVT quadratic forms" begin
    # GR limit: M_TT = κp², det(M_scalar) = -4κ²k⁴
    tc_gr = timed_compute() do
        r = svt_quadratic_forms_6deriv(κ=1.0, ω²=2.0, k²=1.5)
        p2 = 2.0 - 1.5
        @test isapprox(r.tensor.matrix[1,1], p2; rtol=1e-12)
        M = r.scalar.matrix
        det_val = M[1,1]*M[2,2] - M[1,2]*M[2,1]
        @test isapprox(det_val, -4*1.5^2; rtol=1e-10)
        @test r.vector_vanishes === true
        r
    end
    println("  SVT GR limit: $(round(tc_gr.time*1e6; digits=0))μs")

    # Tensor sector = κp²f₂(p²) at random params
    Random.seed!(1301)
    tc_tensor = timed_compute() do
        for _ in 1:50
            κ = rand()*3+0.5; α₂ = rand()*2-1.0; β₂ = rand()*0.5
            ω2 = rand()*5+0.1; k2 = rand()*5+0.1
            p2 = ω2 - k2
            r = svt_quadratic_forms_6deriv(κ=κ, α₂=α₂, β₂=β₂, ω²=ω2, k²=k2)
            expected = κ*p2 - α₂*p2^2 - β₂*p2^3
            @test isapprox(r.tensor.matrix[1,1], expected; rtol=1e-10)
        end
    end
    println("  SVT tensor sector (50 pts): $(round(tc_tensor.time*1e3; digits=1))ms")

    # Scalar det vanishes at f₀ roots (on mass shell)
    Random.seed!(1302)
    n_tested = 0
    tc_scalar = timed_compute() do
        for _ in 1:100
            κ = rand()*3+0.5; α₁ = rand()*2-1.0; α₂ = rand()*2-1.0
            β₁ = rand()*0.5; β₂ = rand()*0.5
            c1 = (6α₁ + 2α₂) / κ; c2 = (6β₁ + 2β₂) / κ
            abs(c2) < 1e-12 && continue
            disc = c1^2 - 4c2
            disc < 0 && continue
            z₁ = (-c1 + sqrt(disc)) / (2c2)
            abs(z₁) < 1e-6 && continue
            k2 = rand()*5 + 0.5; ω2 = k2 + z₁
            r = svt_quadratic_forms_6deriv(κ=κ, α₁=α₁, α₂=α₂, β₁=β₁, β₂=β₂,
                                            ω²=ω2, k²=k2)
            M = r.scalar.matrix
            det_val = M[1,1]*M[2,2] - M[1,2]*M[2,1]
            scale = max(1.0, abs(κ^2 * k2^2))
            @test isapprox(det_val, 0.0; atol=1e-6 * scale)
            n_tested += 1
        end
    end
    println("  SVT scalar det at f₀ roots ($n_tested pts): $(round(tc_scalar.time*1e3; digits=1))ms")
end

# ── Tier 3: Cross-check Path A vs Path B ─────────────────────────────────

@testset "bench_13: Path A vs B cross-check" begin
    # Path A: spin projection form factors f₂(p²), f₀(p²)
    # Path B: SVT M_TT = κp²f₂(p²), det(M_scalar) vanishes at f₀ roots
    # Agreement: tensor poles match f₂ zeros, scalar det zeros match f₀ zeros

    tc = timed_compute() do
        Random.seed!(1303)
        n_tensor = 0
        n_scalar = 0
        for _ in 1:50
            κ = rand()*3+0.5; α₁ = rand()*2-1.0; α₂ = rand()*2+0.1
            β₁ = rand()*0.5; β₂ = rand()*0.5+0.01

            # Form factors
            f₂(z) = 1 - (α₂/κ)*z - (β₂/κ)*z^2
            f₀(z) = 1 + (6α₁+2α₂)*z/κ + (6β₁+2β₂)*z^2/κ

            # Tensor: find f₂ roots, check M_TT vanishes
            a_c = -β₂/κ; b_c = -α₂/κ
            disc2 = b_c^2 - 4a_c
            if disc2 >= 0
                z₁ = (-b_c + sqrt(disc2)) / (2a_c)
                k2 = rand()*5+0.5; ω2 = k2 + z₁
                r = svt_quadratic_forms_6deriv(κ=κ, α₂=α₂, β₂=β₂, ω²=ω2, k²=k2)
                @test isapprox(r.tensor.matrix[1,1], 0.0; atol=1e-8*max(1.0, abs(κ*z₁)))
                n_tensor += 1
            end

            # Scalar: find f₀ roots, check det vanishes
            c1 = (6α₁+2α₂)/κ; c2 = (6β₁+2β₂)/κ
            abs(c2) < 1e-12 && continue
            disc0 = c1^2 - 4c2
            disc0 < 0 && continue
            z₁ = (-c1 + sqrt(disc0)) / (2c2)
            abs(z₁) < 1e-6 && continue
            k2 = rand()*5+0.5; ω2 = k2 + z₁
            r = svt_quadratic_forms_6deriv(κ=κ, α₁=α₁, α₂=α₂, β₁=β₁, β₂=β₂,
                                            ω²=ω2, k²=k2)
            M = r.scalar.matrix
            det_val = M[1,1]*M[2,2] - M[1,2]*M[2,1]
            @test isapprox(det_val, 0.0; atol=1e-6*max(1.0, abs(κ^2*k2^2)))
            n_scalar += 1
        end
        (n_tensor, n_scalar)
    end
    nt, ns = tc.result
    println("  Cross-check tensor=$nt scalar=$ns: $(round(tc.time*1e3; digits=1))ms")

    # Buoninfante form factors via Path A match SVT tensor sector (Path B)
    tc2 = timed_compute() do
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            sp = flat_6deriv_spin_projections(reg; κ=1//1, α₂=3//10, β₂=1//10)
            K_GR = build_FP_momentum_kernel(reg)
            gr2 = spin_project(K_GR, :spin2; registry=reg)

            for k2 in [1.0, 2.0, 3.0, 5.0]
                # Path A: spin projection gives Tr(K·P²) = (5/2)k²·f₂(k²)
                v_A = _eval_spin_scalar(sp.spin2, k2)
                # Path B: M_TT = κp²f₂(p²), with p²=k² (on-shell ω²=2k²)
                r = svt_quadratic_forms_6deriv(κ=1.0, α₂=0.3, β₂=0.1, ω²=2*k2, k²=k2)
                # Normalize both to f₂: Path A via GR, Path B via κp²
                f2_A = v_A / _eval_spin_scalar(gr2, k2)
                f2_B = r.tensor.matrix[1,1] / k2
                @test isapprox(f2_A, f2_B; atol=1e-12)
            end
        end
    end
    println("  Form factor Path A≡B: $(round(tc2.time; digits=1))s")
end
