# ============================================================================
# Benchmark 05: Second-Order Perturbation of Schwarzschild
#
# Paper: Brizuela, Martin-Garcia, Tiglio (arXiv:0903.1134)
# Reproduces: Gauge-invariant 2nd-order perturbations, Zerilli/RW equations
# Exercises: multi-order perturbation, curved background, expression swell
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))

@testset "Bench 05: Schwarzschild 2nd-Order Perturbations" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)

        # Schwarzschild: Ric = 0, R = 0, but Riem ≠ 0
        vacuum_background!(reg, :M4; metric=:g)

        # Curved-background perturbation
        mp = define_metric_perturbation!(reg, :g, :h; curved=true)

        # ── 5.1: δ¹Ricci on Schwarzschild ─────────────────────────────
        @testset "δ¹Ricci on vacuum background" begin
            δ1Ric = δricci(mp, down(:a), down(:b), 1)
            @test δ1Ric != TScalar(0 // 1)
            # On vacuum bg, Ric₀ = 0, so δ¹Ric is purely from
            # the perturbation of Christoffel symbols
            result = simplify(δ1Ric)
            n = count_terms(result)
            @test n > 0
            println("  δ¹Ric on Schwarzschild: $n terms")
        end

        # ── 5.2: δ²Ricci on curved background ─────────────────────────
        @testset "δ²Ricci on curved background" begin
            δ2Ric = δricci(mp, down(:a), down(:b), 2)
            @test δ2Ric != TScalar(0 // 1)

            # Second-order Ricci has bilinear δΓ·δΓ terms and
            # Γ₀ cross terms from the curved background
            n = count_terms(δ2Ric)
            @test n > 1  # must have multiple terms
            println("  δ²Ric on Schwarzschild: $n terms (before simplify)")
        end

        # ── 5.3: Zerilli master equation ───────────────────────────────
        @testset "Zerilli master equation" begin
            @test_skip "Requires spherical harmonic decomposition (B3a)"
        end

        # ── 5.4: Regge-Wheeler master equation ─────────────────────────
        @testset "Regge-Wheeler master equation" begin
            @test_skip "Requires spherical harmonic decomposition (B3a)"
        end
    end
end
