# ============================================================================
# Benchmark 01: xPert — Perturbation of Curvature Tensors
#
# Paper: Brizuela, Martin-Garcia, Mena Marugan (arXiv:0807.0824)
# Reproduces: Fig. 3 (Riemann term counts), Fig. 4 (2nd-order Einstein),
#             Eq. 18 (first-order Ricci), Sec. 5 (background field method)
# Exercises: perturbation engine, canonicalization, xperm.c, dummy indices
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ground_truth.jl"))

@testset "Bench 01: xPert — Perturbation of Curvature Tensors" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)
        @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
        mp = define_metric_perturbation!(reg, :g, :h)

        # ── 1.1: First-order Christoffel ─────────────────────────────────
        @testset "delta^1[Christoffel]" begin
            dG = δchristoffel(mp, up(:a), down(:b), down(:c), 1)
            @test dG isa TProduct
            @test dG != TScalar(0 // 1)
            @test count_terms(dG) == XPERT_CHRISTOFFEL1_TERMS
        end

        # ── 1.2: First-order Ricci (Palatini identity) ───────────────────
        @testset "delta^1[Ricci] — Palatini identity" begin
            dRic = δricci(mp, down(:a), down(:b), 1)
            @test dRic != TScalar(0 // 1)
            @test count_terms(dRic) == XPERT_RICCI1_RAW_TERMS

            # On flat background, simplify
            background_solution!(reg, [:Ric, :RicScalar, :Ein])
            dRic_s = simplify(dRic)
            @test dRic_s != TScalar(0 // 1)
            @test count_terms(dRic_s) == XPERT_RICCI1_SIMPLIFIED_TERMS
        end

        # ── 1.3: First-order Ricci scalar ────────────────────────────────
        @testset "delta^1[RicciScalar]" begin
            dR = δricci_scalar(mp, 1)
            @test dR != TScalar(0 // 1)
            @test count_terms(dR) == XPERT_RICCI_SCALAR1_TERMS
        end

        # ── 1.4: δⁿRiemann term counts (Fig. 3) ─────────────────────────
        @testset "delta^n[Riemann] term counts — Fig. 3" begin
            for n in 1:3
                tc = timed_compute() do
                    δriemann(mp, up(:a), down(:b), down(:c), down(:d), n)
                end
                @test tc.result != TScalar(0 // 1)
                @test count_terms(tc.result) == XPERT_RIEMANN_TERMS[n]
            end
        end

        # ── 1.5: Second-order Einstein (Fig. 4) ──────────────────────────
        @testset "delta^2[Einstein] — Fig. 4" begin
            tc = timed_compute() do
                dRic2 = δricci(mp, down(:a), down(:b), 2)
                dR2 = δricci_scalar(mp, 2)
                g_ab = Tensor(:g, [down(:a), down(:b)])
                tsum([dRic2, tproduct(-1 // 2, [g_ab, dR2])])
            end
            @test tc.result != TScalar(0 // 1)
            @test count_terms(tc.result) == XPERT_EINSTEIN2_TERMS
        end

        # ── 1.6: Higher orders (n=4,5) — optional perf test ──────────────
        if PERF_MODE
            for n in 4:5
                @testset "delta^$n[Riemann] — perf" begin
                    tc = timed_compute() do
                        δriemann(mp, up(:a), down(:b), down(:c), down(:d), n)
                    end
                    @test tc.result != TScalar(0 // 1)
                    @test count_terms(tc.result) > 0
                end
            end
        end
    end
end
