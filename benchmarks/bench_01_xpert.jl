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
            # (1/2) g^{ad} (d_b h_{cd} + d_c h_{bd} - d_d h_{bc})
            # = 3 derivative terms, each with g^{ad} * d h
            @test dG != TScalar(0 // 1)
            @test dG isa TensorExpr
            println("  δ¹Γ: $(count_terms(dG)) terms (raw)")
        end

        # ── 1.2: First-order Ricci (Palatini identity) ───────────────────
        @testset "delta^1[Ricci] — Palatini identity" begin
            dRic = δricci(mp, down(:a), down(:b), 1)
            @test dRic != TScalar(0 // 1)
            nterms_raw = count_terms(dRic)
            println("  δ¹Ric: $nterms_raw terms (raw)")

            # On flat background, simplify and count
            background_solution!(reg, [:Ric, :RicScalar, :Ein])
            dRic_s = simplify(dRic)
            nterms_simplified = count_terms(dRic_s)
            println("  δ¹Ric: $nterms_simplified terms (simplified, flat bg)")
            # Should be the Lichnerowicz operator: 4 terms
            # (1/2)(-Box h_{ab} - d_a d_b h + d_a d^c h_{bc} + d_b d^c h_{ac})
        end

        # ── 1.3: First-order Ricci scalar ────────────────────────────────
        @testset "delta^1[RicciScalar]" begin
            dR = δricci_scalar(mp, 1)
            @test dR != TScalar(0 // 1)
            nterms = count_terms(dR)
            println("  δ¹R: $nterms terms (raw)")
        end

        # ── 1.4: First-order Riemann ─────────────────────────────────────
        @testset "delta^1[Riemann]" begin
            tc = timed_compute() do
                δriemann(mp, up(:a), down(:b), down(:c), down(:d), 1)
            end
            @test tc.result != TScalar(0 // 1)
            n1 = count_terms(tc.result)
            println("  δ¹Riem: $n1 terms ($(round(tc.time, digits=3))s)")
        end

        # ── 1.5: Second-order Riemann ────────────────────────────────────
        @testset "delta^2[Riemann]" begin
            tc = timed_compute() do
                δriemann(mp, up(:a), down(:b), down(:c), down(:d), 2)
            end
            @test tc.result != TScalar(0 // 1)
            n2 = count_terms(tc.result)
            println("  δ²Riem: $n2 terms ($(round(tc.time, digits=3))s)")
        end

        # ── 1.6: Third-order Riemann ─────────────────────────────────────
        @testset "delta^3[Riemann]" begin
            tc = timed_compute() do
                δriemann(mp, up(:a), down(:b), down(:c), down(:d), 3)
            end
            @test tc.result != TScalar(0 // 1)
            n3 = count_terms(tc.result)
            println("  δ³Riem: $n3 terms ($(round(tc.time, digits=3))s)")
            # xPert paper: n=3 canonicalizes in ~1s on 3 GHz Pentium IV (2008)
        end

        # ── 1.7: Second-order Einstein (Fig. 4) ─────────────────────────
        @testset "delta^2[Einstein] — Fig. 4" begin
            # δ²G_{ab} = δ²R_{ab} - (1/2)(δ²(g^{cd}R_{cd}) g_{ab} + ...)
            # The paper says this "is constructed and canonicalized in less
            # than one second."
            tc = timed_compute() do
                dRic2 = δricci(mp, down(:a), down(:b), 2)
                dR2 = δricci_scalar(mp, 2)
                # Einstein = Ricci - (1/2) g R
                g_ab = Tensor(:g, [down(:a), down(:b)])
                tsum([dRic2, tproduct(-1 // 2, [g_ab, dR2])])
            end
            @test tc.result != TScalar(0 // 1)
            n_ein2 = count_terms(tc.result)
            println("  δ²Ein: $n_ein2 terms ($(round(tc.time, digits=3))s)")
        end

        # ── 1.8: Higher orders (n=4,5) — optional perf test ─────────────
        if PERF_MODE
            for n in 4:5
                @testset "delta^$n[Riemann] — perf" begin
                    tc = timed_compute() do
                        δriemann(mp, up(:a), down(:b), down(:c), down(:d), n)
                    end
                    nn = count_terms(tc.result)
                    println("  δ$(n)Riem: $nn terms ($(round(tc.time, digits=2))s, $(round(tc.bytes/1e6, digits=1))MB)")
                end
            end
        end
    end
end
