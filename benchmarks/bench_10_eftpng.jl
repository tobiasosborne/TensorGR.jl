# ============================================================================
# Benchmark 10: EFTofPNG — Post-Newtonian Gravity Scalability
#
# Paper: Levi, Steinhoff (arXiv:1705.06309)
# Reproduces: Performance under expression swell at high PN order
# Exercises: scalability, large expression handling
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))

@testset "Bench 10: EFTofPNG — Scalability" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)

        # ── 10.1: Worldline construction ──────────────────────────────
        @testset "Worldline tensor formalism" begin
            wl = Worldline(:A; manifold=:M4)
            define_worldline!(reg, wl; metric=:g)

            @test has_tensor(reg, :vA)
            v = Tensor(:vA, [up(:a)])
            @test v isa Tensor
            println("  Worldline A: velocity vA registered")

            # PN order counting
            expr0 = Tensor(:g, [down(:a), down(:b)])  # no velocity → 0PN
            @test pn_order(expr0, :vA) == 0

            expr1 = Tensor(:vA, [up(:a)])  # one velocity → order 1
            @test pn_order(expr1, :vA) == 1

            expr2 = Tensor(:vA, [up(:a)]) * Tensor(:vA, [up(:b)])  # v² → 1PN
            @test pn_order(expr2, :vA) == 2
            println("  PN order counting: verified")
        end

        # ── 10.2: Expression growth tracking ──────────────────────────
        @testset "PN expansion expression growth" begin
            # Build a simple EFT interaction: graviton exchange between
            # two worldlines at increasing PN order.
            #
            # The point-particle action has the structure:
            #   S_pp = -m ∫ ds √(-g_{μν} v^μ v^ν)
            # Expanding g = η + h and keeping h interactions:
            #   S_pp ≈ -m ∫ ds (1 + h_{μν}v^μ v^ν/2 + ...)
            #
            # At each PN order, the number of terms grows.

            mp = define_metric_perturbation!(reg, :g, :h)
            wl_A = Worldline(:A; manifold=:M4)
            wl_B = Worldline(:B; manifold=:M4)
            define_worldline!(reg, wl_A; metric=:g)
            define_worldline!(reg, wl_B; metric=:g)

            # Track expression growth: δⁿRicci gives a proxy for PN swell
            term_counts = Int[]
            for order in 1:3
                tc = timed_compute() do
                    δricci(mp, down(:a), down(:b), order)
                end
                n = count_terms(tc.result)
                push!(term_counts, n)
                println("  δ$(order)Ric: $n terms ($(round(tc.time, digits=3))s)")
            end

            # Expression growth should be superlinear but manageable
            @test term_counts[1] > 0
            @test term_counts[2] > term_counts[1]  # grows with order
            @test term_counts[3] > term_counts[2]

            # Truncation works
            v = Tensor(:vA, [up(:a)])
            expr = v * v * Tensor(:h, [down(:a), down(:b)])  # 2 velocities
            trunc = truncate_pn(tsum(TensorExpr[expr, Tensor(:h, [down(:c), down(:d)])]),
                                 0, :vA)
            # Only the term without velocities survives at 0PN
            @test count_terms(trunc) == 1
            println("  PN truncation: verified")
        end
    end
end
