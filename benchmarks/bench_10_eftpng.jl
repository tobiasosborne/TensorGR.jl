# ============================================================================
# Benchmark 10: EFTofPNG — Post-Newtonian Gravity Scalability
#
# Paper: Levi, Steinhoff (arXiv:1705.06309)
# Reproduces: Performance under expression swell at high PN order
# Exercises: scalability, large expression handling
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ground_truth.jl"))

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

            # PN order counting
            @test pn_order(Tensor(:g, [down(:a), down(:b)]), :vA) == 0
            @test pn_order(Tensor(:vA, [up(:a)]), :vA) == 1
            @test pn_order(Tensor(:vA, [up(:a)]) * Tensor(:vA, [up(:b)]), :vA) == 2
        end

        # ── 10.2: δⁿRicci term counts pinned ─────────────────────────
        @testset "δⁿRicci term counts" begin
            mp = define_metric_perturbation!(reg, :g, :h)

            for order in 1:3
                dr = δricci(mp, down(:a), down(:b), order)
                @test dr != TScalar(0 // 1)
                @test count_terms(dr) == EFTPNG_DRICCI_TERMS[order]
            end
        end

        # ── 10.3: Expression growth is superlinear ────────────────────
        @testset "Expression growth with perturbation order" begin
            mp = define_metric_perturbation!(reg, :g, :h)

            term_counts = Int[]
            for order in 1:3
                dr = δricci(mp, down(:a), down(:b), order)
                push!(term_counts, count_terms(dr))
            end

            @test term_counts[2] > term_counts[1]
            @test term_counts[3] > term_counts[2]
        end

        # ── 10.4: PN truncation ───────────────────────────────────────
        @testset "PN truncation drops velocity terms" begin
            wl_A = Worldline(:A; manifold=:M4)
            wl_B = Worldline(:B; manifold=:M4)
            define_worldline!(reg, wl_A; metric=:g)
            define_worldline!(reg, wl_B; metric=:g)

            v = Tensor(:vA, [up(:a)])
            # expr with v² terms + a bare h term
            expr_v2 = v * v * Tensor(:h, [down(:a), down(:b)])
            expr_bare = Tensor(:h, [down(:c), down(:d)])

            trunc = truncate_pn(tsum(TensorExpr[expr_v2, expr_bare]), 0, :vA)
            # Only the term without velocities survives at 0PN
            @test count_terms(trunc) == 1

            # Higher truncation keeps both
            trunc2 = truncate_pn(tsum(TensorExpr[expr_v2, expr_bare]), 2, :vA)
            @test count_terms(trunc2) == 2
        end
    end
end
