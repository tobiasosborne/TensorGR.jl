# ============================================================================
# Benchmark 07: Riemann Correlator on de Sitter
#
# Paper: Frob, Roura, Verdaguer (arXiv:1403.3335)
# Reproduces: Linearised Riemann on maximally symmetric background,
#             Weyl decomposition of perturbation
# Exercises: curved-background perturbation, Weyl decomposition
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ground_truth.jl"))

@testset "Bench 07: Riemann Correlator (de Sitter)" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)

        # de Sitter: maximally symmetric, R_{ab} = Λ g_{ab}
        maximally_symmetric_background!(reg, :M4; metric=:g, cosmological_constant=:Λ)

        # Curved-background perturbation
        mp = define_metric_perturbation!(reg, :g, :h; curved=true)

        # ── 7.1: δ¹Riemann on de Sitter ──────────────────────────────
        @testset "δ¹Riemann on de Sitter" begin
            δ1R = δriemann(mp, up(:a), down(:b), down(:c), down(:d), 1)
            @test δ1R != TScalar(0 // 1)
            @test count_terms(δ1R) == DS_D1RIEM_RAW_TERMS

            δ1R_simplified = simplify(δ1R)
            @test δ1R_simplified != TScalar(0 // 1)
            @test count_terms(δ1R_simplified) == DS_D1RIEM_SIMPLIFIED_TERMS

            # Background rules fully applied: no bare Ric or RicScalar remain
            @test !TensorGR._contains_tensor(δ1R_simplified, :Ric)
            @test !TensorGR._contains_tensor(δ1R_simplified, :RicScalar)
        end

        # ── 7.2: Weyl decomposition of linearised Riemann ─────────────
        @testset "Weyl decomposition of linearised Riemann" begin
            decomp = weyl_to_riemann(down(:a), down(:b), down(:c), down(:d), :g; dim=4)
            @test decomp isa TSum
            @test count_terms(decomp) == WEYL_DECOMPOSITION_TERMS
        end

        # ── 7.3: Weyl trace-free (identity) ──────────────────────────
        @testset "Weyl trace-free on de Sitter" begin
            W = Tensor(:Weyl, [up(:a), down(:b), down(:a), down(:d)])
            W_expanded = to_riemann(W)
            W_contracted = contract_curvature(W_expanded)
            W_trace = simplify(W_contracted)
            @test W_trace == TScalar(0 // 1)
        end

        # ── 7.4: Two-point function tensor structure ───────────────────
        @testset "Two-point function tensor structure" begin
            @test_skip "Requires bitensor framework (parallel propagator, Synge world function)"
        end
    end
end
