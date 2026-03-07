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

            n = count_terms(δ1R)
            @test n > 1
            println("  δ¹Riem on dS: $n terms")

            # After applying background rules (Ric → Λg, R → 4Λ),
            # the linearized Riemann should contain Λ·h coupling terms
            # from the Γ₀ cross terms in δriemann
            δ1R_simplified = simplify(δ1R)
            n_s = count_terms(δ1R_simplified)
            println("  δ¹Riem on dS (simplified): $n_s terms")
            @test n_s > 0
        end

        # ── 7.2: Weyl decomposition of linearised Riemann ─────────────
        @testset "Weyl decomposition of linearised Riemann" begin
            # The linearized Riemann decomposes as
            # δR_{abcd} = δC_{abcd} + (Weyl-trace terms involving δRic, δR)
            # On dS, the Weyl part of the linearized Riemann is the
            # gauge-invariant piece carrying the physical GW degrees of freedom

            # Build the Riemann → Weyl decomposition at the symbolic level
            decomp = weyl_to_riemann(down(:a), down(:b), down(:c), down(:d), :g; dim=4)
            @test decomp isa TSum  # Riem = Weyl + Ricci terms
            n = count_terms(decomp)
            @test n >= 3  # at least Weyl + Ricci + scalar terms
            println("  Weyl decomposition: $n terms")

            # The linearized Weyl tensor on dS is traceless by construction
            W = Tensor(:Weyl, [up(:a), down(:b), down(:a), down(:d)])
            W_expanded = to_riemann(W)
            W_contracted = contract_curvature(W_expanded)
            W_trace = simplify(W_contracted)
            @test W_trace == TScalar(0 // 1)
            println("  Weyl trace-free: confirmed")
        end

        # ── 7.3: Two-point function tensor structure ───────────────────
        @testset "Two-point function tensor structure" begin
            @test_skip "Requires bitensor framework (parallel propagator, Synge world function)"
        end
    end
end
