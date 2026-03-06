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
    @testset "δ¹Riemann on de Sitter" begin
        @test_skip "Requires perturbation with R_{ab} = Lambda g_{ab} substitution"
    end
    @testset "Weyl decomposition of linearised Riemann" begin
        @test_skip "Requires Weyl projection of perturbation output"
    end
    @testset "Two-point function tensor structure" begin
        @test_skip "Requires bitensor framework"
    end
end
