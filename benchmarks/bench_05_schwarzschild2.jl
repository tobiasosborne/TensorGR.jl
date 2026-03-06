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
    @testset "δ²Ricci on curved background" begin
        @test_skip "Requires curved-background perturbation at scale"
    end
    @testset "Zerilli master equation" begin
        @test_skip "Requires spherical harmonic decomposition"
    end
    @testset "Regge-Wheeler master equation" begin
        @test_skip "Requires spherical harmonic decomposition"
    end
end
