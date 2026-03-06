# ============================================================================
# Benchmark 06: GW Stress-Energy in Modified Gravity
#
# Paper: Stein, Yunes (arXiv:1012.3144)
# Reproduces: Effective stress-energy tensor in Chern-Simons gravity
# Exercises: Levi-Civita tensor, Hodge dual, modified gravity Lagrangians
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))

@testset "Bench 06: GW Stress-Energy (Chern-Simons)" begin
    @testset "Pontryagin density construction" begin
        @test_skip "Requires epsilon tensor in curvature products"
    end
    @testset "Chern-Simons field equations" begin
        @test_skip "Requires Pontryagin density + metric variation"
    end
    @testset "Isaacson stress-energy tensor" begin
        @test_skip "Requires second-order perturbation averaging"
    end
end
