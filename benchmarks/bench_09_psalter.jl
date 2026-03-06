# ============================================================================
# Benchmark 09: PSALTer — Spin-Projection Operators
#
# Paper: Barker, Marzo, Rigouzzo (arXiv:2406.09500)
# Reproduces: Spin-projection operator algebra, Fierz-Pauli spectrum
# Exercises: projectors, SVT decomposition, symbolic matrix inversion
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))

@testset "Bench 09: PSALTer — Spin-Projection Operators" begin
    @testset "Spin-2 projector algebra" begin
        @test_skip "Requires full spin-projection operator framework"
    end
    @testset "Fierz-Pauli particle spectrum" begin
        @test_skip "Requires wave operator decomposition"
    end
end
