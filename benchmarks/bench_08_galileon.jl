# ============================================================================
# Benchmark 08: Covariant Galileon Lagrangians
#
# Paper: Deffayet, Esposito-Farese, Vikman (arXiv:0901.1314)
# Reproduces: L_1 through L_5 Galileon Lagrangians,
#             EOM structure (2nd order on flat background)
# Exercises: higher-derivative scalar-tensor, CovD, IBP
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))

@testset "Bench 08: Covariant Galileon" begin
    @testset "L_2 = (nabla pi)^2" begin
        @test_skip "Requires scalar field CovD helpers"
    end
    @testset "L_3 = Box(pi) (nabla pi)^2" begin
        @test_skip "Requires Box operator for scalars"
    end
    @testset "L_4 (4 terms, 6 derivatives)" begin
        @test_skip "Requires nested CovD products"
    end
    @testset "L_5 (7 terms, 8 derivatives)" begin
        @test_skip "Requires nested CovD products"
    end
    @testset "EOM from L_4 is 2nd order on flat bg" begin
        @test_skip "Requires VarD + CovD commutation"
    end
end
