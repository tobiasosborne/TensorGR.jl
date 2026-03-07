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
include(joinpath(@__DIR__, "ground_truth.jl"))

@testset "Bench 08: Covariant Galileon" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)

        register_tensor!(reg, TensorProperties(
            name=:π, manifold=:M4, rank=(0, 0), symmetries=Any[]))

        π = Tensor(:π, TIndex[])

        # ── 8.1: L₂ = (∇π)² ──────────────────────────────────────────
        @testset "L_2 = (nabla pi)^2" begin
            L2 = grad_squared(π, :g)
            @test L2 isa TProduct
            @test length(L2.factors) == GALILEON_L2_FACTORS
        end

        # ── 8.2: L₃ = □π (∇π)² ───────────────────────────────────────
        @testset "L_3 = Box(pi) (nabla pi)^2" begin
            □π = box(π, :g)
            L3 = □π * grad_squared(π, :g)
            @test L3 isa TProduct
            @test count_terms(L3) >= 1
        end

        # ── 8.3: L₄ = (□π)² − (∇∇π)² ────────────────────────────────
        @testset "L_4 term count" begin
            □π = box(π, :g)
            nab_ab = covd_chain(π, [down(:a), down(:b)])
            nab_AB = covd_chain(π, [up(:a), up(:b)])
            L4 = □π * box(π, :g) - nab_ab * nab_AB

            @test L4 isa TSum
            @test count_terms(L4) == GALILEON_L4_TERMS
        end

        # ── 8.4: L₅ ──────────────────────────────────────────────────
        @testset "L_5 structure" begin
            □π = box(π, :g)

            box_cubed = □π * box(π, :g) * box(π, :g)

            nab_ab = covd_chain(π, [down(:a), down(:b)])
            nab_AB = covd_chain(π, [up(:a), up(:b)])
            term2 = □π * nab_ab * nab_AB

            nab_ab2 = covd_chain(π, [down(:a), down(:b)])
            nab_Bc  = covd_chain(π, [up(:b), down(:c)])
            nab_CA  = covd_chain(π, [up(:c), up(:a)])
            term3 = nab_ab2 * nab_Bc * nab_CA

            L5 = box_cubed - (3 // 1) * term2 + (2 // 1) * term3
            @test L5 isa TSum
            @test count_terms(L5) == 3  # three structural groups
        end

        # ── 8.5: EOM from L₄ ─────────────────────────────────────────
        @testset "EOM from L_4 derivative order" begin
            □π = box(π, :g)
            nab_ab = covd_chain(π, [down(:c), down(:d)])
            nab_AB = covd_chain(π, [up(:c), up(:d)])
            L4 = □π * box(π, :g) - nab_ab * nab_AB

            eom = variational_derivative(L4, :π)
            @test eom != TScalar(0 // 1)
            @test count_terms(eom) == GALILEON_L4_EOM_TERMS

            # Structural derivative order before CovD commutation
            dord = derivative_order(eom)
            @test dord == GALILEON_L4_EOM_DERIV_ORDER
        end
    end
end
