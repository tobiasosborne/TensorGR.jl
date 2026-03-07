# ============================================================================
# Benchmark 06: GW Stress-Energy in Modified Gravity
#
# Paper: Stein, Yunes (arXiv:1012.3144)
# Reproduces: Effective stress-energy tensor in Chern-Simons gravity
# Exercises: Levi-Civita tensor, Hodge dual, modified gravity Lagrangians
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ground_truth.jl"))

@testset "Bench 06: GW Stress-Energy (Chern-Simons)" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)

        # ── 6.1: Pontryagin density ───────────────────────────────────
        @testset "Pontryagin density construction" begin
            pont = pontryagin_density(:g; registry=reg)

            @test pont isa TProduct
            names = [f.name for f in pont.factors if f isa Tensor]
            @test :εg in names
            @test count(==(:Riem), names) == 2
            @test length(pont.factors) == CS_PONTRYAGIN_FACTORS
        end

        # ── 6.2: Euler density structure ─────────────────────────────
        @testset "Euler density = Riem^2 - 4Ric^2 + R^2" begin
            euler = euler_density(:g; registry=reg)
            @test euler isa TSum
            @test count_terms(euler) == CS_EULER_TERMS
        end

        # ── 6.3: CS scalar EOM = Pontryagin density ─────────────────
        @testset "CS scalar EOM = Pontryagin (identity)" begin
            register_tensor!(reg, TensorProperties(
                name=:ϑ, manifold=:M4, rank=(0, 0), symmetries=Any[]))
            ϑ = Tensor(:ϑ, TIndex[])

            cs_lagrangian = chern_simons_action(ϑ, :g; registry=reg)
            @test cs_lagrangian isa TProduct

            # δS/δϑ = ★(R∧R) = Pontryagin density
            eom_theta = variational_derivative(cs_lagrangian, :ϑ)
            pont = pontryagin_density(:g; registry=reg)

            # Identity test: EOM - Pont = 0
            diff = simplify(tsum(TensorExpr[eom_theta, tproduct(-1 // 1, TensorExpr[pont])]))
            @test diff == TScalar(0 // 1)
        end

        # ── 6.4: Isaacson averaging ──────────────────────────────────
        @testset "Isaacson stress-energy tensor" begin
            mp = define_metric_perturbation!(reg, :g, :h; curved=true)

            # δ²Ricci
            δ2Ric = δricci(mp, down(:a), down(:b), 2)
            @test δ2Ric != TScalar(0 // 1)

            # Isaacson averaging keeps only bilinear h·h terms
            T_eff = isaacson_average(δ2Ric, :h)
            @test T_eff != TScalar(0 // 1)
            @test count_terms(T_eff) == CS_ISAACSON_TERMS

            # Linear terms must average to zero
            δ1Ric = δricci(mp, down(:a), down(:b), 1)
            T_lin = isaacson_average(δ1Ric, :h)
            @test T_lin == TScalar(0 // 1)
        end
    end
end
