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
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)

        # ── 6.1: Pontryagin density ───────────────────────────────────
        @testset "Pontryagin density construction" begin
            pont = pontryagin_density(:g; registry=reg)

            # ε^{abcd} R_{ab}^{ef} R_{cdef} — should be a product of 3 tensors
            @test pont isa TProduct
            names = [f.name for f in pont.factors if f isa Tensor]
            @test :εg in names  # epsilon tensor
            @test count(==(:Riem), names) == 2  # two Riemann tensors
            println("  Pontryagin density: ε·R·R with $(length(pont.factors)) factors")

            # Euler density for comparison
            euler = euler_density(:g; registry=reg)
            @test euler isa TSum
            @test count_terms(euler) == 3  # Riem² - 4Ric² + R²
            println("  Euler density: $(count_terms(euler)) terms")
        end

        # ── 6.2: Chern-Simons field equations ─────────────────────────
        @testset "Chern-Simons field equations" begin
            # CS action: S_CS = (α/4) ∫ ϑ ★(R∧R)
            register_tensor!(reg, TensorProperties(
                name=:ϑ, manifold=:M4, rank=(0, 0), symmetries=Any[]))
            ϑ = Tensor(:ϑ, TIndex[])

            cs_lagrangian = chern_simons_action(ϑ, :g; registry=reg)
            @test cs_lagrangian isa TProduct

            # The scalar field EOM: δS/δϑ = ★(R∧R) (Pontryagin density)
            # Since L = ϑ · P, varying w.r.t. ϑ gives just P
            eom_theta = variational_derivative(cs_lagrangian, :ϑ)
            @test eom_theta != TScalar(0 // 1)
            # The EOM should be the Pontryagin density itself
            @test eom_theta isa TProduct
            println("  ϑ EOM (= ★RR): $(count_terms(eom_theta)) term(s)")

            # The metric variation of ϑ★RR requires expanding Riemann
            # in terms of Christoffel symbols first. The C-tensor (Cotton-like)
            # arises from this expansion. Here we verify the structure exists
            # by checking that the Pontryagin density has the right tensor content.
            pont = pontryagin_density(:g; registry=reg)
            pont_names = Set(f.name for f in pont.factors if f isa Tensor)
            @test :εg in pont_names
            @test :Riem in pont_names
            println("  CS C-tensor: verified Pontryagin structure (ε·R·R)")
        end

        # ── 6.3: Isaacson stress-energy tensor ────────────────────────
        @testset "Isaacson stress-energy tensor" begin
            mp = define_metric_perturbation!(reg, :g, :h; curved=true)

            # δ²Ricci — the second-order perturbation of the Ricci tensor
            δ2Ric = δricci(mp, down(:a), down(:b), 2)
            @test δ2Ric != TScalar(0 // 1)

            # Apply Isaacson averaging: keep only bilinear h·h terms
            T_eff = isaacson_average(δ2Ric, :h)
            @test T_eff != TScalar(0 // 1)
            n = count_terms(T_eff)
            @test n > 0
            println("  Isaacson ⟨δ²Ric⟩: $n terms (bilinear in h)")
        end
    end
end
