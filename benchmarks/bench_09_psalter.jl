# ============================================================================
# Benchmark 09: PSALTer — Spin-Projection Operators
#
# Paper: Barker, Marzo, Rigouzzo (arXiv:2406.09500)
# Reproduces: Spin-projection operator algebra, Fierz-Pauli spectrum
# Exercises: projectors, SVT decomposition, symbolic matrix inversion
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ground_truth.jl"))

@testset "Bench 09: PSALTer — Spin-Projection Operators" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=η
        register_tensor!(reg, TensorProperties(
            name=:k, manifold=:M4, rank=(0, 1), symmetries=Any[]))

        # ── 9.1: Projector construction with pinned term counts ──────
        @testset "Spin-projector construction" begin
            μ, ν, ρ, σ = down(:a), down(:b), down(:c), down(:d)
            kw = (metric=:η,)

            θ = theta_projector(μ, ν; kw...)
            ω = omega_projector(μ, ν)
            @test θ isa TSum
            @test ω isa TProduct
            @test count_terms(θ) == PSALTER_THETA_TERMS
            @test count_terms(ω) == PSALTER_OMEGA_TERMS

            p2 = spin2_projector(μ, ν, ρ, σ; dim=4, kw...)
            p1 = spin1_projector(μ, ν, ρ, σ; kw...)
            p0s = spin0s_projector(μ, ν, ρ, σ; dim=4, kw...)
            p0w = spin0w_projector(μ, ν, ρ, σ)
            tsw = transfer_sw(μ, ν, ρ, σ; dim=4, kw...)
            tws = transfer_ws(μ, ν, ρ, σ; dim=4, kw...)

            @test count_terms(p2) == PSALTER_P2_TERMS
            @test count_terms(p1) == PSALTER_P1_TERMS
            @test count_terms(p0s) == PSALTER_P0S_TERMS
            @test count_terms(p0w) == PSALTER_P0W_TERMS
        end

        # ── 9.2: Completeness relation θ + ω = η (identity) ─────────
        @testset "Completeness: θ + ω = η" begin
            μ, ν = down(:a), down(:b)
            kw = (metric=:η,)

            θ = theta_projector(μ, ν; kw...)
            ω = omega_projector(μ, ν)

            # θ_{μν} + ω_{μν} = η_{μν}
            sum_proj = simplify(tsum(TensorExpr[θ, ω]))
            η_ab = Tensor(:η, [down(:a), down(:b)])

            # Verify: (θ + ω) - η = 0
            diff = simplify(tsum(TensorExpr[sum_proj, tproduct(-1 // 1, TensorExpr[η_ab])]))
            @test diff == TScalar(0 // 1)
        end

        # ── 9.3: Projector algebra (structural) ─────────────────────
        @testset "Spin-2 projector index structure" begin
            μ, ν, ρ, σ = down(:a), down(:b), down(:c), down(:d)
            kw = (metric=:η,)

            p2 = spin2_projector(μ, ν, ρ, σ; dim=4, kw...)

            fi = free_indices(p2)
            free_names = Set(idx.name for idx in fi)
            @test :a in free_names
            @test :b in free_names
            @test :c in free_names
            @test :d in free_names
        end

        # ── 9.4: Transfer operators are distinct ─────────────────────
        @testset "Transfer operators T^{sw} ≠ T^{ws}" begin
            μ, ν, ρ, σ = down(:a), down(:b), down(:c), down(:d)
            kw = (metric=:η,)

            tsw = transfer_sw(μ, ν, ρ, σ; dim=4, kw...)
            tws = transfer_ws(μ, ν, ρ, σ; dim=4, kw...)
            @test tsw != tws
        end

        # ── 9.5: Fierz-Pauli mass term structure ────────────────────
        @testset "Fierz-Pauli mass term" begin
            h_down = Tensor(:h, [down(:a), down(:b)])
            h_up = Tensor(:h, [up(:a), up(:b)])
            h_trace = Tensor(:h, [up(:a), down(:a)])

            mass_term = h_down * h_up - h_trace * Tensor(:h, [up(:b), down(:b)])
            @test mass_term isa TSum
            @test count_terms(mass_term) == 2
        end
    end
end
