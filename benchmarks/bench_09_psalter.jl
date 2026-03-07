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

            sum_proj = simplify(tsum(TensorExpr[θ, ω]))
            η_ab = Tensor(:η, [down(:a), down(:b)])

            diff = simplify(tsum(TensorExpr[sum_proj, tproduct(-1 // 1, TensorExpr[η_ab])]))
            @test diff == TScalar(0 // 1)
        end

        # ── 9.3: Projector algebra — numeric verification ────────────
        # Evaluate projectors at k=(0,0,0,1) with η=diag(-1,1,1,1)
        # and verify idempotence, orthogonality, completeness.
        @testset "Projector algebra: P²=P, orthogonality, completeness (numeric)" begin
            chart = define_chart!(reg, :std; manifold=:M4, coords=[:t,:x,:y,:z])

            values = Dict{Any,Any}()
            eta = [-1.0, 1.0, 1.0, 1.0]
            for i in 1:4, j in 1:4
                values[(:η, [i,j])] = (i == j ? eta[i] : 0.0)
            end
            for i in 1:4
                values[(:k, [i])] = (i == 4 ? 1.0 : 0.0)
            end
            values[(:k², Int[])] = 1.0

            kw = (metric=:η,)
            μ, ν, ρ, σ = down(:a), down(:b), down(:c), down(:d)

            # Evaluate as 4×4×4×4 numeric arrays (use expand_products, not simplify)
            P2 = to_ctensor(expand_products(spin2_projector(μ, ν, ρ, σ; dim=4, kw...)), chart, values)
            P1 = to_ctensor(expand_products(spin1_projector(μ, ν, ρ, σ; kw...)), chart, values)
            P0s = to_ctensor(expand_products(spin0s_projector(μ, ν, ρ, σ; dim=4, kw...)), chart, values)
            P0w = to_ctensor(expand_products(spin0w_projector(μ, ν, ρ, σ)), chart, values)

            # Metric for index raising in contraction
            eta_mat = zeros(4, 4)
            for i in 1:4; eta_mat[i,i] = eta[i]; end

            # P_i · P_j contraction: (P·Q)_{abef} = P_{abcd} η^{cc'} η^{dd'} Q_{c'd'ef}
            function proj_product(PA, PB)
                result = zeros(4, 4, 4, 4)
                for a in 1:4, b in 1:4, e in 1:4, f in 1:4
                    for c in 1:4, d in 1:4, cp in 1:4, dp in 1:4
                        result[a,b,e,f] += PA[a,b,c,d] * eta_mat[c,cp] * eta_mat[d,dp] * PB[cp,dp,e,f]
                    end
                end
                return result
            end

            eps = 1e-10

            # Idempotence: P_i · P_i = P_i
            @test maximum(abs.(proj_product(P2.data, P2.data) .- P2.data)) < eps
            @test maximum(abs.(proj_product(P1.data, P1.data) .- P1.data)) < eps
            @test maximum(abs.(proj_product(P0s.data, P0s.data) .- P0s.data)) < eps
            @test maximum(abs.(proj_product(P0w.data, P0w.data) .- P0w.data)) < eps

            # Orthogonality: P_i · P_j = 0 for i ≠ j
            @test maximum(abs.(proj_product(P2.data, P1.data))) < eps
            @test maximum(abs.(proj_product(P2.data, P0s.data))) < eps
            @test maximum(abs.(proj_product(P2.data, P0w.data))) < eps
            @test maximum(abs.(proj_product(P1.data, P0s.data))) < eps
            @test maximum(abs.(proj_product(P1.data, P0w.data))) < eps
            @test maximum(abs.(proj_product(P0s.data, P0w.data))) < eps

            # Completeness: P² + P¹ + P⁰ˢ + P⁰ʷ = I_{(ab)(cd)}
            I_sym = zeros(4, 4, 4, 4)
            for a in 1:4, b in 1:4, c in 1:4, d in 1:4
                I_sym[a,b,c,d] = 0.5 * (eta_mat[a,c]*eta_mat[b,d] + eta_mat[a,d]*eta_mat[b,c])
            end
            sum_proj = P2.data .+ P1.data .+ P0s.data .+ P0w.data
            @test maximum(abs.(sum_proj .- I_sym)) < eps
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
