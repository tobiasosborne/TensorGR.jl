# ============================================================================
# Benchmark 02: xTras — Spin-2 Lagrangian on Flat Background
#
# Paper: Nutma (arXiv:1308.3493)
# Reproduces: Section 6.1 — most general gauge-invariant free spin-2 action
# Exercises: contraction enumeration, variational derivative, VarD, ansatz
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ground_truth.jl"))

@testset "Bench 02: xTras — Spin-2 Lagrangian" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

        background_solution!(reg, [:Ric, :RicScalar, :Ein])

        # ── 2.1: Contraction enumeration ────────────────────────────────
        @testset "5 independent contractions (manual)" begin
            t1 = Tensor(:h, [up(:a), up(:b)]) *
                 TDeriv(down(:b), TDeriv(down(:a),
                    Tensor(:h, [up(:c), down(:c)])))

            t2 = Tensor(:h, [up(:a), up(:b)]) *
                 TDeriv(down(:c), TDeriv(down(:b),
                    Tensor(:h, [down(:a), up(:c)])))

            t3 = Tensor(:h, [up(:a), down(:a)]) *
                 TDeriv(down(:c), TDeriv(down(:b),
                    Tensor(:h, [up(:b), up(:c)])))

            t4 = Tensor(:h, [up(:a), up(:b)]) *
                 TDeriv(down(:c), TDeriv(up(:c),
                    Tensor(:h, [down(:a), down(:b)])))

            t5 = Tensor(:h, [up(:a), down(:a)]) *
                 TDeriv(down(:c), TDeriv(up(:c),
                    Tensor(:h, [up(:b), down(:b)])))

            terms = [t1, t2, t3, t4, t5]
            @test length(terms) == XTRAS_SPIN2_CONTRACTIONS

            # All 5 are non-zero (independent)
            for t in terms
                @test simplify(t) != TScalar(0 // 1)
            end
        end

        # ── 2.2: Automatic contraction enumeration ──────────────────────
        @testset "all_contractions (automatic)" begin
            t_h = Tensor(:h, [down(:a), down(:b)])
            results = all_contractions([t_h, t_h], TIndex[])
            @test length(results) == XTRAS_HH_CONTRACTIONS
            for r in results
                @test simplify(r) != TScalar(0 // 1)
            end
        end

        # ── 2.3: Variational derivative of Lagrangian ───────────────────
        @testset "VarD of h^{ab} Box h_{ab}" begin
            L = Tensor(:h, [up(:a), up(:b)]) *
                TDeriv(down(:c), TDeriv(up(:c),
                    Tensor(:h, [down(:a), down(:b)])))
            eom = variational_derivative(L, :h)
            @test eom != TScalar(0 // 1)
            @test count_terms(eom) == XTRAS_VARD_HBOXH_TERMS
        end

        # ── 2.4: Gauss-Bonnet EOM vanishes ──────────────────────────────
        @testset "Gauss-Bonnet EOM = 0 (Lovelock)" begin
            Riem_sq = Tensor(:Riem, [down(:c), down(:d), down(:e), down(:f)]) *
                      Tensor(:Riem, [up(:c), up(:d), up(:e), up(:f)])
            Ric_sq = Tensor(:Ric, [down(:c), down(:d)]) *
                     Tensor(:Ric, [up(:c), up(:d)])
            R_sq = Tensor(:RicScalar, TIndex[]) * Tensor(:RicScalar, TIndex[])

            gb = tsum(TensorExpr[Riem_sq,
                                  tproduct(-4 // 1, TensorExpr[Ric_sq]),
                                  R_sq])

            gb_eom = metric_variation(gb, :g, down(:a), down(:b))
            gb_eom_s = simplify(gb_eom)
            @test gb_eom_s == TScalar(0 // 1)
        end
    end
end
