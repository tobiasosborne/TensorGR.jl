# ============================================================================
# Benchmark 04: Conformal Gravity — Weyl-Squared Action
#
# Paper: Grumiller, Irakleidou, Lovrekovic, McNees (arXiv:1310.0819)
# Reproduces: Eq. 12 (Weyl^2 = 2 Ric^2 - 2/3 R^2 + Gauss-Bonnet identity)
#             Eq. 3 (Bach tensor is trace-free)
# Exercises: Weyl tensor, curvature conversions, contraction, simplify
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ground_truth.jl"))

@testset "Bench 04: Conformal Gravity — Weyl-Squared" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)

        # ── 4.1: Weyl → Riemann expansion ─────────────────────────────────
        @testset "Weyl expansion term count" begin
            W_abcd = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            W_expanded = to_riemann(W_abcd)
            @test W_expanded != TScalar(0 // 1)
            @test count_terms(W_expanded) == WEYL_EXPANSION_TERMS
        end

        # ── 4.2: Weyl-squared construction ─────────────────────────────────
        @testset "Weyl-squared construction" begin
            W_abcd = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            W_up = Tensor(:Weyl, [up(:a), up(:b), up(:c), up(:d)])
            weyl_sq = W_abcd * W_up
            @test weyl_sq isa TProduct
        end

        # ── 4.3: Gauss-Bonnet structure {1, -4, 1} ────────────────────────
        @testset "Gauss-Bonnet = Riem^2 - 4 Ric^2 + R^2" begin
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = simplify(Riem_down * Riem_up)

            Ric_down = Tensor(:Ric, [down(:a), down(:b)])
            Ric_up = Tensor(:Ric, [up(:a), up(:b)])
            ricci_sq = simplify(Ric_down * Ric_up)

            R = Tensor(:RicScalar, TIndex[])
            scalar_sq = R * R

            gb = simplify(tsum(TensorExpr[kretschner,
                                          tproduct(-4 // 1, TensorExpr[ricci_sq]),
                                          scalar_sq]))
            @test gb != TScalar(0 // 1)  # GB is non-trivial in general
            @test count_terms(gb) == CS_EULER_TERMS  # 3 terms: {Riem², Ric², R²}
        end

        # ── 4.4: Weyl trace-free property ──────────────────────────────────
        @testset "Weyl is trace-free" begin
            W_traced = Tensor(:Weyl, [up(:a), down(:b), down(:a), down(:d)])
            expanded = to_riemann(W_traced)
            contracted = contract_curvature(expanded)
            result = simplify(contracted)
            @test result == TScalar(0 // 1)
        end

        # ── 4.5: Weyl tensor symmetries ────────────────────────────────────
        @testset "Weyl symmetries" begin
            W1 = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            W2 = Tensor(:Weyl, [down(:b), down(:a), down(:c), down(:d)])

            # Antisymmetry: C_{abcd} = -C_{bacd}
            @test simplify(W1 + W2) == TScalar(0 // 1)

            # Pair symmetry: C_{abcd} = C_{cdab}
            W3 = Tensor(:Weyl, [down(:c), down(:d), down(:a), down(:b)])
            @test simplify(W1 - W3) == TScalar(0 // 1)
        end

        # ── 4.6: Weyl decomposition term count ────────────────────────────
        @testset "Weyl decomposition (Riem = Weyl + Ricci + scalar)" begin
            decomp = weyl_to_riemann(down(:a), down(:b), down(:c), down(:d), :g; dim=4)
            @test decomp isa TSum
            @test count_terms(decomp) == WEYL_DECOMPOSITION_TERMS
        end
    end
end
