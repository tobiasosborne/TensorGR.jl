#= Weyl completeness identity validation.
#
# Verify C_{abcd}C^{abcd} = R_{abcd}R^{abcd} - 2R_{ab}R^{ab} + R²/3  in d=4.
#
# Ground truth: standard GR identity (Wald, General Relativity, Eq 3.2.28;
#   Fulling et al., Class. Quantum Grav. 9 (1992) 1151).
#
# Method: expand Weyl tensor via weyl_to_riemann, form C²=W·W, simplify,
# and verify the result equals K - 2Ric² + R²/3.
=#

@testset "Weyl Completeness Identity" begin

    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)

        # Build C² by expanding Weyl into Riemann decomposition
        W_low  = weyl_to_riemann(down(:a), down(:b), down(:c), down(:d), :g; dim=4)
        W_high = weyl_to_riemann(up(:a), up(:b), up(:c), up(:d), :g; dim=4)
        Weyl_sq = tproduct(1//1, TensorExpr[W_low, W_high])

        result = simplify(Weyl_sq; registry=reg)

        # Expected: K - 2Ric² + R²/3
        Riem_sq = tproduct(1//1, TensorExpr[
            Tensor(:Riem, [up(:e), up(:f), up(:g), up(:h)]),
            Tensor(:Riem, [down(:e), down(:f), down(:g), down(:h)])])
        Ric_sq = tproduct(1//1, TensorExpr[
            Tensor(:Ric, [up(:i), up(:j)]),
            Tensor(:Ric, [down(:i), down(:j)])])
        R_sq = tproduct(1//1, TensorExpr[
            Tensor(:RicScalar, TIndex[]),
            Tensor(:RicScalar, TIndex[])])

        expected = tsum(TensorExpr[
            Riem_sq,
            tproduct(-2//1, TensorExpr[Ric_sq]),
            tproduct(1//3, TensorExpr[R_sq])])

        expected_s = simplify(expected; registry=reg)

        # STRING MATCH: C² - (K - 2Ric² + R²/3) = 0
        diff = tsum(TensorExpr[result, tproduct(-1//1, TensorExpr[expected_s])])
        diff_s = simplify(diff; registry=reg)

        @test diff_s == TScalar(0 // 1)
    end

end
