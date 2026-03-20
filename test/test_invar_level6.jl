#= Invar Level 6: Dual invariant product relations.
#
# Verify that simplify_level6 correctly applies dual Riemann reduction
# rules after Levels 1-5 in the Invar simplification pipeline.
#
# Ground truth:
#   - Garcia-Parrado & Martin-Garcia (2007) Sec 4, Level 6
#   - Zakhary & McIntosh (1997) GRG 29, 539
#   - Identity: *R*_{abcd} = R_{abcd} in d=4
=#

using Test
using TensorGR

# simplify_level6 and helpers are not yet exported; access via module prefix
const simplify_level6 = TensorGR.simplify_level6
const double_dual_identity = TensorGR.double_dual_identity
const register_dual_rules! = TensorGR.register_dual_rules!
const has_dual_rules = TensorGR.has_dual_rules

@testset "Invar Level 6: Dual Invariant Product Relations" begin

    # ── Setup helpers ─────────────────────────────────────────────────

    function _l6_reg(; dim=4)
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=dim metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
        end
        reg
    end

    """Check if an expression contains a tensor with the given name."""
    function _has_tensor(expr::TensorExpr, name::Symbol)
        found = Ref(false)
        walk(expr) do node
            if node isa Tensor && node.name == name
                found[] = true
            end
            node
        end
        found[]
    end

    # ── double_dual_identity construction ─────────────────────────────

    @testset "double_dual_identity returns well-formed expression" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            dd = double_dual_identity(; metric=:g, registry=reg)

            # Should be a TensorExpr (specifically a TSum: (*R*)^2 - R^2)
            @test dd isa TensorExpr

            # Should have no free indices (it is a scalar identity)
            @test isempty(free_indices(dd))

            # Should contain Riemann tensors
            @test _has_tensor(dd, :Riem)
        end
    end

    @testset "double_dual_identity contains epsilon tensors" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            dd = double_dual_identity(; metric=:g, registry=reg)

            # The double-dual term involves epsilon tensors
            has_eps = _has_tensor(dd, :εg)
            @test has_eps
        end
    end

    # ── register_dual_rules! ──────────────────────────────────────────

    @testset "register_dual_rules! creates rules (d=4)" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            rules = register_dual_rules!(reg; dim=4, metric=:g)

            @test !isempty(rules)
            @test all(r -> r isa RewriteRule, rules)
        end
    end

    @testset "register_dual_rules! returns empty for non-4D" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            # d=3: no dual rules
            rules3 = register_dual_rules!(reg; dim=3, metric=:g)
            @test isempty(rules3)

            # d=5: no dual rules (currently)
            rules5 = register_dual_rules!(reg; dim=5, metric=:g)
            @test isempty(rules5)
        end
    end

    @testset "has_dual_rules tracking" begin
        reg = _l6_reg(; dim=4)

        # Before Level 6: no dual rules
        @test !has_dual_rules(reg; dim=4, metric=:g)

        # After simplify_level6 call: dual rules should be registered
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            simplify_level6(R; dim=4, registry=reg)
        end
        @test has_dual_rules(reg; dim=4, metric=:g)
    end

    # ── simplify_level6 subsumes Levels 1-5 ───────────────────────────

    @testset "Level 6 subsumes Level 1 (canonicalization)" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            result = simplify_level6(R; dim=4, registry=reg)
            @test result isa TensorExpr
        end
    end

    @testset "Level 6 subsumes Level 2 (first Bianchi)" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            bianchi = bianchi_relation(down(:a), down(:b), down(:c), down(:d))
            @test bianchi isa TSum
            @test length(bianchi.terms) == 3

            result = simplify_level6(bianchi; dim=4, registry=reg)
            @test result isa TensorExpr
        end
    end

    @testset "Level 6 subsumes Level 5 (DDIs: Gauss-Bonnet)" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            # Gauss-Bonnet identity: Riem^2 - 4 Ric^2 + R^2 = 0
            gb = gauss_bonnet_ddi(; metric=:g, registry=reg)
            result = simplify_level6(gb; dim=4, registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Level 6 subsumes Level 5 (Kretschner elimination)" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            result = simplify_level6(kretschner; dim=4, registry=reg)

            # Riemann should be eliminated by Gauss-Bonnet DDI
            @test !_has_tensor(result, :Riem)

            # Result should contain Ricci invariants
            @test _has_tensor(result, :Ric) || _has_tensor(result, :RicScalar)

            # Still a scalar
            @test isempty(free_indices(result))
        end
    end

    # ── Level 6 idempotency ──────────────────────────────────────────

    @testset "Idempotent: repeated calls produce same result" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            Ric_down = Tensor(:Ric, [down(:a), down(:b)])
            Ric_up = Tensor(:Ric, [up(:a), up(:b)])
            ric_sq = Ric_down * Ric_up

            result1 = simplify_level6(ric_sq; dim=4, registry=reg)
            result2 = simplify_level6(result1; dim=4, registry=reg)

            @test result1 == result2
        end
    end

    # ── Pontryagin density ───────────────────────────────────────────

    @testset "Pontryagin density is a well-formed scalar" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            pont = pontryagin_density(:g; registry=reg)

            # Pontryagin density should have no free indices
            @test isempty(free_indices(pont))

            # It involves Riemann tensors
            @test _has_tensor(pont, :Riem)

            # It involves epsilon tensors
            @test _has_tensor(pont, :εg)
        end
    end

    @testset "Pontryagin density passes through Level 6" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            pont = pontryagin_density(:g; registry=reg)
            result = simplify_level6(pont; dim=4, registry=reg)

            # Result should still be a valid expression
            @test result isa TensorExpr

            # Should still be a scalar (no free indices)
            @test isempty(free_indices(result))
        end
    end

    # ── DualRInv integration ─────────────────────────────────────────

    @testset "DualRInv to_tensor_expr produces correct structure" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            # Construct the Pontryagin density as a DualRInv
            p = pontryagin_rinv()
            @test p isa DualRInv
            @test p.base.degree == 2

            # Convert to TensorExpr
            expr = to_tensor_expr(p; registry=reg, metric=:g)
            @test expr isa TensorExpr

            # Should be a scalar (no free indices)
            @test isempty(free_indices(expr))
        end
    end

    @testset "Double-dual DualRInv construction" begin
        # Construct a DualRInv with double dual on both factors
        kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

        # Double dual on factor 1
        dd1 = double_dual(kr, 1)
        @test dd1 isa DualRInv
        @test dd1.dual_positions == [(1, :double)]

        # Both-factor double dual via constructor
        dd_both = DualRInv(kr, [(1, :double), (2, :double)])
        @test dd_both isa DualRInv
        @test length(dd_both.dual_positions) == 2
    end

    @testset "Double-dual DualRInv to_tensor_expr" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            dd_both = DualRInv(kr, [(1, :double), (2, :double)])

            expr = to_tensor_expr(dd_both; registry=reg, metric=:g)
            @test expr isa TensorExpr

            # Should have epsilon tensors (4 of them: 2 per double-dual factor)
            eps_count = Ref(0)
            walk(expr) do node
                if node isa Tensor && node.name == :εg
                    eps_count[] += 1
                end
                node
            end
            @test eps_count[] == 4

            # Should be a scalar
            @test isempty(free_indices(expr))
        end
    end

    # ── Level 6 on expressions without duals ─────────────────────────

    @testset "Level 6 on pure Ricci expression (no dual)" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            # Pure Ricci scalar squared: no duals involved
            R = Tensor(:RicScalar, TIndex[])
            r_sq = R * R
            result = simplify_level6(r_sq; dim=4, registry=reg)

            # Should pass through unchanged (it is already simplified)
            @test result isa TensorExpr
            @test isempty(free_indices(result))
        end
    end

    @testset "Level 6 on Ricci squared (no dual)" begin
        reg = _l6_reg(; dim=4)
        with_registry(reg) do
            Ric_down = Tensor(:Ric, [down(:e), down(:f)])
            Ric_up = Tensor(:Ric, [up(:e), up(:f)])
            ric_sq = Ric_down * Ric_up

            result = simplify_level6(ric_sq; dim=4, registry=reg)
            @test result isa TensorExpr
            @test isempty(free_indices(result))
        end
    end

    # ── Dimension variations ─────────────────────────────────────────

    @testset "Level 6 in d=3 (no dual rules, Weyl vanishes)" begin
        reg = _l6_reg(; dim=3)
        with_registry(reg) do
            weyl = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify_level6(weyl; dim=3, registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

end
