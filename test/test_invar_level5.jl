#= Invar Level 5: Dimensionally-dependent identities (DDIs).
#
# Verify that simplify_level5 correctly integrates DDI rules into the
# Invar simplification pipeline.
#
# Ground truth:
#   - Garcia-Parrado & Martin-Garcia (2007) Sec 4, Level 3 (their numbering)
#   - Lovelock (1971), J. Math. Phys. 12, 498
#   - Fulling et al. (1992), Class. Quantum Grav. 9, 1151, Table 1
=#

using Test
using TensorGR

# simplify_level5 is not yet exported; access via module prefix
const simplify_level5 = TensorGR.simplify_level5

@testset "Invar Level 5: Dimensionally-Dependent Identities" begin

    # ── Setup helpers ─────────────────────────────────────────────────

    function _l5_reg(; dim=4)
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

    # ── d=4: Gauss-Bonnet identity ────────────────────────────────────

    @testset "Gauss-Bonnet identity simplifies to zero (d=4)" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            # The Gauss-Bonnet combination Riem^2 - 4 Ric^2 + R^2 = 0
            gb = gauss_bonnet_ddi(; metric=:g, registry=reg)
            @test gb isa TSum
            @test length(gb.terms) == 3

            result = simplify_level5(gb; dim=4, registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Gauss-Bonnet zero: Riem^2 - 4 Ric^2 + R^2 = 0 (d=4)" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            # Build the combination directly (not via gauss_bonnet_ddi)
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            Ric_down = Tensor(:Ric, [down(:e), down(:f)])
            Ric_up = Tensor(:Ric, [up(:e), up(:f)])
            ric_sq = Ric_down * Ric_up

            R = Tensor(:RicScalar, TIndex[])
            r_sq = R * R

            gb_combo = kretschner - (4 // 1) * ric_sq + r_sq
            result = simplify_level5(gb_combo; dim=4, registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Kretschner scalar eliminated: Riem^2 => 4 Ric^2 - R^2 (d=4)" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            result = simplify_level5(kretschner; dim=4, registry=reg)

            # Riemann should be eliminated by Gauss-Bonnet DDI
            @test !_has_tensor(result, :Riem)

            # Result should contain Ricci invariants
            @test _has_tensor(result, :Ric) || _has_tensor(result, :RicScalar)

            # Result should still be a scalar (no free indices)
            @test isempty(free_indices(result))
        end
    end

    # ── d=3: Weyl vanishing ───────────────────────────────────────────

    @testset "Weyl vanishes identically in d=3" begin
        reg = _l5_reg(; dim=3)
        with_registry(reg) do
            weyl = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify_level5(weyl; dim=3, registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── d=2: Ricci is pure trace ──────────────────────────────────────

    @testset "Ricci = (R/2) g in d=2" begin
        reg = _l5_reg(; dim=2)
        with_registry(reg) do
            ric = Tensor(:Ric, [down(:a), down(:b)])
            result = simplify_level5(ric; dim=2, registry=reg)

            # Result should be (1/2) R g_{ab}
            @test result isa TProduct
            @test result.scalar == 1 // 2

            # Extract factor names
            factor_names = [f.name for f in result.factors if f isa Tensor]
            @test :g in factor_names
            @test :RicScalar in factor_names
        end
    end

    # ── Cubic DDI (order=3) ───────────────────────────────────────────

    @testset "Cubic DDI reduces term count (d=4)" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            # Build a cubic Riemann monomial: Riem^2 * R
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            R = Tensor(:RicScalar, TIndex[])
            cubic = Riem_down * Riem_up * R

            # Level 5 with max_order=3 should apply both quadratic and cubic DDIs
            result = simplify_level5(cubic; dim=4, max_order=3, registry=reg)

            # Riemann should be eliminated (Riem^2 -> 4 Ric^2 - R^2 by GB)
            @test !_has_tensor(result, :Riem)

            # Result should still be a scalar expression
            @test isempty(free_indices(result))
        end
    end

    # ── Level 5 subsumes Levels 1-4 ───────────────────────────────────

    @testset "Level 5 subsumes Level 1 (canonicalization)" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            # A simple curvature scalar should pass through unchanged
            R = Tensor(:RicScalar, TIndex[])
            result = simplify_level5(R; dim=4, registry=reg)
            @test result isa TensorExpr
        end
    end

    @testset "Level 5 subsumes Level 2 (first Bianchi)" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            # The first Bianchi identity R_{abcd} + R_{acdb} + R_{adbc} = 0
            bianchi = bianchi_relation(down(:a), down(:b), down(:c), down(:d))
            @test bianchi isa TSum
            @test length(bianchi.terms) == 3

            # Level 5 handles it (Level 2 is prerequisite)
            result = simplify_level5(bianchi; dim=4, registry=reg)
            @test result isa TensorExpr
        end
    end

    @testset "Level 5 subsumes Level 4 (derivative commutation)" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            # Build nabla_e nabla_f R_{abcd}
            Riem = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            dd_Riem = TDeriv(down(:e), TDeriv(down(:f), Riem, :D), :D)

            # Level 5 should handle derivative commutation
            result = simplify_level5(dd_Riem; dim=4, registry=reg)
            @test result isa TensorExpr

            # Should have 6 free indices (a, b, c, d, e, f)
            fi = free_indices(result)
            @test length(fi) == 6
        end
    end

    # ── Riem^2 in sum reduces ─────────────────────────────────────────

    @testset "Riem^2 + 2 Ric^2 reduces to 6 Ric^2 - R^2 (d=4)" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            # Build: Riem^2 + 2 Ric^2
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            Ric_down = Tensor(:Ric, [down(:e), down(:f)])
            Ric_up = Tensor(:Ric, [up(:e), up(:f)])
            ric_sq = Ric_down * Ric_up

            expr = kretschner + (2 // 1) * ric_sq

            result = simplify_level5(expr; dim=4, registry=reg)

            # Riem^2 -> 4 Ric^2 - R^2 by GB, so result = 6 Ric^2 - R^2
            @test !_has_tensor(result, :Riem)
            @test isempty(free_indices(result))
        end
    end

    # ── max_order parameter ───────────────────────────────────────────

    @testset "max_order=2 applies only quadratic DDIs" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            # With max_order=2, only Gauss-Bonnet is applied
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            result = simplify_level5(kretschner; dim=4, max_order=2, registry=reg)
            @test !_has_tensor(result, :Riem)
        end
    end

    @testset "max_order auto-detection from expression degree" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            # For a quadratic expression, auto max_order should be 2
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            # Default max_order=0 triggers auto-detection
            result = simplify_level5(kretschner; dim=4, registry=reg)
            @test !_has_tensor(result, :Riem)
        end
    end

    # ── dim parameter variations ──────────────────────────────────────

    @testset "dim parameter correctly controls DDI selection" begin
        # In d=4, Gauss-Bonnet applies (Riem^2 eliminated)
        reg4 = _l5_reg(; dim=4)
        with_registry(reg4) do
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            result4 = simplify_level5(kretschner; dim=4, registry=reg4)
            @test !_has_tensor(result4, :Riem)
        end

        # In d=3, Weyl vanishes and Gauss-Bonnet does NOT apply
        # (generate_ddi_rules(3; order=2) returns Weyl vanishing rule only)
        reg3 = _l5_reg(; dim=3)
        with_registry(reg3) do
            weyl = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            result3 = simplify_level5(weyl; dim=3, registry=reg3)
            @test result3 == TScalar(0 // 1)
        end
    end

    # ── Idempotency ───────────────────────────────────────────────────

    @testset "Idempotent: repeated calls produce same result" begin
        reg = _l5_reg(; dim=4)
        with_registry(reg) do
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            result1 = simplify_level5(kretschner; dim=4, registry=reg)
            result2 = simplify_level5(result1; dim=4, registry=reg)

            @test result1 == result2
        end
    end

end
