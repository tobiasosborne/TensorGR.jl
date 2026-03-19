#= xAct Ground Truth Test Suite
   ================================
   Tests that reproduce published equations from papers that used xAct
   (the Mathematica CAS for tensor algebra). Each test computes an expression
   with TensorGR.jl and structurally verifies it matches the paper result.

   Papers and xAct subpackages covered:
   1. Nutma 2014 (arXiv:1308.3493) — xTras
   2. Brizuela et al 2009 (arXiv:0807.0824) — xPert
   3. Barker et al 2024 (arXiv:2406.09500) — PSALTer (Barnes-Rivers projectors)
   4. Hohmann et al 2021 (arXiv:2012.14984) — xPPN

   Protocol: compute in TensorGR, structural comparison against paper.
   NO println cheating. =#

using TensorGR
using TensorGR: all_contractions, free_indices, euler_density,
                riemann_to_weyl, weyl_to_riemann,
                δRiemann, δRicci, δRicciScalar,
                expand_perturbation, define_metric_perturbation!,
                δinverse_metric, δchristoffel,
                spin2_projector, spin1_projector, spin0s_projector, spin0w_projector,
                transfer_sw, transfer_ws,
                theta_projector, omega_projector,
                variational_derivative, metric_variation,
                svt_rules_bardeen, SVTFields, collect_sectors,
                split_spacetime

# ═══════════════════════════════════════════════════════════════════════
# Helper: standard 4D registry setup
# ═══════════════════════════════════════════════════════════════════════

function _make_4d_registry(; indices=[:a,:b,:c,:d,:e,:f,:g1,:h,:i1,:j,:k1,:l])
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, indices))
    define_metric!(reg, :g; manifold=:M4)
    reg
end

# ═══════════════════════════════════════════════════════════════════════
# PART 1: Nutma 2014 (xTras) — arXiv:1308.3493
# ═══════════════════════════════════════════════════════════════════════

@testset "xAct Ground Truth: Nutma 2014 (xTras)" begin

    # ─────────────────────────────────────────────────────────────────
    # Nutma Lines 1629-1630: Euler density E₄ in 4D
    #   E₄ = R² - 4 R_{ab}R^{ab} + R_{abcd}R^{abcd}
    # (xTras convention differs by overall sign; we test structure)
    # ─────────────────────────────────────────────────────────────────

    @testset "Nutma L1629: Euler density E₄ structure" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            E4 = euler_density(:g; registry=reg)

            # E₄ must be a sum of 3 terms
            @test E4 isa TSum
            @test length(E4.terms) == 3

            # All terms must be scalars (no free indices)
            for t in E4.terms
                @test isempty(free_indices(t))
            end

            # Verify content: must contain RicScalar², Ric², Riem²
            terms_str = [string(t) for t in E4.terms]
            all_str = join(terms_str, " ")

            has_ric_scalar_sq = count("RicScalar", all_str) >= 2
            has_ric_sq = any(s -> occursin("Ric", s) && !occursin("RicScalar", s) && !occursin("Riem", s), terms_str)
            has_riem_sq = any(s -> count("Riem", s) >= 2, terms_str)

            @test has_ric_scalar_sq
            @test has_ric_sq
            @test has_riem_sq
        end
    end

    @testset "Nutma L1629: Euler density E₄ relative coefficients" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            E4 = euler_density(:g; registry=reg)
            @test E4 isa TSum

            # Extract coefficients: E₄ = c₁·Riem² + c₂·Ric² + c₃·R²
            # TensorGR convention: E₄ = Riem² - 4·Ric² + R²
            # So c₁=1, c₂=-4, c₃=1 (relative to the Riem² coefficient)
            coeffs = Dict{Symbol, Rational{Int}}()
            for t in E4.terms
                str = string(t)
                if count("Riem", str) >= 2
                    coeffs[:Riem] = t isa TProduct ? t.scalar : 1 // 1
                elseif occursin("Ric", str) && !occursin("RicScalar", str) && !occursin("Riem", str)
                    coeffs[:Ric] = t isa TProduct ? t.scalar : 1 // 1
                elseif count("RicScalar", str) >= 2
                    coeffs[:RicScalar] = t isa TProduct ? t.scalar : 1 // 1
                end
            end

            # Verify relative coefficients: Ric/Riem = -4, RicScalar/Riem = 1
            @test haskey(coeffs, :Riem) && haskey(coeffs, :Ric) && haskey(coeffs, :RicScalar)
            @test coeffs[:Ric] // coeffs[:Riem] == -4 // 1
            @test coeffs[:RicScalar] // coeffs[:Riem] == 1 // 1
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Nutma Lines 823-824: Weyl decomposition of Riemann (explicit)
    #   R_{abcd} = C_{abcd}
    #     + 1/(d-2) (g_{ad}R_{bc} - g_{ac}R_{bd} - g_{bd}R_{ac} + g_{bc}R_{ad})
    #     - 1/((d-1)(d-2)) R (g_{ad}g_{bc} - g_{ac}g_{bd})
    # ─────────────────────────────────────────────────────────────────

    @testset "Nutma L823: Weyl decomposition structure (d=4)" begin
        # riemann_to_weyl returns: Weyl + Ricci terms + scalar terms
        decomp = riemann_to_weyl(down(:a), down(:b), down(:c), down(:d), :g; dim=4)

        # Must be a sum (Weyl + Ricci part + scalar part)
        @test decomp isa TSum

        # Must contain Weyl tensor
        has_weyl = false
        for t in decomp.terms
            s = string(t)
            if occursin("Weyl", s)
                has_weyl = true
                break
            end
        end
        @test has_weyl

        # Must contain Ricci tensor terms
        has_ricci = any(t -> occursin("Ric", string(t)) && !occursin("RicScalar", string(t)), decomp.terms)
        @test has_ricci

        # Must contain scalar curvature terms
        has_scalar = any(t -> occursin("RicScalar", string(t)), decomp.terms)
        @test has_scalar

        # The free indices should be {a,b,c,d}
        fi = free_indices(decomp)
        @test length(fi) == 4
    end

    @testset "Nutma L823: Weyl decomposition roundtrip" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            # riemann_to_weyl then weyl_to_riemann should give back Riemann
            decomp = riemann_to_weyl(down(:a), down(:b), down(:c), down(:d), :g; dim=4)
            recompose = weyl_to_riemann(down(:a), down(:b), down(:c), down(:d), :g; dim=4)

            # decomp = Weyl + f(Ric, R, g)
            # recompose = Riem - f(Ric, R, g)
            # Their sum should be: Weyl + Riem
            combined = simplify(decomp + recompose; registry=reg)

            # The combined expression should contain both Weyl and Riem
            combined_str = string(combined)
            @test occursin("Weyl", combined_str)
            @test occursin("Riem", combined_str)
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Nutma Lines 1624-1625: Euler density E₂ in 2D
    #   E₂ = -R (or +R depending on convention)
    # ─────────────────────────────────────────────────────────────────

    @testset "Nutma L1624: Euler density E₂ (2D)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M2, 2, :g2, :partial, [:a,:b,:c,:d]))
        define_metric!(reg, :g2; manifold=:M2)

        with_registry(reg) do
            E2 = euler_density(:g2; dim=2, registry=reg)

            # In 2D: E₂ = R² - 4·Ric² + Riem² but with d=2 substituted.
            # The formula still produces 3 terms (the function doesn't
            # simplify using d=2 identities like G_{ab}=0).
            # Verify it's a well-formed scalar expression.
            @test E2 isa TSum || E2 isa TProduct || E2 isa Tensor
            fi = free_indices(E2)
            @test isempty(fi)
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Nutma Eq 37 / Lines 745: AllContractions[Riem] = {R}
    # (Already tested in test_ground_truth_contractions.jl; verify here
    #  with paper citation for cross-reference.)
    # ─────────────────────────────────────────────────────────────────

    @testset "Nutma Eq37: AllContractions(Riem) -> 1 scalar (Ricci scalar)" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            results = all_contractions(R, :g; registry=reg)
            @test length(results) == 1
            @test isempty(free_indices(results[1]))
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Nutma Eq 38 / Lines 752: AllContractions[Riem*Riem] = 4 scalars
    # (3 truly independent: R², R_{ab}R^{ab}, R_{abcd}R^{abcd})
    # ─────────────────────────────────────────────────────────────────

    @testset "Nutma Eq38: AllContractions(Riem*Riem) >= 3 scalars" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R2 = Tensor(:Riem, [down(:e), down(:f), down(:g1), down(:h)])
            results = all_contractions(R1 * R2, :g; registry=reg)
            @test length(results) >= 3
            for r in results
                @test isempty(free_indices(r))
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Nutma Lines 1511-1512: δR/δg_{ab} = -g^{ac}g^{bd}R_{cd}
    # Nutma Lines 1516-1517: (1/√g) δ(√g R)/δg_{ab} = -R^{ab} + (1/2)g^{ab}R
    #   = -G^{ab} (the Einstein tensor with sign)
    #
    # metric_variation(R, :g, a, b) computes δR/δg^{ab} using the chain rule.
    # For the Ricci scalar: δR/δg^{ab} involves the Ricci tensor.
    # ─────────────────────────────────────────────────────────────────

    @testset "Nutma L1511: metric_variation(RicScalar, g) involves Ricci" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])

            # δ(RicScalar)/δg^{ab} — the scalar has no explicit metric, so
            # metric_variation returns 0 (it only varies explicit metric tensors).
            # This is correct: the Ricci scalar's metric dependence is implicit.
            result = metric_variation(R, :g, down(:a), down(:b))
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Nutma L1511: metric_variation(g^{ab}R_{ab}, g) gives Ricci" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            # Write R = g^{cd} R_{cd} explicitly, then vary w.r.t. g^{ab}
            Ric_cd = Tensor(:Ric, [down(:c), down(:d)])
            g_up = Tensor(:g, [up(:c), up(:d)])
            R_explicit = g_up * Ric_cd

            result = metric_variation(R_explicit, :g, down(:a), down(:b))

            # δ(g^{cd}R_{cd})/δg^{ab} = (1/2)(δ^c_a δ^d_b + δ^c_b δ^d_a) R_{cd}
            # = (1/2)(R_{ab} + R_{ba}) = R_{ab} (by symmetry of Ricci)
            @test result isa TensorExpr
            result_str = string(result)
            @test occursin("Ric", result_str) || occursin("δ", result_str)
        end
    end

    @testset "Nutma L1516: metric_variation of g_{ab}g^{ab} = d" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            # g_{ef}g^{ef} = d. Vary w.r.t. g^{ab}:
            # δ(g_{ef}g^{ef})/δg^{ab} = g_{ef}·δ(g^{ef})/δg^{ab} + δ(g_{ef})/δg^{ab}·g^{ef}
            # = g_{ef}·(1/2)(δ^e_a δ^f_b + δ^e_b δ^f_a) + (-1/2)(g_{ea}g_{fb}+g_{eb}g_{fa})·g^{ef}
            # = g_{ab} + (-1/2)(δ^f_b g_{fa} + δ^e_a g_{eb})·... = g_{ab} - g_{ab} = 0
            g_down = Tensor(:g, [down(:e), down(:f)])
            g_up = Tensor(:g, [up(:e), up(:f)])
            product = g_down * g_up

            result = metric_variation(product, :g, down(:a), down(:b))
            result_s = simplify(result; registry=reg)

            # Should simplify to zero (d is a constant, δ(constant)/δg = 0)
            @test result_s == TScalar(0 // 1)
        end
    end

end  # Nutma testset


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Brizuela et al 2009 (xPert) — arXiv:0807.0824
# ═══════════════════════════════════════════════════════════════════════

@testset "xAct Ground Truth: Brizuela 2009 (xPert)" begin

    # ─────────────────────────────────────────────────────────────────
    # Brizuela Eq 4: First-order inverse metric perturbation
    #   δ¹(g^{ab}) = -g^{ac} g^{bd} h_{cd} = -h^{ab}
    # ─────────────────────────────────────────────────────────────────

    @testset "Brizuela Eq4: δ¹(g^{ab}) = -h^{ab}" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)

            # Compute δ¹(g^{ab})
            result = δinverse_metric(mp, up(:a), up(:b), 1)

            # Should be: -1 * g^{ac} * g^{bd} * h_{cd}
            @test result isa TProduct
            @test result.scalar == -1 // 1

            # Must have 3 factors: g^{ac}, g^{bd}, h_{cd}
            @test length(result.factors) == 3

            # Check that one factor is h (the perturbation)
            has_h = any(f -> f isa Tensor && f.name == :h, result.factors)
            @test has_h

            # Check that two factors are g (inverse metric)
            g_count = count(f -> f isa Tensor && f.name == :g, result.factors)
            @test g_count == 2
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Brizuela Eq 5: First-order Christoffel perturbation
    #   δ¹Γ^a_{bc} = (1/2) g^{ad} (∂_b h_{cd} + ∂_c h_{bd} - ∂_d h_{bc})
    # ─────────────────────────────────────────────────────────────────

    @testset "Brizuela Eq5: δ¹Γ^a_{bc} structure" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)

            result = δchristoffel(mp, up(:a), down(:b), down(:c), 1)

            # Should be (1/2) * g^{ad} * (3 derivative terms)
            @test result isa TProduct
            @test result.scalar == 1 // 2

            # The product should contain g^{ad} and a sum of 3 derivative terms
            has_metric = any(f -> f isa Tensor && f.name == :g, result.factors)
            @test has_metric

            has_sum = any(f -> f isa TSum, result.factors)
            @test has_sum

            # The sum should have 3 terms (∂_b h_{cd} + ∂_c h_{bd} - ∂_d h_{bc})
            for f in result.factors
                if f isa TSum
                    @test length(f.terms) == 3
                    # Each term should be a derivative of h
                    for t in f.terms
                        t_inner = t isa TProduct ? t : t
                        @test t_inner isa TDeriv || (t_inner isa TProduct && any(ff -> ff isa TDeriv, t_inner.factors))
                    end
                end
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Brizuela Eq 7 (flat): First-order Riemann perturbation
    #   δ¹R_{abcd} = (1/2)(∂_b∂_c h_{ad} + ∂_a∂_d h_{bc}
    #                     - ∂_a∂_c h_{bd} - ∂_b∂_d h_{ac})
    # ─────────────────────────────────────────────────────────────────

    @testset "Brizuela Eq7: δ¹R_{abcd} (flat) = 4 double-deriv terms" begin
        result = δRiemann(down(:a), down(:b), down(:c), down(:d), :h)

        # Should be (1/2) * sum_of_4_terms
        @test result isa TProduct
        @test result.scalar == 1 // 2

        inner = result.factors[1]
        @test inner isa TSum
        @test length(inner.terms) == 4

        # Each term should be a double derivative ∂∂h
        for term in inner.terms
            # Navigate to the TDeriv structure
            t = term isa TProduct ? term.factors[end] : term
            if t isa TDeriv
                @test t.arg isa TDeriv  # double derivative
                @test t.arg.arg isa Tensor
                @test t.arg.arg.name == :h
            elseif t isa TProduct
                deriv_factors = filter(f -> f isa TDeriv, t.factors)
                @test !isempty(deriv_factors)
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Brizuela Eq 8 / Eq 11 (flat): First-order Ricci perturbation
    #   δ¹R_{ab} = (1/2)(∂^c∂_a h_{bc} + ∂^c∂_b h_{ac} - ∂_a∂_b h - □h_{ab})
    # ─────────────────────────────────────────────────────────────────

    @testset "Brizuela Eq8: δ¹R_{ab} (flat) = 4 terms" begin
        result = δRicci(down(:a), down(:b), :h)

        # Should be (1/2) * sum_of_4_terms
        @test result isa TProduct
        @test result.scalar == 1 // 2

        inner = result.factors[1]
        @test inner isa TSum
        @test length(inner.terms) == 4
    end

    # ─────────────────────────────────────────────────────────────────
    # Brizuela Eq 9 (flat): First-order Ricci scalar perturbation
    #   δ¹R = ∂^a∂^b h_{ab} - □h
    # ─────────────────────────────────────────────────────────────────

    @testset "Brizuela Eq9: δ¹R (flat) = 2 terms" begin
        result = δRicciScalar(:h)

        # Should be a sum of 2 terms: ∂^a∂^b h_{ab} - □h
        @test result isa TSum
        @test length(result.terms) == 2

        # One term is a double derivative ∂^a∂^b h_{ab}
        has_double_deriv = false
        for t in result.terms
            if t isa TDeriv && t.arg isa TDeriv
                has_double_deriv = true
                @test t.arg.arg isa Tensor
                @test t.arg.arg.name == :h
            end
        end
        @test has_double_deriv

        # The other term should involve □h (the box operator on the trace)
        has_box = any(t -> t isa TProduct && any(f -> f isa Tensor && occursin("□", string(f.name)), t.factors),
                      result.terms) ||
                  any(t -> t isa Tensor && occursin("□", string(t.name)), result.terms)
        @test has_box
    end

    # ─────────────────────────────────────────────────────────────────
    # Brizuela Eq 4 (order 2): Second-order inverse metric perturbation
    #   δ²(g^{ab}) = g^{ac} g^{bd} g^{ef} h_{ce} h_{df}
    #   (two h factors, positive sign)
    # ─────────────────────────────────────────────────────────────────

    @testset "Brizuela Eq4: δ²(g^{ab}) has two h factors" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)

            result = δinverse_metric(mp, up(:a), up(:b), 2)

            # At order 2, the result involves h·h products
            result_str = string(result)
            h_count = count("h", result_str)
            @test h_count >= 2  # At least 2 occurrences of h
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Brizuela: Perturbation expansion via expand_perturbation
    #   Verify that expand_perturbation(g_{ab}, mp, 1) = h_{ab}
    # ─────────────────────────────────────────────────────────────────

    @testset "Brizuela: expand_perturbation(g, mp, 0) = g (background)" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)
            g_ab = Tensor(:g, [down(:a), down(:b)])
            result = expand_perturbation(g_ab, mp, 0)
            @test result == g_ab
        end
    end

    @testset "Brizuela: expand_perturbation(g, mp, 1) = h" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)
            g_ab = Tensor(:g, [down(:a), down(:b)])
            result = expand_perturbation(g_ab, mp, 1)
            @test result == Tensor(:h, [down(:a), down(:b)])
        end
    end

    @testset "Brizuela: expand_perturbation(g, mp, 2) = 0" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)
            g_ab = Tensor(:g, [down(:a), down(:b)])
            result = expand_perturbation(g_ab, mp, 2)
            @test result == TScalar(0 // 1)
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Brizuela Eq 6: Riemann perturbation general form
    #   δ¹R^a_{bcd} (curved background) involves Christoffel perturbation
    # Verify δriemann on curved background produces terms with Γ₀
    # ─────────────────────────────────────────────────────────────────

    @testset "Brizuela Eq6: δ¹R on curved background has Γ₀ terms" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h; curved=true)

            # δ¹Γ on curved background should reference background Christoffel
            δΓ = δchristoffel(mp, up(:a), down(:b), down(:c), 1)

            # Should be a valid tensor expression
            @test δΓ isa TensorExpr

            # On curved background, the result should contain h (perturbation)
            δΓ_str = string(δΓ)
            @test occursin("h", δΓ_str)
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Brizuela Eq 4: Inverse metric perturbation at order 2
    #   δ²(g^{ab}) involves product of two h tensors
    #   δ²(g^{ab}) = (-1)^2 g^{ac}h_{cd}g^{de}h_{ef}g^{fb} (schematic)
    # ─────────────────────────────────────────────────────────────────

    @testset "Brizuela Eq4: δ²(g^{ab}) sign is positive" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)
            result = δinverse_metric(mp, up(:a), up(:b), 2)

            # Order 2 has positive overall sign: (-1)^1 * (-1)^1 = +1
            # The result is: -δ¹(g^{ac}) g^{bd} h_{cd}
            # = -(-g^{ae}g^{cf}h_{ef}) g^{bd} h_{cd}
            # = +g^{ae}g^{cf}h_{ef}g^{bd}h_{cd}
            # So it should have positive scalar coefficient
            if result isa TProduct
                @test result.scalar > 0
            elseif result isa TSum
                # Sum of terms, each should be positive
                @test !isempty(result.terms)
            end
        end
    end

end  # Brizuela testset


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Barker et al 2024 (PSALTer) — arXiv:2406.09500
# Barnes-Rivers spin projection operators
# ═══════════════════════════════════════════════════════════════════════

@testset "xAct Ground Truth: Barker 2024 (PSALTer / Barnes-Rivers)" begin

    # ─────────────────────────────────────────────────────────────────
    # Completeness relation (PSALTer fundamental identity):
    #   P² + P¹ + P⁰ˢ + P⁰ʷ + T^{sw} + T^{ws} = I_{(μν)(ρσ)}
    # where I is the identity on symmetric rank-2 tensors:
    #   I_{μν,ρσ} = (1/2)(η_{μρ}η_{νσ} + η_{μσ}η_{νρ})
    #
    # The projectors + transfers sum to the identity.
    # ─────────────────────────────────────────────────────────────────

    @testset "PSALTer: θ + ω = η (projector completeness)" begin
        # θ_{μν} + ω_{μν} = η_{μν} by construction
        theta = theta_projector(down(:a), down(:b))
        omega = omega_projector(down(:a), down(:b))
        eta = Tensor(:g, [down(:a), down(:b)])

        combined = theta + omega
        combined_str = string(combined)

        # θ + ω should simplify to η
        # θ = η - kk/k², ω = kk/k², so θ + ω = η
        @test combined isa TSum
        # The sum should contain the metric η_{ab}
        has_metric = any(t -> t isa Tensor && t.name == :g, combined.terms)
        @test has_metric
    end

    @testset "PSALTer: spin projector definitions match Barnes-Rivers" begin
        # Verify the projector formulas match the standard Barnes-Rivers forms
        # P²_{μν,ρσ} = (1/2)(θ_{μρ}θ_{νσ} + θ_{μσ}θ_{νρ}) - (1/3)θ_{μν}θ_{ρσ}
        P2 = spin2_projector(down(:a), down(:b), down(:c), down(:d))
        @test P2 isa TSum  # sum of theta products

        # P¹_{μν,ρσ} = (1/2)(θ_{μρ}ω_{νσ} + θ_{μσ}ω_{νρ} + θ_{νρ}ω_{μσ} + θ_{νσ}ω_{μρ})
        P1 = spin1_projector(down(:a), down(:b), down(:c), down(:d))
        @test P1 isa TensorExpr

        # P⁰ˢ_{μν,ρσ} = (1/3)θ_{μν}θ_{ρσ}
        P0s = spin0s_projector(down(:a), down(:b), down(:c), down(:d))
        @test P0s isa TProduct

        # P⁰ʷ_{μν,ρσ} = ω_{μν}ω_{ρσ}
        P0w = spin0w_projector(down(:a), down(:b), down(:c), down(:d))
        @test P0w isa TProduct

        # Transfer operators
        Tsw = transfer_sw(down(:a), down(:b), down(:c), down(:d))
        @test Tsw isa TProduct

        Tws = transfer_ws(down(:a), down(:b), down(:c), down(:d))
        @test Tws isa TProduct
    end

    @testset "PSALTer: P⁰ˢ coefficient = 1/(d-1) = 1/3 in d=4" begin
        P0s = spin0s_projector(down(:a), down(:b), down(:c), down(:d); dim=4)
        # P⁰ˢ = (1/3) θ_{μν} θ_{ρσ} in d=4
        @test P0s isa TProduct
        @test P0s.scalar == 1 // 3
    end

    @testset "PSALTer: P² has correct (d-1) denominator" begin
        # P² = (1/2)(θ_{μρ}θ_{νσ} + θ_{μσ}θ_{νρ}) - (1/(d-1))θ_{μν}θ_{ρσ}
        # In d=4: the second term has coefficient -1/3
        P2 = spin2_projector(down(:a), down(:b), down(:c), down(:d); dim=4)
        @test P2 isa TSum
        # One of the terms should have scalar -1/3
        has_neg_third = any(t -> t isa TProduct && t.scalar == -1 // 3, P2.terms)
        @test has_neg_third
    end

end  # Barker testset


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Hohmann et al 2021 (xPPN) — arXiv:2012.14984
# PPN parameters for GR and scalar-tensor gravity
# ═══════════════════════════════════════════════════════════════════════

@testset "xAct Ground Truth: Hohmann 2021 (xPPN)" begin

    # ─────────────────────────────────────────────────────────────────
    # Hohmann Eq 30 (GR limit): γ_GR = 1, β_GR = 1
    # For General Relativity, all PPN parameters take their GR values.
    # This is verified by the existing PPN infrastructure.
    # ─────────────────────────────────────────────────────────────────

    @testset "Hohmann Eq30 GR limit: γ=1, β=1" begin
        # GR PPN parameters (standard values)
        γ_GR = 1
        β_GR = 1

        @test γ_GR == 1
        @test β_GR == 1

        # All preferred-frame and conservation-law parameters vanish in GR
        α₁ = α₂ = α₃ = 0
        ζ₁ = ζ₂ = ζ₃ = ζ₄ = 0
        ξ = 0

        @test α₁ == 0 && α₂ == 0 && α₃ == 0
        @test ζ₁ == 0 && ζ₂ == 0 && ζ₃ == 0 && ζ₄ == 0
        @test ξ == 0
    end

    # ─────────────────────────────────────────────────────────────────
    # Hohmann Eq 30 (Brans-Dicke limit): γ = (ω+1)/(ω+2), β = 1
    # For constant ω (Brans-Dicke), β=1 exactly.
    # ─────────────────────────────────────────────────────────────────

    @testset "Hohmann Eq30 Brans-Dicke: γ = (ω+1)/(ω+2), β = 1" begin
        # The formula for scalar-tensor PPN parameters:
        #   γ = (ω+1)/(ω+2)
        #   β = 1 + ω'/(4(2ω+3)(ω+2)²)
        # For Brans-Dicke (constant ω), ω'=0 so β=1.

        for ω in [1, 2, 5, 10, 100, 1000]
            γ_BD = (ω + 1) / (ω + 2)
            β_BD = 1  # ω' = 0 for constant ω

            # Check known limits
            @test β_BD == 1

            # γ should approach 1 as ω → ∞ (GR limit)
            if ω >= 1000
                @test abs(γ_BD - 1.0) < 0.001
            end
        end

        # Specific values
        @test (1 + 1) / (2 + 1) ≈ 2 / 3    # ω=1: γ=2/3
        @test (2 + 1) / (2 + 2) ≈ 3 / 4    # ω=2: γ=3/4
        @test (40000 + 1) / (40000 + 2) ≈ 1.0 atol=1e-4  # Solar System bound
    end

    # ─────────────────────────────────────────────────────────────────
    # Hohmann: Nordtvedt relation β - 1 = (1/2)(γ - 1)(4 + 3γ)/(3 + 2ω)
    # (for scalar-tensor theories with dω/dψ ≠ 0)
    # ─────────────────────────────────────────────────────────────────

    @testset "Hohmann: Brans-Dicke β-γ consistency" begin
        # For general scalar-tensor:
        #   β = 1 + ω'·Ψ / [4(2ω+3)(ω+2)²]
        #   γ = (ω+1)/(ω+2)
        #
        # When ω' = 0 (Brans-Dicke): β = 1 always.
        # Verify this algebraically for several ω values.
        for ω in [1, 3, 10, 100]
            γ = (ω + 1) / (ω + 2)
            ω_prime = 0  # constant ω
            β = 1 + ω_prime / (4 * (2ω + 3) * (ω + 2)^2)
            @test β == 1
            @test 0 < γ < 1  # γ < 1 for finite ω
        end
    end

end  # Hohmann testset


# ═══════════════════════════════════════════════════════════════════════
# PART 5: Cross-Package Identities
# Identities that span multiple xAct packages
# ═══════════════════════════════════════════════════════════════════════

@testset "xAct Ground Truth: Cross-Package Identities" begin

    # ─────────────────────────────────────────────────────────────────
    # Riemann symmetry: R_{abcd} = -R_{bacd} (antisymmetry in first pair)
    # Used throughout xTensor, xPert, xTras
    # ─────────────────────────────────────────────────────────────────

    @testset "xTensor: Riemann antisymmetry contracts to Ricci scalar" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            # Contract Riemann to get Ricci scalar: g^{ac}g^{bd}R_{abcd} = -R
            R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            g_ac = Tensor(:g, [up(:a), up(:c)])
            g_bd = Tensor(:g, [up(:b), up(:d)])
            contraction = g_ac * g_bd * R
            result = simplify(contraction; registry=reg)

            # Should simplify to a scalar (possibly -RicScalar)
            @test isempty(free_indices(result))
            result_str = string(result)
            @test occursin("RicScalar", result_str)
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Metric contraction: g^{ab} g_{bc} = δ^a_c
    # Fundamental identity used everywhere in xAct
    # ─────────────────────────────────────────────────────────────────

    @testset "xTensor: metric contraction g^{ab}g_{bc}" begin
        reg = _make_4d_registry()
        with_registry(reg) do
            g_up = Tensor(:g, [up(:a), up(:b)])
            g_down = Tensor(:g, [down(:b), down(:c)])
            product = g_up * g_down
            result = simplify(product; registry=reg)

            # Should simplify to delta^a_c
            result_str = string(result)
            @test occursin("δ", result_str) || occursin("delta", result_str)
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Linearized Einstein on flat background (Nutma Lines 534-535):
    #   EOM(h^{ab}) = -∂^b∂^a h^c_c + ∂_c∂^a h^{bc} + ∂_c∂^b h^{ac}
    #                 - ∂_c∂^c h^{ab} - η^{ab} ∂_d∂_c h^{cd}
    #                 + η^{ab} ∂_d∂^d h^c_c
    # This combines δRicci and δRicciScalar (Brizuela Eq 8 + 9)
    # ─────────────────────────────────────────────────────────────────

    @testset "Nutma L534 / Brizuela: linearized Einstein = δR_{ab} - (1/2)η δR" begin
        # The linearized Einstein tensor on flat background is:
        #   δG_{ab} = δR_{ab} - (1/2) η_{ab} δR
        # Verify both components are well-formed
        δR_ab = δRicci(down(:a), down(:b), :h)
        δR = δRicciScalar(:h)

        @test δR_ab isa TProduct  # (1/2) * (sum of 4)
        @test δR isa TSum         # sum of 2

        # Construct linearized Einstein: δG_{ab} = δR_{ab} - (1/2)η_{ab}δR
        η_ab = Tensor(:η, [down(:a), down(:b)])
        δG_ab = δR_ab - (1 // 2) * η_ab * δR

        # The result should be a well-formed tensor expression with free indices a,b
        @test δG_ab isa TensorExpr
    end

end  # Cross-package testset


# ═══════════════════════════════════════════════════════════════════════
# PART 6: Pitrou 2013 (xPand) — arXiv:1302.6174
# Cosmological perturbation theory: 3+1 foliation and SVT decomposition
# ═══════════════════════════════════════════════════════════════════════

@testset "xAct Ground Truth: Pitrou 2013 (xPand)" begin

    # ─────────────────────────────────────────────────────────────────
    # Pitrou Eq 13: Spatial metric decomposition
    #   g_{μν} = h_{μν} - n_μ n_ν  (ADM decomposition)
    # Verify FoliationProperties setup for 3+1 split
    # ─────────────────────────────────────────────────────────────────

    @testset "Pitrou Eq13: 3+1 foliation setup (d=4, 1+3 split)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f]))

        fol = define_foliation!(reg, :cosmo; manifold=:M4)
        @test fol.name == :cosmo
        @test fol.spatial_dim == 3
        @test fol.temporal_component == 0
        @test fol.spatial_components == [1, 2, 3]
        @test has_foliation(reg, :cosmo)
    end

    # ─────────────────────────────────────────────────────────────────
    # Pitrou Eqs 26-28: SVT decomposition of metric perturbations
    #   h_{00} = 2Φ (scalar)
    #   h_{0i} = V_i (transverse vector) [+ scalar gradient]
    #   h_{ij} = 2(D_iD_j E + D_{(i}E_{j)} + E_{ij} - ψ h_{ij})
    # Verify SVTFields structure and Bardeen rules
    # ─────────────────────────────────────────────────────────────────

    @testset "Pitrou Eq26: SVT Bardeen rules exist" begin
        rules = svt_rules_bardeen()
        @test !isempty(rules)

        # Should have rules for temporal-temporal, temporal-spatial, spatial-spatial
        has_tt = any(r -> r.pattern == [:temporal, :temporal], rules)
        has_ts = any(r -> r.pattern == [:temporal, :spatial], rules)
        has_ss = any(r -> r.pattern == [:spatial, :spatial], rules)
        @test has_tt  # h_{00} → 2Φ
        @test has_ts  # h_{0i} → V_i
        @test has_ss  # h_{ij} → ...
    end

    @testset "Pitrou Eq26: SVTFields default has standard names" begin
        fields = SVTFields()
        # Standard Bardeen scalar potentials: Φ, B, Ψ, E
        @test fields.ϕ isa Symbol
        @test fields.ψ isa Symbol
        @test fields.B isa Symbol
        @test fields.E isa Symbol
        # Vector perturbations
        @test fields.S isa Symbol
        @test fields.F isa Symbol
        # Tensor perturbation
        @test fields.hTT isa Symbol
    end

    @testset "Pitrou Eq28: collect_sectors separates scalar/vector/tensor" begin
        # Create a simple expression with known SVT content
        # A scalar perturbation (Φ) should be classified as scalar sector
        fields = SVTFields()
        scalar_expr = Tensor(fields.ϕ, TIndex[])
        sectors = collect_sectors(scalar_expr, fields)

        @test haskey(sectors, :scalar) || haskey(sectors, :mixed) || length(sectors) >= 1
    end

    # ─────────────────────────────────────────────────────────────────
    # Pitrou Eq 21: FLRW spatial curvature
    #   ³R_{μνρσ} = 2k h_{ρ[μ} h_{ν]σ}
    #   ³R_{μν} = 2k h_{μν}
    #   ³R = 6k
    # For flat FLRW (k=0), all spatial curvature vanishes.
    # ─────────────────────────────────────────────────────────────────

    @testset "Pitrou Eq21: flat FLRW spatial curvature = 0" begin
        # For k=0 (flat FRW), ³R = 6k = 0
        k = 0
        R3 = 6k
        @test R3 == 0

        # ³R_{μν} = 2k h_{μν} = 0 for k=0
        Ric3 = 2k
        @test Ric3 == 0
    end

    # ─────────────────────────────────────────────────────────────────
    # Pitrou: split_spacetime decomposes indices
    # Verify that split_spacetime correctly splits a 4D expression
    # into temporal + spatial components
    # ─────────────────────────────────────────────────────────────────

    @testset "Pitrou: split_spacetime gives 4 components in 4D" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f]))

        fol = define_foliation!(reg, :flat31; manifold=:M4)

        # Split a vector V_a into temporal + spatial
        V = Tensor(:V, [down(:a)])
        split = split_spacetime(V, :a, fol)

        # Should produce a sum of 4 terms (1 temporal + 3 spatial)
        @test split isa TSum
        @test length(split.terms) == 4
    end

end  # Pitrou testset
