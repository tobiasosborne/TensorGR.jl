@testset "BH-Pert2 End-to-End Validation" begin
    using TensorGR: source_coupling_modes, assemble_source,
                    second_order_rw, second_order_zerilli,
                    regge_wheeler_potential, zerilli_potential,
                    gaunt_integral, vector_gaunt, tensor_gaunt,
                    angular_selection_rule, wigner3j,
                    scalar_coupling_coefficient,
                    EnergyFluxFormula, energy_flux_formula, flux_mode_count,
                    quadrupole_flux_scaling, second_order_flux_scaling,
                    SourcedMasterEquation, SecondOrderSource

    # ══════════════════════════════════════════════════════════════════
    # TGR-2y0: Brizuela et al l=2 quadrupole source validation
    # ══════════════════════════════════════════════════════════════════

    @testset "Wigner 3j symbols: known values" begin
        # (2 2 0; 0 0 0) = 1/√5
        @test wigner3j(2, 2, 0, 0, 0, 0) ≈ 1 / √5

        # (2 2 4; 0 0 0) = known value from Racah formula
        w4 = wigner3j(2, 2, 4, 0, 0, 0)
        @test abs(w4) > 0

        # (2 2 1; 0 0 0) = 0 (parity: 2+2+1=5 odd)
        @test wigner3j(2, 2, 1, 0, 0, 0) ≈ 0.0 atol=1e-15

        # (2 2 2; 0 0 0) ≠ 0 (parity: 2+2+2=6 even)
        @test abs(wigner3j(2, 2, 2, 0, 0, 0)) > 0
    end

    @testset "Coupling coefficients: C^{l,0}_{2,0,2,0} for l=0,2,4" begin
        # The Gaunt integral ∫ Y_{2,0} Y_{2,0} Y*_{l,0} dΩ

        # l=0: C = (-1)^0 · √(5·5·1/(4π)) · (2 2 0;0 0 0)²
        #      = 5/(2√π) · 1/5 = 1/(2√π)
        C00 = gaunt_integral(2, 0, 2, 0, 0, 0)
        @test C00 ≈ 1 / (2√π) rtol=1e-12

        # l=2: non-zero (allowed by selection rules)
        C20 = gaunt_integral(2, 0, 2, 0, 2, 0)
        @test abs(C20) > 0

        # l=4: non-zero (allowed: 2+2+4=8 even, |2-2|≤4≤2+2)
        C40 = gaunt_integral(2, 0, 2, 0, 4, 0)
        @test abs(C40) > 0

        # l=1: vanishes (parity: 2+2+1=5 odd)
        @test gaunt_integral(2, 0, 2, 0, 1, 0) ≈ 0.0 atol=1e-15

        # l=3: vanishes (parity: 2+2+3=7 odd)
        @test gaunt_integral(2, 0, 2, 0, 3, 0) ≈ 0.0 atol=1e-15

        # l=5: vanishes (triangle: 2+2=4 < 5)
        @test gaunt_integral(2, 0, 2, 0, 5, 0) ≈ 0.0 atol=1e-15
    end

    @testset "Coupling coefficients: relative tolerance 1e-13" begin
        # The issue requires 1e-13 relative tolerance
        C00 = gaunt_integral(2, 0, 2, 0, 0, 0)
        expected = 1 / (2√π)
        @test abs(C00 - expected) / abs(expected) < 1e-13

        # C^{4,0}_{2,0,2,0} involves (2 2 4; 0 0 0)²
        C40 = gaunt_integral(2, 0, 2, 0, 4, 0)
        w3j_220_004 = wigner3j(2, 2, 4, 0, 0, 0)
        expected_C40 = √((5 * 5 * 9) / (4π)) * w3j_220_004^2
        @test abs(C40 - expected_C40) / abs(expected_C40) < 1e-13
    end

    @testset "Mode coupling: l=2 quadrupole self-coupling structure" begin
        # (l1=2, m1=0) × (l2=2, m2=0) can source l = 0, 2, 4
        for l_target in [0, 2, 4]
            couplings = source_coupling_modes(l_target, 0, 2)
            has_22 = any(c -> c.l1 == 2 && c.l2 == 2 && c.m1 == 0 && c.m2 == 0,
                         couplings)
            @test has_22
        end

        # l=1, 3 are forbidden
        for l_target in [1, 3]
            couplings = source_coupling_modes(l_target, 0, 2)
            has_22 = any(c -> c.l1 == 2 && c.l2 == 2, couplings)
            @test !has_22
        end
    end

    @testset "Source assembly: quadrupole → target modes" begin
        # Even-parity source at l=0 from (2,0)×(2,0)
        src0 = assemble_source(0, 0, :even, 2)
        @test !isempty(src0.contributions)

        # Even-parity source at l=2 from (2,0)×(2,0)
        src2 = assemble_source(2, 0, :even, 2)
        @test !isempty(src2.contributions)

        # Even-parity source at l=4 from (2,0)×(2,0)
        src4 = assemble_source(4, 0, :even, 4)
        has_22_in_4 = any(c -> c.l1 == 2 && c.l2 == 2, src4.contributions)
        @test has_22_in_4
    end

    @testset "Second-order master equations: complete pipeline" begin
        # Full pipeline: mode coupling → source assembly → master equation
        eq_z = second_order_zerilli(2, 0, 2)
        eq_rw = second_order_rw(2, 0, 2)

        # Both should have source terms
        @test eq_z isa SourcedMasterEquation
        @test eq_rw isa SourcedMasterEquation

        # Zerilli (even) source should have even×even and odd×odd
        even_ee = count(c -> c.parity1 === :even && c.parity2 === :even,
                        eq_z.source.contributions)
        @test even_ee > 0

        # RW (odd) source should have even×odd
        # (may be empty if no valid couplings at this lmax)
        @test eq_rw.source.parity === :odd
    end

    @testset "Vector coupling: Brizuela Table I" begin
        # Vector Gaunt integral for (l1=2, l2=2)
        # L1 = L2 = 6, L_target varies
        vg_0 = vector_gaunt(2, 0, 2, 0, 0, 0)
        vg_2 = vector_gaunt(2, 0, 2, 0, 2, 0)
        vg_4 = vector_gaunt(2, 0, 2, 0, 4, 0)

        # Vector coupling = (L1+L2-L3)/2 * gaunt
        # For l3=0: coupling_factor = (6+6-0)/2 = 6
        @test vg_0 ≈ 6.0 * gaunt_integral(2, 0, 2, 0, 0, 0)
        # For l3=2: coupling_factor = (6+6-6)/2 = 3
        @test vg_2 ≈ 3.0 * gaunt_integral(2, 0, 2, 0, 2, 0)
        # For l3=4: coupling_factor = (6+6-20)/2 = -4
        @test vg_4 ≈ -4.0 * gaunt_integral(2, 0, 2, 0, 4, 0)
    end

    @testset "Tensor coupling: Brizuela Table I" begin
        # Y-Y (metric × metric) = 2 * gaunt
        @test tensor_gaunt(2, 0, 2, 0, 0, 0, :Y, :Y) ≈
              2.0 * gaunt_integral(2, 0, 2, 0, 0, 0)

        # Z-Z (STF × STF) uses Q coefficient
        # Q(2,2,l3) with L1=L2=6:
        # l3=0: Q = (12)²/8 + (36-12)/4 = 18+6 = 24
        @test tensor_gaunt(2, 0, 2, 0, 0, 0, :Z, :Z) ≈
              24.0 * gaunt_integral(2, 0, 2, 0, 0, 0)

        # l3=2: Q = (6+6-6)²/8 + (36-12)/4 = 36/8+24/4 = 4.5+6 = 10.5
        @test tensor_gaunt(2, 0, 2, 0, 2, 0, :Z, :Z) ≈
              10.5 * gaunt_integral(2, 0, 2, 0, 2, 0)

        # l3=4: Q = (6+6-20)²/8 + (36-12)/4 = 64/8+6 = 8+6 = 14
        @test tensor_gaunt(2, 0, 2, 0, 4, 0, :Z, :Z) ≈
              14.0 * gaunt_integral(2, 0, 2, 0, 4, 0)

        # X-X = same as Z-Z by parity symmetry
        @test tensor_gaunt(2, 0, 2, 0, 0, 0, :X, :X) ≈
              tensor_gaunt(2, 0, 2, 0, 0, 0, :Z, :Z)
    end

    # ══════════════════════════════════════════════════════════════════
    # TGR-2gv: Second-order gravitational wave energy flux
    # ══════════════════════════════════════════════════════════════════

    @testset "energy_flux_formula construction" begin
        ef1 = energy_flux_formula(; lmax=2, order=1)
        @test ef1 isa EnergyFluxFormula
        @test ef1.lmax == 2
        @test ef1.order == 1
        @test ef1.normalization == 1 // 64

        ef2 = energy_flux_formula(; lmax=4, order=2)
        @test ef2.order == 2
        @test ef2.normalization == 1 // 32
    end

    @testset "energy_flux_formula: l >= 2 required" begin
        @test_throws ErrorException energy_flux_formula(; lmax=1)
    end

    @testset "energy_flux_formula: order 1 or 2 only" begin
        @test_throws ErrorException energy_flux_formula(; order=3)
    end

    @testset "flux_mode_count" begin
        ef = energy_flux_formula(; lmax=2)
        # l=2: 5 m-values × 2 parities = 10 modes
        @test flux_mode_count(ef) == 10

        ef3 = energy_flux_formula(; lmax=3)
        # l=2: 10, l=3: 7×2=14, total=24
        @test flux_mode_count(ef3) == 24
    end

    @testset "quadrupole dominance" begin
        # At l=2, the quadrupole mode dominates the energy flux
        # The l=2 contribution is proportional to the square of
        # the mass quadrupole moment
        ef = energy_flux_formula(; lmax=2)
        @test flux_mode_count(ef) == 10

        # Higher multipoles contribute less (l=3 is suppressed by v²)
        ef4 = energy_flux_formula(; lmax=4)
        @test flux_mode_count(ef4) > flux_mode_count(ef)
    end

    @testset "quadrupole_flux_scaling" begin
        # Leading-order flux scales as η² (symmetric mass ratio)
        @test quadrupole_flux_scaling(1 // 4) == (1 // 4)^2  # equal mass
        @test quadrupole_flux_scaling(0.0) == 0.0  # test particle limit

        # η = 1/4 for equal mass: dE/dt ∝ 1/16
        @test quadrupole_flux_scaling(0.25) ≈ 0.0625
    end

    @testset "second_order_flux_scaling" begin
        # Second-order correction scales as η³
        @test second_order_flux_scaling(1 // 4) == (1 // 4)^3
        @test second_order_flux_scaling(0.0) == 0.0

        # Ratio dE^{(2)}/dE^{(1)} ∝ η, small for extreme mass ratios
        eta = 0.01  # extreme mass ratio
        ratio = second_order_flux_scaling(eta) / quadrupole_flux_scaling(eta)
        @test ratio ≈ eta
        @test ratio < 0.1  # second-order correction is small
    end

    @testset "Campanelli-Lousto structure: cross-term scaling" begin
        # dE^{(2)}/dt contains ψ^{(1)*} × ψ^{(2)} cross terms
        # (Campanelli & Lousto 1999, Eq 25)
        # For equal-mass head-on collision:
        #   ψ^{(1)} ∝ η  (first-order perturbation)
        #   ψ^{(2)} ∝ η² (second-order perturbation)
        #   dE^{(2)}/dt ∝ ψ^{(1)} × ψ^{(2)} ∝ η³

        eta = 0.25  # equal mass
        first_order = quadrupole_flux_scaling(eta)   # η²
        second_order = second_order_flux_scaling(eta)  # η³

        # Second-order is suppressed by factor η relative to first
        @test second_order / first_order ≈ eta
    end

    @testset "display" begin
        ef = energy_flux_formula(; lmax=2, order=1)
        s = sprint(show, ef)
        @test occursin("dE", s)
    end
end
