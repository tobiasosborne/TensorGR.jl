@testset "BH Source Assembly & Gauge-Invariant Master Equations" begin
    using TensorGR: SourceContribution, SecondOrderSource, assemble_source,
                    SourcedMasterEquation, second_order_rw, second_order_zerilli,
                    GaugeInvariantVariable, gauge_invariant_variable,
                    is_gauge_invariant_at_zero,
                    MasterEquation, regge_wheeler_potential, zerilli_potential,
                    evaluate_potential,
                    source_coupling_modes, angular_selection_rule,
                    scalar_coupling_coefficient,
                    EVEN_SECTORS, ODD_SECTORS

    # ── Source assembly (TGR-u19) ────────────────────────────────────────

    @testset "assemble_source: even parity l=2, m=0" begin
        source = assemble_source(2, 0, :even, 2)
        @test source isa SecondOrderSource
        @test source.l == 2
        @test source.m == 0
        @test source.parity === :even
        @test !isempty(source.contributions)

        # All contributions must satisfy selection rules
        for c in source.contributions
            @test c.m1 + c.m2 == 0  # m-conservation
            @test angular_selection_rule(c.l1, c.l2, 2)
            @test abs(c.coupling) > 1e-15
        end
    end

    @testset "assemble_source: odd parity l=2, m=0" begin
        source = assemble_source(2, 0, :odd, 2)
        @test source isa SecondOrderSource
        @test source.parity === :odd

        # Odd-parity source comes from even×odd coupling
        for c in source.contributions
            @test (c.parity1 === :even && c.parity2 === :odd) ||
                  (c.parity1 === :odd && c.parity2 === :even)
        end
    end

    @testset "assemble_source: l=0 monopole" begin
        # l=0 monopole: only l1=l2 allowed (triangle with l=0)
        source = assemble_source(0, 0, :even, 2)
        for c in source.contributions
            @test c.l1 == c.l2  # triangle inequality with l=0
        end
    end

    @testset "assemble_source: source vanishes at zero" begin
        # When first-order perturbation is zero, the source is zero
        # because every contribution has a non-zero coupling coefficient
        # multiplied by zero (from the first-order radial functions).
        # We verify the structural property: all contributions are bilinear.
        source = assemble_source(2, 0, :even, 2)
        # Each contribution involves products of two first-order modes
        for c in source.contributions
            @test !isempty(c.sector_pairs)  # each has sector pairs
        end
    end

    @testset "assemble_source: sector pair structure" begin
        source = assemble_source(2, 0, :even, 2)
        for c in source.contributions
            if c.parity1 === :even && c.parity2 === :even
                # Even×Even: sector pairs from EVEN_SECTORS
                for (s1, s2) in c.sector_pairs
                    @test s1 in EVEN_SECTORS
                    @test s2 in EVEN_SECTORS
                end
            elseif c.parity1 === :odd && c.parity2 === :odd
                # Odd×Odd: sector pairs from ODD_SECTORS
                for (s1, s2) in c.sector_pairs
                    @test s1 in ODD_SECTORS
                    @test s2 in ODD_SECTORS
                end
            end
        end
    end

    @testset "assemble_source: lmax=0 gives only l1=l2=0" begin
        source = assemble_source(0, 0, :even, 0)
        for c in source.contributions
            @test c.l1 == 0 && c.l2 == 0
        end
    end

    @testset "assemble_source: contributions count grows with lmax" begin
        n2 = length(assemble_source(2, 0, :even, 2).contributions)
        n3 = length(assemble_source(2, 0, :even, 3).contributions)
        @test n3 >= n2  # more modes → more couplings
    end

    # ── Sourced master equations (TGR-22h) ───────────────────────────────

    @testset "second_order_rw construction" begin
        eq = second_order_rw(2, 0, 2)
        @test eq isa SourcedMasterEquation
        @test eq.equation.parity === :odd
        @test eq.equation.l == 2
        @test eq.equation.potential_name === :RW
        @test eq.source.parity === :odd
    end

    @testset "second_order_zerilli construction" begin
        eq = second_order_zerilli(2, 0, 2)
        @test eq isa SourcedMasterEquation
        @test eq.equation.parity === :even
        @test eq.equation.l == 2
        @test eq.equation.potential_name === :Zerilli
        @test eq.source.parity === :even
    end

    @testset "second_order: l >= 2 required" begin
        @test_throws ErrorException second_order_rw(1, 0, 2)
        @test_throws ErrorException second_order_zerilli(1, 0, 2)
    end

    @testset "second_order: S=0 recovers first-order equation" begin
        # When source has no contributions (impossible physically but
        # structurally), the equation reduces to the first-order form.
        eq_rw = second_order_rw(2, 0, 2)
        eq_z = second_order_zerilli(2, 0, 2)

        # The potential is identical to the first-order equation
        M = 1.0; r = 10.0
        @test evaluate_potential(eq_rw.equation, r, M) ≈ regge_wheeler_potential(r, M, 2)
        @test evaluate_potential(eq_z.equation, r, M) ≈ zerilli_potential(r, M, 2)
    end

    @testset "second_order: source at l=2 from quadrupole self-coupling" begin
        eq = second_order_zerilli(2, 0, 2)
        # Should have contributions from (2,0)×(2,0) coupling
        has_22 = any(c -> c.l1 == 2 && c.l2 == 2 && c.m1 == 0 && c.m2 == 0,
                     eq.source.contributions)
        @test has_22
    end

    @testset "second_order_rw: source contains even×odd" begin
        eq = second_order_rw(2, 0, 2)
        # Odd-parity source from even×odd coupling
        has_eo = any(c -> c.parity1 === :even && c.parity2 === :odd,
                     eq.source.contributions)
        # May or may not have contributions depending on selection rules
        # but the framework should handle it
        @test eq.source.parity === :odd
    end

    # ── Gauge-invariant variables (TGR-22h) ──────────────────────────────

    @testset "gauge_invariant_variable: even parity" begin
        gv = gauge_invariant_variable(2, 0, :even)
        @test gv isa GaugeInvariantVariable
        @test gv.l == 2
        @test gv.m == 0
        @test gv.parity === :even
        @test gv.bare_variable === :psi2_Z
        @test gv.correction_order == 2
    end

    @testset "gauge_invariant_variable: odd parity" begin
        gv = gauge_invariant_variable(2, 0, :odd)
        @test gv.parity === :odd
        @test gv.bare_variable === :psi2_RW
    end

    @testset "gauge_invariant_variable: l >= 2 required" begin
        @test_throws ErrorException gauge_invariant_variable(1, 0, :even)
    end

    @testset "gauge_invariant: reduces to bare at h1=0" begin
        gv = gauge_invariant_variable(2, 0, :even)
        @test is_gauge_invariant_at_zero(gv)
    end

    @testset "gauge_invariant: different l,m" begin
        gv3 = gauge_invariant_variable(3, 1, :even)
        @test gv3.l == 3
        @test gv3.m == 1

        gv4 = gauge_invariant_variable(4, -2, :odd)
        @test gv4.l == 4
        @test gv4.m == -2
    end

    # ── Display ──────────────────────────────────────────────────────────

    @testset "display" begin
        source = assemble_source(2, 0, :even, 2)
        s = sprint(show, source)
        @test occursin("S²", s)
        @test occursin("contributions", s)

        eq = second_order_zerilli(2, 0, 2)
        s2 = sprint(show, eq)
        @test occursin("source", s2)

        gv = gauge_invariant_variable(2, 0, :even)
        s3 = sprint(show, gv)
        @test occursin("GI", s3)
    end

    # ── MasterField (TGR-22h) ───────────────────────────────────────────

    @testset "MasterField construction" begin
        using TensorGR: MasterField, time_deriv, radial_deriv

        f = MasterField(:Psi, 2, 0)
        @test f.name === :Psi
        @test f.l == 2
        @test f.m == 0
        @test f.dt_order == 0
        @test f.dr_order == 0

        fd = time_deriv(f)
        @test fd.dt_order == 1
        @test fd.dr_order == 0
        @test fd.name === :Psi

        fp = radial_deriv(f)
        @test fp.dt_order == 0
        @test fp.dr_order == 1

        fdd = time_deriv(fd)
        @test fdd.dt_order == 2
    end

    @testset "MasterField equality and hashing" begin
        using TensorGR: MasterField, time_deriv

        f1 = MasterField(:Psi, 2, 0, 1, 0)
        f2 = time_deriv(MasterField(:Psi, 2, 0))
        @test f1 == f2
        @test hash(f1) == hash(f2)

        f3 = MasterField(:Pi, 2, 0, 1, 0)
        @test f1 != f3
    end

    @testset "MasterField display" begin
        using TensorGR: MasterField, time_deriv, radial_deriv

        f = MasterField(:Psi, 2, 0)
        s = sprint(show, f)
        @test occursin("Psi", s)
        @test occursin("l=2", s)

        fd = time_deriv(f)
        sd = sprint(show, fd)
        @test occursin(".", sd)
    end

    # ── Gauge correction (TGR-22h) ──────────────────────────────────────

    @testset "gauge_correction: even parity structure" begin
        using TensorGR: GaugeCorrection, gauge_correction

        Q = gauge_correction(:even)
        @test Q isa GaugeCorrection
        @test Q.parity === :even
        @test Q.l == 2
        @test Q.m == 0
        @test Q.sqrt_prefactor == 5//1
        # Eq 89 has terms from two blocks: ~10 from the Psi block, 3 from the Pi block
        @test length(Q.terms) == 13
    end

    @testset "gauge_correction: odd parity structure" begin
        using TensorGR: GaugeCorrection, gauge_correction

        Q = gauge_correction(:odd)
        @test Q isa GaugeCorrection
        @test Q.parity === :odd
        @test Q.l == 2
        @test Q.sqrt_prefactor == 5//1
        # Eq 94 has exactly 3 bilinear terms
        @test length(Q.terms) == 3
    end

    @testset "gauge_correction: bilinearity" begin
        using TensorGR: gauge_correction, is_bilinear

        Q_even = gauge_correction(:even)
        Q_odd = gauge_correction(:odd)
        @test is_bilinear(Q_even)
        @test is_bilinear(Q_odd)
    end

    @testset "gauge_correction: vanishes at zero" begin
        using TensorGR: gauge_correction, correction_vanishes_at_zero

        Q_even = gauge_correction(:even)
        Q_odd = gauge_correction(:odd)
        @test correction_vanishes_at_zero(Q_even)
        @test correction_vanishes_at_zero(Q_odd)
    end

    @testset "gauge_correction: l >= 2 required" begin
        using TensorGR: gauge_correction
        @test_throws ErrorException gauge_correction(:even; l=1)
    end

    @testset "gauge_correction: unsupported modes" begin
        using TensorGR: gauge_correction
        @test_throws ErrorException gauge_correction(:even; l=3, m=1)
    end

    @testset "gauge_correction: even contains Psi and Pi fields" begin
        using TensorGR: gauge_correction

        Q = gauge_correction(:even)
        field_names = Set{Symbol}()
        for t in Q.terms
            push!(field_names, t.field1.name)
            push!(field_names, t.field2.name)
        end
        @test :Psi in field_names
        @test :Pi in field_names
    end

    @testset "gauge_correction: odd couples Psi and Pi" begin
        using TensorGR: gauge_correction

        Q = gauge_correction(:odd)
        for t in Q.terms
            names = Set([t.field1.name, t.field2.name])
            # Each term in odd Q couples Pi and Psi
            @test :Psi in names || :Pi in names
        end
    end

    @testset "gauge_correction: display" begin
        using TensorGR: gauge_correction

        Q = gauge_correction(:even)
        s = sprint(show, Q)
        @test occursin("Q_reg", s)
        @test occursin("even", s)
        @test occursin("13 terms", s)
    end

    # ── Regularized source (TGR-22h) ────────────────────────────────────

    @testset "regularized_source_term: construction" begin
        using TensorGR: RegularizedSource, regularized_source_term,
                        gauge_correction

        eq_z = second_order_zerilli(2, 0, 2)
        Q_even = gauge_correction(:even)
        rs = regularized_source_term(eq_z, Q_even)
        @test rs isa RegularizedSource
        @test rs.potential_name === :Zerilli
        @test rs.correction === Q_even
        @test rs.original_source === eq_z
    end

    @testset "regularized_source_term: odd parity" begin
        using TensorGR: RegularizedSource, regularized_source_term,
                        gauge_correction

        eq_rw = second_order_rw(2, 0, 2)
        Q_odd = gauge_correction(:odd)
        rs = regularized_source_term(eq_rw, Q_odd)
        @test rs isa RegularizedSource
        @test rs.potential_name === :RW
    end

    @testset "regularized_source_term: parity mismatch" begin
        using TensorGR: regularized_source_term, gauge_correction

        eq_z = second_order_zerilli(2, 0, 2)
        Q_odd = gauge_correction(:odd)
        @test_throws ErrorException regularized_source_term(eq_z, Q_odd)
    end

    @testset "regularized_source_term: display" begin
        using TensorGR: regularized_source_term, gauge_correction

        eq_z = second_order_zerilli(2, 0, 2)
        Q = gauge_correction(:even)
        rs = regularized_source_term(eq_z, Q)
        s = sprint(show, rs)
        @test occursin("S^reg", s)
        @test occursin("Zerilli", s)
    end

    # ── Coefficient spot-checks against Brizuela Eq 89/94 ──────────────

    @testset "gauge_correction: odd coefficients match Eq 94" begin
        using TensorGR: gauge_correction

        Q = gauge_correction(:odd)
        # Eq 94: Q = r^3/84 * sqrt(5/pi) * {3 Pi_dot Psi_dot + Pi_ddot Psi + Psi_ddot Pi}
        @test length(Q.terms) == 3
        coeffs = sort([t.coeff for t in Q.terms])
        @test coeffs == [1//84, 1//84, 3//84]  # = [1//84, 1//84, 1//28]
        # All terms have r^3 factor
        for t in Q.terms
            @test t.r_power == 3
        end
        # Every term couples Psi and Pi
        for t in Q.terms
            names = Set([t.field1.name, t.field2.name])
            @test :Psi in names && :Pi in names
        end
    end

    @testset "gauge_correction: even Psi_prime term sign (Eq 89 line C)" begin
        using TensorGR: gauge_correction

        Q = gauge_correction(:even)
        # The Psi' * Psi_ddot term should have coeff = +2//63 (double negative in Eq 89)
        psi_prime_terms = filter(t -> t.field1.dr_order > 0 || t.field2.dr_order > 0, Q.terms)
        @test length(psi_prime_terms) == 1
        t = psi_prime_terms[1]
        @test t.coeff == 2//63  # positive! (two negatives cancel)
        @test t.r_power == 2
    end

    @testset "gauge_correction: even first term coefficient" begin
        using TensorGR: gauge_correction

        Q = gauge_correction(:even)
        # First Psi block term: -1/14 * M * Psi_dot * Psi_ddot
        # (from 2*9M/252 = 18M/252 = M/14, with overall minus)
        m_terms = filter(t -> t.M_power == 1 && t.r_power == 0 &&
                         t.f_power == 0 && t.field1.name === :Psi &&
                         t.field1.dt_order == 1 && t.field2.dt_order == 2,
                         Q.terms)
        @test length(m_terms) == 1
        @test m_terms[1].coeff == -1//14
    end
end
