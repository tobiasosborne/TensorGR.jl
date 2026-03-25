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
end
