# Tests for Regge-Wheeler and Zerilli master equations.
# Ground truth: Martel & Poisson (2005), PRD 71, 104003, Eqs 4.25-4.26, 5.14-5.15.

@testset "Regge-Wheeler and Zerilli master equations" begin

    @testset "MasterEquation construction and display" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_odd = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == ODD, decomps)
        me = master_equation(l2_odd[1])
        @test me.parity == ODD
        @test me.l == 2
        @test me.mass == :M
        s = sprint(show, me)
        @test occursin("Regge-Wheeler", s)
        @test occursin("l=2", s)
    end

    @testset "MasterEquation equality" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        sp_odd = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == ODD, decomps)[1]
        sp_even = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == EVEN, decomps)[1]
        me1 = master_equation(sp_odd)
        me2 = master_equation(sp_odd)
        me3 = master_equation(sp_even)
        @test me1 == me2
        @test me1 != me3
    end

    @testset "master_equation requires l >= 2" begin
        decomps = decompose_schwarzschild(:h, :M, 1)
        sp_l1_even = filter(sp -> sp.l == 1 && sp.m == 0 && sp.parity == EVEN, decomps)[1]
        @test_throws ArgumentError master_equation(sp_l1_even)

        sp_l0 = filter(sp -> sp.l == 0, decomps)[1]
        @test_throws ArgumentError master_equation(sp_l0)
    end

    @testset "master_equation with explicit parity" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        sp = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == EVEN, decomps)[1]
        # Can construct RW equation from an even-parity SchwarzschildPerturbation
        me_odd = master_equation(sp, ODD)
        @test me_odd.parity == ODD
        me_even = master_equation(sp, EVEN)
        @test me_even.parity == EVEN
    end

    # ── Regge-Wheeler potential ───────────────────────────────────────────────

    @testset "regge_wheeler_potential: l=2 at r=6M (ISCO)" begin
        # V_RW(r) = (1-2M/r)[l(l+1)/r^2 - 6M/r^3]
        # At r=6M, l=2, M=1:
        #   f = 1 - 2/6 = 2/3
        #   V = (2/3)[6/36 - 6/216] = (2/3)[1/6 - 1/36] = (2/3)(5/36) = 10/108 = 5/54
        M = 1
        r = 6M
        V = regge_wheeler_potential(M, 2, r)
        @test V ≈ 5 / 54 atol=1e-14
    end

    @testset "regge_wheeler_potential: vanishes as r -> infinity" begin
        # As r -> inf, f -> 1 and both 1/r^2, 1/r^3 -> 0
        M = 1
        V_far = regge_wheeler_potential(M, 2, 1e6)
        @test abs(V_far) < 1e-10
    end

    @testset "regge_wheeler_potential: vanishes at horizon r=2M" begin
        # f(2M) = 0, so V_RW(2M) = 0
        M = 1.0
        V_horizon = regge_wheeler_potential(M, 2, 2M)
        @test V_horizon ≈ 0.0 atol=1e-14
    end

    @testset "regge_wheeler_potential: requires l >= 2" begin
        @test_throws ArgumentError regge_wheeler_potential(1, 1, 10)
        @test_throws ArgumentError regge_wheeler_potential(1, 0, 10)
    end

    @testset "regge_wheeler_potential: positive definite outside horizon for l=2" begin
        # V_RW = f[l(l+1)/r^2 - 6M/r^3] = f/r^3[l(l+1)r - 6M]
        # For l=2: 6r - 6M > 0 when r > M. Since r > 2M outside horizon, V_RW > 0.
        M = 1.0
        for r in [2.1, 3.0, 6.0, 10.0, 100.0]
            @test regge_wheeler_potential(M, 2, r) > 0
        end
    end

    @testset "regge_wheeler_potential: scaling with l" begin
        # For large r, V_RW ~ l(l+1)/r^2 -- grows with l
        M = 1.0
        r = 100.0
        V2 = regge_wheeler_potential(M, 2, r)
        V3 = regge_wheeler_potential(M, 3, r)
        V4 = regge_wheeler_potential(M, 4, r)
        @test V3 > V2
        @test V4 > V3
    end

    # ── Zerilli potential ─────────────────────────────────────────────────────

    @testset "zerilli_potential: vanishes as r -> infinity" begin
        M = 1
        V_far = zerilli_potential(M, 2, 1e6)
        @test abs(V_far) < 1e-10
    end

    @testset "zerilli_potential: vanishes at horizon r=2M" begin
        M = 1.0
        V_horizon = zerilli_potential(M, 2, 2M)
        @test V_horizon ≈ 0.0 atol=1e-14
    end

    @testset "zerilli_potential: requires l >= 2" begin
        @test_throws ArgumentError zerilli_potential(1, 1, 10)
        @test_throws ArgumentError zerilli_potential(1, 0, 10)
    end

    @testset "zerilli_potential: l=2 at r=6M matches analytic value" begin
        # lambda = (2-1)(2+2)/2 = 2
        # Lambda_r = 2*6 + 3 = 15
        # f = 2/3
        # num = 2*4*3*216 + 6*4*1*36 + 18*2*1*6 + 18*1 = 5184 + 864 + 216 + 18 = 6282
        # V = (2/3) * 6282 / (216 * 225) = (2/3) * 6282 / 48600 = 4188 / 48600
        #   = 1047/12150 = 349/4050
        M = 1
        r = 6
        V = zerilli_potential(M, 2, r)
        expected = Rational{Int}(2, 3) * (2 * 4 * 3 * 216 + 6 * 4 * 36 + 18 * 2 * 6 + 18) //
                   (216 * (2 * 6 + 3)^2)
        @test V ≈ Float64(expected) atol=1e-14
    end

    @testset "zerilli_potential: positive definite outside horizon for l=2" begin
        M = 1.0
        for r in [2.1, 3.0, 6.0, 10.0, 100.0]
            @test zerilli_potential(M, 2, r) > 0
        end
    end

    @testset "zerilli_potential: scaling with l" begin
        M = 1.0
        r = 100.0
        V2 = zerilli_potential(M, 2, r)
        V3 = zerilli_potential(M, 3, r)
        V4 = zerilli_potential(M, 4, r)
        @test V3 > V2
        @test V4 > V3
    end

    # ── Cross-checks between RW and Zerilli potentials ────────────────────────

    @testset "RW and Zerilli potentials agree at large r (leading order)" begin
        # Both potentials behave as l(l+1)/r^2 for r >> M
        M = 1.0
        r = 1e5
        for l in 2:5
            V_rw = regge_wheeler_potential(M, l, r)
            V_z = zerilli_potential(M, l, r)
            expected = l * (l + 1) / r^2
            @test V_rw ≈ expected rtol=1e-4
            @test V_z ≈ expected rtol=1e-4
        end
    end

    @testset "RW and Zerilli potentials: same peak location structure" begin
        # Both have a single peak near r ~ 3M (the photon sphere).
        # The peak values differ but the qualitative structure is the same.
        M = 1.0
        l = 2
        r_grid = range(2.01, 20.0; length=1000)
        V_rw = [regge_wheeler_potential(M, l, r) for r in r_grid]
        V_z = [zerilli_potential(M, l, r) for r in r_grid]
        # Both should have a unique maximum
        i_rw = argmax(V_rw)
        i_z = argmax(V_z)
        # Peak near r ~ 3M (within [2.5M, 4M])
        @test 2.5 < r_grid[i_rw] < 4.0
        @test 2.5 < r_grid[i_z] < 4.0
    end

    # ── MasterEquation from SchwarzschildPerturbation ─────────────────────────

    @testset "master_equation: vacuum source is zero" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        for sp in decomps
            sp.l < 2 && continue
            me = master_equation(sp)
            @test me.source(10.0) == 0
            @test me.source(100.0) == 0
        end
    end

    @testset "master_equation: potential evaluates correctly" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        sp_odd = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == ODD, decomps)[1]
        sp_even = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == EVEN, decomps)[1]

        me_rw = master_equation(sp_odd)
        me_z = master_equation(sp_even)

        # M=1 (unit mass in the closures)
        @test me_rw.potential(6.0) ≈ regge_wheeler_potential(1, 2, 6.0)
        @test me_z.potential(6.0) ≈ zerilli_potential(1, 2, 6.0)
    end

    @testset "master_equation: Zerilli display" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        sp_even = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == EVEN, decomps)[1]
        me = master_equation(sp_even)
        s = sprint(show, me)
        @test occursin("Zerilli", s)
        @test occursin("l=2", s)
    end

    @testset "master_equation: all l=2 modes produce valid equations" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_modes = filter(sp -> sp.l == 2, decomps)
        @test length(l2_modes) == 10  # 5 even + 5 odd (m=-2..2)
        for sp in l2_modes
            me = master_equation(sp)
            @test me.l == 2
            @test me.potential(10.0) > 0
        end
    end

end
