@testset "Geodesics" begin

    # Helper: build a diagonal matrix without LinearAlgebra
    _diag(v) = Float64[i == j ? v[i] : 0.0 for i in 1:length(v), j in 1:length(v)]

    @testset "GeodesicEquation struct" begin
        mfn = x -> (_diag(ones(2)), _diag(ones(2)))
        cfn = x -> zeros(2, 2, 2)
        geq = GeodesicEquation(mfn, cfn, 2, true)
        @test geq.dim == 2
        @test geq.is_timelike == true
    end

    @testset "setup_geodesic with metric_fn" begin
        # Flat 2D Euclidean metric
        function flat2d_setup(x)
            g = _diag([1.0, 1.0])
            ginv = _diag([1.0, 1.0])
            (g, ginv)
        end
        geq = setup_geodesic(flat2d_setup; dim=2, is_timelike=false)
        @test geq.dim == 2
        @test geq.is_timelike == false
        # Christoffel should be zero for flat metric
        Gamma = geq.christoffel_fn([1.0, 1.0])
        for a in 1:2, b in 1:2, c in 1:2
            @test abs(Gamma[a, b, c]) < 1e-10
        end
    end

    @testset "setup_geodesic with custom christoffel_fn" begin
        mfn = x -> (_diag(ones(3)), _diag(ones(3)))
        custom_gamma = x -> zeros(3, 3, 3)
        geq = setup_geodesic(mfn; dim=3, christoffel_fn=custom_gamma)
        @test geq.dim == 3
        Gamma = geq.christoffel_fn([0.0, 0.0, 0.0])
        @test all(Gamma .== 0.0)
    end

    @testset "Flat Minkowski: straight-line geodesics" begin
        function minkowski4d(x)
            g = _diag([-1.0, 1.0, 1.0, 1.0])
            ginv = _diag([-1.0, 1.0, 1.0, 1.0])
            (g, ginv)
        end
        geq = setup_geodesic(minkowski4d; dim=4)

        # All Christoffels should vanish
        Gamma = geq.christoffel_fn([0.0, 1.0, 2.0, 3.0])
        for a in 1:4, b in 1:4, c in 1:4
            @test abs(Gamma[a, b, c]) < 1e-10
        end

        # Test RHS: with zero Christoffels, acceleration = 0, velocities propagate
        u = [0.0, 0.0, 0.0, 0.0,   # position
             1.0, 0.5, 0.3, 0.1]    # velocity
        du = similar(u)
        geodesic_rhs!(du, u, geq, 0.0)

        # du[1:4] should be velocities
        @test du[1] == 1.0
        @test du[2] == 0.5
        @test du[3] == 0.3
        @test du[4] == 0.1

        # du[5:8] should be accelerations = 0
        for i in 5:8
            @test abs(du[i]) < 1e-10
        end
    end

    @testset "RHS output format" begin
        function flat3d_fmt(x)
            g = _diag([1.0, 1.0, 1.0])
            ginv = _diag([1.0, 1.0, 1.0])
            (g, ginv)
        end
        geq = setup_geodesic(flat3d_fmt; dim=3, is_timelike=false)

        u = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        du = similar(u)
        geodesic_rhs!(du, u, geq, 0.0)

        # du has same length as u
        @test length(du) == length(u)
        # Velocities propagate to position derivatives
        @test du[1] == u[4]
        @test du[2] == u[5]
        @test du[3] == u[6]
    end

    @testset "Diagonal metric: Schwarzschild-like Christoffel" begin
        # Schwarzschild-like 2D radial slice: ds^2 = -f(r)dt^2 + dr^2/f(r)
        # with f(r) = 1 - 2M/r. Use M=1, evaluate at r=4.
        # g = diag(-f, 1/f), ginv = diag(-1/f, f)
        # Non-trivial Christoffels:
        #   Gamma^t_{tr} = Gamma^t_{rt} = f'/(2f)
        #   Gamma^r_{tt} = f*f'/2
        #   Gamma^r_{rr} = -f'/(2f)
        # where f' = df/dr = 2M/r^2

        M_bh = 1.0

        function schwarzschild_2d(x)
            # x[1] = t, x[2] = r
            r = x[2]
            f = 1.0 - 2.0 * M_bh / r
            g = _diag([-f, 1.0 / f])
            ginv = _diag([-1.0 / f, f])
            (g, ginv)
        end

        geq = setup_geodesic(schwarzschild_2d; dim=2)

        r0 = 4.0
        x0 = [0.0, r0]
        Gamma = geq.christoffel_fn(x0)

        f0 = 1.0 - 2.0 * M_bh / r0       # = 0.5
        fp = 2.0 * M_bh / r0^2            # = 0.125

        # Gamma^t_{tr} = Gamma^t_{rt} = f'/(2f) = 0.125 / (2*0.5) = 0.125
        @test abs(Gamma[1, 1, 2] - fp / (2 * f0)) < 1e-5
        @test abs(Gamma[1, 2, 1] - fp / (2 * f0)) < 1e-5

        # Gamma^r_{tt} = f * f' / 2 = 0.5 * 0.125 / 2 = 0.03125
        @test abs(Gamma[2, 1, 1] - f0 * fp / 2) < 1e-5

        # Gamma^r_{rr} = -f'/(2f) = -0.125
        @test abs(Gamma[2, 2, 2] - (-fp / (2 * f0))) < 1e-5

        # Off-diagonal and remaining should be near zero
        @test abs(Gamma[1, 2, 2]) < 1e-5
        @test abs(Gamma[2, 1, 2]) < 1e-5
        @test abs(Gamma[2, 2, 1]) < 1e-5
        @test abs(Gamma[1, 1, 1]) < 1e-5
    end

    @testset "Geodesic RHS with non-zero Christoffel" begin
        # Use polar coordinates: ds^2 = dr^2 + r^2 d(theta)^2
        # Gamma^r_{theta,theta} = -r, Gamma^theta_{r,theta} = Gamma^theta_{theta,r} = 1/r

        function polar2d(x)
            r = x[1]
            g = _diag([1.0, r^2])
            ginv = _diag([1.0, 1.0 / r^2])
            (g, ginv)
        end

        geq = setup_geodesic(polar2d; dim=2, is_timelike=false)

        # At r=2.0, theta=0
        r0 = 2.0
        u = [r0, 0.0,   # position: r, theta
             0.0, 1.0]   # velocity: v^r=0, v^theta=1
        du = similar(u)
        geodesic_rhs!(du, u, geq, 0.0)

        # Velocities propagate
        @test du[1] == 0.0
        @test du[2] == 1.0

        # Acceleration in r: a^r = -Gamma^r_{theta,theta} * v^theta * v^theta = r * 1 = 2.0
        @test abs(du[3] - r0) < 1e-5

        # Acceleration in theta: a^theta = -2 * Gamma^theta_{r,theta} * v^r * v^theta = 0
        # (since v^r = 0)
        @test abs(du[4]) < 1e-5
    end

    @testset "Euler integration: flat space straight line" begin
        # Simple Euler integration in flat 2D: should follow straight line
        function flat2d_euler(x)
            g = _diag([1.0, 1.0])
            ginv = _diag([1.0, 1.0])
            (g, ginv)
        end

        geq = setup_geodesic(flat2d_euler; dim=2, is_timelike=false)

        u = [0.0, 0.0, 1.0, 2.0]  # start at origin, velocity (1, 2)
        du = similar(u)
        dt = 0.01
        nsteps = 100

        for _ in 1:nsteps
            geodesic_rhs!(du, u, geq, 0.0)
            u .+= dt .* du
        end

        # After time T=1.0, should be at (1.0, 2.0) with same velocity
        @test abs(u[1] - 1.0) < 1e-10
        @test abs(u[2] - 2.0) < 1e-10
        @test abs(u[3] - 1.0) < 1e-10
        @test abs(u[4] - 2.0) < 1e-10
    end

    # ================================================================
    # Integration validation tests (require DifferentialEquations.jl)
    # ================================================================

    # Helper: compute the norm g_mu_nu v^mu v^nu at a point
    function _geodesic_norm(metric_fn, x, v)
        g, _ = metric_fn(x)
        s = 0.0
        dim = length(x)
        for mu in 1:dim, nu in 1:dim
            s += g[mu, nu] * v[mu] * v[nu]
        end
        s
    end

    # Full 4D Schwarzschild metric in (t, r, theta, phi) coordinates.
    # ds^2 = -f dt^2 + dr^2/f + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2
    # with f(r) = 1 - 2M/r.
    function _schwarzschild_4d(x; M=1.0)
        r = x[2]
        theta = x[3]
        f = 1.0 - 2.0 * M / r
        sth2 = sin(theta)^2
        g = _diag([-f, 1.0/f, r^2, r^2 * sth2])
        ginv = _diag([-1.0/f, f, 1.0/r^2, 1.0/(r^2 * sth2)])
        (g, ginv)
    end

    _diffeq_available = try
        @eval using DifferentialEquations
        true
    catch
        false
    end

    if _diffeq_available

    @testset "Schwarzschild ISCO circular orbit (r=6M)" begin
        # Circular orbit at the innermost stable circular orbit (ISCO)
        # for Schwarzschild with M=1: r_ISCO = 6M = 6.
        #
        # At r=6M, theta=pi/2 (equatorial plane):
        #   f = 1 - 2/6 = 2/3
        #   Angular momentum per unit mass: L^2 = 12 M^2 => L = 2*sqrt(3)
        #   Energy per unit mass: E^2 = 8/9 => E = 2*sqrt(2)/3
        #   Omega = dphi/dt = sqrt(M/r^3) = 1/(6*sqrt(6))
        #
        # For a massive particle, normalize: g_mu_nu v^mu v^nu = -1.
        # With v^r = 0, v^theta = 0, the normalization gives:
        #   -f (v^t)^2 + r^2 (v^phi)^2 = -1
        # Combined with L = r^2 v^phi and E = f v^t:
        #   v^t = E/f, v^phi = L/r^2

        M_bh = 1.0
        r_isco = 6.0 * M_bh
        f_isco = 1.0 - 2.0 * M_bh / r_isco  # 2/3

        E_isco = sqrt(8.0 / 9.0)
        L_isco = 2.0 * sqrt(3.0)

        vt = E_isco / f_isco
        vphi = L_isco / r_isco^2

        x0 = [0.0, r_isco, pi/2, 0.0]
        v0 = [vt, 0.0, 0.0, vphi]

        metric_fn = x -> _schwarzschild_4d(x; M=M_bh)
        geq = setup_geodesic(metric_fn; dim=4, is_timelike=true)

        # Verify initial norm is -1
        norm0 = _geodesic_norm(metric_fn, x0, v0)
        @test abs(norm0 - (-1.0)) < 1e-10

        # Integrate for one full orbit: period in proper time
        # dphi/dtau = vphi, full orbit = 2*pi => tau_orbit = 2*pi / vphi
        tau_orbit = 2.0 * pi / vphi
        sol = integrate_geodesic(geq, x0, v0, (0.0, tau_orbit);
                                 abstol=1e-12, reltol=1e-12)

        @test sol.retcode == :Success

        # Check norm conservation at every saved point (should stay at -1)
        max_norm_err = 0.0
        for i in eachindex(sol.t)
            norm_i = _geodesic_norm(metric_fn, sol.x[i], sol.v[i])
            max_norm_err = max(max_norm_err, abs(norm_i - (-1.0)))
        end
        @test max_norm_err < 1e-6

        # Radius should remain at r=6M throughout (circular orbit)
        max_r_err = 0.0
        for i in eachindex(sol.t)
            max_r_err = max(max_r_err, abs(sol.x[i][2] - r_isco))
        end
        @test max_r_err < 1e-6

        # Theta should remain at pi/2 (equatorial plane)
        max_theta_err = 0.0
        for i in eachindex(sol.t)
            max_theta_err = max(max_theta_err, abs(sol.x[i][3] - pi/2))
        end
        @test max_theta_err < 1e-6
    end

    @testset "Schwarzschild null geodesic (norm = 0)" begin
        # A radial null geodesic in Schwarzschild spacetime.
        # At theta=pi/2 with only radial motion: ds^2 = 0 requires
        #   -f (v^t)^2 + (v^r)^2/f = 0
        # so v^r = f * v^t (outgoing) or v^r = -f * v^t (ingoing).
        #
        # Choose outgoing null ray starting at r=10M.

        M_bh = 1.0
        r0 = 10.0
        f0 = 1.0 - 2.0 * M_bh / r0  # 0.8

        vt_null = 1.0
        vr_null = f0 * vt_null  # outgoing

        x0 = [0.0, r0, pi/2, 0.0]
        v0 = [vt_null, vr_null, 0.0, 0.0]

        metric_fn = x -> _schwarzschild_4d(x; M=M_bh)
        geq = setup_geodesic(metric_fn; dim=4, is_timelike=false)

        # Verify initial norm is 0
        norm0 = _geodesic_norm(metric_fn, x0, v0)
        @test abs(norm0) < 1e-10

        # Short integration (affine parameter)
        sol = integrate_geodesic(geq, x0, v0, (0.0, 20.0);
                                 abstol=1e-12, reltol=1e-12)

        @test sol.retcode == :Success

        # Norm should stay 0 along the entire trajectory
        max_norm_err = 0.0
        for i in eachindex(sol.t)
            norm_i = _geodesic_norm(metric_fn, sol.x[i], sol.v[i])
            max_norm_err = max(max_norm_err, abs(norm_i))
        end
        @test max_norm_err < 1e-6

        # Outgoing ray: r should be increasing
        @test sol.x[end][2] > r0
    end

    @testset "Schwarzschild energy conservation (eccentric orbit)" begin
        # A slightly eccentric orbit (not circular) to verify that
        # g_mu_nu v^mu v^nu remains constant throughout the trajectory.
        #
        # Start at r=10M with a small radial velocity perturbation from
        # the circular orbit values. Circular at r=10M:
        #   L^2 = M r^2 / (r - 3M) = 100/7
        #   E^2 = f^2 / (1 - 3M/r) = (0.8)^2 / 0.7 = 64/70
        #   v^t = E/f, v^phi = L/r^2

        M_bh = 1.0
        r0 = 10.0
        f0 = 1.0 - 2.0 * M_bh / r0  # 0.8

        L_circ = sqrt(M_bh * r0^2 / (r0 - 3.0 * M_bh))
        E_circ = f0 / sqrt(1.0 - 3.0 * M_bh / r0)
        vt_circ = E_circ / f0
        vphi_circ = L_circ / r0^2

        # Add a small radial kick to make it eccentric
        vr_kick = 0.01

        x0 = [0.0, r0, pi/2, 0.0]
        v0_ecc = [vt_circ, vr_kick, 0.0, vphi_circ]

        metric_fn = x -> _schwarzschild_4d(x; M=M_bh)

        # Re-normalize v^t so that g_mu_nu v^mu v^nu = -1 exactly
        # -f (v^t)^2 + (v^r)^2/f + r^2 (v^phi)^2 = -1
        # => (v^t)^2 = (1 + (v^r)^2/f + r^2 (v^phi)^2) / f
        vt_sq = (1.0 + vr_kick^2 / f0 + r0^2 * vphi_circ^2) / f0
        v0_ecc[1] = sqrt(vt_sq)

        # Verify initial norm is -1
        norm0 = _geodesic_norm(metric_fn, x0, v0_ecc)
        @test abs(norm0 - (-1.0)) < 1e-10

        geq = setup_geodesic(metric_fn; dim=4, is_timelike=true)

        # Integrate for a modest proper time interval
        sol = integrate_geodesic(geq, x0, v0_ecc, (0.0, 200.0);
                                 abstol=1e-12, reltol=1e-12)

        @test sol.retcode == :Success

        # Norm should remain -1 throughout
        max_norm_err = 0.0
        for i in eachindex(sol.t)
            norm_i = _geodesic_norm(metric_fn, sol.x[i], sol.v[i])
            max_norm_err = max(max_norm_err, abs(norm_i - (-1.0)))
        end
        @test max_norm_err < 1e-6

        # The orbit should remain bounded (no escape or plunge for this mild kick)
        for i in eachindex(sol.t)
            @test sol.x[i][2] > 4.0 * M_bh   # stays well outside horizon
            @test sol.x[i][2] < 20.0 * M_bh   # stays reasonably close
        end
    end

    else
        @info "Skipping geodesic integration tests: DifferentialEquations.jl not available"
    end  # _diffeq_available

end
