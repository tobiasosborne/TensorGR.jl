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

end
