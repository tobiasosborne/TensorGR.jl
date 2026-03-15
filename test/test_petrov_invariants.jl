@testset "Petrov Invariants (I, J)" begin
    # ================================================================
    # Helper: build Schwarzschild curvature numerically at a given (r, theta)
    # ================================================================
    function schwarzschild_curvature(M, r, theta)
        f = 1.0 - 2M / r
        g = zeros(4, 4)
        g[1, 1] = -f
        g[2, 2] = 1.0 / f
        g[3, 3] = r^2
        g[4, 4] = r^2 * sin(theta)^2

        ginv = zeros(4, 4)
        for i in 1:4; ginv[i, i] = 1.0 / g[i, i]; end

        function schw_metric(point)
            t, rv, th, ph = point
            fv = 1.0 - 2M / rv
            gp = zeros(4, 4)
            gp[1, 1] = -fv
            gp[2, 2] = 1.0 / fv
            gp[3, 3] = rv^2
            gp[4, 4] = rv^2 * sin(th)^2
            gp
        end

        function schw_ginv(point)
            gp = schw_metric(point)
            gi = zeros(4, 4)
            for i in 1:4; gi[i, i] = 1.0 / gp[i, i]; end
            gi
        end

        x0 = [0.0, r, theta, 0.0]
        eps_fd = 1e-5

        function gamma_at(point)
            gp = schw_metric(point)
            gi = schw_ginv(point)
            G = Array{Float64}(undef, 4, 4, 4)
            for a in 1:4, b in 1:4, c in 1:4
                s = 0.0
                for d in 1:4
                    xp = copy(point); xp[b] += eps_fd
                    xm = copy(point); xm[b] -= eps_fd
                    dg1 = (schw_metric(xp)[c,d] - schw_metric(xm)[c,d]) / (2eps_fd)
                    xp = copy(point); xp[c] += eps_fd
                    xm = copy(point); xm[c] -= eps_fd
                    dg2 = (schw_metric(xp)[b,d] - schw_metric(xm)[b,d]) / (2eps_fd)
                    xp = copy(point); xp[d] += eps_fd
                    xm = copy(point); xm[d] -= eps_fd
                    dg3 = (schw_metric(xp)[b,c] - schw_metric(xm)[b,c]) / (2eps_fd)
                    s += gi[a,d] * (dg1 + dg2 - dg3)
                end
                G[a,b,c] = s / 2
            end
            G
        end

        Gamma = gamma_at(x0)

        Riem = Array{Float64}(undef, 4, 4, 4, 4)
        for a in 1:4, b in 1:4, c in 1:4, d in 1:4
            xp = copy(x0); xp[c] += eps_fd
            xm = copy(x0); xm[c] -= eps_fd
            dG1 = (gamma_at(xp)[a,d,b] - gamma_at(xm)[a,d,b]) / (2eps_fd)
            xp = copy(x0); xp[d] += eps_fd
            xm = copy(x0); xm[d] -= eps_fd
            dG2 = (gamma_at(xp)[a,c,b] - gamma_at(xm)[a,c,b]) / (2eps_fd)
            val = dG1 - dG2
            for e in 1:4
                val += Gamma[a,c,e] * Gamma[e,d,b] - Gamma[a,d,e] * Gamma[e,c,b]
            end
            Riem[a,b,c,d] = val
        end

        Ric = metric_ricci(Riem, 4)
        R = metric_ricci_scalar(Ric, ginv, 4)
        Weyl = metric_weyl(Riem, Ric, R, g, ginv, 4)

        return (; g, ginv, Riem, Ric, R, Weyl)
    end

    # ================================================================
    # Test 1: Minkowski -- I = 0, J = 0
    # ================================================================
    @testset "Minkowski: I = J = 0" begin
        Weyl = zeros(Float64, 4, 4, 4, 4)
        g = Float64[-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
        inv_t = petrov_invariants(Weyl, g)
        @test isapprox(inv_t.I, 0, atol=1e-14)
        @test isapprox(inv_t.J, 0, atol=1e-14)
        @test is_algebraically_special(inv_t.I, inv_t.J)
    end

    # ================================================================
    # Test 2: Schwarzschild (Type D) -- I^3 = 27 J^2
    # ================================================================
    @testset "Schwarzschild: algebraically special (Type D)" begin
        M = 1.0
        r = 3.0
        theta = pi / 4
        curv = schwarzschild_curvature(M, r, theta)

        inv_t = petrov_invariants(curv.Weyl, curv.g)

        # For Schwarzschild with Kinnersley tetrad: Psi2 = -M/r^3
        # I = 3 Psi2^2 = 3 M^2 / r^6
        I_exact = 3.0 * M^2 / r^6
        @test isapprox(real(inv_t.I), I_exact, rtol=1e-4)

        # Key test: algebraically special criterion I^3 = 27 J^2
        @test isapprox(inv_t.I^3, 27 * inv_t.J^2, rtol=1e-4)
        @test is_algebraically_special(inv_t.I, inv_t.J, atol=1e-4)
    end

    # ================================================================
    # Test 3: Petrov invariants from NP scalars
    # ================================================================
    @testset "From NP scalars: Schwarzschild" begin
        M = 1.0
        r = 3.0
        Psi2 = -M / r^3

        psi = (Psi0=0.0+0im, Psi1=0.0+0im, Psi2=Complex(Psi2),
               Psi3=0.0+0im, Psi4=0.0+0im)
        inv_s = petrov_invariants(psi)

        # I = 3 Psi2^2
        @test isapprox(inv_s.I, 3 * Psi2^2, atol=1e-14)
        # J = -Psi2^3
        @test isapprox(inv_s.J, -Psi2^3, atol=1e-14)
        # Algebraically special
        @test isapprox(inv_s.I^3, 27 * inv_s.J^2, atol=1e-14)
    end

    # ================================================================
    # Test 4: Cross-check tensor vs scalar route (Schwarzschild)
    # ================================================================
    @testset "Tensor vs scalar route agreement" begin
        M = 1.0
        r = 3.0
        theta = pi / 4
        curv = schwarzschild_curvature(M, r, theta)

        # Tensor route (via internal null tetrad)
        inv_t = petrov_invariants(curv.Weyl, curv.g)

        # Scalar route via Kinnersley tetrad
        f = 1.0 - 2M / r
        l    = ComplexF64[1.0/f, 1.0, 0.0, 0.0]
        n    = ComplexF64[0.5, -f/2, 0.0, 0.0]
        m    = ComplexF64[0.0, 0.0, 1.0, im/sin(theta)] / (r * sqrt(2.0))
        mbar = ComplexF64[0.0, 0.0, 1.0, -im/sin(theta)] / (r * sqrt(2.0))

        psi = weyl_scalars(Float64.(curv.Weyl), l, n, m, mbar)
        inv_s = petrov_invariants(psi)

        # Both routes should agree on algebraic specialness
        @test isapprox(inv_t.I^3, 27 * inv_t.J^2, rtol=1e-4)
        @test isapprox(inv_s.I^3, 27 * inv_s.J^2, atol=1e-10)

        # The I values should agree (both go through NP scalars, but with
        # different tetrads -- the invariants are tetrad-independent)
        @test isapprox(real(inv_t.I), real(inv_s.I), rtol=1e-4)
        @test isapprox(real(inv_t.J), real(inv_s.J), rtol=1e-4)
    end

    # ================================================================
    # Test 5: CTensor convenience method
    # ================================================================
    @testset "CTensor convenience method" begin
        Weyl = zeros(Float64, 4, 4, 4, 4)
        g = Float64[-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
        W_ct = CTensor(Weyl, :Mink, fill(Down, 4))
        g_ct = CTensor(g, :Mink, [Down, Down])
        inv_ct = petrov_invariants(W_ct, g_ct)
        @test isapprox(inv_ct.I, 0, atol=1e-14)
        @test isapprox(inv_ct.J, 0, atol=1e-14)
    end

    # ================================================================
    # Test 6: is_algebraically_special edge cases
    # ================================================================
    @testset "is_algebraically_special" begin
        # Type O (flat): trivially special
        @test is_algebraically_special(0.0, 0.0)
        # Type D: I^3 = 27 J^2
        I_d = 3.0
        J_d = sqrt(I_d^3 / 27)
        @test is_algebraically_special(I_d, J_d)
        # Type I (generic): I^3 != 27 J^2
        @test !is_algebraically_special(1.0, 0.5)
    end

    # ================================================================
    # Test 7: NP scalar determinant formula (explicit check)
    # ================================================================
    @testset "Determinant formula explicit" begin
        P0, P1, P2, P3, P4 = 1.0, 2.0, 3.0, 4.0, 5.0
        psi = (Psi0=P0+0im, Psi1=P1+0im, Psi2=P2+0im, Psi3=P3+0im, Psi4=P4+0im)
        inv_g = petrov_invariants(psi)

        # Check I
        I_check = P0*P4 - 4*P1*P3 + 3*P2^2
        @test isapprox(real(inv_g.I), I_check, atol=1e-12)

        # Check J = det of the Q matrix
        Q = [P0 P1 P2; P1 P2 P3; P2 P3 P4]
        using LinearAlgebra: det
        J_check = det(Q)
        @test isapprox(real(inv_g.J), J_check, atol=1e-12)
    end

    # ================================================================
    # Test 8: Weyl contraction invariants (Kretschmann-type)
    # ================================================================
    @testset "Weyl contraction invariants" begin
        M = 1.0
        r = 3.0
        theta = pi / 4
        curv = schwarzschild_curvature(M, r, theta)
        wci = weyl_contraction_invariants(curv.Weyl, curv.ginv)

        # For vacuum: I2 = K/2 where K = 48 M^2 / r^6
        K_exact = 48.0 * M^2 / r^6
        @test isapprox(wci.I2, K_exact / 2, rtol=1e-4)
    end
end
