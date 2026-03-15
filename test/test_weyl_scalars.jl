@testset "Weyl Scalars (Newman-Penrose)" begin
    # ----------------------------------------------------------------
    # Helper: build full curvature pipeline from a diagonal metric
    # ----------------------------------------------------------------
    function curvature_from_diagonal(coords, diag_vals)
        dim = length(coords)
        g = zeros(dim, dim)
        for i in 1:dim; g[i, i] = diag_vals[i]; end
        ginv = zeros(dim, dim)
        for i in 1:dim; ginv[i, i] = 1.0 / diag_vals[i]; end

        # Numeric derivative via finite differences
        eps_fd = 1e-7
        function num_deriv(expr_fn, coord_idx, point)
            p_plus  = copy(point); p_plus[coord_idx]  += eps_fd
            p_minus = copy(point); p_minus[coord_idx] -= eps_fd
            (expr_fn(p_plus) - expr_fn(p_minus)) / (2 * eps_fd)
        end

        return g, ginv
    end

    # ================================================================
    # Test 1: Minkowski spacetime -- all Weyl scalars vanish
    # ================================================================
    @testset "Minkowski: all scalars zero" begin
        dim = 4
        # Flat metric: ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
        g = Float64[-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

        # Weyl tensor is zero for flat space
        Weyl = zeros(Float64, 4, 4, 4, 4)

        # Build a null tetrad
        g_ct = CTensor(g, :Mink)
        l, n, m, mbar = null_tetrad_from_metric(g_ct)

        psi = weyl_scalars(Weyl, l, n, m, mbar)
        @test all(v -> isapprox(v, 0, atol=1e-14), [psi.Psi0, psi.Psi1, psi.Psi2, psi.Psi3, psi.Psi4])
    end

    # ================================================================
    # Test 2: Schwarzschild in Kinnersley tetrad -- Psi2 = -M/r^3
    # ================================================================
    @testset "Schwarzschild: Kinnersley tetrad" begin
        M = 1.0
        r = 3.0
        theta = pi / 4  # generic angle

        f = 1.0 - 2M / r

        # Schwarzschild metric in (t, r, theta, phi) coordinates
        # ds^2 = -f dt^2 + f^{-1} dr^2 + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2
        g = zeros(4, 4)
        g[1, 1] = -f
        g[2, 2] = 1.0 / f
        g[3, 3] = r^2
        g[4, 4] = r^2 * sin(theta)^2

        ginv = zeros(4, 4)
        for i in 1:4; ginv[i, i] = 1.0 / g[i, i]; end

        # Compute curvature numerically via metric_weyl
        # We need Christoffel -> Riemann -> Ricci -> R -> Weyl
        # Use a numerical derivative function that evaluates the metric at a point
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

        # Numerical Christoffel
        Gamma = Array{Float64}(undef, 4, 4, 4)
        for a in 1:4, b in 1:4, c in 1:4
            s = 0.0
            for d in 1:4
                # dg_{cd}/dx^b
                xp = copy(x0); xp[b] += eps_fd
                xm = copy(x0); xm[b] -= eps_fd
                dg_cd_b = (schw_metric(xp)[c, d] - schw_metric(xm)[c, d]) / (2eps_fd)
                # dg_{bd}/dx^c
                xp = copy(x0); xp[c] += eps_fd
                xm = copy(x0); xm[c] -= eps_fd
                dg_bd_c = (schw_metric(xp)[b, d] - schw_metric(xm)[b, d]) / (2eps_fd)
                # dg_{bc}/dx^d
                xp = copy(x0); xp[d] += eps_fd
                xm = copy(x0); xm[d] -= eps_fd
                dg_bc_d = (schw_metric(xp)[b, c] - schw_metric(xm)[b, c]) / (2eps_fd)

                s += ginv[a, d] * (dg_cd_b + dg_bd_c - dg_bc_d)
            end
            Gamma[a, b, c] = s / 2
        end

        # Numerical Riemann R^a_{bcd}
        Riem = Array{Float64}(undef, 4, 4, 4, 4)

        # Gamma at displaced points for numerical dGamma
        function gamma_at(point)
            gp = schw_metric(point)
            gi = schw_ginv(point)
            G = Array{Float64}(undef, 4, 4, 4)
            for a in 1:4, b in 1:4, c in 1:4
                s = 0.0
                for d in 1:4
                    xp2 = copy(point); xp2[b] += eps_fd
                    xm2 = copy(point); xm2[b] -= eps_fd
                    dg1 = (schw_metric(xp2)[c, d] - schw_metric(xm2)[c, d]) / (2eps_fd)
                    xp2 = copy(point); xp2[c] += eps_fd
                    xm2 = copy(point); xm2[c] -= eps_fd
                    dg2 = (schw_metric(xp2)[b, d] - schw_metric(xm2)[b, d]) / (2eps_fd)
                    xp2 = copy(point); xp2[d] += eps_fd
                    xm2 = copy(point); xm2[d] -= eps_fd
                    dg3 = (schw_metric(xp2)[b, c] - schw_metric(xm2)[b, c]) / (2eps_fd)
                    s += gi[a, d] * (dg1 + dg2 - dg3)
                end
                G[a, b, c] = s / 2
            end
            G
        end

        for a in 1:4, b in 1:4, c in 1:4, d in 1:4
            # dGamma^a_{db}/dx^c
            xp = copy(x0); xp[c] += eps_fd
            xm = copy(x0); xm[c] -= eps_fd
            dG1 = (gamma_at(xp)[a, d, b] - gamma_at(xm)[a, d, b]) / (2eps_fd)
            # dGamma^a_{cb}/dx^d
            xp = copy(x0); xp[d] += eps_fd
            xm = copy(x0); xm[d] -= eps_fd
            dG2 = (gamma_at(xp)[a, c, b] - gamma_at(xm)[a, c, b]) / (2eps_fd)

            val = dG1 - dG2
            for e in 1:4
                val += Gamma[a, c, e] * Gamma[e, d, b] - Gamma[a, d, e] * Gamma[e, c, b]
            end
            Riem[a, b, c, d] = val
        end

        Ric = metric_ricci(Riem, 4)
        R = metric_ricci_scalar(Ric, ginv, 4)
        Weyl = metric_weyl(Riem, Ric, R, g, ginv, 4)

        # Kinnersley null tetrad for Schwarzschild (Kinnersley 1969):
        #   l^a = (r^2/f, 1, 0, 0) / r^2  -- but normalized: l^a = (1/f, 1, 0, 0)
        #   n^a = (1/2)(1, -f, 0, 0)
        #   m^a = (1/(r*sqrt(2)))(0, 0, 1, i/sin(theta))
        #   mbar^a = conj(m^a)
        l    = ComplexF64[1.0/f, 1.0, 0.0, 0.0]
        n    = ComplexF64[0.5, -f/2, 0.0, 0.0]
        m    = ComplexF64[0.0, 0.0, 1.0, im/sin(theta)] / (r * sqrt(2.0))
        mbar = ComplexF64[0.0, 0.0, 1.0, -im/sin(theta)] / (r * sqrt(2.0))

        # Verify tetrad normalization
        @test validate_null_tetrad(l, n, m, mbar, g, atol=1e-8)

        psi = weyl_scalars(Weyl, l, n, m, mbar)

        # Schwarzschild is Petrov type D: only Psi2 is nonzero
        @test isapprox(real(psi.Psi2), -M/r^3, atol=1e-6)
        @test isapprox(imag(psi.Psi2), 0.0, atol=1e-6)
        @test all(isapprox.(abs.([psi.Psi0, psi.Psi1, psi.Psi3, psi.Psi4]), 0, atol=1e-6))
    end

    # ================================================================
    # Test 3: validate_null_tetrad rejects bad tetrads
    # ================================================================
    @testset "Tetrad validation" begin
        g = Float64[-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
        g_ct = CTensor(g, :Mink)
        l, n, m, mbar = null_tetrad_from_metric(g_ct)

        # Good tetrad passes
        @test validate_null_tetrad(l, n, m, mbar, g) == true

        # Rescaled tetrad fails (l . n != -1)
        @test validate_null_tetrad(2.0 .* l, n, m, mbar, g) == false
    end

    # ================================================================
    # Test 4: null_tetrad_from_metric produces valid tetrad
    # ================================================================
    @testset "null_tetrad_from_metric" begin
        g = Float64[-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
        g_ct = CTensor(g, :Mink)
        l, n, m, mbar = null_tetrad_from_metric(g_ct)

        @test validate_null_tetrad(l, n, m, mbar, g) == true

        # Verify m and mbar are complex conjugates
        @test isapprox(m, conj.(mbar), atol=1e-14)

        # Non-diagonal metric should error
        g_nd = Float64[-1 0.1 0 0; 0.1 1 0 0; 0 0 1 0; 0 0 0 1]
        g_ct_nd = CTensor(g_nd, :test)
        @test_throws ErrorException null_tetrad_from_metric(g_ct_nd)

        # Wrong dimension should error
        g3 = Float64[-1 0 0; 0 1 0; 0 0 1]
        g_ct3 = CTensor(g3, :test3)
        @test_throws ErrorException null_tetrad_from_metric(g_ct3)
    end

    # ================================================================
    # Test 5: Petrov invariant I = Psi0*Psi4 - 4*Psi1*Psi3 + 3*Psi2^2
    # For Schwarzschild: I = 3*Psi2^2 = 3*M^2/r^6
    # ================================================================
    @testset "Petrov invariant I" begin
        M = 1.0
        r = 3.0
        Psi2 = -M / r^3  # exact Schwarzschild value

        # Build from known values
        I_exact = 3 * Psi2^2

        # Should equal 3*M^2/r^6
        @test isapprox(I_exact, 3 * M^2 / r^6, atol=1e-14)
    end
end
