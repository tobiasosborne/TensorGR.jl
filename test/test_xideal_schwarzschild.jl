#= xIdeal Validation: Schwarzschild = Petrov Type D, Segre [(1,1)(11)].
#
# End-to-end integration test of the Petrov + Segre + energy condition
# pipeline for the Schwarzschild metric. All curvature tensors are computed
# numerically via finite differences, then classified.
#
# Ground truth:
#   Stephani et al., "Exact Solutions" (2003), Table 15.1
#   Wald, "General Relativity" (1984), Ch 6
#   Kinnersley, J. Math. Phys. 10, 1195 (1969), Eqs (3)-(4)
=#

using LinearAlgebra: diagm, norm

@testset "xIdeal Validation: Schwarzschild" begin

    # ================================================================
    # Schwarzschild numeric curvature pipeline
    # ================================================================
    # Evaluate at M=1, r=3, theta=pi/2 (equatorial plane, outside horizon).

    M_bh  = 1.0
    r_val = 3.0
    theta_val = pi / 2

    f = 1.0 - 2M_bh / r_val   # = 1/3

    # Metric g_{ab} = diag(-(1-2M/r), (1-2M/r)^{-1}, r^2, r^2 sin^2 theta)
    g_schw = zeros(4, 4)
    g_schw[1, 1] = -f
    g_schw[2, 2] = 1.0 / f
    g_schw[3, 3] = r_val^2
    g_schw[4, 4] = r_val^2 * sin(theta_val)^2

    ginv_schw = zeros(4, 4)
    for i in 1:4; ginv_schw[i, i] = 1.0 / g_schw[i, i]; end

    # -- Finite-difference curvature computation --
    function schw_metric_at(point)
        _, rv, th, _ = point
        fv = 1.0 - 2M_bh / rv
        gp = zeros(4, 4)
        gp[1, 1] = -fv
        gp[2, 2] = 1.0 / fv
        gp[3, 3] = rv^2
        gp[4, 4] = rv^2 * sin(th)^2
        gp
    end

    function schw_ginv_at(point)
        gp = schw_metric_at(point)
        gi = zeros(4, 4)
        for i in 1:4; gi[i, i] = 1.0 / gp[i, i]; end
        gi
    end

    x0 = [0.0, r_val, theta_val, 0.0]
    h_fd = 1e-5   # finite-difference step

    function gamma_at(point)
        gi = schw_ginv_at(point)
        G = Array{Float64}(undef, 4, 4, 4)
        for a in 1:4, b in 1:4, c in 1:4
            s = 0.0
            for d in 1:4
                xp = copy(point); xp[b] += h_fd
                xm = copy(point); xm[b] -= h_fd
                dg1 = (schw_metric_at(xp)[c, d] - schw_metric_at(xm)[c, d]) / (2h_fd)
                xp = copy(point); xp[c] += h_fd
                xm = copy(point); xm[c] -= h_fd
                dg2 = (schw_metric_at(xp)[b, d] - schw_metric_at(xm)[b, d]) / (2h_fd)
                xp = copy(point); xp[d] += h_fd
                xm = copy(point); xm[d] -= h_fd
                dg3 = (schw_metric_at(xp)[b, c] - schw_metric_at(xm)[b, c]) / (2h_fd)
                s += gi[a, d] * (dg1 + dg2 - dg3)
            end
            G[a, b, c] = s / 2
        end
        G
    end

    Gamma = gamma_at(x0)

    Riem = Array{Float64}(undef, 4, 4, 4, 4)
    for a in 1:4, b in 1:4, c in 1:4, d in 1:4
        xp = copy(x0); xp[c] += h_fd
        xm = copy(x0); xm[c] -= h_fd
        dG1 = (gamma_at(xp)[a, d, b] - gamma_at(xm)[a, d, b]) / (2h_fd)
        xp = copy(x0); xp[d] += h_fd
        xm = copy(x0); xm[d] -= h_fd
        dG2 = (gamma_at(xp)[a, c, b] - gamma_at(xm)[a, c, b]) / (2h_fd)
        val = dG1 - dG2
        for e in 1:4
            val += Gamma[a, c, e] * Gamma[e, d, b] - Gamma[a, d, e] * Gamma[e, c, b]
        end
        Riem[a, b, c, d] = val
    end

    Ric_schw  = metric_ricci(Riem, 4)
    R_schw    = metric_ricci_scalar(Float64.(Ric_schw), ginv_schw, 4)
    Weyl_schw = metric_weyl(Riem, Ric_schw, R_schw, g_schw, ginv_schw, 4)

    # ================================================================
    # 1. Kretschmann scalar cross-check: K = 48 M^2 / r^6
    # ================================================================
    @testset "Kretschmann scalar" begin
        K_numeric = metric_kretschmann(Riem, g_schw, ginv_schw, 4)
        K_exact   = 48.0 * M_bh^2 / r_val^6   # = 48/729
        @test isapprox(Float64(K_numeric), K_exact, rtol=1e-4)
    end

    # ================================================================
    # 2. Vacuum: Ricci tensor and scalar vanish
    # ================================================================
    @testset "Vacuum: Ricci = 0, R = 0" begin
        @test isapprox(Float64(R_schw), 0.0, atol=1e-6)
        for a in 1:4, b in 1:4
            @test isapprox(Float64(Ric_schw[a, b]), 0.0, atol=1e-6)
        end
    end

    # ================================================================
    # 3. Kinnersley null tetrad and Weyl scalars
    # ================================================================
    @testset "Kinnersley tetrad and Weyl scalars" begin
        # Kinnersley tetrad (Kinnersley 1969):
        #   l^a = (1/f, 1, 0, 0)
        #   n^a = (1/2)(1, -f, 0, 0)
        #   m^a = (1/(r sqrt(2)))(0, 0, 1, i/sin(theta))
        l_kin    = ComplexF64[1.0 / f, 1.0, 0.0, 0.0]
        n_kin    = ComplexF64[0.5, -f / 2, 0.0, 0.0]
        m_kin    = ComplexF64[0.0, 0.0, 1.0, im / sin(theta_val)] / (r_val * sqrt(2.0))
        mbar_kin = conj.(m_kin)

        # Tetrad normalization: l.n = -1, m.mbar = +1, all others = 0
        @test validate_null_tetrad(l_kin, n_kin, m_kin, mbar_kin, g_schw; atol=1e-10)

        # Weyl scalars
        psi = weyl_scalars(Float64.(Weyl_schw), l_kin, n_kin, m_kin, mbar_kin)

        # Psi2 = -M/r^3 = -1/27 (Kinnersley 1969, Eq 4)
        @test isapprox(real(psi.Psi2), -M_bh / r_val^3, atol=1e-6)
        @test isapprox(imag(psi.Psi2), 0.0, atol=1e-6)

        # All other scalars vanish (Type D signature)
        @test isapprox(abs(psi.Psi0), 0.0, atol=1e-6)
        @test isapprox(abs(psi.Psi1), 0.0, atol=1e-6)
        @test isapprox(abs(psi.Psi3), 0.0, atol=1e-6)
        @test isapprox(abs(psi.Psi4), 0.0, atol=1e-6)
    end

    # ================================================================
    # 4. Petrov invariants: I = 3 Psi2^2, J = -Psi2^3, I^3 = 27 J^2
    # ================================================================
    @testset "Petrov invariants (algebraically special)" begin
        Psi2_exact = -M_bh / r_val^3
        I_exact = 3.0 * Psi2_exact^2       # = 3/729 = 1/243
        J_exact = -Psi2_exact^3             # = 1/19683

        inv_t = petrov_invariants(Float64.(Weyl_schw), g_schw)
        @test isapprox(real(inv_t.I), I_exact, rtol=1e-4)
        @test isapprox(real(inv_t.J), J_exact, rtol=1e-4)

        # Algebraically special criterion: I^3 = 27 J^2
        @test isapprox(inv_t.I^3, 27 * inv_t.J^2, rtol=1e-4)
        @test is_algebraically_special(inv_t.I, inv_t.J; atol=1e-4)
    end

    # ================================================================
    # 5. Petrov classification: Type D
    # ================================================================
    @testset "Petrov type = D" begin
        @test petrov_classify(Float64.(Weyl_schw), g_schw; atol=1e-4) == TypeD
    end

    # ================================================================
    # 6. Segre classification: vacuum {(1111)}
    # ================================================================
    @testset "Segre type = vacuum" begin
        # The finite-difference Ricci has O(h^2) ~ 1e-7 residuals, so we
        # need atol > that to group the near-zero eigenvalues correctly.
        st = segre_classify(Float64.(Ric_schw), ginv_schw; atol=1e-6)
        # Vacuum: all Ricci eigenvalues zero, single group of multiplicity 4
        @test st.notation == "{(1111)}"
        @test length(st.eigenvalues) == 1
        @test abs(st.eigenvalues[1]) < 1e-6
        @test st.multiplicities == [4]
        @test st.is_degenerate == false
    end

    # ================================================================
    # 7. Energy conditions: all satisfied (vacuum)
    # ================================================================
    @testset "Energy conditions (vacuum)" begin
        # Relax atol to accommodate finite-difference residuals in Ricci.
        ec = check_energy_conditions(Float64.(Ric_schw), Float64(R_schw),
                                     g_schw, ginv_schw; atol=1e-6)
        @test ec.NEC == true
        @test ec.WEC == true
        @test ec.SEC == true
        @test ec.DEC == true
        @test abs(ec.rho) < 1e-6
    end

    # ================================================================
    # 8. Consistency at multiple radii
    # ================================================================
    @testset "Type D at r = 6M, 10M" begin
        for r_test in [6.0, 10.0]
            f_test = 1.0 - 2M_bh / r_test
            g_test = diagm([-f_test, 1.0 / f_test, r_test^2,
                            r_test^2 * sin(pi / 2)^2])
            ginv_test = diagm([-1.0 / f_test, f_test, 1.0 / r_test^2,
                               1.0 / r_test^2])
            x_test = [0.0, r_test, pi / 2, 0.0]

            function gamma_test(pt)
                gi = diagm([1.0 / (-1.0 + 2M_bh / pt[2]),
                            1.0 - 2M_bh / pt[2],
                            1.0 / pt[2]^2,
                            1.0 / (pt[2]^2 * sin(pt[3])^2)])
                G = Array{Float64}(undef, 4, 4, 4)
                for a in 1:4, b in 1:4, c in 1:4
                    s = 0.0
                    for d in 1:4
                        xp = copy(pt); xp[b] += h_fd
                        xm = copy(pt); xm[b] -= h_fd
                        dg1 = (schw_metric_at(xp)[c, d] - schw_metric_at(xm)[c, d]) / (2h_fd)
                        xp = copy(pt); xp[c] += h_fd
                        xm = copy(pt); xm[c] -= h_fd
                        dg2 = (schw_metric_at(xp)[b, d] - schw_metric_at(xm)[b, d]) / (2h_fd)
                        xp = copy(pt); xp[d] += h_fd
                        xm = copy(pt); xm[d] -= h_fd
                        dg3 = (schw_metric_at(xp)[b, c] - schw_metric_at(xm)[b, c]) / (2h_fd)
                        s += gi[a, d] * (dg1 + dg2 - dg3)
                    end
                    G[a, b, c] = s / 2
                end
                G
            end

            Gamma_t = gamma_test(x_test)
            Riem_t = Array{Float64}(undef, 4, 4, 4, 4)
            for a in 1:4, b in 1:4, c in 1:4, d in 1:4
                xp = copy(x_test); xp[c] += h_fd
                xm = copy(x_test); xm[c] -= h_fd
                dG1 = (gamma_test(xp)[a, d, b] - gamma_test(xm)[a, d, b]) / (2h_fd)
                xp = copy(x_test); xp[d] += h_fd
                xm = copy(x_test); xm[d] -= h_fd
                dG2 = (gamma_test(xp)[a, c, b] - gamma_test(xm)[a, c, b]) / (2h_fd)
                val = dG1 - dG2
                for e in 1:4
                    val += Gamma_t[a, c, e] * Gamma_t[e, d, b] -
                           Gamma_t[a, d, e] * Gamma_t[e, c, b]
                end
                Riem_t[a, b, c, d] = val
            end

            Ric_t = Float64.(metric_ricci(Riem_t, 4))
            R_t   = metric_ricci_scalar(Ric_t, ginv_test, 4)
            W_t   = Float64.(metric_weyl(Riem_t, Ric_t, R_t, g_test, ginv_test, 4))

            @test petrov_classify(W_t, g_test; atol=1e-4) == TypeD
        end
    end

end
