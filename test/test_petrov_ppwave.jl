#= xIdeal Validation: pp-wave = Petrov Type N.
#
# End-to-end integration test of the Petrov + Segre + energy condition
# pipeline for the pp-wave spacetime in Brinkmann coordinates.
#
# Metric:
#   ds^2 = H(u,x,y) du^2 - 2 du dv + dx^2 + dy^2
# with H = x^2 - y^2 (linearly polarized vacuum GW, harmonic in transverse coords).
# Coordinates: (u, v, x, y) = indices (1, 2, 3, 4).
#
# Ground truth:
#   Stephani et al., "Exact Solutions" (2003), Ch 24 (pp-waves)
#   Griffiths & Podolsky, "Exact Space-Times in Einstein's GR" (2009), Ch 17
#   Ehlers & Kundt, in "Gravitation: an introduction..." (1962), pp-wave classification
#
# Key properties:
#   - Vacuum: R_{ab} = 0 (H is harmonic: d^2H/dx^2 + d^2H/dy^2 = 0)
#   - Petrov Type N: single quadruple principal null direction l = dv
#   - Weyl tensor non-zero (gravitational wave)
#   - Only Psi_4 != 0 in the adapted null tetrad
#   - Petrov invariants I = J = 0 (hallmark of Type N)
#   - All polynomial curvature invariants vanish (VSI spacetime)
=#

using LinearAlgebra: norm, I as Id4

@testset "xIdeal Validation: pp-wave (Type N)" begin

    # ================================================================
    # pp-wave metric in Brinkmann coordinates
    # ================================================================
    # H(u,x,y) = x^2 - y^2, evaluated at x0=1, y0=0.5
    # Coordinate ordering: (u, v, x, y) = (1, 2, 3, 4)

    x0_val = 1.0
    y0_val = 0.5
    H_val = x0_val^2 - y0_val^2   # = 0.75

    # Metric g_{ab} with g_{uv} = -1 (standard Brinkmann normalization)
    #   g = [H   -1  0  0;
    #        -1   0  0  0;
    #         0   0  1  0;
    #         0   0  0  1]
    g_pp = zeros(4, 4)
    g_pp[1, 1] = H_val
    g_pp[1, 2] = -1.0; g_pp[2, 1] = -1.0
    g_pp[3, 3] = 1.0
    g_pp[4, 4] = 1.0

    # Inverse metric:
    #   ginv = [ 0  -1  0  0;
    #           -1  -H  0  0;
    #            0   0  1  0;
    #            0   0  0  1]
    ginv_pp = zeros(4, 4)
    ginv_pp[1, 2] = -1.0; ginv_pp[2, 1] = -1.0
    ginv_pp[2, 2] = -H_val
    ginv_pp[3, 3] = 1.0
    ginv_pp[4, 4] = 1.0

    # Verify g * ginv = identity
    @test isapprox(g_pp * ginv_pp, Matrix{Float64}(Id4, 4, 4), atol=1e-14)

    # ================================================================
    # Finite-difference curvature pipeline
    # ================================================================
    # H(u,x,y) = x^2 - y^2 (independent of u,v)

    function ppwave_H(point)
        # point = [u, v, x, y]
        point[3]^2 - point[4]^2
    end

    function ppwave_metric_at(point)
        Hv = ppwave_H(point)
        gp = zeros(4, 4)
        gp[1, 1] = Hv
        gp[1, 2] = -1.0; gp[2, 1] = -1.0
        gp[3, 3] = 1.0
        gp[4, 4] = 1.0
        gp
    end

    function ppwave_ginv_at(point)
        Hv = ppwave_H(point)
        gi = zeros(4, 4)
        gi[1, 2] = -1.0; gi[2, 1] = -1.0
        gi[2, 2] = -Hv
        gi[3, 3] = 1.0
        gi[4, 4] = 1.0
        gi
    end

    pt0 = [0.0, 0.0, x0_val, y0_val]  # evaluation point
    h_fd = 1e-5   # finite-difference step

    function gamma_pp(point)
        gi = ppwave_ginv_at(point)
        G = Array{Float64}(undef, 4, 4, 4)
        for a in 1:4, b in 1:4, c in 1:4
            s = 0.0
            for d in 1:4
                xp = copy(point); xp[b] += h_fd
                xm = copy(point); xm[b] -= h_fd
                dg1 = (ppwave_metric_at(xp)[c, d] - ppwave_metric_at(xm)[c, d]) / (2h_fd)
                xp = copy(point); xp[c] += h_fd
                xm = copy(point); xm[c] -= h_fd
                dg2 = (ppwave_metric_at(xp)[b, d] - ppwave_metric_at(xm)[b, d]) / (2h_fd)
                xp = copy(point); xp[d] += h_fd
                xm = copy(point); xm[d] -= h_fd
                dg3 = (ppwave_metric_at(xp)[b, c] - ppwave_metric_at(xm)[b, c]) / (2h_fd)
                s += gi[a, d] * (dg1 + dg2 - dg3)
            end
            G[a, b, c] = s / 2
        end
        G
    end

    Gamma_pp = gamma_pp(pt0)

    Riem_pp = Array{Float64}(undef, 4, 4, 4, 4)
    for a in 1:4, b in 1:4, c in 1:4, d in 1:4
        xp = copy(pt0); xp[c] += h_fd
        xm = copy(pt0); xm[c] -= h_fd
        dG1 = (gamma_pp(xp)[a, d, b] - gamma_pp(xm)[a, d, b]) / (2h_fd)
        xp = copy(pt0); xp[d] += h_fd
        xm = copy(pt0); xm[d] -= h_fd
        dG2 = (gamma_pp(xp)[a, c, b] - gamma_pp(xm)[a, c, b]) / (2h_fd)
        val = dG1 - dG2
        for e in 1:4
            val += Gamma_pp[a, c, e] * Gamma_pp[e, d, b] -
                   Gamma_pp[a, d, e] * Gamma_pp[e, c, b]
        end
        Riem_pp[a, b, c, d] = val
    end

    Ric_pp  = Float64.(metric_ricci(Riem_pp, 4))
    R_pp    = Float64(metric_ricci_scalar(Ric_pp, ginv_pp, 4))
    Weyl_pp = Float64.(metric_weyl(Riem_pp, Ric_pp, R_pp, g_pp, ginv_pp, 4))

    # ================================================================
    # 1. Weyl tensor non-zero (gravitational wave)
    # ================================================================
    @testset "Weyl tensor != 0 (gravitational wave)" begin
        weyl_max = maximum(abs, Weyl_pp)
        @test weyl_max > 0.1   # definitively non-zero
    end

    # ================================================================
    # 2. Vacuum: Ricci tensor and scalar vanish
    # ================================================================
    @testset "Vacuum: Ricci = 0, R = 0" begin
        @test isapprox(R_pp, 0.0, atol=1e-6)
        for a in 1:4, b in 1:4
            @test isapprox(Ric_pp[a, b], 0.0, atol=1e-6)
        end
    end

    # ================================================================
    # 3. Kretschmann scalar = 0 (VSI spacetime)
    # ================================================================
    @testset "Kretschmann scalar = 0 (VSI)" begin
        K_pp = Float64(metric_kretschmann(Riem_pp, g_pp, ginv_pp, 4))
        # pp-waves are VSI (vanishing scalar invariant) spacetimes:
        # ALL polynomial curvature invariants vanish, including Kretschmann,
        # despite non-zero Weyl tensor. This is the hallmark of Type N/III.
        # See Coley et al., Class. Quantum Grav. 21, 5519 (2004).
        @test isapprox(K_pp, 0.0, atol=1e-6)
    end

    # ================================================================
    # 4. Null tetrad and Weyl scalars: only Psi4 != 0
    # ================================================================
    @testset "Adapted null tetrad and Weyl scalars" begin
        # Adapted null tetrad for pp-wave (Griffiths & Podolsky, Ch 17):
        #   l^a = (0, 1, 0, 0)     -- along dv (repeated PND)
        #   n^a = (1, H/2, 0, 0)   -- second null vector
        #   m^a = (0, 0, 1, i) / sqrt(2)  -- complex spatial
        #   mbar^a = conj(m^a)
        #
        # Normalization check:
        #   l^a = (l^u, l^v, l^x, l^y) = (0, 1, 0, 0)
        #   n^a = (n^u, n^v, n^x, n^y) = (1, H/2, 0, 0)
        #   l.n = g_{vu} l^v n^u = (-1)(1)(1) = -1. Good.
        #   n.n = g_{uu} + 2 g_{uv} (H/2) = H - H = 0. Good.

        l_pp    = ComplexF64[0.0, 1.0, 0.0, 0.0]
        n_pp    = ComplexF64[1.0, H_val/2, 0.0, 0.0]
        m_pp    = ComplexF64[0.0, 0.0, 1.0, im] / sqrt(2.0)
        mbar_pp = conj.(m_pp)

        # Tetrad normalization: l.n = -1, m.mbar = +1, all others = 0
        @test validate_null_tetrad(l_pp, n_pp, m_pp, mbar_pp, g_pp; atol=1e-10)

        # Weyl scalars
        psi = weyl_scalars(Weyl_pp, l_pp, n_pp, m_pp, mbar_pp)

        # Type N signature: only Psi4 != 0
        @test isapprox(abs(psi.Psi0), 0.0, atol=1e-4)
        @test isapprox(abs(psi.Psi1), 0.0, atol=1e-4)
        @test isapprox(abs(psi.Psi2), 0.0, atol=1e-4)
        @test isapprox(abs(psi.Psi3), 0.0, atol=1e-4)
        @test abs(psi.Psi4) > 0.1   # definitively non-zero

        # For H = x^2 - y^2 with g_{uv} = -1:
        # Psi4 = C_{abcd} n^a mbar^b n^c mbar^d
        # The non-zero Weyl components are C_{1313} = -1, C_{1414} = +1
        # (corresponding to d_{xx}H/2 and d_{yy}H/2).
        # Psi4 = -(1/2)(d_{xx}H - d_{yy}H) mbar^x mbar^x
        #       -(1/2)(d_{yy}H - d_{xx}H) mbar^y mbar^y
        # With our tetrad: |Psi4| = 1
        @test isapprox(abs(psi.Psi4), 1.0, rtol=1e-3)
    end

    # ================================================================
    # 5. Petrov invariants: I = J = 0 (Type N hallmark)
    # ================================================================
    @testset "Petrov invariants I = J = 0" begin
        l_pp    = ComplexF64[0.0, 1.0, 0.0, 0.0]
        n_pp    = ComplexF64[1.0, H_val/2, 0.0, 0.0]
        m_pp    = ComplexF64[0.0, 0.0, 1.0, im] / sqrt(2.0)
        mbar_pp = conj.(m_pp)

        psi = weyl_scalars(Weyl_pp, l_pp, n_pp, m_pp, mbar_pp)
        inv_pp = petrov_invariants(psi)

        # I = Psi0*Psi4 - 4*Psi1*Psi3 + 3*Psi2^2
        # With only Psi4 != 0: I = 0*Psi4 - 0 + 0 = 0
        @test isapprox(abs(inv_pp.I), 0.0, atol=1e-6)

        # J = det(Q) with Q = [0 0 0; 0 0 0; 0 0 Psi4] -> J = 0
        @test isapprox(abs(inv_pp.J), 0.0, atol=1e-6)

        # Algebraically special (trivially, since I = J = 0)
        @test is_algebraically_special(inv_pp.I, inv_pp.J; atol=1e-4)
    end

    # ================================================================
    # 6. Petrov classification: Type N
    # ================================================================
    @testset "Petrov type = N" begin
        l_pp    = ComplexF64[0.0, 1.0, 0.0, 0.0]
        n_pp    = ComplexF64[1.0, H_val/2, 0.0, 0.0]
        m_pp    = ComplexF64[0.0, 0.0, 1.0, im] / sqrt(2.0)
        mbar_pp = conj.(m_pp)

        psi = weyl_scalars(Weyl_pp, l_pp, n_pp, m_pp, mbar_pp)
        @test petrov_classify(psi; atol=1e-4) == TypeN
    end

    # ================================================================
    # 7. Segre classification: vacuum {(1111)}
    # ================================================================
    @testset "Segre type = vacuum" begin
        st = segre_classify(Ric_pp, ginv_pp; atol=1e-6)
        @test st.notation == "{(1111)}"
        @test length(st.eigenvalues) == 1
        @test abs(st.eigenvalues[1]) < 1e-6
        @test st.multiplicities == [4]
    end

    # ================================================================
    # 8. Energy conditions: all satisfied (vacuum)
    # ================================================================
    @testset "Energy conditions (vacuum)" begin
        ec = check_energy_conditions(Ric_pp, R_pp, g_pp, ginv_pp; atol=1e-6)
        @test ec.NEC == true
        @test ec.WEC == true
        @test ec.SEC == true
        @test ec.DEC == true
        @test abs(ec.rho) < 1e-6
    end

    # ================================================================
    # 9. Consistency at different evaluation points
    # ================================================================
    @testset "Type N at (x,y) = (2, 1) and (0.3, 0.7)" begin
        for (xv, yv) in [(2.0, 1.0), (0.3, 0.7)]
            pt_test = [0.0, 0.0, xv, yv]
            H_test = xv^2 - yv^2

            g_test = zeros(4, 4)
            g_test[1, 1] = H_test
            g_test[1, 2] = -1.0; g_test[2, 1] = -1.0
            g_test[3, 3] = 1.0
            g_test[4, 4] = 1.0

            ginv_test = zeros(4, 4)
            ginv_test[1, 2] = -1.0; ginv_test[2, 1] = -1.0
            ginv_test[2, 2] = -H_test
            ginv_test[3, 3] = 1.0
            ginv_test[4, 4] = 1.0

            Gamma_t = gamma_pp(pt_test)
            Riem_t = Array{Float64}(undef, 4, 4, 4, 4)
            for a in 1:4, b in 1:4, c in 1:4, d in 1:4
                xp = copy(pt_test); xp[c] += h_fd
                xm = copy(pt_test); xm[c] -= h_fd
                dG1 = (gamma_pp(xp)[a, d, b] - gamma_pp(xm)[a, d, b]) / (2h_fd)
                xp = copy(pt_test); xp[d] += h_fd
                xm = copy(pt_test); xm[d] -= h_fd
                dG2 = (gamma_pp(xp)[a, c, b] - gamma_pp(xm)[a, c, b]) / (2h_fd)
                val = dG1 - dG2
                for e in 1:4
                    val += Gamma_t[a, c, e] * Gamma_t[e, d, b] -
                           Gamma_t[a, d, e] * Gamma_t[e, c, b]
                end
                Riem_t[a, b, c, d] = val
            end

            Ric_t = Float64.(metric_ricci(Riem_t, 4))
            R_t   = Float64(metric_ricci_scalar(Ric_t, ginv_test, 4))
            W_t   = Float64.(metric_weyl(Riem_t, Ric_t, R_t, g_test, ginv_test, 4))

            # Construct adapted tetrad at new point
            l_t    = ComplexF64[0.0, 1.0, 0.0, 0.0]
            n_t    = ComplexF64[1.0, H_test/2, 0.0, 0.0]
            m_t    = ComplexF64[0.0, 0.0, 1.0, im] / sqrt(2.0)
            mbar_t = conj.(m_t)

            psi_t = weyl_scalars(W_t, l_t, n_t, m_t, mbar_t)
            @test petrov_classify(psi_t; atol=1e-4) == TypeN

            # Vacuum at all points
            @test isapprox(R_t, 0.0, atol=1e-6)
        end
    end

end
