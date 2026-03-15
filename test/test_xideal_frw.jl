#= xIdeal Validation: FRW = Petrov Type O (conformally flat).
#
# End-to-end test of the classification pipeline for the Friedmann-Robertson-
# Walker metric. FRW is conformally flat for ALL a(t) and k, so the Weyl
# tensor vanishes identically.
#
# Ground truth:
#   Stephani et al., "Exact Solutions" (2003), Ch 14, Table 5.2
#   Wald, "General Relativity" (1984), Sec 5.1
=#

using LinearAlgebra: diagm

@testset "xIdeal Validation: FRW" begin

    # ================================================================
    # FRW numeric curvature pipeline (matter-dominated, k=0)
    # ================================================================
    # a(t) = t^(2/3), evaluate at t=1, r=0.5, theta=pi/2.

    t0 = 1.0; r0 = 0.5; theta0 = pi / 2
    a0    = t0^(2/3)          # a(t0) = 1
    adot  = (2/3) * t0^(-1/3) # da/dt
    addot = -(2/9) * t0^(-4/3) # d^2a/dt^2

    function frw_metric_at(point)
        t, r, th, _ = point
        a = t^(2/3)
        gp = zeros(4, 4)
        gp[1, 1] = -1.0
        gp[2, 2] = a^2
        gp[3, 3] = a^2 * r^2
        gp[4, 4] = a^2 * r^2 * sin(th)^2
        gp
    end

    function frw_ginv_at(point)
        gp = frw_metric_at(point)
        gi = zeros(4, 4)
        for i in 1:4; gi[i, i] = 1.0 / gp[i, i]; end
        gi
    end

    g_frw    = frw_metric_at([t0, r0, theta0, 0.0])
    ginv_frw = frw_ginv_at([t0, r0, theta0, 0.0])

    x0 = [t0, r0, theta0, 0.0]
    h_fd = 1e-5

    function gamma_frw(point)
        gi = frw_ginv_at(point)
        G = Array{Float64}(undef, 4, 4, 4)
        for a in 1:4, b in 1:4, c in 1:4
            s = 0.0
            for d in 1:4
                xp = copy(point); xp[b] += h_fd
                xm = copy(point); xm[b] -= h_fd
                dg1 = (frw_metric_at(xp)[c, d] - frw_metric_at(xm)[c, d]) / (2h_fd)
                xp = copy(point); xp[c] += h_fd
                xm = copy(point); xm[c] -= h_fd
                dg2 = (frw_metric_at(xp)[b, d] - frw_metric_at(xm)[b, d]) / (2h_fd)
                xp = copy(point); xp[d] += h_fd
                xm = copy(point); xm[d] -= h_fd
                dg3 = (frw_metric_at(xp)[b, c] - frw_metric_at(xm)[b, c]) / (2h_fd)
                s += gi[a, d] * (dg1 + dg2 - dg3)
            end
            G[a, b, c] = s / 2
        end
        G
    end

    Gamma_frw = gamma_frw(x0)

    Riem_frw = Array{Float64}(undef, 4, 4, 4, 4)
    for a in 1:4, b in 1:4, c in 1:4, d in 1:4
        xp = copy(x0); xp[c] += h_fd
        xm = copy(x0); xm[c] -= h_fd
        dG1 = (gamma_frw(xp)[a, d, b] - gamma_frw(xm)[a, d, b]) / (2h_fd)
        xp = copy(x0); xp[d] += h_fd
        xm = copy(x0); xm[d] -= h_fd
        dG2 = (gamma_frw(xp)[a, c, b] - gamma_frw(xm)[a, c, b]) / (2h_fd)
        val = dG1 - dG2
        for e in 1:4
            val += Gamma_frw[a, c, e] * Gamma_frw[e, d, b] -
                   Gamma_frw[a, d, e] * Gamma_frw[e, c, b]
        end
        Riem_frw[a, b, c, d] = val
    end

    Ric_frw  = Float64.(metric_ricci(Riem_frw, 4))
    R_frw    = Float64(metric_ricci_scalar(Ric_frw, ginv_frw, 4))
    Weyl_frw = Float64.(metric_weyl(Riem_frw, Ric_frw, R_frw, g_frw, ginv_frw, 4))

    # ================================================================
    # 1. Weyl tensor vanishes (conformally flat)
    # ================================================================
    @testset "Weyl tensor = 0 (conformally flat)" begin
        @test all(x -> isapprox(x, 0, atol=1e-6), Weyl_frw)
    end

    # ================================================================
    # 2. Petrov type = O
    # ================================================================
    @testset "Petrov type = O" begin
        @test petrov_classify(Weyl_frw, g_frw; atol=1e-4) == TypeO
    end

    # ================================================================
    # 3. Petrov invariants I = J = 0
    # ================================================================
    @testset "Petrov invariants I = J = 0" begin
        inv_frw = petrov_invariants(Weyl_frw, g_frw)
        @test isapprox(real(inv_frw.I), 0.0, atol=1e-10)
        @test isapprox(real(inv_frw.J), 0.0, atol=1e-10)
    end

    # ================================================================
    # 4. Ricci tensor non-zero (non-vacuum), Ricci scalar check
    # ================================================================
    @testset "Non-vacuum: Ricci != 0" begin
        # For dust (a = t^(2/3), k=0):
        # R = 6(addot/a + (adot/a)^2) = 6(-2/9 + 4/9) = 6*2/9 = 4/3
        R_exact = 6.0 * (addot / a0 + (adot / a0)^2)
        @test isapprox(R_frw, R_exact, rtol=1e-4)
        @test abs(R_frw) > 0.1   # definitively non-zero
        # R_{tt} = -3 addot/a = -3*(-2/9)/1 = 2/3
        Rtt_exact = -3.0 * addot / a0
        @test isapprox(Ric_frw[1, 1], Rtt_exact, rtol=1e-3)
    end

    # ================================================================
    # 5. Segre type: {1,(111)} (perfect fluid)
    # ================================================================
    @testset "Segre type = perfect fluid" begin
        st = segre_classify(Ric_frw, ginv_frw; atol=1e-4)
        @test st.notation == "{1,(111)}"
    end

    # ================================================================
    # 6. Energy conditions: dust (rho > 0, p = 0) -> all satisfied
    # ================================================================
    @testset "Energy conditions (dust)" begin
        ec = check_energy_conditions(Ric_frw, R_frw, g_frw, ginv_frw; atol=1e-4)
        @test ec.NEC == true
        @test ec.WEC == true
        @test ec.SEC == true
        @test ec.DEC == true
        @test ec.rho > 0   # positive energy density
    end

end
