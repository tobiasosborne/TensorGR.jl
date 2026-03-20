#= Riemann tensor identities validation on Schwarzschild.
#
# Numerically verify the fundamental identities of the Riemann tensor:
# 1. Pair antisymmetry: R_{abcd} = -R_{abdc} = -R_{bacd}
# 2. Pair symmetry: R_{abcd} = R_{cdab}
# 3. First Bianchi identity: R_{a[bcd]} = 0 (cyclic)
# 4. Contracted Bianchi identity: ∇^a G_{ab} = 0
# 5. Einstein tensor trace: G^a_a = -R (in d=4)
#
# These are verified on the Schwarzschild metric at a generic point
# (r=4, θ=π/3) using finite differences, confirming the component
# computation pipeline is correct.
#
# Ground truth: Wald, "General Relativity" (1984), Ch 3, Eqs 3.2.13-3.2.16.
=#

using LinearAlgebra: diagm

@testset "Riemann Tensor Identities (Schwarzschild)" begin

    M_bh = 1.0; r_val = 4.0; θ_val = π / 3
    h_fd = 1e-5

    function schw_g(pt)
        fv = 1 - 2M_bh / pt[2]
        diagm([-fv, 1 / fv, pt[2]^2, pt[2]^2 * sin(pt[3])^2])
    end
    function schw_ginv(pt)
        fv = 1 - 2M_bh / pt[2]
        diagm([-1 / fv, fv, 1 / pt[2]^2, 1 / (pt[2]^2 * sin(pt[3])^2)])
    end

    x0 = [0.0, r_val, θ_val, 0.0]

    function fd_chris(pt)
        gi = schw_ginv(pt); G = zeros(4, 4, 4)
        for a in 1:4, b in 1:4, c in 1:4
            s = 0.0
            for d in 1:4
                xp = copy(pt); xp[b] += h_fd; xm = copy(pt); xm[b] -= h_fd
                dg1 = (schw_g(xp)[c, d] - schw_g(xm)[c, d]) / (2h_fd)
                xp = copy(pt); xp[c] += h_fd; xm = copy(pt); xm[c] -= h_fd
                dg2 = (schw_g(xp)[b, d] - schw_g(xm)[b, d]) / (2h_fd)
                xp = copy(pt); xp[d] += h_fd; xm = copy(pt); xm[d] -= h_fd
                dg3 = (schw_g(xp)[b, c] - schw_g(xm)[b, c]) / (2h_fd)
                s += gi[a, d] * (dg1 + dg2 - dg3)
            end; G[a, b, c] = s / 2
        end; G
    end

    Gamma = fd_chris(x0)
    g0 = schw_g(x0); gi0 = schw_ginv(x0)

    # R^a_{bcd}
    Riem = zeros(4, 4, 4, 4)
    for a in 1:4, b in 1:4, c in 1:4, d in 1:4
        xp = copy(x0); xp[c] += h_fd; xm = copy(x0); xm[c] -= h_fd
        dG1 = (fd_chris(xp)[a, d, b] - fd_chris(xm)[a, d, b]) / (2h_fd)
        xp = copy(x0); xp[d] += h_fd; xm = copy(x0); xm[d] -= h_fd
        dG2 = (fd_chris(xp)[a, c, b] - fd_chris(xm)[a, c, b]) / (2h_fd)
        val = dG1 - dG2
        for e in 1:4
            val += Gamma[a, c, e] * Gamma[e, d, b] - Gamma[a, d, e] * Gamma[e, c, b]
        end
        Riem[a, b, c, d] = val
    end

    # Lower first index: R_{abcd} = g_{ae} R^e_{bcd}
    Riem_down = zeros(4, 4, 4, 4)
    for a in 1:4, b in 1:4, c in 1:4, d in 1:4
        for e in 1:4; Riem_down[a, b, c, d] += g0[a, e] * Riem[e, b, c, d]; end
    end

    Ric = Float64.(metric_ricci(Riem, 4))
    R_scalar = Float64(metric_ricci_scalar(Ric, gi0, 4))

    # ================================================================
    # 1. Pair antisymmetry: R_{abcd} = -R_{abdc} = -R_{bacd}
    # ================================================================
    @testset "Pair antisymmetry" begin
        max_err = 0.0
        for a in 1:4, b in 1:4, c in 1:4, d in 1:4
            max_err = max(max_err,
                abs(Riem_down[a, b, c, d] + Riem_down[a, b, d, c]),
                abs(Riem_down[a, b, c, d] + Riem_down[b, a, c, d]))
        end
        @test max_err < 1e-4
    end

    # ================================================================
    # 2. Pair symmetry: R_{abcd} = R_{cdab}
    # ================================================================
    @testset "Pair symmetry" begin
        max_err = 0.0
        for a in 1:4, b in 1:4, c in 1:4, d in 1:4
            max_err = max(max_err, abs(Riem_down[a, b, c, d] - Riem_down[c, d, a, b]))
        end
        @test max_err < 1e-4
    end

    # ================================================================
    # 3. First Bianchi identity: R_{a[bcd]} = 0
    # ================================================================
    @testset "First Bianchi identity (cyclic)" begin
        max_err = 0.0
        for a in 1:4, b in 1:4, c in 1:4, d in 1:4
            cyclic = Riem_down[a, b, c, d] + Riem_down[a, c, d, b] + Riem_down[a, d, b, c]
            max_err = max(max_err, abs(cyclic))
        end
        @test max_err < 1e-4
    end

    # ================================================================
    # 4. Einstein tensor trace: G^a_a = -R (Wald Eq 3.2.28)
    # ================================================================
    @testset "Einstein trace G^a_a = -R" begin
        G = zeros(4, 4)
        for a in 1:4, b in 1:4
            G[a, b] = Ric[a, b] - 0.5 * R_scalar * g0[a, b]
        end
        trG = sum(gi0[a, b] * G[b, a] for a in 1:4, b in 1:4)
        @test isapprox(trG, -R_scalar, atol=1e-6)
    end

    # ================================================================
    # 5. Contracted Bianchi identity: ∇_a G^{ab} = 0 (Wald Eq 3.2.16)
    # ================================================================
    @testset "Contracted Bianchi: ∇_a G^{ab} = 0" begin
        # G^a_b
        G = zeros(4, 4)
        for a in 1:4, b in 1:4
            Ric_ab = Ric[a, b]
            G_low_ab = Ric_ab - 0.5 * R_scalar * g0[a, b]
            for c in 1:4; G[a, b] += gi0[a, c] * (Ric[c, b] - 0.5 * R_scalar * g0[c, b]); end
        end
        # Recompute G^a_b properly
        G_low = zeros(4, 4)
        for a in 1:4, b in 1:4; G_low[a, b] = Ric[a, b] - 0.5 * R_scalar * g0[a, b]; end
        Gup = zeros(4, 4)
        for a in 1:4, b in 1:4
            for c in 1:4; Gup[a, b] += gi0[a, c] * G_low[c, b]; end
        end

        function Gup_at(pt)
            Γp = fd_chris(pt); gp = schw_g(pt); gip = schw_ginv(pt)
            Rp = zeros(4, 4, 4, 4)
            for a in 1:4, b in 1:4, c in 1:4, d in 1:4
                xp = copy(pt); xp[c] += h_fd; xm = copy(pt); xm[c] -= h_fd
                dG1 = (fd_chris(xp)[a, d, b] - fd_chris(xm)[a, d, b]) / (2h_fd)
                xp = copy(pt); xp[d] += h_fd; xm = copy(pt); xm[d] -= h_fd
                dG2 = (fd_chris(xp)[a, c, b] - fd_chris(xm)[a, c, b]) / (2h_fd)
                val = dG1 - dG2
                for e in 1:4; val += Γp[a, c, e] * Γp[e, d, b] - Γp[a, d, e] * Γp[e, c, b]; end
                Rp[a, b, c, d] = val
            end
            Ricp = Float64.(metric_ricci(Rp, 4))
            Rsc = Float64(metric_ricci_scalar(Ricp, gip, 4))
            Gupp = zeros(4, 4)
            for a in 1:4, b in 1:4
                for c in 1:4; Gupp[a, b] += gip[a, c] * (Ricp[c, b] - 0.5 * Rsc * gp[c, b]); end
            end
            Gupp
        end

        divG = zeros(4)
        for b in 1:4, a in 1:4
            xp = copy(x0); xp[a] += h_fd; xm = copy(x0); xm[a] -= h_fd
            divG[b] += (Gup_at(xp)[a, b] - Gup_at(xm)[a, b]) / (2h_fd)
            for c in 1:4; divG[b] += Gamma[a, a, c] * Gup[c, b]; end
            for c in 1:4; divG[b] -= Gamma[c, a, b] * Gup[a, c]; end
        end

        @test maximum(abs.(divG)) < 1e-4
    end

    # ================================================================
    # 6. Kretschner scalar: K = 48 M²/r⁶ (Schwarzschild)
    # ================================================================
    @testset "Kretschner K = 48M²/r⁶" begin
        K = metric_kretschmann(Riem, g0, gi0, 4)
        K_exact = 48.0 * M_bh^2 / r_val^6
        @test isapprox(Float64(K), K_exact, rtol=1e-4)
    end

end
