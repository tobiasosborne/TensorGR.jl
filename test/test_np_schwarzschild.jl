#= NP Validation: Schwarzschild metric with Kinnersley null tetrad.
#
# Compute all NP quantities for Schwarzschild in Boyer-Lindquist coordinates
# (t, r, θ, φ) using the Kinnersley null tetrad. Verify Weyl scalars, NP Ricci
# scalars, and all 12 spin coefficients.
#
# Ground truth:
#   Teukolsky, Astrophys. J. 185, 635 (1973), Eqs 4.4-4.6.
#   Local copy: reference/papers/teukolsky_1973ApJ_185_635.pdf
#
# Sign convention:
#   Teukolsky uses (+,-,-,-) with l·n = +1, m·m* = -1.
#   Our code uses (-,+,+,+) with l·n = -1, m·m̄ = +1.
#   The tetrad vectors (contravariant) are identical in both conventions.
#   Spin coefficients (which involve lowered vectors) are NEGATED.
#   The Weyl scalar ψ₂ = -M/r³ has the same sign in both conventions.
=#

using LinearAlgebra: diagm

@testset "NP Validation: Schwarzschild" begin

    # ================================================================
    # Setup: Schwarzschild at M=1, r=3, θ=π/4
    # ================================================================
    # Use θ ≠ π/2 to test cotangent terms.
    M_bh  = 1.0
    r_val = 3.0
    θ_val = π / 4

    f  = 1.0 - 2M_bh / r_val          # = 1/3
    Δ  = r_val^2 - 2M_bh * r_val      # = 3
    sinθ = sin(θ_val)
    cosθ = cos(θ_val)

    g_schw = diagm([-f, 1.0 / f, r_val^2, r_val^2 * sinθ^2])
    ginv_schw = diagm([-1.0 / f, f, 1.0 / r_val^2, 1.0 / (r_val^2 * sinθ^2)])

    # ================================================================
    # Curvature via finite differences
    # ================================================================
    h_fd = 1e-6

    function schw_metric(pt)
        _, rv, th, _ = pt
        fv = 1.0 - 2M_bh / rv
        diagm([-fv, 1.0 / fv, rv^2, rv^2 * sin(th)^2])
    end

    function schw_ginv(pt)
        _, rv, th, _ = pt
        fv = 1.0 - 2M_bh / rv
        diagm([-1.0 / fv, fv, 1.0 / rv^2, 1.0 / (rv^2 * sin(th)^2)])
    end

    x0 = [0.0, r_val, θ_val, 0.0]

    function fd_christoffel(pt)
        gi = schw_ginv(pt); G = zeros(4, 4, 4)
        for a in 1:4, b in 1:4, c in 1:4
            s = 0.0
            for d in 1:4
                xp = copy(pt); xp[b] += h_fd; xm = copy(pt); xm[b] -= h_fd
                dg1 = (schw_metric(xp)[c, d] - schw_metric(xm)[c, d]) / (2h_fd)
                xp = copy(pt); xp[c] += h_fd; xm = copy(pt); xm[c] -= h_fd
                dg2 = (schw_metric(xp)[b, d] - schw_metric(xm)[b, d]) / (2h_fd)
                xp = copy(pt); xp[d] += h_fd; xm = copy(pt); xm[d] -= h_fd
                dg3 = (schw_metric(xp)[b, c] - schw_metric(xm)[b, c]) / (2h_fd)
                s += gi[a, d] * (dg1 + dg2 - dg3)
            end
            G[a, b, c] = s / 2
        end
        G
    end

    Gamma = fd_christoffel(x0)

    Riem = zeros(4, 4, 4, 4)
    for a in 1:4, b in 1:4, c in 1:4, d in 1:4
        xp = copy(x0); xp[c] += h_fd; xm = copy(x0); xm[c] -= h_fd
        dG1 = (fd_christoffel(xp)[a, d, b] - fd_christoffel(xm)[a, d, b]) / (2h_fd)
        xp = copy(x0); xp[d] += h_fd; xm = copy(x0); xm[d] -= h_fd
        dG2 = (fd_christoffel(xp)[a, c, b] - fd_christoffel(xm)[a, c, b]) / (2h_fd)
        val = dG1 - dG2
        for e in 1:4
            val += Gamma[a, c, e] * Gamma[e, d, b] - Gamma[a, d, e] * Gamma[e, c, b]
        end
        Riem[a, b, c, d] = val
    end

    Ric_schw  = Float64.(metric_ricci(Riem, 4))
    R_schw    = Float64(metric_ricci_scalar(Ric_schw, ginv_schw, 4))
    Weyl_schw = Float64.(metric_weyl(Riem, Ric_schw, R_schw, g_schw, ginv_schw, 4))

    # ================================================================
    # 1. Kinnersley null tetrad (Teukolsky 1973, Eq 4.4 with a=0)
    # ================================================================
    # l^μ = (1/f, 1, 0, 0)
    # n^μ = (1/2)(1, -f, 0, 0)
    # m^μ = (0, 0, 1, i/sinθ) / (r√2)
    l_kin    = ComplexF64[1.0 / f, 1.0, 0.0, 0.0]
    n_kin    = ComplexF64[0.5, -f / 2, 0.0, 0.0]
    m_kin    = ComplexF64[0.0, 0.0, 1.0, im / sinθ] / (r_val * sqrt(2.0))
    mbar_kin = conj.(m_kin)

    @testset "Kinnersley tetrad normalization" begin
        @test validate_null_tetrad(l_kin, n_kin, m_kin, mbar_kin, g_schw; atol=1e-10)
    end

    # ================================================================
    # 2. Weyl scalars: Ψ₂ = -M/r³, all others zero (Teukolsky Eq 4.6)
    # ================================================================
    @testset "Weyl scalars (Teukolsky Eq 4.6)" begin
        psi = weyl_scalars(Weyl_schw, l_kin, n_kin, m_kin, mbar_kin)

        # Ψ₂ = -M/r³ (same sign in both signature conventions)
        @test isapprox(real(psi.Psi2), -M_bh / r_val^3, rtol=1e-3)
        @test isapprox(imag(psi.Psi2), 0.0, atol=1e-8)

        # Type D: all other Weyl scalars vanish (Teukolsky Eq 2.1)
        @test isapprox(abs(psi.Psi0), 0.0, atol=1e-5)
        @test isapprox(abs(psi.Psi1), 0.0, atol=1e-5)
        @test isapprox(abs(psi.Psi3), 0.0, atol=1e-5)
        @test isapprox(abs(psi.Psi4), 0.0, atol=1e-5)
    end

    # ================================================================
    # 3. Vacuum: Φ_{ij} = 0, Λ = 0
    # ================================================================
    @testset "NP Ricci scalars (vacuum)" begin
        @test isapprox(R_schw / 24, 0.0, atol=1e-4)
        for a in 1:4, b in 1:4
            @test isapprox(Ric_schw[a, b], 0.0, atol=1e-4)
        end
    end

    # ================================================================
    # 4. Spin coefficients (Teukolsky 1973, Eq 4.5 with a=0)
    # ================================================================
    # Ground truth: Teukolsky Eq 4.5 gives spin coefficients in (+,-,-,-)
    # signature with the standard NP sign convention (negative sign on
    # n-type coefficients ν, λ, μ, π).
    #
    # Our code uses (-,+,+,+) and defines ALL simple coefficients without
    # the NP negative sign (src/spinors/np.jl). For l-type coefficients
    # (κ, σ, ρ, τ), the metric flip negates them. For n-type coefficients
    # (ν, λ, μ, π), the metric flip and the missing NP negative cancel,
    # giving the SAME values as Teukolsky. For compound coefficients
    # (ε, γ, α, β), both terms involve l/m-type derivatives, so they
    # are negated.
    #
    # Result for Schwarzschild (a=0):
    #   ρ = +1/r (negated), β = -cotθ/(2√2 r) (negated),
    #   μ = -Δ/(2r³) (same), γ = -M/(2r²) (negated),
    #   α = -β = +cotθ/(2√2 r), all others = 0

    @testset "Spin coefficients (Teukolsky Eq 4.5)" begin
        # Helper: tetrad vectors (lowered) as function of position
        function tetrad_at(pt)
            _, rv, th, _ = pt
            fv = 1.0 - 2M_bh / rv; sv = sin(th)
            lv = ComplexF64[1.0 / fv, 1.0, 0.0, 0.0]
            nv = ComplexF64[0.5, -fv / 2, 0.0, 0.0]
            mv = ComplexF64[0.0, 0.0, 1.0, im / sv] / (rv * sqrt(2.0))
            gv = schw_metric(pt)
            ld = ComplexF64[sum(gv[a, b] * lv[b] for b in 1:4) for a in 1:4]
            nd = ComplexF64[sum(gv[a, b] * nv[b] for b in 1:4) for a in 1:4]
            md = ComplexF64[sum(gv[a, b] * mv[b] for b in 1:4) for a in 1:4]
            (l=lv, n=nv, m=mv, mbar=conj.(mv),
             ld=ld, nd=nd, md=md, mbard=conj.(md))
        end

        function covd_lowered(vfn, b_idx)
            xp = copy(x0); xp[b_idx] += h_fd
            xm = copy(x0); xm[b_idx] -= h_fd
            dv = (vfn(xp) .- vfn(xm)) ./ (2h_fd)
            v0 = vfn(x0)
            result = zeros(ComplexF64, 4)
            for a in 1:4
                result[a] = dv[a]
                for c in 1:4
                    result[a] -= Gamma[c, b_idx, a] * v0[c]
                end
            end
            result
        end

        ld_fn(pt) = tetrad_at(pt).ld
        nd_fn(pt) = tetrad_at(pt).nd
        md_fn(pt) = tetrad_at(pt).md

        t0 = tetrad_at(x0)

        # v1^a v2^b ∇_b v3_a
        function sc(vfn, ca, cb)
            s = zero(ComplexF64)
            for b in 1:4
                nv = covd_lowered(vfn, b)
                for a in 1:4
                    s += ca[a] * cb[b] * nv[a]
                end
            end
            s
        end

        # Simple spin coefficients (NP 1962, Eq 4.2)
        κ     = sc(ld_fn, t0.m, t0.l)
        σ_sc  = sc(ld_fn, t0.m, t0.m)
        ρ_sc  = sc(ld_fn, t0.m, t0.mbar)
        τ_sc  = sc(ld_fn, t0.m, t0.n)
        ν_sc  = sc(nd_fn, t0.mbar, t0.n)
        λ_sc  = sc(nd_fn, t0.mbar, t0.mbar)
        μ_sc  = sc(nd_fn, t0.mbar, t0.m)
        π_sc  = sc(nd_fn, t0.mbar, t0.l)

        # Compound spin coefficients
        ε_sc = (sc(ld_fn, t0.n, t0.l) - sc(md_fn, t0.mbar, t0.l)) / 2
        γ_sc = (sc(ld_fn, t0.n, t0.n) - sc(md_fn, t0.mbar, t0.n)) / 2
        α_sc = (sc(ld_fn, t0.n, t0.mbar) - sc(md_fn, t0.mbar, t0.mbar)) / 2
        β_sc = (sc(ld_fn, t0.n, t0.m) - sc(md_fn, t0.mbar, t0.m)) / 2

        # Ground truth: Teukolsky Eq 4.5, adapted for our convention
        ρ_exact = 1.0 / r_val                             # negated from Teukolsky -1/r
        μ_exact = -Δ / (2 * r_val^3)                      # same as Teukolsky -Δ/(2r³)
        γ_exact = -M_bh / (2 * r_val^2)                   # negated from Teukolsky M/(2r²)
        β_exact = -cosθ / (2 * sqrt(2.0) * r_val * sinθ)  # negated from Teukolsky cotθ/(2√2 r)
        α_exact = -β_exact                                 # α = -β

        # Use atol=1e-4 for finite-difference accuracy
        @testset "Vanishing: κ, σ, λ, ν, τ, π, ε (Eq 2.1 + ε=0 gauge)" begin
            @test isapprox(abs(κ),    0.0, atol=1e-5)
            @test isapprox(abs(σ_sc), 0.0, atol=1e-5)
            @test isapprox(abs(λ_sc), 0.0, atol=1e-5)
            @test isapprox(abs(ν_sc), 0.0, atol=1e-5)
            @test isapprox(abs(τ_sc), 0.0, atol=1e-5)
            @test isapprox(abs(π_sc), 0.0, atol=1e-5)
            @test isapprox(abs(ε_sc), 0.0, atol=1e-5)
        end

        @testset "ρ = +1/r (Teukolsky: -1/r, negated)" begin
            @test isapprox(real(ρ_sc), ρ_exact, rtol=1e-3)
            @test isapprox(imag(ρ_sc), 0.0, atol=1e-6)
        end

        @testset "μ = -Δ/(2r³) (same as Teukolsky)" begin
            @test isapprox(real(μ_sc), μ_exact, rtol=1e-3)
            @test isapprox(imag(μ_sc), 0.0, atol=1e-6)
        end

        @testset "γ = -M/(2r²) (Teukolsky: M/(2r²), negated)" begin
            @test isapprox(real(γ_sc), γ_exact, rtol=1e-3)
            @test isapprox(imag(γ_sc), 0.0, atol=1e-6)
        end

        @testset "β = -cotθ/(2√2 r) (Teukolsky: cotθ/(2√2 r), negated)" begin
            @test isapprox(real(β_sc), β_exact, rtol=1e-3)
            @test isapprox(imag(β_sc), 0.0, atol=1e-6)
        end

        @testset "α = -β (Teukolsky: α = π - β* = -β)" begin
            @test isapprox(real(α_sc), α_exact, rtol=1e-3)
            @test isapprox(imag(α_sc), 0.0, atol=1e-6)
            @test isapprox(real(α_sc + β_sc), 0.0, atol=1e-5)
        end
    end

    # ================================================================
    # 5. Petrov Type D
    # ================================================================
    @testset "Petrov Type D" begin
        @test petrov_classify(Weyl_schw, g_schw; atol=1e-4) == TypeD
    end

    # ================================================================
    # 6. Second evaluation point: r=5, θ=π/3
    # ================================================================
    @testset "Second point: r=5, θ=π/3" begin
        r2 = 5.0; θ2 = π / 3
        f2 = 1.0 - 2M_bh / r2
        s2 = sin(θ2)

        l2 = ComplexF64[1.0 / f2, 1.0, 0.0, 0.0]
        n2 = ComplexF64[0.5, -f2 / 2, 0.0, 0.0]
        m2 = ComplexF64[0.0, 0.0, 1.0, im / s2] / (r2 * sqrt(2.0))

        g2 = diagm([-f2, 1.0 / f2, r2^2, r2^2 * s2^2])
        ginv2 = diagm([-1.0 / f2, f2, 1.0 / r2^2, 1.0 / (r2^2 * s2^2)])
        x02 = [0.0, r2, θ2, 0.0]

        function fd_chris2(pt)
            gi = schw_ginv(pt); G = zeros(4, 4, 4)
            for a in 1:4, b in 1:4, c in 1:4
                s = 0.0
                for d in 1:4
                    xp = copy(pt); xp[b] += h_fd; xm = copy(pt); xm[b] -= h_fd
                    dg1 = (schw_metric(xp)[c, d] - schw_metric(xm)[c, d]) / (2h_fd)
                    xp = copy(pt); xp[c] += h_fd; xm = copy(pt); xm[c] -= h_fd
                    dg2 = (schw_metric(xp)[b, d] - schw_metric(xm)[b, d]) / (2h_fd)
                    xp = copy(pt); xp[d] += h_fd; xm = copy(pt); xm[d] -= h_fd
                    dg3 = (schw_metric(xp)[b, c] - schw_metric(xm)[b, c]) / (2h_fd)
                    s += gi[a, d] * (dg1 + dg2 - dg3)
                end; G[a, b, c] = s / 2
            end; G
        end

        Gamma2 = fd_chris2(x02)
        Riem2 = zeros(4, 4, 4, 4)
        for a in 1:4, b in 1:4, c in 1:4, d in 1:4
            xp = copy(x02); xp[c] += h_fd; xm = copy(x02); xm[c] -= h_fd
            dG1 = (fd_chris2(xp)[a, d, b] - fd_chris2(xm)[a, d, b]) / (2h_fd)
            xp = copy(x02); xp[d] += h_fd; xm = copy(x02); xm[d] -= h_fd
            dG2 = (fd_chris2(xp)[a, c, b] - fd_chris2(xm)[a, c, b]) / (2h_fd)
            val = dG1 - dG2
            for e in 1:4
                val += Gamma2[a, c, e] * Gamma2[e, d, b] - Gamma2[a, d, e] * Gamma2[e, c, b]
            end
            Riem2[a, b, c, d] = val
        end

        Ric2 = Float64.(metric_ricci(Riem2, 4))
        R2 = Float64(metric_ricci_scalar(Ric2, ginv2, 4))
        W2 = Float64.(metric_weyl(Riem2, Ric2, R2, g2, ginv2, 4))

        psi2 = weyl_scalars(W2, l2, n2, m2, conj.(m2))

        # Ψ₂ = -M/r³ (Teukolsky Eq 4.6)
        @test isapprox(real(psi2.Psi2), -M_bh / r2^3, rtol=1e-3)
        @test isapprox(abs(psi2.Psi0), 0.0, atol=1e-5)
        @test isapprox(abs(psi2.Psi1), 0.0, atol=1e-5)
        @test isapprox(abs(psi2.Psi3), 0.0, atol=1e-5)
        @test isapprox(abs(psi2.Psi4), 0.0, atol=1e-5)
        @test petrov_classify(W2, g2; atol=1e-4) == TypeD
    end

end
