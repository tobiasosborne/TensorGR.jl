#= NP Validation: Kerr metric with Kinnersley null tetrad.
#
# Compute all NP quantities for Kerr in Boyer-Lindquist coordinates
# (t, r, θ, φ) using the Kinnersley null tetrad. Verify Weyl scalars
# and all 12 spin coefficients.
#
# Ground truth:
#   Teukolsky, Astrophys. J. 185, 635 (1973), Eqs 4.1, 4.4-4.6.
#   Local copy: reference/papers/teukolsky_1973ApJ_185_635.pdf
#
# Sign convention notes: see test_np_schwarzschild.jl header.
#   l-type (κ,σ,ρ,τ): negated from Teukolsky
#   n-type (ν,λ,μ,π): same as Teukolsky
#   compound (ε,γ,α,β): negated from Teukolsky
#   ψ₂: same sign in both conventions
=#

using LinearAlgebra: diagm

@testset "NP Validation: Kerr" begin

    # ================================================================
    # Setup: Kerr at M=1, a=0.5, r=4, θ=π/4
    # ================================================================
    M_bh = 1.0
    a    = 0.5
    r_val = 4.0
    θ_val = π / 4

    Σ  = r_val^2 + a^2 * cos(θ_val)^2
    Δ  = r_val^2 - 2M_bh * r_val + a^2
    sinθ = sin(θ_val)
    cosθ = cos(θ_val)

    # Kerr metric in Boyer-Lindquist (Teukolsky Eq 4.1, signature -,+,+,+)
    g_kerr = zeros(4, 4)
    g_kerr[1, 1] = -(1 - 2M_bh * r_val / Σ)
    g_kerr[1, 4] = -2M_bh * a * r_val * sinθ^2 / Σ
    g_kerr[4, 1] = g_kerr[1, 4]
    g_kerr[2, 2] = Σ / Δ
    g_kerr[3, 3] = Σ
    g_kerr[4, 4] = (r_val^2 + a^2 + 2M_bh * a^2 * r_val * sinθ^2 / Σ) * sinθ^2

    # Inverse metric
    det_g = g_kerr[1, 1] * g_kerr[4, 4] - g_kerr[1, 4]^2
    ginv_kerr = zeros(4, 4)
    ginv_kerr[1, 1] = g_kerr[4, 4] / det_g
    ginv_kerr[1, 4] = -g_kerr[1, 4] / det_g
    ginv_kerr[4, 1] = ginv_kerr[1, 4]
    ginv_kerr[4, 4] = g_kerr[1, 1] / det_g
    ginv_kerr[2, 2] = 1.0 / g_kerr[2, 2]
    ginv_kerr[3, 3] = 1.0 / g_kerr[3, 3]

    # ================================================================
    # Kinnersley null tetrad (Teukolsky 1973, Eq 4.4)
    # ================================================================
    l_kin = ComplexF64[(r_val^2 + a^2) / Δ, 1.0, 0.0, a / Δ]
    n_kin = ComplexF64[r_val^2 + a^2, -Δ, 0.0, a] ./ (2Σ)
    m_kin = ComplexF64[im * a * sinθ, 0.0, 1.0, im / sinθ] ./
            (sqrt(2.0) * (r_val + im * a * cosθ))
    mbar_kin = conj.(m_kin)

    @testset "Kinnersley tetrad normalization" begin
        dot(v, w) = sum(g_kerr[i, j] * v[i] * w[j] for i in 1:4, j in 1:4)
        @test isapprox(dot(l_kin, n_kin), -1.0, atol=1e-10)
        @test isapprox(dot(m_kin, mbar_kin), 1.0, atol=1e-10)
        @test isapprox(abs(dot(l_kin, l_kin)), 0.0, atol=1e-10)
        @test isapprox(abs(dot(n_kin, n_kin)), 0.0, atol=1e-10)
        @test isapprox(abs(dot(m_kin, m_kin)), 0.0, atol=1e-10)
        @test isapprox(abs(dot(l_kin, m_kin)), 0.0, atol=1e-10)
        @test isapprox(abs(dot(n_kin, m_kin)), 0.0, atol=1e-10)
    end

    # ================================================================
    # Curvature via finite differences
    # ================================================================
    h_fd = 1e-6

    function kerr_metric(pt)
        _, rv, th, _ = pt
        sv = sin(th); cv = cos(th)
        Σv = rv^2 + a^2 * cv^2
        Δv = rv^2 - 2M_bh * rv + a^2
        gv = zeros(4, 4)
        gv[1, 1] = -(1 - 2M_bh * rv / Σv)
        gv[1, 4] = -2M_bh * a * rv * sv^2 / Σv
        gv[4, 1] = gv[1, 4]
        gv[2, 2] = Σv / Δv
        gv[3, 3] = Σv
        gv[4, 4] = (rv^2 + a^2 + 2M_bh * a^2 * rv * sv^2 / Σv) * sv^2
        gv
    end

    function kerr_ginv(pt)
        gv = kerr_metric(pt)
        det_tphi = gv[1, 1] * gv[4, 4] - gv[1, 4]^2
        gi = zeros(4, 4)
        gi[1, 1] = gv[4, 4] / det_tphi
        gi[1, 4] = -gv[1, 4] / det_tphi
        gi[4, 1] = gi[1, 4]
        gi[4, 4] = gv[1, 1] / det_tphi
        gi[2, 2] = 1.0 / gv[2, 2]
        gi[3, 3] = 1.0 / gv[3, 3]
        gi
    end

    x0 = [0.0, r_val, θ_val, 0.0]

    function fd_christoffel(pt)
        gi = kerr_ginv(pt); G = zeros(4, 4, 4)
        for aa in 1:4, bb in 1:4, cc in 1:4
            s = 0.0
            for dd in 1:4
                xp = copy(pt); xp[bb] += h_fd; xm = copy(pt); xm[bb] -= h_fd
                dg1 = (kerr_metric(xp)[cc, dd] - kerr_metric(xm)[cc, dd]) / (2h_fd)
                xp = copy(pt); xp[cc] += h_fd; xm = copy(pt); xm[cc] -= h_fd
                dg2 = (kerr_metric(xp)[bb, dd] - kerr_metric(xm)[bb, dd]) / (2h_fd)
                xp = copy(pt); xp[dd] += h_fd; xm = copy(pt); xm[dd] -= h_fd
                dg3 = (kerr_metric(xp)[bb, cc] - kerr_metric(xm)[bb, cc]) / (2h_fd)
                s += gi[aa, dd] * (dg1 + dg2 - dg3)
            end
            G[aa, bb, cc] = s / 2
        end
        G
    end

    Gamma = fd_christoffel(x0)

    Riem = zeros(4, 4, 4, 4)
    for aa in 1:4, bb in 1:4, cc in 1:4, dd in 1:4
        xp = copy(x0); xp[cc] += h_fd; xm = copy(x0); xm[cc] -= h_fd
        dG1 = (fd_christoffel(xp)[aa, dd, bb] - fd_christoffel(xm)[aa, dd, bb]) / (2h_fd)
        xp = copy(x0); xp[dd] += h_fd; xm = copy(x0); xm[dd] -= h_fd
        dG2 = (fd_christoffel(xp)[aa, cc, bb] - fd_christoffel(xm)[aa, cc, bb]) / (2h_fd)
        val = dG1 - dG2
        for ee in 1:4
            val += Gamma[aa, cc, ee] * Gamma[ee, dd, bb] - Gamma[aa, dd, ee] * Gamma[ee, cc, bb]
        end
        Riem[aa, bb, cc, dd] = val
    end

    Ric_kerr = Float64.(metric_ricci(Riem, 4))
    R_kerr   = Float64(metric_ricci_scalar(Ric_kerr, ginv_kerr, 4))
    Weyl_kerr = Float64.(metric_weyl(Riem, Ric_kerr, R_kerr, g_kerr, ginv_kerr, 4))

    # ================================================================
    # 2. Weyl scalars: Ψ₂ = Mρ³ = -M/(r-ia cosθ)³ (Teukolsky Eq 4.6)
    # ================================================================
    @testset "Weyl scalars (Teukolsky Eq 4.6)" begin
        psi = weyl_scalars(Weyl_kerr, l_kin, n_kin, m_kin, mbar_kin)

        # ρ_T = -1/(r - ia cosθ) (Teukolsky convention)
        ρ_T = -1.0 / (r_val - im * a * cosθ)
        Psi2_exact = M_bh * ρ_T^3

        @test isapprox(real(psi.Psi2), real(Psi2_exact), rtol=1e-3)
        @test isapprox(imag(psi.Psi2), imag(Psi2_exact), rtol=1e-3)

        # Type D: all other Weyl scalars vanish (Teukolsky Eq 2.1)
        @test isapprox(abs(psi.Psi0), 0.0, atol=1e-4)
        @test isapprox(abs(psi.Psi1), 0.0, atol=1e-4)
        @test isapprox(abs(psi.Psi3), 0.0, atol=1e-4)
        @test isapprox(abs(psi.Psi4), 0.0, atol=1e-4)
    end

    # ================================================================
    # 3. Vacuum: Ricci = 0
    # ================================================================
    @testset "Vacuum: Ricci = 0" begin
        @test isapprox(R_kerr, 0.0, atol=1e-3)
        for i in 1:4, j in 1:4
            @test isapprox(Ric_kerr[i, j], 0.0, atol=1e-3)
        end
    end

    # ================================================================
    # 4. Spin coefficients (Teukolsky 1973, Eq 4.5)
    # ================================================================
    @testset "Spin coefficients (Teukolsky Eq 4.5)" begin
        # Tetrad vectors as functions of position (for finite-diff derivatives)
        function tetrad_at(pt)
            _, rv, th, _ = pt
            sv = sin(th); cv = cos(th)
            Σv = rv^2 + a^2 * cv^2
            Δv = rv^2 - 2M_bh * rv + a^2
            lv = ComplexF64[(rv^2 + a^2) / Δv, 1.0, 0.0, a / Δv]
            nv = ComplexF64[rv^2 + a^2, -Δv, 0.0, a] ./ (2Σv)
            mv = ComplexF64[im * a * sv, 0.0, 1.0, im / sv] ./
                 (sqrt(2.0) * (rv + im * a * cv))
            gv = kerr_metric(pt)
            ld = ComplexF64[sum(gv[i, j] * lv[j] for j in 1:4) for i in 1:4]
            nd = ComplexF64[sum(gv[i, j] * nv[j] for j in 1:4) for i in 1:4]
            md = ComplexF64[sum(gv[i, j] * mv[j] for j in 1:4) for i in 1:4]
            (l=lv, n=nv, m=mv, mbar=conj.(mv),
             ld=ld, nd=nd, md=md, mbard=conj.(md))
        end

        function covd_lowered(vfn, b_idx)
            xp = copy(x0); xp[b_idx] += h_fd
            xm = copy(x0); xm[b_idx] -= h_fd
            dv = (vfn(xp) .- vfn(xm)) ./ (2h_fd)
            v0 = vfn(x0)
            result = zeros(ComplexF64, 4)
            for i in 1:4
                result[i] = dv[i]
                for c in 1:4
                    result[i] -= Gamma[c, b_idx, i] * v0[c]
                end
            end
            result
        end

        ld_fn(pt) = tetrad_at(pt).ld
        nd_fn(pt) = tetrad_at(pt).nd
        md_fn(pt) = tetrad_at(pt).md

        t0 = tetrad_at(x0)

        function sc(vfn, ca, cb)
            s = zero(ComplexF64)
            for b in 1:4
                nv = covd_lowered(vfn, b)
                for i in 1:4
                    s += ca[i] * cb[b] * nv[i]
                end
            end
            s
        end

        # Compute all 12 coefficients
        κ     = sc(ld_fn, t0.m, t0.l)
        σ_sc  = sc(ld_fn, t0.m, t0.m)
        ρ_sc  = sc(ld_fn, t0.m, t0.mbar)
        τ_sc  = sc(ld_fn, t0.m, t0.n)
        ν_sc  = sc(nd_fn, t0.mbar, t0.n)
        λ_sc  = sc(nd_fn, t0.mbar, t0.mbar)
        μ_sc  = sc(nd_fn, t0.mbar, t0.m)
        π_sc  = sc(nd_fn, t0.mbar, t0.l)
        ε_sc = (sc(ld_fn, t0.n, t0.l) - sc(md_fn, t0.mbar, t0.l)) / 2
        γ_sc = (sc(ld_fn, t0.n, t0.n) - sc(md_fn, t0.mbar, t0.n)) / 2
        α_sc = (sc(ld_fn, t0.n, t0.mbar) - sc(md_fn, t0.mbar, t0.mbar)) / 2
        β_sc = (sc(ld_fn, t0.n, t0.m) - sc(md_fn, t0.mbar, t0.m)) / 2

        # Teukolsky Eq 4.5 ground truth (in +,-,-,- convention)
        ρ_T = -1.0 / (r_val - im * a * cosθ)
        β_T = -conj(ρ_T) * cosθ / (2 * sqrt(2.0) * sinθ)
        π_T = im * a * ρ_T^2 * sinθ / sqrt(2.0)
        τ_T = -im * a * ρ_T * conj(ρ_T) * sinθ / sqrt(2.0)
        μ_T = ρ_T^2 * conj(ρ_T) * Δ / 2
        γ_T = μ_T + ρ_T * conj(ρ_T) * (r_val - M_bh) / 2
        α_T = π_T - conj(β_T)

        # Our convention: l-type negated, n-type same, compound negated
        ρ_exact = -ρ_T      # negated
        τ_exact = -τ_T      # negated
        μ_exact = μ_T       # same
        π_exact = π_T       # same
        γ_exact = -γ_T      # negated
        β_exact = -β_T      # negated
        α_exact = -α_T      # negated

        @testset "Vanishing: κ, σ, ν, λ, ε (Type D + ε=0 gauge)" begin
            @test isapprox(abs(κ),    0.0, atol=1e-4)
            @test isapprox(abs(σ_sc), 0.0, atol=1e-4)
            @test isapprox(abs(ν_sc), 0.0, atol=1e-4)
            @test isapprox(abs(λ_sc), 0.0, atol=1e-4)
            @test isapprox(abs(ε_sc), 0.0, atol=1e-4)
        end

        @testset "ρ (complex, negated from Teukolsky)" begin
            @test isapprox(real(ρ_sc), real(ρ_exact), rtol=1e-3)
            @test isapprox(imag(ρ_sc), imag(ρ_exact), rtol=1e-3)
        end

        @testset "τ (purely imaginary, negated from Teukolsky)" begin
            @test isapprox(real(τ_sc), 0.0, atol=1e-5)
            @test isapprox(imag(τ_sc), imag(τ_exact), rtol=1e-3)
        end

        @testset "μ (complex, same as Teukolsky)" begin
            @test isapprox(real(μ_sc), real(μ_exact), rtol=1e-3)
            @test isapprox(imag(μ_sc), imag(μ_exact), rtol=1e-3)
        end

        @testset "π (complex, same as Teukolsky)" begin
            @test isapprox(real(π_sc), real(π_exact), rtol=1e-3)
            @test isapprox(imag(π_sc), imag(π_exact), rtol=1e-3)
        end

        @testset "γ (complex, negated from Teukolsky)" begin
            @test isapprox(real(γ_sc), real(γ_exact), rtol=1e-3)
            @test isapprox(imag(γ_sc), imag(γ_exact), rtol=1e-3)
        end

        @testset "β (complex, negated from Teukolsky)" begin
            @test isapprox(real(β_sc), real(β_exact), rtol=1e-3)
            @test isapprox(imag(β_sc), imag(β_exact), rtol=1e-3)
        end

        @testset "α (complex, negated from Teukolsky)" begin
            @test isapprox(real(α_sc), real(α_exact), rtol=1e-3)
            @test isapprox(imag(α_sc), imag(α_exact), rtol=1e-3)
        end
    end

    # ================================================================
    # 5. Petrov Type D (via Weyl scalars — petrov_classify requires
    #    diagonal metric, so we use the already-computed Kinnersley
    #    tetrad Weyl scalars directly)
    # ================================================================
    @testset "Petrov Type D" begin
        psi = weyl_scalars(Weyl_kerr, l_kin, n_kin, m_kin, mbar_kin)
        psi_nt = (Psi0=psi.Psi0, Psi1=psi.Psi1, Psi2=psi.Psi2,
                  Psi3=psi.Psi3, Psi4=psi.Psi4)
        inv_t = petrov_invariants(psi_nt)
        @test is_algebraically_special(inv_t.I, inv_t.J; atol=1e-3)
        # Type D: Ψ₀=Ψ₁=Ψ₃=Ψ₄=0, Ψ₂≠0
        @test abs(psi.Psi2) > 1e-6
        @test abs(psi.Psi0) < 1e-4
        @test abs(psi.Psi4) < 1e-4
    end

end
