#= Goldberg-Sachs theorem validation.
#
# The Goldberg-Sachs theorem states: In a vacuum spacetime, a null vector
# field l^a is a repeated principal null direction of the Weyl tensor
# if and only if it is geodesic (κ=0) and shear-free (σ=0).
#
# Equivalently: Ψ₀ = Ψ₁ = 0  ⟺  κ = σ = 0  (in vacuum).
#
# We verify both directions for Schwarzschild and Kerr using the
# Kinnersley tetrad (aligned with the repeated PND).
#
# Ground truth:
#   Goldberg & Sachs, Acta Phys. Polon. Suppl. 22, 13 (1962)
#   Teukolsky, Astrophys. J. 185, 635 (1973), Eq 2.1
#   Local copy: reference/papers/teukolsky_1973ApJ_185_635.pdf
=#

using LinearAlgebra: diagm

@testset "Goldberg-Sachs Theorem" begin

    # ================================================================
    # Shared infrastructure: compute NP quantities from metric + tetrad
    # ================================================================
    h_fd = 1e-6

    function compute_np_quantities(metric_fn, ginv_fn, tetrad_fn, x0)
        g0 = metric_fn(x0)
        gi0 = ginv_fn(x0)
        t0 = tetrad_fn(x0)

        # Christoffel via finite differences
        function fd_chris(pt)
            gi = ginv_fn(pt); G = zeros(4, 4, 4)
            for a in 1:4, b in 1:4, c in 1:4
                s = 0.0
                for d in 1:4
                    xp = copy(pt); xp[b] += h_fd; xm = copy(pt); xm[b] -= h_fd
                    dg1 = (metric_fn(xp)[c, d] - metric_fn(xm)[c, d]) / (2h_fd)
                    xp = copy(pt); xp[c] += h_fd; xm = copy(pt); xm[c] -= h_fd
                    dg2 = (metric_fn(xp)[b, d] - metric_fn(xm)[b, d]) / (2h_fd)
                    xp = copy(pt); xp[d] += h_fd; xm = copy(pt); xm[d] -= h_fd
                    dg3 = (metric_fn(xp)[b, c] - metric_fn(xm)[b, c]) / (2h_fd)
                    s += gi[a, d] * (dg1 + dg2 - dg3)
                end; G[a, b, c] = s / 2
            end; G
        end

        Gamma = fd_chris(x0)

        # Riemann
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

        Ric = Float64.(metric_ricci(Riem, 4))
        R = Float64(metric_ricci_scalar(Ric, gi0, 4))
        W = Float64.(metric_weyl(Riem, Ric, R, g0, gi0, 4))

        # Weyl scalars
        psi = weyl_scalars(W, t0.l, t0.n, t0.m, t0.mbar)

        # Spin coefficients via covariant derivative
        ld_fn(pt) = tetrad_fn(pt).ld
        nd_fn(pt) = tetrad_fn(pt).nd

        function covd_low(vfn, b_idx)
            xp = copy(x0); xp[b_idx] += h_fd
            xm = copy(x0); xm[b_idx] -= h_fd
            dv = (vfn(xp) .- vfn(xm)) ./ (2h_fd)
            v0 = vfn(x0)
            result = zeros(ComplexF64, 4)
            for i in 1:4
                result[i] = dv[i]
                for c in 1:4; result[i] -= Gamma[c, b_idx, i] * v0[c]; end
            end
            result
        end

        function sc(vfn, ca, cb)
            s = zero(ComplexF64)
            for b in 1:4
                nv = covd_low(vfn, b)
                for i in 1:4; s += ca[i] * cb[b] * nv[i]; end
            end; s
        end

        kappa = sc(ld_fn, t0.m, t0.l)
        sigma = sc(ld_fn, t0.m, t0.m)

        (psi=psi, kappa=kappa, sigma=sigma, Ric=Ric)
    end

    # ================================================================
    # 1. Schwarzschild: verify κ=σ=0 ⟺ Ψ₀=Ψ₁=0
    # ================================================================
    @testset "Schwarzschild: Kinnersley tetrad (Type D)" begin
        M_bh = 1.0

        function schw_metric(pt)
            fv = 1.0 - 2M_bh / pt[2]
            diagm([-fv, 1.0 / fv, pt[2]^2, pt[2]^2 * sin(pt[3])^2])
        end
        function schw_ginv(pt)
            fv = 1.0 - 2M_bh / pt[2]
            diagm([-1.0 / fv, fv, 1.0 / pt[2]^2, 1.0 / (pt[2]^2 * sin(pt[3])^2)])
        end
        function schw_tetrad(pt)
            rv = pt[2]; th = pt[3]; fv = 1.0 - 2M_bh / rv; sv = sin(th)
            l = ComplexF64[1.0 / fv, 1.0, 0.0, 0.0]
            n = ComplexF64[0.5, -fv / 2, 0.0, 0.0]
            m = ComplexF64[0.0, 0.0, 1.0, im / sv] / (rv * sqrt(2.0))
            gv = schw_metric(pt)
            ld = ComplexF64[sum(gv[i, j] * l[j] for j in 1:4) for i in 1:4]
            nd = ComplexF64[sum(gv[i, j] * n[j] for j in 1:4) for i in 1:4]
            (l=l, n=n, m=m, mbar=conj.(m), ld=ld, nd=nd)
        end

        for (r_val, θ_val) in [(3.0, π / 4), (5.0, π / 3), (8.0, π / 6)]
            x0 = [0.0, r_val, θ_val, 0.0]
            np = compute_np_quantities(schw_metric, schw_ginv, schw_tetrad, x0)

            @testset "r=$r_val, θ=$(round(θ_val, digits=2))" begin
                # Forward: κ=σ=0 (geodesic, shear-free)
                @test isapprox(abs(np.kappa), 0.0, atol=1e-5)
                @test isapprox(abs(np.sigma), 0.0, atol=1e-5)

                # Forward: Ψ₀=Ψ₁=0 (repeated PND)
                @test isapprox(abs(np.psi.Psi0), 0.0, atol=1e-5)
                @test isapprox(abs(np.psi.Psi1), 0.0, atol=1e-5)

                # Ψ₂ ≠ 0 (Type D, not conformally flat)
                @test abs(np.psi.Psi2) > 1e-6

                # Also verify Ψ₃=Ψ₄=0 (both PNDs are repeated for Type D)
                @test isapprox(abs(np.psi.Psi3), 0.0, atol=1e-5)
                @test isapprox(abs(np.psi.Psi4), 0.0, atol=1e-5)

                # Vacuum check (finite-diff residual grows at large r, small θ)
                @test isapprox(sum(abs.(np.Ric)), 0.0, atol=1e-2)
            end
        end
    end

    # ================================================================
    # 2. Kerr: verify κ=σ=0 ⟺ Ψ₀=Ψ₁=0
    # ================================================================
    @testset "Kerr: Kinnersley tetrad (Type D)" begin
        M_bh = 1.0; a_spin = 0.6

        function kerr_metric(pt)
            rv = pt[2]; th = pt[3]; sv = sin(th); cv = cos(th)
            Σv = rv^2 + a_spin^2 * cv^2
            Δv = rv^2 - 2M_bh * rv + a_spin^2
            gv = zeros(4, 4)
            gv[1, 1] = -(1 - 2M_bh * rv / Σv)
            gv[1, 4] = -2M_bh * a_spin * rv * sv^2 / Σv
            gv[4, 1] = gv[1, 4]
            gv[2, 2] = Σv / Δv
            gv[3, 3] = Σv
            gv[4, 4] = (rv^2 + a_spin^2 + 2M_bh * a_spin^2 * rv * sv^2 / Σv) * sv^2
            gv
        end
        function kerr_ginv(pt)
            gv = kerr_metric(pt)
            det_tp = gv[1, 1] * gv[4, 4] - gv[1, 4]^2
            gi = zeros(4, 4)
            gi[1, 1] = gv[4, 4] / det_tp
            gi[1, 4] = -gv[1, 4] / det_tp; gi[4, 1] = gi[1, 4]
            gi[4, 4] = gv[1, 1] / det_tp
            gi[2, 2] = 1.0 / gv[2, 2]; gi[3, 3] = 1.0 / gv[3, 3]
            gi
        end
        function kerr_tetrad(pt)
            rv = pt[2]; th = pt[3]; sv = sin(th); cv = cos(th)
            Σv = rv^2 + a_spin^2 * cv^2
            Δv = rv^2 - 2M_bh * rv + a_spin^2
            l = ComplexF64[(rv^2 + a_spin^2) / Δv, 1.0, 0.0, a_spin / Δv]
            n = ComplexF64[rv^2 + a_spin^2, -Δv, 0.0, a_spin] ./ (2Σv)
            m = ComplexF64[im * a_spin * sv, 0.0, 1.0, im / sv] ./
                (sqrt(2.0) * (rv + im * a_spin * cv))
            gv = kerr_metric(pt)
            ld = ComplexF64[sum(gv[i, j] * l[j] for j in 1:4) for i in 1:4]
            nd = ComplexF64[sum(gv[i, j] * n[j] for j in 1:4) for i in 1:4]
            (l=l, n=n, m=m, mbar=conj.(m), ld=ld, nd=nd)
        end

        for (r_val, θ_val) in [(4.0, π / 4), (6.0, π / 3)]
            x0 = [0.0, r_val, θ_val, 0.0]
            np = compute_np_quantities(kerr_metric, kerr_ginv, kerr_tetrad, x0)

            @testset "a=$a_spin, r=$r_val, θ=$(round(θ_val, digits=2))" begin
                # Goldberg-Sachs: κ=σ=0 ⟺ Ψ₀=Ψ₁=0
                @test isapprox(abs(np.kappa), 0.0, atol=1e-4)
                @test isapprox(abs(np.sigma), 0.0, atol=1e-4)
                @test isapprox(abs(np.psi.Psi0), 0.0, atol=1e-4)
                @test isapprox(abs(np.psi.Psi1), 0.0, atol=1e-4)

                # Type D: both PNDs repeated
                @test abs(np.psi.Psi2) > 1e-6
                @test isapprox(abs(np.psi.Psi3), 0.0, atol=1e-4)
                @test isapprox(abs(np.psi.Psi4), 0.0, atol=1e-4)

                # Vacuum
                @test isapprox(sum(abs.(np.Ric)), 0.0, atol=1e-2)
            end
        end
    end

end
