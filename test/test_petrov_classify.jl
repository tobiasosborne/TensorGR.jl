@testset "Petrov Classification" begin
    # ================================================================
    # Helper: Schwarzschild curvature (reused from test_petrov_invariants.jl)
    # ================================================================
    function schw_curvature(M, r, theta)
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
        (; g, ginv, Riem, Ric, R, Weyl)
    end

    # ================================================================
    # Test 1: Minkowski -> Type O
    # ================================================================
    @testset "Minkowski: Type O" begin
        Weyl = zeros(Float64, 4, 4, 4, 4)
        g = Float64[-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
        @test petrov_classify(Weyl, g) == TypeO
    end

    # ================================================================
    # Test 2: Schwarzschild -> Type D (tensor route)
    # ================================================================
    @testset "Schwarzschild: Type D (tensor)" begin
        curv = schw_curvature(1.0, 3.0, pi / 4)
        @test petrov_classify(curv.Weyl, curv.g; atol=1e-4) == TypeD
    end

    # ================================================================
    # Test 3: Schwarzschild -> Type D (from NP scalars)
    # ================================================================
    @testset "Schwarzschild: Type D (NP scalars)" begin
        M = 1.0; r = 3.0
        Psi2 = -M / r^3
        psi = (Psi0=0.0+0im, Psi1=0.0+0im, Psi2=Complex(Psi2),
               Psi3=0.0+0im, Psi4=0.0+0im)
        @test petrov_classify(psi) == TypeD
    end

    # ================================================================
    # Test 4: pp-wave -> Type N (from NP scalars)
    # ================================================================
    @testset "pp-wave: Type N (NP scalars)" begin
        # pp-wave: only Psi4 nonzero in Brinkmann coordinates
        psi = (Psi0=0.0+0im, Psi1=0.0+0im, Psi2=0.0+0im,
               Psi3=0.0+0im, Psi4=1.0+0im)
        @test petrov_classify(psi) == TypeN
    end

    # ================================================================
    # Test 5: Type III (from NP scalars)
    # ================================================================
    @testset "Type III (NP scalars)" begin
        # Type III in canonical frame: only Psi3 nonzero
        # But I=J=0 only if Psi2=0 and at most Psi3 nonzero.
        # Actually for I=J=0 with Psi3 nonzero: I = 3*Psi2^2 = 0, check.
        # With Psi3 nonzero and all others zero: I = 0, J = 0.
        # Q = [0 0 0; 0 0 Psi3; 0 Psi3 0] -- 2x2 minor [2,3]x[2,3] = -Psi3^2 != 0
        psi = (Psi0=0.0+0im, Psi1=0.0+0im, Psi2=0.0+0im,
               Psi3=1.0+0im, Psi4=0.0+0im)
        @test petrov_classify(psi) == TypeIII
    end

    # ================================================================
    # Test 6: Type II (from NP scalars)
    # ================================================================
    @testset "Type II (NP scalars)" begin
        # Type II: I^3 = 27 J^2 but not Type D.
        # Canonical frame: Psi0 = Psi1 = 0, Psi2 != 0, Psi3 != 0, Psi4 = anything.
        # With Psi0=Psi1=0: I = 3*Psi2^2, J = -Psi2^3.
        # I^3 = 27*Psi2^6 = 27*Psi2^6 = 27*J^2. Algebraically special.
        # Q = [0 0 Psi2; 0 Psi2 Psi3; Psi2 Psi3 Psi4].
        # rank(Q) >= 2 if Psi3 != 0 -> Type II.
        Psi2 = 1.0 + 0im
        Psi3 = 0.5 + 0im
        Psi4 = 0.3 + 0im
        psi = (Psi0=0.0+0im, Psi1=0.0+0im, Psi2=Psi2,
               Psi3=Psi3, Psi4=Psi4)
        @test petrov_classify(psi) == TypeII
    end

    # ================================================================
    # Test 7: Type I -- algebraically general
    # ================================================================
    @testset "Type I (NP scalars)" begin
        # Choose scalars where I^3 != 27 J^2 (algebraically general).
        # (2, 0, 1, 0, 3): I = 9, J = 5, I^3 = 729, 27*J^2 = 675 -> general.
        psi = (Psi0=2.0+0im, Psi1=0.0+0im, Psi2=1.0+0im,
               Psi3=0.0+0im, Psi4=3.0+0im)
        inv_g = petrov_invariants(psi)
        @test !is_algebraically_special(inv_g.I, inv_g.J)
        @test petrov_classify(psi) == TypeI
    end

    # ================================================================
    # Test 8: All zero NP scalars -> Type O
    # ================================================================
    @testset "Zero NP scalars: Type O" begin
        psi = (Psi0=0.0+0im, Psi1=0.0+0im, Psi2=0.0+0im,
               Psi3=0.0+0im, Psi4=0.0+0im)
        @test petrov_classify(psi) == TypeO
    end

    # ================================================================
    # Test 9: pp-wave variant -- Psi0 nonzero only -> Type N
    # ================================================================
    @testset "pp-wave variant: Psi0 only -> Type N" begin
        psi = (Psi0=2.5+0im, Psi1=0.0+0im, Psi2=0.0+0im,
               Psi3=0.0+0im, Psi4=0.0+0im)
        @test petrov_classify(psi) == TypeN
    end

    # ================================================================
    # Test 10: PetrovType enum values
    # ================================================================
    @testset "PetrovType enum" begin
        @test TypeI isa PetrovType
        @test TypeII isa PetrovType
        @test TypeIII isa PetrovType
        @test TypeD isa PetrovType
        @test TypeN isa PetrovType
        @test TypeO isa PetrovType
        # All six types are distinct
        types = [TypeI, TypeII, TypeIII, TypeD, TypeN, TypeO]
        @test length(unique(types)) == 6
    end

    # ================================================================
    # Test 11: Schwarzschild at different radii -> always Type D
    # ================================================================
    @testset "Schwarzschild Type D at multiple radii" begin
        for r in [2.5, 5.0, 10.0]
            curv = schw_curvature(1.0, r, pi / 3)
            @test petrov_classify(curv.Weyl, curv.g; atol=1e-4) == TypeD
        end
    end
end
