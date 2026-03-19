@testset "GHP spin/boost weights" begin

    @testset "GHPWeight arithmetic" begin
        w1 = GHPWeight(3, 1)
        w2 = GHPWeight(1, -1)

        @test w1 + w2 == GHPWeight(4, 0)
        @test w1 - w2 == GHPWeight(2, 2)
        @test -w1 == GHPWeight(-3, -1)
        @test spin_weight(w1) == 1
        @test boost_weight(w1) == 2
    end

    @testset "Weyl scalar weights (GHP 1973, Table 1)" begin
        @test ghp_weight(:Psi_0) == GHPWeight(4, 0)
        @test ghp_weight(:Psi_1) == GHPWeight(2, 0)
        @test ghp_weight(:Psi_2) == GHPWeight(0, 0)
        @test ghp_weight(:Psi_3) == GHPWeight(-2, 0)
        @test ghp_weight(:Psi_4) == GHPWeight(-4, 0)

        # All Weyl scalars have zero boost weight (q=0)
        for n in 0:4
            w = WEYL_SCALAR_WEIGHTS[n]
            @test w.q == 0
        end

        # Spin weights: s = 2, 1, 0, -1, -2
        for (n, expected_s) in [(0, 2), (1, 1), (2, 0), (3, -1), (4, -2)]
            @test spin_weight(WEYL_SCALAR_WEIGHTS[n]) == expected_s
        end
    end

    @testset "Ricci scalar weights" begin
        # Phi_{00} has type {2, 2}
        @test ghp_weight(:Phi_00) == GHPWeight(2, 2)
        # Phi_{22} has type {-2, -2}
        @test ghp_weight(:Phi_22) == GHPWeight(-2, -2)
        # Phi_{11} has type {0, 0}
        @test ghp_weight(:Phi_11) == GHPWeight(0, 0)
        # Lambda has type {0, 0}
        @test ghp_weight(:Lambda) == GHPWeight(0, 0)

        # Hermiticity: weight of Phi_{ij} = conjugate of weight of Phi_{ji}
        # Conjugation: {p,q} -> {q,p}
        for i in 0:2, j in 0:2
            w_ij = RICCI_SCALAR_WEIGHTS[(i, j)]
            w_ji = RICCI_SCALAR_WEIGHTS[(j, i)]
            @test w_ij.p == w_ji.q
            @test w_ij.q == w_ji.p
        end
    end

    @testset "Spin coefficient weights (GHP 1973, Table 1)" begin
        @test ghp_weight(:kappa) == GHPWeight(3, 1)
        @test ghp_weight(:sigma_np) == GHPWeight(3, -1)
        @test ghp_weight(:rho_np) == GHPWeight(1, 1)
        @test ghp_weight(:tau_np) == GHPWeight(1, -1)
        @test ghp_weight(:nu_np) == GHPWeight(-3, -1)
        @test ghp_weight(:lambda_np) == GHPWeight(-3, 1)
        @test ghp_weight(:mu_np) == GHPWeight(-1, -1)
        @test ghp_weight(:pi_np) == GHPWeight(-1, 1)
    end

    @testset "Proper GHP quantities" begin
        # The 8 proper spin coefficients
        for name in [:kappa, :sigma_np, :rho_np, :tau_np,
                     :nu_np, :lambda_np, :mu_np, :pi_np]
            @test is_proper_ghp(name)
        end
        # epsilon, gamma, alpha, beta are NOT proper
        for name in [:epsilon_np, :gamma_np, :alpha_np, :beta_np]
            @test !is_proper_ghp(name)
        end
    end

    @testset "Product rule: weight of product = sum of weights" begin
        # Psi_0 * rho should have weight {4,0} + {1,1} = {5,1}
        w_prod = ghp_weight(:Psi_0) + ghp_weight(:rho_np)
        @test w_prod == GHPWeight(5, 1)

        # kappa * nu should have weight {3,1} + {-3,-1} = {0,0}
        w_kn = ghp_weight(:kappa) + ghp_weight(:nu_np)
        @test w_kn == GHPWeight(0, 0)
    end

    @testset "Display" begin
        w = GHPWeight(3, -1)
        s = sprint(show, w)
        @test occursin("3", s)
        @test occursin("-1", s)
    end

    @testset "Invalid name" begin
        @test_throws ErrorException ghp_weight(:nonexistent)
    end
end
