@testset "NP Bianchi identities" begin

    @testset "Equation count" begin
        eqs = np_bianchi_identities()
        @test length(eqs) == 11  # 8 Weyl-sector + 3 contracted
    end

    @testset "Equation labels 4.5a through 4.5k" begin
        eqs = np_bianchi_identities()
        labels = [eq.label for eq in eqs]
        expected = ["4.5a", "4.5b", "4.5c", "4.5d", "4.5e", "4.5f",
                    "4.5g", "4.5h", "4.5i", "4.5j", "4.5k"]
        @test labels == expected
    end

    @testset "All equations have valid directional derivative operators" begin
        valid_derivs = Set([:D, :Delta, :delta, :deltabar])
        for eq in np_bianchi_identities()
            @test eq.deriv1 in valid_derivs
            @test eq.deriv2 in valid_derivs
            for (_, d, _) in eq.ricci_derivs
                @test d in valid_derivs
            end
        end
    end

    @testset "All RHS terms have nonzero integer coefficients" begin
        for eq in np_bianchi_identities()
            for (c, fs) in eq.rhs
                @test c != 0
                @test !isempty(fs)
            end
        end
    end

    @testset "All RHS factors are recognized NP symbols" begin
        sc_names = Set([:kappa, :sigma, :rho, :tau, :nu, :lambda, :mu, :pi,
                        :epsilon, :gamma, :alpha, :beta,
                        :kappa_bar, :sigma_bar, :rho_bar, :tau_bar,
                        :nu_bar, :lambda_bar, :mu_bar, :pi_bar,
                        :epsilon_bar, :gamma_bar, :alpha_bar, :beta_bar])
        all_valid = union(sc_names, NP_CURVATURE_SYMBOLS)

        for eq in np_bianchi_identities()
            for (c, fs) in eq.rhs
                for f in fs
                    @test f in all_valid
                end
            end
        end
    end

    @testset "Bianchi RHS has spin_coeff x curvature products (4.5a-4.5h)" begin
        # Key difference from field equations: Bianchi RHS has terms like
        # (spin_coeff, Psi_n) -- products of spin coefficients WITH curvature scalars
        eqs = np_bianchi_identities()
        for eq in eqs[1:8]
            has_curv_product = any(eq.rhs) do (c, fs)
                any(f -> f in NP_CURVATURE_SYMBOLS, fs)
            end
            @test has_curv_product
        end
    end

    @testset "LHS Weyl scalar structure (4.5a-4.5h)" begin
        eqs = np_bianchi_identities()
        eq_by_label = Dict(eq.label => eq for eq in eqs)

        # Set 1: D, deltabar on Weyl scalars
        # (4.5a) DPsi1 - deltabar(Psi0)
        @test eq_by_label["4.5a"].deriv1 == :D
        @test eq_by_label["4.5a"].lhs1 == :Psi1
        @test eq_by_label["4.5a"].deriv2 == :deltabar
        @test eq_by_label["4.5a"].lhs2 == :Psi0

        # (4.5c) DPsi2 - deltabar(Psi1)
        @test eq_by_label["4.5c"].deriv1 == :D
        @test eq_by_label["4.5c"].lhs1 == :Psi2
        @test eq_by_label["4.5c"].deriv2 == :deltabar
        @test eq_by_label["4.5c"].lhs2 == :Psi1

        # (4.5e) DPsi3 - deltabar(Psi2)
        @test eq_by_label["4.5e"].deriv1 == :D
        @test eq_by_label["4.5e"].lhs1 == :Psi3
        @test eq_by_label["4.5e"].deriv2 == :deltabar
        @test eq_by_label["4.5e"].lhs2 == :Psi2

        # (4.5g) DPsi4 - deltabar(Psi3)
        @test eq_by_label["4.5g"].deriv1 == :D
        @test eq_by_label["4.5g"].lhs1 == :Psi4
        @test eq_by_label["4.5g"].deriv2 == :deltabar
        @test eq_by_label["4.5g"].lhs2 == :Psi3

        # Set 2: Delta, delta on Weyl scalars
        # (4.5b) Delta(Psi0) - delta(Psi1)
        @test eq_by_label["4.5b"].deriv1 == :Delta
        @test eq_by_label["4.5b"].lhs1 == :Psi0
        @test eq_by_label["4.5b"].deriv2 == :delta
        @test eq_by_label["4.5b"].lhs2 == :Psi1

        # (4.5d) Delta(Psi1) - delta(Psi2)
        @test eq_by_label["4.5d"].deriv1 == :Delta
        @test eq_by_label["4.5d"].lhs1 == :Psi1
        @test eq_by_label["4.5d"].deriv2 == :delta
        @test eq_by_label["4.5d"].lhs2 == :Psi2

        # (4.5f) Delta(Psi2) - delta(Psi3)
        @test eq_by_label["4.5f"].deriv1 == :Delta
        @test eq_by_label["4.5f"].lhs1 == :Psi2
        @test eq_by_label["4.5f"].deriv2 == :delta
        @test eq_by_label["4.5f"].lhs2 == :Psi3

        # (4.5h) Delta(Psi3) - delta(Psi4)
        @test eq_by_label["4.5h"].deriv1 == :Delta
        @test eq_by_label["4.5h"].lhs1 == :Psi3
        @test eq_by_label["4.5h"].deriv2 == :delta
        @test eq_by_label["4.5h"].lhs2 == :Psi4
    end

    @testset "Ladder structure: vacuum equation n involves nearest-neighbor Psi" begin
        vac = vacuum_np_bianchi_identities()

        function weyl_scalars_in(eq)
            ws = Set{Symbol}()
            for (c, fs) in eq.rhs
                for f in fs
                    f in NP_WEYL_SYMBOLS && push!(ws, f)
                end
            end
            ws
        end

        eq_by_label = Dict(eq.label => eq for eq in vac)

        # (4.5a) DPsi1 - deltabar(Psi0): RHS has Psi0, Psi1, Psi2 only
        ws_a = weyl_scalars_in(eq_by_label["4.5a"])
        @test :Psi0 in ws_a
        @test :Psi1 in ws_a
        @test :Psi2 in ws_a
        @test :Psi3 ∉ ws_a
        @test :Psi4 ∉ ws_a

        # (4.5c) DPsi2 - deltabar(Psi1): RHS has Psi0, Psi1, Psi2
        ws_c = weyl_scalars_in(eq_by_label["4.5c"])
        @test :Psi0 in ws_c
        @test :Psi1 in ws_c
        @test :Psi2 in ws_c
        @test :Psi4 ∉ ws_c

        # (4.5g) DPsi4 - deltabar(Psi3): RHS has Psi2, Psi3, Psi4 only
        ws_g = weyl_scalars_in(eq_by_label["4.5g"])
        @test :Psi0 ∉ ws_g
        @test :Psi1 ∉ ws_g
        @test :Psi2 in ws_g
        @test :Psi3 in ws_g
        @test :Psi4 in ws_g

        # (4.5h) Delta(Psi3) - delta(Psi4): RHS has Psi2, Psi3, Psi4 only
        ws_h = weyl_scalars_in(eq_by_label["4.5h"])
        @test :Psi0 ∉ ws_h
        @test :Psi1 ∉ ws_h
        @test :Psi2 in ws_h
        @test :Psi3 in ws_h
        @test :Psi4 in ws_h
    end

    @testset "Vacuum reduction gives 8 equations with no Ricci terms" begin
        vac_eqs = vacuum_np_bianchi_identities()
        @test length(vac_eqs) == 8

        for eq in vac_eqs
            # No Ricci derivative terms in vacuum
            @test isempty(eq.ricci_derivs)

            # No Ricci symbols in RHS
            for (c, fs) in eq.rhs
                for f in fs
                    @test f ∉ NP_RICCI_SYMBOLS
                end
            end
        end
    end

    @testset "l<->n symmetry: (4.5a) <-> (4.5h)" begin
        eq_a = np_bianchi_identity("4.5a")
        eq_h = np_bianchi_identity("4.5h")

        # 4.5a: D(Psi1) - deltabar(Psi0)
        # prime: Delta(Psi3) - delta(Psi4) = 4.5h
        @test eq_a.deriv1 == :D && eq_h.deriv1 == :Delta
        @test eq_a.deriv2 == :deltabar && eq_h.deriv2 == :delta
        @test eq_a.lhs1 == :Psi1 && eq_h.lhs1 == :Psi3  # Psi1 <-> Psi3
        @test eq_a.lhs2 == :Psi0 && eq_h.lhs2 == :Psi4  # Psi0 <-> Psi4

        # Both should have the same number of Ricci derivative terms
        @test length(eq_a.ricci_derivs) == length(eq_h.ricci_derivs)
    end

    @testset "l<->n symmetry: (4.5b) <-> (4.5g)" begin
        eq_b = np_bianchi_identity("4.5b")
        eq_g = np_bianchi_identity("4.5g")

        # 4.5b: Delta(Psi0) - delta(Psi1)
        # prime: D(Psi4) - deltabar(Psi3) = 4.5g
        @test eq_b.deriv1 == :Delta && eq_g.deriv1 == :D
        @test eq_b.deriv2 == :delta && eq_g.deriv2 == :deltabar
        @test eq_b.lhs1 == :Psi0 && eq_g.lhs1 == :Psi4
        @test eq_b.lhs2 == :Psi1 && eq_g.lhs2 == :Psi3
    end

    @testset "l<->n symmetry: (4.5c) <-> (4.5f)" begin
        eq_c = np_bianchi_identity("4.5c")
        eq_f = np_bianchi_identity("4.5f")

        # 4.5c: D(Psi2) - deltabar(Psi1)
        # prime: Delta(Psi2) - delta(Psi3) = 4.5f
        @test eq_c.deriv1 == :D && eq_f.deriv1 == :Delta
        @test eq_c.lhs1 == :Psi2 && eq_f.lhs1 == :Psi2
        @test eq_c.lhs2 == :Psi1 && eq_f.lhs2 == :Psi3
    end

    @testset "l<->n symmetry: (4.5d) <-> (4.5e)" begin
        eq_d = np_bianchi_identity("4.5d")
        eq_e = np_bianchi_identity("4.5e")

        # 4.5d: Delta(Psi1) - delta(Psi2)
        # prime: D(Psi3) - deltabar(Psi2) = 4.5e
        @test eq_d.deriv1 == :Delta && eq_e.deriv1 == :D
        @test eq_d.lhs1 == :Psi1 && eq_e.lhs1 == :Psi3
        @test eq_d.lhs2 == :Psi2 && eq_e.lhs2 == :Psi2
    end

    @testset "Ricci derivative terms present in full equations" begin
        eqs = np_bianchi_identities()
        eq_by_label = Dict(eq.label => eq for eq in eqs)

        # 4.5a has 2 Ricci deriv terms: -D(Phi01) + delta(Phi00)
        @test length(eq_by_label["4.5a"].ricci_derivs) == 2

        # 4.5c has 3 Ricci deriv terms: Delta(Phi00) - deltabar(Phi01) + 2D(Lambda)
        @test length(eq_by_label["4.5c"].ricci_derivs) == 3

        # 4.5g has 2 Ricci deriv terms (no Lambda)
        @test length(eq_by_label["4.5g"].ricci_derivs) == 2
        # 4.5h has 2 Ricci deriv terms (no Lambda)
        @test length(eq_by_label["4.5h"].ricci_derivs) == 2
    end

    @testset "Contracted Bianchi identities (4.5i-4.5k)" begin
        eqs = np_bianchi_identities()
        eq_by_label = Dict(eq.label => eq for eq in eqs)

        # 4.5i: D(Phi11) - delta(Phi10) + ...
        @test eq_by_label["4.5i"].lhs1 == :Phi11
        @test eq_by_label["4.5i"].lhs2 == :Phi10

        # 4.5j: D(Phi12) - delta(Phi11) + ...
        @test eq_by_label["4.5j"].lhs1 == :Phi12
        @test eq_by_label["4.5j"].lhs2 == :Phi11

        # 4.5k: D(Phi22) - delta(Phi21) + ...
        @test eq_by_label["4.5k"].lhs1 == :Phi22
        @test eq_by_label["4.5k"].lhs2 == :Phi21

        # Contracted identities should have NO Weyl scalars on RHS
        for label in ["4.5i", "4.5j", "4.5k"]
            eq = eq_by_label[label]
            for (c, fs) in eq.rhs
                for f in fs
                    @test f ∉ NP_WEYL_SYMBOLS
                end
            end
        end
    end

    @testset "np_bianchi_identity lookup" begin
        eq_a = np_bianchi_identity("4.5a")
        @test eq_a.label == "4.5a"
        @test eq_a.deriv1 == :D
        @test eq_a.lhs1 == :Psi1

        eq_k = np_bianchi_identity("4.5k")
        @test eq_k.label == "4.5k"

        @test_throws ErrorException np_bianchi_identity("4.5z")
    end

    @testset "Display method" begin
        eq = np_bianchi_identity("4.5a")
        s = sprint(show, eq)
        @test occursin("4.5a", s)
        @test occursin("D", s)
        @test occursin("Psi1", s)
        @test occursin("Bianchi", s)
    end

    @testset "NPBianchiIdentity struct type" begin
        eq = np_bianchi_identity("4.5a")
        @test eq isa NPBianchiIdentity
        @test eq.label isa String
        @test eq.deriv1 isa Symbol
        @test eq.ricci_derivs isa Vector{Tuple{Int, Symbol, Symbol}}
        @test eq.rhs isa Vector{Tuple{Int, Vector{Symbol}}}
    end

    @testset "Vacuum (4.5a): specific terms check" begin
        vac = vacuum_np_bianchi_identities()
        eq_a = first(filter(eq -> eq.label == "4.5a", vac))

        # Should have (pi - 4alpha)Psi0 terms, i.e., -pi*Psi0 + 4*alpha*Psi0
        has_pi_psi0 = any(t -> t == (-1, [:pi, :Psi0]), eq_a.rhs)
        @test has_pi_psi0

        has_4alpha_psi0 = any(t -> t == (4, [:alpha, :Psi0]), eq_a.rhs)
        @test has_4alpha_psi0

        # -2(2rho + epsilon)Psi1 = -4rho*Psi1 - 2epsilon*Psi1
        has_m4rho_psi1 = any(t -> t == (-4, [:rho, :Psi1]), eq_a.rhs)
        @test has_m4rho_psi1

        has_m2eps_psi1 = any(t -> t == (-2, [:epsilon, :Psi1]), eq_a.rhs)
        @test has_m2eps_psi1

        # 3kappa*Psi2
        has_3k_psi2 = any(t -> t == (3, [:kappa, :Psi2]), eq_a.rhs)
        @test has_3k_psi2
    end

    @testset "Vacuum (4.5b): specific terms check" begin
        vac = vacuum_np_bianchi_identities()
        eq_b = first(filter(eq -> eq.label == "4.5b", vac))

        # -(4gamma - mu)Psi0 = -4gamma*Psi0 + mu*Psi0
        has_m4g_psi0 = any(t -> t == (-4, [:gamma, :Psi0]), eq_b.rhs)
        @test has_m4g_psi0

        has_mu_psi0 = any(t -> t == (1, [:mu, :Psi0]), eq_b.rhs)
        @test has_mu_psi0

        # 2(2tau + beta)Psi1 = 4tau*Psi1 + 2beta*Psi1
        has_4tau_psi1 = any(t -> t == (4, [:tau, :Psi1]), eq_b.rhs)
        @test has_4tau_psi1

        # -3sigma*Psi2
        has_m3s_psi2 = any(t -> t == (-3, [:sigma, :Psi2]), eq_b.rhs)
        @test has_m3s_psi2
    end

    @testset "Ricci deriv terms in 4.5a" begin
        eq = np_bianchi_identity("4.5a")
        # -D(Phi01) + delta(Phi00)
        @test (-1, :D, :Phi01) in eq.ricci_derivs
        @test (1, :delta, :Phi00) in eq.ricci_derivs
    end

    @testset "Lambda derivative terms" begin
        eqs = np_bianchi_identities()
        eq_by_label = Dict(eq.label => eq for eq in eqs)

        # 4.5c has +2D(Lambda)
        lambda_derivs_c = filter(rd -> rd[3] == :Lambda, eq_by_label["4.5c"].ricci_derivs)
        @test length(lambda_derivs_c) == 1
        @test lambda_derivs_c[1] == (2, :D, :Lambda)

        # 4.5d has -2delta(Lambda)
        lambda_derivs_d = filter(rd -> rd[3] == :Lambda, eq_by_label["4.5d"].ricci_derivs)
        @test length(lambda_derivs_d) == 1
        @test lambda_derivs_d[1] == (-2, :delta, :Lambda)

        # 4.5e has -2deltabar(Lambda)
        lambda_derivs_e = filter(rd -> rd[3] == :Lambda, eq_by_label["4.5e"].ricci_derivs)
        @test length(lambda_derivs_e) == 1
        @test lambda_derivs_e[1] == (-2, :deltabar, :Lambda)

        # 4.5f has +2Delta(Lambda)
        lambda_derivs_f = filter(rd -> rd[3] == :Lambda, eq_by_label["4.5f"].ricci_derivs)
        @test length(lambda_derivs_f) == 1
        @test lambda_derivs_f[1] == (2, :Delta, :Lambda)
    end

    @testset "NP_CURVATURE_SYMBOLS constant" begin
        @test :Psi0 in NP_CURVATURE_SYMBOLS
        @test :Psi4 in NP_CURVATURE_SYMBOLS
        @test :Phi00 in NP_CURVATURE_SYMBOLS
        @test :Lambda in NP_CURVATURE_SYMBOLS
        @test :kappa ∉ NP_CURVATURE_SYMBOLS
    end
end
