@testset "NP field equations (Ricci identities)" begin

    @testset "Equation count" begin
        eqs = np_field_equations()
        @test length(eqs) == 18
    end

    @testset "Equation labels 4.2a through 4.2r" begin
        eqs = np_field_equations()
        labels = [eq.label for eq in eqs]
        expected = ["4.2a", "4.2b", "4.2c", "4.2d", "4.2e", "4.2f",
                    "4.2g", "4.2h", "4.2i", "4.2j", "4.2k", "4.2l",
                    "4.2m", "4.2n", "4.2o", "4.2p", "4.2q", "4.2r"]
        @test labels == expected
    end

    @testset "All equations have valid directional derivative operators" begin
        valid_derivs = Set([:D, :Delta, :delta, :deltabar])
        for eq in np_field_equations()
            @test eq.deriv1 in valid_derivs
            @test eq.deriv2 in valid_derivs
        end
    end

    @testset "All RHS terms have nonzero integer coefficients" begin
        for eq in np_field_equations()
            for (c, fs) in eq.rhs
                @test c != 0
                @test !isempty(fs)
            end
        end
    end

    @testset "All RHS factors are recognized NP symbols" begin
        # Valid spin coefficient names (and their conjugates)
        sc_names = Set([:kappa, :sigma, :rho, :tau, :nu, :lambda, :mu, :pi,
                        :epsilon, :gamma, :alpha, :beta,
                        :kappa_bar, :sigma_bar, :rho_bar, :tau_bar,
                        :nu_bar, :lambda_bar, :mu_bar, :pi_bar,
                        :epsilon_bar, :gamma_bar, :alpha_bar, :beta_bar])
        curv_names = union(NP_WEYL_SYMBOLS, NP_RICCI_SYMBOLS)
        all_valid = union(sc_names, curv_names)

        for eq in np_field_equations()
            for (c, fs) in eq.rhs
                for f in fs
                    @test f in all_valid
                end
            end
        end
    end

    @testset "Each equation has bilinear spin-coefficient terms" begin
        # Every equation should have at least some quadratic (2-factor) terms
        for eq in np_field_equations()
            bilinear = filter(t -> length(t[2]) == 2, eq.rhs)
            @test !isempty(bilinear)
        end
    end

    @testset "Curvature term content per equation" begin
        eqs = np_field_equations()
        eq_by_label = Dict(eq.label => eq for eq in eqs)

        # Extract curvature symbols from an equation's RHS
        function curv_terms(eq)
            curv = Symbol[]
            all_curv = union(NP_WEYL_SYMBOLS, NP_RICCI_SYMBOLS)
            for (c, fs) in eq.rhs
                for f in fs
                    f in all_curv && push!(curv, f)
                end
            end
            curv
        end

        # (4.2a): Φ₀₀ only
        @test curv_terms(eq_by_label["4.2a"]) == [:Phi00]
        # (4.2b): Ψ₀ only
        @test curv_terms(eq_by_label["4.2b"]) == [:Psi0]
        # (4.2c): Ψ₁, Φ₀₁
        ct_c = curv_terms(eq_by_label["4.2c"])
        @test :Psi1 in ct_c && :Phi01 in ct_c
        # (4.2f): Ψ₂, Φ₁₁, Λ
        ct_f = curv_terms(eq_by_label["4.2f"])
        @test :Psi2 in ct_f && :Phi11 in ct_f && :Lambda in ct_f
        # (4.2j): Ψ₄ only (with negative sign)
        @test curv_terms(eq_by_label["4.2j"]) == [:Psi4]
        # (4.2n): Φ₂₂ only
        @test curv_terms(eq_by_label["4.2n"]) == [:Phi22]
    end

    @testset "D equations: first 9 have deriv1 = :D" begin
        eqs = np_field_equations()
        for i in 1:9
            @test eqs[i].deriv1 == :D
        end
    end

    @testset "LHS structure of specific equations" begin
        eqs = np_field_equations()
        eq_by_label = Dict(eq.label => eq for eq in eqs)

        # (4.2a): Dρ − δ̄κ
        @test eq_by_label["4.2a"].deriv1 == :D
        @test eq_by_label["4.2a"].sc1 == :rho
        @test eq_by_label["4.2a"].deriv2 == :deltabar
        @test eq_by_label["4.2a"].sc2 == :kappa

        # (4.2b): Dσ − δκ
        @test eq_by_label["4.2b"].sc1 == :sigma
        @test eq_by_label["4.2b"].deriv2 == :delta

        # (4.2c): Dτ − Δκ
        @test eq_by_label["4.2c"].sc1 == :tau
        @test eq_by_label["4.2c"].deriv2 == :Delta

        # (4.2j): Δλ − δ̄ν
        @test eq_by_label["4.2j"].deriv1 == :Delta
        @test eq_by_label["4.2j"].sc1 == :lambda
        @test eq_by_label["4.2j"].sc2 == :nu

        # (4.2q): Δρ − δ̄τ
        @test eq_by_label["4.2q"].deriv1 == :Delta
        @test eq_by_label["4.2q"].sc1 == :rho
        @test eq_by_label["4.2q"].sc2 == :tau

        # (4.2r): Δα − δ̄γ
        @test eq_by_label["4.2r"].sc1 == :alpha
        @test eq_by_label["4.2r"].sc2 == :gamma
    end

    @testset "Vacuum reduction removes all Ricci terms" begin
        vac_eqs = vacuum_np_field_equations()
        @test length(vac_eqs) == 18

        for eq in vac_eqs
            for (c, fs) in eq.rhs
                for f in fs
                    @test f ∉ NP_RICCI_SYMBOLS
                end
            end
        end
    end

    @testset "Vacuum: only Weyl scalars remain as curvature" begin
        vac_eqs = vacuum_np_field_equations()
        all_curv = union(NP_WEYL_SYMBOLS, NP_RICCI_SYMBOLS)

        for eq in vac_eqs
            for (c, fs) in eq.rhs
                curv_in_term = filter(f -> f in all_curv, fs)
                # Any curvature symbol must be a Weyl scalar
                for f in curv_in_term
                    @test f in NP_WEYL_SYMBOLS
                end
            end
        end
    end

    @testset "Vacuum equations with specific Weyl scalar content" begin
        vac = vacuum_np_field_equations()
        eq_by_label = Dict(eq.label => eq for eq in vac)

        function weyl_in(eq)
            ws = Symbol[]
            for (c, fs) in eq.rhs
                for f in fs
                    f in NP_WEYL_SYMBOLS && push!(ws, f)
                end
            end
            ws
        end

        # Vacuum (4.2a): no curvature (Φ₀₀ removed)
        @test isempty(weyl_in(eq_by_label["4.2a"]))
        # Vacuum (4.2b): Ψ₀ survives
        @test weyl_in(eq_by_label["4.2b"]) == [:Psi0]
        # Vacuum (4.2c): Ψ₁ survives (Φ₀₁ removed)
        @test weyl_in(eq_by_label["4.2c"]) == [:Psi1]
        # Vacuum (4.2f): Ψ₂ survives (Φ₁₁, Λ removed)
        @test weyl_in(eq_by_label["4.2f"]) == [:Psi2]
        # Vacuum (4.2j): Ψ₄ survives
        @test weyl_in(eq_by_label["4.2j"]) == [:Psi4]
    end

    @testset "np_field_equation lookup" begin
        eq_a = np_field_equation("4.2a")
        @test eq_a.label == "4.2a"
        @test eq_a.deriv1 == :D
        @test eq_a.sc1 == :rho

        eq_r = np_field_equation("4.2r")
        @test eq_r.label == "4.2r"

        @test_throws ErrorException np_field_equation("4.2z")
    end

    @testset "Display method" begin
        eq = np_field_equation("4.2a")
        s = sprint(show, eq)
        @test occursin("4.2a", s)
        @test occursin("D", s)
        @test occursin("rho", s)
    end

    @testset "l↔n symmetry: (4.2a) ↔ (4.2n)" begin
        # Under l↔n exchange: κ↔−ν, σ↔−λ, ρ↔−μ, τ↔−π, ε↔−γ, α↔−β̄, β↔−ᾱ
        # D↔Δ, δ↔δ̄, Ψ₀↔Ψ₄, Ψ₁↔Ψ₃, Φ₀₀↔Φ₂₂
        # (4.2a) Dρ − δ̄κ should map to δν − Δμ = (4.2n)
        eq_a = np_field_equation("4.2a")
        eq_n = np_field_equation("4.2n")

        # LHS: D→Δ on ρ→−μ gives Δ(−μ); δ̄→δ on κ→−ν gives δ(−ν)
        # So −Δμ + δν = δν − Δμ which is eq 4.2n
        @test eq_n.deriv1 == :delta
        @test eq_n.sc1 == :nu
        @test eq_n.deriv2 == :Delta
        @test eq_n.sc2 == :mu

        # Both equations should have the same number of bilinear terms
        bilinear_a = length(filter(t -> length(t[2]) == 2, eq_a.rhs))
        bilinear_n = length(filter(t -> length(t[2]) == 2, eq_n.rhs))
        @test bilinear_a == bilinear_n
    end

    @testset "l↔n symmetry: (4.2b) ↔ (4.2j)" begin
        # (4.2b) Dσ − δκ maps to −Δλ + δ̄ν = −(Δλ − δ̄ν) under l↔n
        eq_b = np_field_equation("4.2b")
        eq_j = np_field_equation("4.2j")

        @test eq_j.deriv1 == :Delta
        @test eq_j.sc1 == :lambda
        @test eq_j.deriv2 == :deltabar
        @test eq_j.sc2 == :nu

        # Both have 1 Weyl scalar (Ψ₀ and Ψ₄ respectively)
        weyl_b = count(t -> any(f -> f in NP_WEYL_SYMBOLS, t[2]), eq_b.rhs)
        weyl_j = count(t -> any(f -> f in NP_WEYL_SYMBOLS, t[2]), eq_j.rhs)
        @test weyl_b == 1
        @test weyl_j == 1
    end

    @testset "Lambda appears in exactly 4 equations" begin
        eqs = np_field_equations()
        lambda_eqs = filter(eqs) do eq
            any(t -> :Lambda in t[2], eq.rhs)
        end
        @test length(lambda_eqs) == 4
        lambda_labels = Set(eq.label for eq in lambda_eqs)
        @test lambda_labels == Set(["4.2f", "4.2h", "4.2l", "4.2q"])
    end

    @testset "Lambda signs: −Λ in 4.2f, +2Λ in 4.2h, +Λ in 4.2l, −2Λ in 4.2q" begin
        eq_by_label = Dict(eq.label => eq for eq in np_field_equations())

        function lambda_coeff(eq)
            for (c, fs) in eq.rhs
                if fs == [:Lambda]
                    return c
                end
            end
            return 0
        end

        @test lambda_coeff(eq_by_label["4.2f"]) == -1
        @test lambda_coeff(eq_by_label["4.2h"]) == 2
        @test lambda_coeff(eq_by_label["4.2l"]) == 1
        @test lambda_coeff(eq_by_label["4.2q"]) == -2
    end

    @testset "Sachs optical equation structure (4.2a)" begin
        # (4.2a) Dρ − δ̄κ = ρ² + σσ̄ + (ε+ε̄)ρ − κ̄τ − κ(3α+β̄−π) + Φ₀₀
        eq = np_field_equation("4.2a")

        # Should contain ρ² term
        has_rho_sq = any(t -> t == (1, [:rho, :rho]), eq.rhs)
        @test has_rho_sq

        # Should contain σσ̄ term
        has_sigma_sigmabar = any(t -> t == (1, [:sigma, :sigma_bar]), eq.rhs)
        @test has_sigma_sigmabar

        # Should contain Φ₀₀
        has_phi00 = any(t -> t == (1, [:Phi00]), eq.rhs)
        @test has_phi00
    end

    @testset "Shear equation structure (4.2b)" begin
        # (4.2b) Dσ − δκ = ... + Ψ₀
        eq = np_field_equation("4.2b")
        has_psi0 = any(t -> t == (1, [:Psi0]), eq.rhs)
        @test has_psi0
    end

    @testset "NPFieldEquation struct type" begin
        eq = np_field_equation("4.2a")
        @test eq isa NPFieldEquation
        @test eq.label isa String
        @test eq.deriv1 isa Symbol
        @test eq.rhs isa Vector{Tuple{Int, Vector{Symbol}}}
    end
end
