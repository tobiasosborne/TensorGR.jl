# Ground truth: GHP field equations derived from NP (1962) Eqs 4.2a-4.2r
# using GHP covariant derivatives (GHP 1973).
#
# The 12 GHP field equations are the NP Ricci identities where both LHS
# spin coefficients are proper GHP quantities, rewritten with thorn, thorn',
# edth, edth' absorbing the improper spin coefficients epsilon, gamma, alpha, beta.

# Import non-exported names for testing
const _ghp_field_equations = TensorGR.ghp_field_equations
const _vacuum_ghp_field_equations = TensorGR.vacuum_ghp_field_equations
const _ghp_field_equation = TensorGR.ghp_field_equation
const _GHP_IMPROPER_SYMBOLS = TensorGR.GHP_IMPROPER_SYMBOLS
const _ghp_field_equation_weight_consistent = TensorGR.ghp_field_equation_weight_consistent
const _ghp_eq_factor_weight = TensorGR._ghp_eq_factor_weight

@testset "GHP field equations" begin

    @testset "Equation count" begin
        eqs = _ghp_field_equations()
        # Exactly 12 equations (18 NP minus 6 with improper LHS coefficients)
        @test length(eqs) == 12
    end

    @testset "Valid GHP derivative operators" begin
        valid_ops = Set([:thorn, :thorn_prime, :edth, :edth_prime])
        for eq in _ghp_field_equations()
            @test eq.deriv1 in valid_ops
            @test eq.deriv2 in valid_ops
        end
    end

    @testset "Proper spin coefficients on LHS" begin
        proper_sc = Set([:kappa, :sigma, :rho, :tau, :nu, :lambda, :mu, :pi])
        for eq in _ghp_field_equations()
            @test eq.sc1 in proper_sc
            @test eq.sc2 in proper_sc
        end
    end

    @testset "No improper spin coefficients in any RHS term" begin
        improper = _GHP_IMPROPER_SYMBOLS
        for eq in _ghp_field_equations()
            for (coeff, factors) in eq.rhs
                for f in factors
                    @test !(f in improper)
                end
            end
        end
    end

    @testset "GHP weight consistency: all terms have same weight" begin
        for eq in _ghp_field_equations()
            @test _ghp_field_equation_weight_consistent(eq)
        end
    end

    @testset "Expected equation weights" begin
        eqs = _ghp_field_equations()
        eq_weights = Dict(eq.label => eq.weight for eq in eqs)

        @test eq_weights["GHP.1"]  == (2, 2)    # thorn(rho) - edth'(kappa)
        @test eq_weights["GHP.2"]  == (4, 0)    # thorn(sigma) - edth(kappa)
        @test eq_weights["GHP.3"]  == (2, 0)    # thorn(tau) - thorn'(kappa)
        @test eq_weights["GHP.4"]  == (-2, 2)   # thorn(lambda) - edth'(pi)
        @test eq_weights["GHP.5"]  == (0, 0)    # thorn(mu) - edth(pi)
        @test eq_weights["GHP.6"]  == (-2, 0)   # thorn(nu) - thorn'(pi)
        @test eq_weights["GHP.7"]  == (-4, 0)   # thorn'(lambda) - edth'(nu)
        @test eq_weights["GHP.8"]  == (2, 0)    # edth(rho) - edth'(sigma)
        @test eq_weights["GHP.9"]  == (-2, 0)   # edth(lambda) - edth'(mu)
        @test eq_weights["GHP.10"] == (-2, -2)  # edth(nu) - thorn'(mu)
        @test eq_weights["GHP.11"] == (2, -2)   # edth(tau) - thorn'(sigma)
        @test eq_weights["GHP.12"] == (0, 0)    # thorn'(rho) - edth'(tau)
    end

    @testset "Weight equals sc1_weight + deriv1_shift" begin
        for eq in _ghp_field_equations()
            sc1_w = _ghp_eq_factor_weight(eq.sc1)
            shift = ghp_weight_shift(eq.deriv1)
            expected = sc1_w + shift
            @test (expected.p, expected.q) == eq.weight
        end
    end

    @testset "NP origin correspondence" begin
        eqs = _ghp_field_equations()
        origins = Set(eq.np_origin for eq in eqs)

        # The 12 equations with proper LHS coefficients
        expected_origins = Set([
            "4.2a", "4.2b", "4.2c",           # D-equations with rho,sigma,tau,kappa
            "4.2g", "4.2h", "4.2i",           # D-equations with lambda,mu,nu,pi
            "4.2j",                             # Delta-equation with lambda,nu
            "4.2k", "4.2m",                    # delta-equations with rho,sigma,lambda,mu
            "4.2n", "4.2p", "4.2q",           # delta/Delta-equations with nu,mu,tau,sigma,rho
        ])
        @test origins == expected_origins

        # The 6 excluded equations (improper LHS)
        excluded = Set(["4.2d", "4.2e", "4.2f", "4.2l", "4.2o", "4.2r"])
        @test isempty(intersect(origins, excluded))
    end

    @testset "Unique labels" begin
        eqs = _ghp_field_equations()
        labels = [eq.label for eq in eqs]
        @test length(labels) == length(unique(labels))

        np_origins = [eq.np_origin for eq in eqs]
        @test length(np_origins) == length(unique(np_origins))
    end

    @testset "Vacuum reduction removes Ricci/Lambda terms" begin
        full_eqs = _ghp_field_equations()
        vac_eqs = _vacuum_ghp_field_equations()

        @test length(vac_eqs) == 12

        for (full, vac) in zip(full_eqs, vac_eqs)
            @test vac.label == full.label
            @test vac.np_origin == full.np_origin
            @test vac.weight == full.weight

            # No Ricci or Lambda terms in vacuum equations
            for (coeff, factors) in vac.rhs
                for f in factors
                    @test !(f in NP_RICCI_SYMBOLS)
                end
            end

            # Vacuum has fewer or equal terms
            @test length(vac.rhs) <= length(full.rhs)
        end
    end

    @testset "Vacuum equations have correct term counts" begin
        vac_eqs = _vacuum_ghp_field_equations()
        vac_dict = Dict(eq.label => eq for eq in vac_eqs)

        # GHP.1: thorn(rho) - edth'(kappa) = rho^2 + sigma*sigma_bar - kappa_bar*tau + kappa*pi (no Phi00)
        @test length(vac_dict["GHP.1"].rhs) == 4

        # GHP.2: thorn(sigma) - edth(kappa) = sigma*rho + sigma*rho_bar - kappa*tau + kappa*pi_bar + Psi0
        @test length(vac_dict["GHP.2"].rhs) == 5

        # GHP.5: thorn(mu) - edth(pi) = rho_bar*mu + sigma*lambda + pi*pi_bar - nu*kappa + Psi2 (no 2*Lambda)
        @test length(vac_dict["GHP.5"].rhs) == 5

        # GHP.12: thorn'(rho) - edth'(tau) = -rho*mu_bar - sigma*lambda - tau_bar*tau + nu*kappa - Psi2 (no -2*Lambda)
        @test length(vac_dict["GHP.12"].rhs) == 5
    end

    @testset "Lookup by label" begin
        eq = _ghp_field_equation("GHP.1")
        @test eq.np_origin == "4.2a"
        @test eq.sc1 == :rho

        # Also lookup by NP origin
        eq2 = _ghp_field_equation("4.2a")
        @test eq2.label == "GHP.1"

        @test_throws ErrorException _ghp_field_equation("GHP.99")
    end

    @testset "Display" begin
        eq = _ghp_field_equations()[1]
        s = sprint(show, eq)
        @test occursin("GHP", s)
        @test occursin("rho", s) || occursin("kappa", s)
    end

    @testset "Prime symmetry (l <-> n exchange)" begin
        # The GHP equations come in prime-related pairs under l <-> n:
        #   thorn <-> thorn', edth <-> edth', {p,q} <-> {-p,-q}
        #   kappa <-> -nu, sigma <-> -lambda, rho <-> -mu, tau <-> -pi
        # Check that GHP.1 (thorn rho - edth' kappa) has a prime partner GHP.10
        # (edth nu - thorn' mu) with negated weight
        eq1 = _ghp_field_equation("GHP.1")   # weight {2,2}
        eq10 = _ghp_field_equation("GHP.10") # weight {-2,-2}
        @test eq1.weight[1] == -eq10.weight[1]
        @test eq1.weight[2] == -eq10.weight[2]

        # GHP.2 (weight {4,0}) and GHP.7 (weight {-4,0})
        eq2 = _ghp_field_equation("GHP.2")
        eq7 = _ghp_field_equation("GHP.7")
        @test eq2.weight[1] == -eq7.weight[1]
        @test eq2.weight[2] == -eq7.weight[2]

        # GHP.3 (weight {2,0}) and GHP.6 (weight {-2,0})
        eq3 = _ghp_field_equation("GHP.3")
        eq6 = _ghp_field_equation("GHP.6")
        @test eq3.weight[1] == -eq6.weight[1]
        @test eq3.weight[2] == -eq6.weight[2]
    end

    @testset "Conjugation symmetry" begin
        # Complex conjugation swaps (p,q):
        # GHP.4 weight {-2,2} is conjugate of GHP.11 weight {2,-2}
        eq4 = _ghp_field_equation("GHP.4")
        eq11 = _ghp_field_equation("GHP.11")
        @test eq4.weight[1] == eq11.weight[2]
        @test eq4.weight[2] == eq11.weight[1]

        # GHP.5 weight {0,0} and GHP.12 weight {0,0} -- both self-conjugate weight
        eq5 = _ghp_field_equation("GHP.5")
        eq12 = _ghp_field_equation("GHP.12")
        @test eq5.weight == (0, 0)
        @test eq12.weight == (0, 0)
    end

    @testset "Specific equation content: GHP.1 (thorn rho - edth' kappa)" begin
        eq = _ghp_field_equation("GHP.1")
        @test eq.deriv1 == :thorn
        @test eq.sc1 == :rho
        @test eq.deriv2 == :edth_prime
        @test eq.sc2 == :kappa
        @test length(eq.rhs) == 5  # rho^2, sigma*sigma_bar, -kappa_bar*tau, kappa*pi, Phi00

        # Check each term exists
        coeffs_and_factors = [(c, Set(f)) for (c, f) in eq.rhs]
        @test (1, Set([:rho, :rho])) in coeffs_and_factors
        @test (1, Set([:sigma, :sigma_bar])) in coeffs_and_factors
        @test (-1, Set([:kappa_bar, :tau])) in coeffs_and_factors
        @test (1, Set([:kappa, :pi])) in coeffs_and_factors
        @test (1, Set([:Phi00])) in coeffs_and_factors
    end

    @testset "Specific equation content: GHP.5 (thorn mu - edth pi)" begin
        eq = _ghp_field_equation("GHP.5")
        @test eq.deriv1 == :thorn
        @test eq.sc1 == :mu
        @test eq.deriv2 == :edth
        @test eq.sc2 == :pi
        @test length(eq.rhs) == 6  # rho_bar*mu, sigma*lambda, pi*pi_bar, -nu*kappa, Psi2, 2*Lambda

        coeffs_and_factors = [(c, Set(f)) for (c, f) in eq.rhs]
        @test (1, Set([:rho_bar, :mu])) in coeffs_and_factors
        @test (1, Set([:sigma, :lambda])) in coeffs_and_factors
        @test (1, Set([:pi, :pi_bar])) in coeffs_and_factors
        @test (-1, Set([:nu, :kappa])) in coeffs_and_factors
        @test (1, Set([:Psi2])) in coeffs_and_factors
        @test (2, Set([:Lambda])) in coeffs_and_factors
    end

    @testset "Specific equation content: GHP.12 (thorn' rho - edth' tau)" begin
        eq = _ghp_field_equation("GHP.12")
        @test eq.deriv1 == :thorn_prime
        @test eq.sc1 == :rho
        @test eq.deriv2 == :edth_prime
        @test eq.sc2 == :tau
        @test length(eq.rhs) == 6  # -rho*mu_bar, -sigma*lambda, -tau_bar*tau, nu*kappa, -Psi2, -2*Lambda

        coeffs_and_factors = [(c, Set(f)) for (c, f) in eq.rhs]
        @test (-1, Set([:rho, :mu_bar])) in coeffs_and_factors
        @test (-1, Set([:sigma, :lambda])) in coeffs_and_factors
        @test (-1, Set([:tau_bar, :tau])) in coeffs_and_factors
        @test (1, Set([:nu, :kappa])) in coeffs_and_factors
        @test (-1, Set([:Psi2])) in coeffs_and_factors
        @test (-2, Set([:Lambda])) in coeffs_and_factors
    end

    @testset "Vacuum weight consistency" begin
        for eq in _vacuum_ghp_field_equations()
            @test _ghp_field_equation_weight_consistent(eq)
        end
    end

end
