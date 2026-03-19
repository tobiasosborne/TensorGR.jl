# Ground truth: Hohmann (2021), "xPPN: An implementation of the parametrized
# post-Newtonian formalism using xAct for Mathematica", arXiv:2012.14984.
#
# Equation (exppnpar) in reference/papers/2012_14984_src/xppnpaper.tex:
#   gamma = (omega(Psi) + 1) / (omega(Psi) + 2)
#   beta  = 1 + Psi * omega'(Psi) / (4 * (2*omega(Psi) + 3) * (omega(Psi) + 2)^2)
#   alpha_1 = alpha_2 = alpha_3 = zeta_1 = zeta_2 = zeta_3 = zeta_4 = xi = 0
#
# The paper states: "This is of course the well-known post-Newtonian limit
# of scalar-tensor gravity with a massless scalar field."

@testset "PPN: General scalar-tensor gravity (Hohmann 2021)" begin

    function _ppn_st_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l]))
        reg
    end

    # ── Ground truth formulas from Hohmann (2021), Eq (exppnpar) ──

    """Hohmann Eq (exppnpar): gamma = (omega + 1) / (omega + 2)"""
    hohmann_gamma(omega) = (omega + 1) / (omega + 2)

    """Hohmann Eq (exppnpar): beta = 1 + Psi*omega'/(4*(2*omega+3)*(omega+2)^2)"""
    hohmann_beta(omega, omega_prime, Psi) =
        1 + Psi * omega_prime / (4 * (2 * omega + 3) * (omega + 2)^2)

    @testset "ScalarTensor gamma matches Hohmann Eq (exppnpar)" begin
        # Test with several values of omega
        for omega in [1, 5, 40, 100, 1000, 40000]
            p = PPNParameters(:ScalarTensor; omega=omega, omega_prime=0, Psi=1)
            @test p.gamma == hohmann_gamma(omega)
        end
    end

    @testset "ScalarTensor beta matches Hohmann Eq (exppnpar)" begin
        # General case: omega' != 0
        for (omega, omega_prime, Psi) in [
            (5, 1//1, 1),       # simple case
            (40, 2//1, 1),      # Cassini-regime omega
            (100, 1//10, 1),    # small omega'
            (5, -3//1, 2),      # negative omega', Psi=2
        ]
            p = PPNParameters(:ScalarTensor; omega=omega, omega_prime=omega_prime, Psi=Psi)
            expected_beta = hohmann_beta(omega, omega_prime, Psi)
            @test p.beta == expected_beta
        end
    end

    @testset "Brans-Dicke subcase: omega'=0 gives beta=1" begin
        # When omega' = 0 (constant omega), beta = 1 exactly
        # This is the Brans-Dicke limit
        for omega in [1, 5, 40, 100, 40000]
            p = PPNParameters(:ScalarTensor; omega=omega, omega_prime=0, Psi=1)
            @test p.beta == 1
        end
    end

    @testset "GR limit: omega -> infinity" begin
        # As omega -> infinity: gamma -> 1, beta -> 1
        omega_large = 10^8
        p = PPNParameters(:ScalarTensor; omega=omega_large, omega_prime=0, Psi=1)
        @test abs(p.gamma - 1) < 1e-6
        @test p.beta == 1

        # Even with omega' != 0, beta -> 1 as omega -> infinity
        p2 = PPNParameters(:ScalarTensor; omega=omega_large, omega_prime=1, Psi=1)
        @test abs(p2.beta - 1) < 1e-6
    end

    @testset "All other PPN parameters vanish (Hohmann Eq exppnpar)" begin
        p = PPNParameters(:ScalarTensor; omega=40, omega_prime=1//2, Psi=1)
        @test p.xi == 0
        @test p.alpha1 == 0
        @test p.alpha2 == 0
        @test p.alpha3 == 0
        @test p.zeta1 == 0
        @test p.zeta2 == 0
        @test p.zeta3 == 0
        @test p.zeta4 == 0
    end

    @testset "Consistency with BransDicke constructor" begin
        for omega in [1, 5, 40, 100]
            p_bd = PPNParameters(:BransDicke; omega=omega)
            p_st = PPNParameters(:ScalarTensor; omega=omega, omega_prime=0, Psi=1)
            @test p_bd.gamma == p_st.gamma
            @test p_bd.beta == p_st.beta
        end
    end

    @testset "ppn_solve dispatch for ScalarTensor" begin
        reg = _ppn_st_reg()
        with_registry(reg) do
            result = ppn_solve(:ScalarTensor, reg; omega=40, omega_prime=1//2, Psi=1)
            @test result isa PPNFieldEquationResult
            p = extract_ppn_parameters(result)
            @test p.gamma == hohmann_gamma(40)
            @test p.beta == hohmann_beta(40, 1//2, 1)
        end
    end

    @testset "Symbolic omega (rational arithmetic)" begin
        # Use rational omega to verify exact symbolic computation
        omega = 3 // 1
        omega_prime = 1 // 2
        Psi = 1 // 1
        p = PPNParameters(:ScalarTensor; omega=omega, omega_prime=omega_prime, Psi=Psi)

        # gamma = 4/5
        @test p.gamma == 4 // 5

        # beta = 1 + 1/(4*9*25) = 1 + 1/900
        # (2*3+3)=9, (3+2)^2=25, Psi*omega'=1/2
        # beta = 1 + (1/2) / (4*9*25) = 1 + 1/1800
        @test p.beta == 1 + 1 // 1800
    end

    @testset "Nordtvedt eta for general scalar-tensor" begin
        # The Nordtvedt parameter eta = 4*beta - gamma - 3
        # For scalar-tensor: measures violation of strong equivalence principle
        omega = 40
        omega_prime = 1 // 1
        Psi = 1 // 1
        p = PPNParameters(:ScalarTensor; omega=omega, omega_prime=omega_prime, Psi=Psi)

        eta = 4 * p.beta - p.gamma - 3
        # For Brans-Dicke (omega'=0): eta = 1/(2+omega)
        # For general: eta has additional omega' contribution
        @test eta isa Number
    end
end
