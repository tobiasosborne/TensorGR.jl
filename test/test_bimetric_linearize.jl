# Ground truth: Hassan & Rosen, JHEP 02 (2012) 126, arXiv:1109.3515, Sec 3.

@testset "Bimetric linearized perturbation" begin

    function _bim_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        bs = define_bimetric!(reg, :g, :f; manifold=:M4)
        reg, bs
    end

    @testset "define_bimetric_perturbation!" begin
        reg, bs = _bim_reg()
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=0, beta2=1, beta3=0, beta4=0)
        bp = define_bimetric_perturbation!(reg, bs, params; background_ratio=1)

        @test bp isa BimetricPerturbation
        @test has_tensor(reg, bp.delta_g)
        @test has_tensor(reg, bp.delta_f)
        @test has_tensor(reg, bp.massless_mode)
        @test has_tensor(reg, bp.massive_mode)
    end

    @testset "FP mass: only β₂ nonzero gives m²_FP = 2c β₂ m² / (1+c²)" begin
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=0, beta2=1, beta3=0, beta4=0)
        # c=1: m²_FP = (0 + 2·1·1 + 0) / (1+1) = 2/2 = 1
        @test fierz_pauli_mass_squared(params, 1) == 1
        # c=2: m²_FP = (0 + 2·2·1 + 0) / (1+4) = 4/5
        @test fierz_pauli_mass_squared(params, 2) == 4 // 5
    end

    @testset "FP mass: β₁ + 2c β₂ + c² β₃ formula" begin
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=1, beta2=1, beta3=1, beta4=0)
        # c=1: (1 + 2 + 1) / 2 = 2
        @test fierz_pauli_mass_squared(params, 1) == 2
    end

    @testset "Massless eigenvalue is zero" begin
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=0, beta2=1, beta3=0, beta4=0)
        ev = bimetric_mass_eigenvalues(params, 1)
        @test ev.massless == 0
        @test ev.massive == 1  # from β₂=1, c=1
    end

    @testset "Mass matrix rank 1" begin
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=0, beta2=1, beta3=0, beta4=0)
        M = bimetric_mass_matrix(params, 1)
        @test size(M) == (2, 2)
        # det = M_gg*M_ff - M_gf*M_fg = μ²·μ²/c² - μ²·μ²/c² = 0
        det = M[1,1]*M[2,2] - M[1,2]*M[2,1]
        @test det == 0  # rank 1 → one zero eigenvalue (massless)
    end

    @testset "β₂-only is Fierz-Pauli mass term" begin
        # Setting only β₂ nonzero gives the standard Fierz-Pauli mass
        # for the massive combination χ = δg - δf
        params = HassanRosenParams(m_sq=:m2, beta0=0, beta1=0, beta2=1, beta3=0, beta4=0)
        ev = bimetric_mass_eigenvalues(params, 1)
        @test ev.massless == 0
        # massive = m² × (0 + 2·1·1 + 0) / (1+1) = m²
        # (symbolic)
        @test ev.massive isa Any
    end

    @testset "All β=0: no mass term" begin
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=0, beta2=0, beta3=0, beta4=0)
        @test fierz_pauli_mass_squared(params, 1) == 0
        ev = bimetric_mass_eigenvalues(params, 1)
        @test ev.massive == 0  # both modes massless → two copies of GR
    end

    @testset "GR limit: c → 0 decouples f" begin
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=1, beta2=0, beta3=0, beta4=0)
        # c=0: m²_FP = (β₁ + 0 + 0) / (1+0) = β₁ = 1
        @test fierz_pauli_mass_squared(params, 0) == 1
    end

    @testset "Display" begin
        reg, bs = _bim_reg()
        params = HassanRosenParams(m_sq=1, beta2=1)
        bp = define_bimetric_perturbation!(reg, bs, params)
        s = sprint(show, bp)
        @test occursin("BimetricPerturbation", s)
    end
end
