@testset "Bellini-Sawicki alpha parameters" begin

    # Helper to set up a standard Horndeski registry
    function _alpha_registry()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d,
            [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q,:r,:s,:t,:u,:v,:w]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true,
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1),
            symmetries=SymmetrySpec[],
            is_delta=true,
            options=Dict{Symbol,Any}(:is_delta => true)))
        reg
    end

    # ── FRWBackground construction ────────────────────────────────

    @testset "FRWBackground construction" begin
        bg = define_frw_background()
        @test bg.H == :H
        @test bg.phi_dot == :phi_dot
        @test bg.phi_ddot == :phi_ddot
        @test bg.scale_factor == :a
        # X_bg = (1/2) phi_dot^2
        @test bg.X_bg !== nothing
    end

    @testset "Custom FRWBackground" begin
        bg = define_frw_background(H=:H0, phi_dot=:v, phi_ddot=:a_phi)
        @test bg.H == :H0
        @test bg.phi_dot == :v
        @test bg.phi_ddot == :a_phi
    end

    # ── BelliniSawickiAlphas struct ───────────────────────────────

    @testset "BelliniSawickiAlphas struct fields" begin
        bg = define_frw_background()
        alphas = BelliniSawickiAlphas(:aM, :aK, :aB, :aT, 0, :Mstar2, bg)
        @test alphas.alpha_M == :aM
        @test alphas.alpha_K == :aK
        @test alphas.alpha_B == :aB
        @test alphas.alpha_T == :aT
        @test alphas.alpha_H == 0
        @test alphas.M_star_sq == :Mstar2
        @test alphas.background === bg
    end

    # ── compute_alphas basic ──────────────────────────────────────

    @testset "compute_alphas returns BelliniSawickiAlphas" begin
        reg = _alpha_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()

            alphas = compute_alphas(ht, bg; registry=reg)
            @test alphas isa BelliniSawickiAlphas
            @test alphas.background === bg
            @test alphas.alpha_H == 0  # Horndeski => no beyond-Horndeski
        end
    end

    @testset "compute_alphas registers additional G-functions" begin
        reg = _alpha_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()

            # Before compute_alphas, extra derivatives should not exist
            @test !has_tensor(reg, :G2_XX)
            @test !has_tensor(reg, :G3_XX)
            @test !has_tensor(reg, :G4_XXX)
            @test !has_tensor(reg, :G5_XXX)

            compute_alphas(ht, bg; registry=reg)

            # After compute_alphas, they should be registered
            @test has_tensor(reg, :G2_XX)
            @test has_tensor(reg, :G3_XX)
            @test has_tensor(reg, :G3_phiX)
            @test has_tensor(reg, :G4_XXX)
            @test has_tensor(reg, :G5_XXX)
        end
    end

    # ── GR limit: all alphas = 0 ──────────────────────────────────

    @testset "GR limit: all alphas vanish" begin
        # GR: G2=0, G3=0, G5=0, G4 = M_Pl^2/2 (constant)
        # => all G-function derivatives = 0 except G4 itself
        reg = _alpha_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            # Substitute GR values: G4 = 1/2 (M_Pl = 1), all else = 0
            gvals = Dict{Symbol,Float64}(
                :G4 => 0.5,
                :G4_X => 0.0, :G4_XX => 0.0, :G4_XXX => 0.0,
                :G4_phi => 0.0, :G4_phiX => 0.0,
                :G2_X => 0.0, :G2_XX => 0.0,
                :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                :G5_X => 0.0, :G5_XX => 0.0, :G5_XXX => 0.0,
                :G5_phi => 0.0, :G5_phiX => 0.0,
                :H => 1.0, :phi_dot => 1.0, :phi_ddot => 0.0,
            )

            result = compute_alphas_numerical(ht, bg, gvals; registry=reg)

            # M_*^2 = 2 * G4 = 1.0 (Planck mass)
            @test result[:M_star_sq] ≈ 1.0
            # All alphas should vanish
            @test result[:alpha_K] ≈ 0.0 atol=1e-14
            @test result[:alpha_B] ≈ 0.0 atol=1e-14
            @test result[:alpha_T] ≈ 0.0 atol=1e-14
            @test result[:alpha_H] ≈ 0.0
        end
    end

    # ── Brans-Dicke: alpha_T = 0 ─────────────────────────────────

    @testset "Brans-Dicke: alpha_T = 0, alpha_B nonzero" begin
        # Brans-Dicke: G2 = omega_BD X / phi, G4 = phi/2
        # => G4_X = 0, G5 = 0 => alpha_T = 0
        # => G4_phi = 1/2, nonzero alpha_B
        reg = _alpha_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            phi0 = 2.0  # background phi value
            pd = 0.5    # phi_dot
            H = 1.0
            X = 0.5 * pd^2

            gvals = Dict{Symbol,Float64}(
                :G4 => phi0 / 2,
                :G4_X => 0.0, :G4_XX => 0.0, :G4_XXX => 0.0,
                :G4_phi => 0.5, :G4_phiX => 0.0,
                :G2_X => 5.0,  # omega_BD / phi (omega_BD = 10, phi=2)
                :G2_XX => 0.0,
                :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                :G5_X => 0.0, :G5_XX => 0.0, :G5_XXX => 0.0,
                :G5_phi => 0.0, :G5_phiX => 0.0,
                :H => H, :phi_dot => pd, :phi_ddot => 0.0,
            )

            result = compute_alphas_numerical(ht, bg, gvals; registry=reg)

            # alpha_T = 0 (no tensor speed excess in BD theory)
            @test result[:alpha_T] ≈ 0.0 atol=1e-14

            # M_*^2 = 2 * G4 = phi0 = 2.0
            @test result[:M_star_sq] ≈ phi0

            # alpha_K should be nonzero (kinetic term from G2_X)
            @test abs(result[:alpha_K]) > 0.0
        end
    end

    # ── f(R) gravity: alpha_B + alpha_M/2 = 0 ────────────────────

    @testset "f(R) gravity: alpha_T = 0" begin
        # f(R) as Horndeski: G2 = f - R f_R, G4 = f_R/2, G3 = G5 = 0
        # Key property: alpha_T = 0 (GW speed = light)
        # Also: in f(R) the scalar field phi = f_R, so G4_X = 0
        reg = _alpha_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            # f(R) with f_R = 1.5 (background value)
            gvals = Dict{Symbol,Float64}(
                :G4 => 0.75,  # f_R / 2
                :G4_X => 0.0, :G4_XX => 0.0, :G4_XXX => 0.0,
                :G4_phi => 0.0, :G4_phiX => 0.0,
                :G2_X => 0.0, :G2_XX => 0.0,
                :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                :G5_X => 0.0, :G5_XX => 0.0, :G5_XXX => 0.0,
                :G5_phi => 0.0, :G5_phiX => 0.0,
                :H => 1.0, :phi_dot => 0.3, :phi_ddot => -0.1,
            )

            result = compute_alphas_numerical(ht, bg, gvals; registry=reg)

            # alpha_T = 0 for f(R)
            @test result[:alpha_T] ≈ 0.0 atol=1e-14
        end
    end

    # ── Quintic galileon: nonzero alpha_T ─────────────────────────

    @testset "Quintic galileon: nonzero alpha_T" begin
        # G5 = c5 X^2 / M^9 => G5_X = 2 c5 X / M^9
        # => alpha_T nonzero (constrained by GW170817)
        reg = _alpha_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            c5 = 0.1
            M9 = 1.0
            pd = 0.5
            H = 1.0
            pdd = -0.1
            X = 0.5 * pd^2

            gvals = Dict{Symbol,Float64}(
                :G4 => 0.5,  # M_Pl^2/2
                :G4_X => 0.0, :G4_XX => 0.0, :G4_XXX => 0.0,
                :G4_phi => 0.0, :G4_phiX => 0.0,
                :G2_X => 0.0, :G2_XX => 0.0,
                :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                :G5_X => 2 * c5 * X / M9,
                :G5_XX => 2 * c5 / M9,
                :G5_XXX => 0.0,
                :G5_phi => 0.0, :G5_phiX => 0.0,
                :H => H, :phi_dot => pd, :phi_ddot => pdd,
            )

            result = compute_alphas_numerical(ht, bg, gvals; registry=reg)

            # alpha_T should be nonzero due to G5_X
            @test abs(result[:alpha_T]) > 0.0

            # Check explicit formula:
            # alpha_T = (2X / M_*^2) [2*0 - 2*0 - (pdd - H*pd) G5X]
            #         = (2X / M_*^2) [-(pdd - H*pd) G5X]
            M_star_sq = 2 * (0.5 - H * pd * X * gvals[:G5_X])
            expected_T = (2 * X / M_star_sq) * (-(pdd - H * pd) * gvals[:G5_X])
            @test result[:alpha_T] ≈ expected_T atol=1e-12
        end
    end

    # ── alpha_H = 0 for Horndeski ────────────────────────────────

    @testset "alpha_H is zero for Horndeski" begin
        reg = _alpha_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)
            @test alphas.alpha_H == 0
        end
    end

    # ── M_*^2 expression correctness ─────────────────────────────

    @testset "M_star_sq reduces to 2*G4 in GR" begin
        reg = _alpha_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            # Evaluate M_*^2 with G4=1, all else zero
            gvals = Dict{Symbol,Float64}(
                :G4 => 1.0,
                :G4_X => 0.0,
                :G5_phi => 0.0, :G5_X => 0.0,
                :H => 1.0, :phi_dot => 0.5,
            )
            Mstar2 = sym_eval(alphas.M_star_sq, gvals)
            @test Mstar2 ≈ 2.0
        end
    end

    @testset "M_star_sq with nonzero G4_X" begin
        reg = _alpha_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            # M_*^2 = 2(G4 - 2X G4X + X G5phi - H pd X G5X)
            pd = 0.6
            H = 0.7
            X = 0.5 * pd^2
            G4 = 1.0
            G4X = 0.3
            G5phi = 0.1
            G5X = 0.05

            gvals = Dict{Symbol,Float64}(
                :G4 => G4, :G4_X => G4X,
                :G5_phi => G5phi, :G5_X => G5X,
                :H => H, :phi_dot => pd,
            )
            Mstar2 = sym_eval(alphas.M_star_sq, gvals)
            expected = 2 * (G4 - 2*X*G4X + X*G5phi - H*pd*X*G5X)
            @test Mstar2 ≈ expected atol=1e-12
        end
    end

end
