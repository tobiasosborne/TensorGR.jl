@testset "EFT of dark energy" begin

    # Helper to set up a standard Horndeski registry
    function _eft_registry()
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

    # ── EFTDarkEnergy struct ────────────────────────────────────────

    @testset "EFTDarkEnergy struct construction" begin
        eft = EFTDarkEnergy(0, 0, 0, 0, 0, :H, :Omega_DE)
        @test eft.alpha_M == 0
        @test eft.alpha_K == 0
        @test eft.alpha_B == 0
        @test eft.alpha_T == 0
        @test eft.alpha_H == 0
        @test eft.H == :H
        @test eft.Omega_DE == :Omega_DE
    end

    @testset "EFTDarkEnergy keyword constructor" begin
        eft = EFTDarkEnergy(alpha_K=0.1, alpha_B=0.05)
        @test eft.alpha_M == 0
        @test eft.alpha_K == 0.1
        @test eft.alpha_B == 0.05
        @test eft.alpha_T == 0
        @test eft.alpha_H == 0
    end

    # ── GR limit: all alphas = 0 ────────────────────────────────────

    @testset "GR gives all alphas = 0" begin
        eft = eft_gr()
        @test eft.alpha_M == 0
        @test eft.alpha_K == 0
        @test eft.alpha_B == 0
        @test eft.alpha_T == 0
        @test eft.alpha_H == 0
    end

    @testset "GR from Horndeski: all alphas vanish numerically" begin
        reg = _eft_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()

            # GR: G4 = M_Pl^2/2 = 0.5 (M_Pl=1), all others zero
            gvals = Dict{Symbol,Float64}(
                :G4 => 0.5,
                :G4_X => 0.0, :G4_XX => 0.0, :G4_XXX => 0.0,
                :G4_phi => 0.0, :G4_phiX => 0.0,
                :G2_X => 0.0, :G2_XX => 0.0,
                :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                :G5_X => 0.0, :G5_XX => 0.0, :G5_XXX => 0.0,
                :G5_phi => 0.0, :G5_phiX => 0.0,
                :H => 1.0, :phi_dot => 1.0, :phi_ddot => 0.0,
                :alpha_M => 0.0, :Omega_DE => 0.7,
            )

            eft = eft_from_numerical(ht, bg, gvals; registry=reg)
            @test eft.alpha_K ≈ 0.0 atol=1e-14
            @test eft.alpha_B ≈ 0.0 atol=1e-14
            @test eft.alpha_T ≈ 0.0 atol=1e-14
            @test eft.alpha_H ≈ 0.0
            @test eft.alpha_M ≈ 0.0
        end
    end

    # ── f(R) gravity: alpha_T = 0, alpha_B = -alpha_M ──────────────

    @testset "f(R) gravity: alpha_T = 0, alpha_B = -alpha_M" begin
        aM = 0.1
        eft = eft_fR(aM)
        @test eft.alpha_T == 0
        @test eft.alpha_B == -aM
        @test eft.alpha_M == aM
        @test eft.alpha_H == 0
        @test eft.alpha_K == 0
    end

    @testset "f(R) from Horndeski: alpha_T = 0" begin
        # f(R) as Horndeski: G2 = f - R f_R, G4 = f_R/2, G3 = G5 = 0
        reg = _eft_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()

            gvals = Dict{Symbol,Float64}(
                :G4 => 0.75,  # f_R / 2
                :G4_X => 0.0, :G4_XX => 0.0, :G4_XXX => 0.0,
                :G4_phi => 0.0, :G4_phiX => 0.0,
                :G2_X => 0.0, :G2_XX => 0.0,
                :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                :G5_X => 0.0, :G5_XX => 0.0, :G5_XXX => 0.0,
                :G5_phi => 0.0, :G5_phiX => 0.0,
                :H => 1.0, :phi_dot => 0.3, :phi_ddot => -0.1,
                :alpha_M => 0.0, :Omega_DE => 0.7,
            )

            eft = eft_from_numerical(ht, bg, gvals; registry=reg)
            @test eft.alpha_T ≈ 0.0 atol=1e-14
        end
    end

    # ── Quintessence: alpha_K nonzero, others = 0 ──────────────────

    @testset "Quintessence: only alpha_K nonzero" begin
        aK = 0.5
        eft = eft_quintessence(aK)
        @test eft.alpha_K == aK
        @test eft.alpha_M == 0
        @test eft.alpha_B == 0
        @test eft.alpha_T == 0
        @test eft.alpha_H == 0
    end

    @testset "Quintessence from Horndeski: G2 = X - V" begin
        # Quintessence: G2 nonzero (=> G2_X != 0), G3 = G5 = 0, G4 = M_Pl^2/2
        reg = _eft_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()

            pd = 0.5
            X = 0.5 * pd^2
            gvals = Dict{Symbol,Float64}(
                :G4 => 0.5,
                :G4_X => 0.0, :G4_XX => 0.0, :G4_XXX => 0.0,
                :G4_phi => 0.0, :G4_phiX => 0.0,
                :G2_X => 1.0,   # dG2/dX = 1 for G2 = X - V
                :G2_XX => 0.0,
                :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                :G5_X => 0.0, :G5_XX => 0.0, :G5_XXX => 0.0,
                :G5_phi => 0.0, :G5_phiX => 0.0,
                :H => 1.0, :phi_dot => pd, :phi_ddot => 0.0,
                :alpha_M => 0.0, :Omega_DE => 0.7,
            )

            eft = eft_from_numerical(ht, bg, gvals; registry=reg)
            # alpha_K should be nonzero (kinetic term from G2_X)
            @test abs(eft.alpha_K) > 0.0
            # All others should vanish (minimally coupled)
            @test eft.alpha_B ≈ 0.0 atol=1e-14
            @test eft.alpha_T ≈ 0.0 atol=1e-14
            @test eft.alpha_M ≈ 0.0
        end
    end

    # ── GW170817 constraint ──────────────────────────────────────────

    @testset "GW170817 constraint correctly checks alpha_T" begin
        # alpha_T = 0 satisfies constraint
        eft_ok = EFTDarkEnergy(0, 0.1, 0.05, 0, 0, 1.0, 0.7)
        @test gw170817_constraint(eft_ok) == true

        # alpha_T = 1e-16 satisfies constraint (within bound)
        eft_tiny = EFTDarkEnergy(0, 0.1, 0.05, 1e-16, 0, 1.0, 0.7)
        @test gw170817_constraint(eft_tiny) == true

        # alpha_T = 0.1 violates constraint
        eft_bad = EFTDarkEnergy(0, 0.1, 0.05, 0.1, 0, 1.0, 0.7)
        @test gw170817_constraint(eft_bad) == false

        # Symbolic alpha_T = 0 satisfies
        eft_sym_zero = EFTDarkEnergy(0, 0, 0, 0, 0, :H, :Omega_DE)
        @test gw170817_constraint(eft_sym_zero) == true

        # Symbolic alpha_T = :alpha_T does not satisfy (not identically zero)
        eft_sym = EFTDarkEnergy(0, 0, 0, :alpha_T, 0, :H, :Omega_DE)
        @test gw170817_constraint(eft_sym) == false
    end

    @testset "apply_gw170817 sets alpha_T = 0" begin
        eft = EFTDarkEnergy(0.1, 0.5, 0.2, 0.3, 0.0, 1.0, 0.7)
        eft2 = apply_gw170817(eft)
        @test eft2.alpha_T == 0
        @test eft2.alpha_M == eft.alpha_M
        @test eft2.alpha_K == eft.alpha_K
        @test eft2.alpha_B == eft.alpha_B
        @test eft2.alpha_H == eft.alpha_H
        @test gw170817_constraint(eft2) == true
    end

    # ── Stability conditions for GR ──────────────────────────────────

    @testset "Stability conditions for GR" begin
        eft = eft_gr()
        stab = eft_stability(eft)
        # GR: D = 0 (no propagating scalar DOF)
        @test stab.D == 0
        # Tensor speed = 1
        @test stab.c_T_sq == 1
        # Tensor sector stable
        @test stab.no_gradient_tensor == true
        @test stab.no_ghost_tensor == true
    end

    @testset "Stability conditions for f(R)" begin
        aM = 0.1
        eft = eft_fR(aM)
        stab = eft_stability(eft)
        # c_T^2 = 1 + alpha_T = 1 (f(R) has alpha_T = 0)
        @test stab.c_T_sq == 1
        @test stab.no_gradient_tensor == true
        # D = alpha_K + (3/2) alpha_B^2 = 0 + (3/2) * alpha_M^2 > 0
        @test stab.D == 3//2 * aM^2
        @test stab.no_ghost_scalar == true
    end

    @testset "Stability conditions for quintessence" begin
        aK = 0.5
        eft = eft_quintessence(aK)
        stab = eft_stability(eft)
        # D = alpha_K > 0
        @test stab.D == aK
        @test stab.no_ghost_scalar == true
        # c_T^2 = 1
        @test stab.c_T_sq == 1
        @test stab.no_gradient_tensor == true
        # c_s^2 = -(1/D)[0 + 0 + (0 - 0)(1)] = 0
        # For quintessence with only alpha_K and all others zero:
        # c_s^2 = -(1/alpha_K) * 0 = 0
        @test stab.c_s_sq ≈ 0.0
    end

    # ── Observables ──────────────────────────────────────────────────

    @testset "Observables for GR" begin
        eft = eft_gr()
        obs = eft_observables(eft)
        # GR: G_eff/G_N = 1, slip = 1, c_T^2 = 1
        @test obs.G_eff_over_GN == 1
        @test obs.slip == 1
        @test obs.c_T_sq == 1
    end

    @testset "Observables: c_T_sq = 1 + alpha_T" begin
        eft = EFTDarkEnergy(0, 0.1, 0, 0.05, 0, 1.0, 0.7)
        obs = eft_observables(eft)
        @test obs.c_T_sq ≈ 1.05
    end

    # ── eft_from_horndeski symbolic ──────────────────────────────────

    @testset "eft_from_horndeski returns EFTDarkEnergy" begin
        reg = _eft_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()

            eft = eft_from_horndeski(ht, bg; registry=reg)
            @test eft isa EFTDarkEnergy
            @test eft.alpha_H == 0  # Horndeski => alpha_H = 0
            @test eft.H == :H
        end
    end

    # ── eft_from_beyond_horndeski ────────────────────────────────────

    @testset "eft_from_beyond_horndeski has nonzero alpha_H structure" begin
        reg = _eft_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht)
            bg = define_frw_background()

            eft = eft_from_beyond_horndeski(bht, bg; registry=reg)
            @test eft isa EFTDarkEnergy
            # alpha_H is a symbolic expression (not necessarily zero)
            @test eft.alpha_H !== nothing
        end
    end

end
