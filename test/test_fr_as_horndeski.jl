#= Validation: f(R) gravity as Horndeski subcase.
   Ground truth: Kobayashi (2019) arXiv:1901.04778, Sec 2.3, Eqs 2.20-2.23;
                 Sotiriou & Faraoni (2010) Rev. Mod. Phys. 82, 451, Sec III.

   f(R) maps to Horndeski via phi = f'(R) (the scalaron):
     G2 = f - R f', G3 = 0, G4 = f'/2 = phi/2, G5 = 0
   No kinetic X dependence: G2_X = G3_X = G4_X = G5_X = 0.
=#

@testset "f(R) gravity as Horndeski subcase" begin

    # Helper: set up a standard Horndeski registry
    function _fR_registry()
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

    # ── Test 1: Lagrangian structure under f(R) mapping ─────────────

    @testset "f(R) Lagrangian: L3=0, L5=0, L4 = (phi/2)*R" begin
        reg = _fR_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            # f(R) mapping: G3 = 0, G5 = 0, G4_X = 0 (no X dependence)
            set_vanishing!(reg, :G3)
            set_vanishing!(reg, :G3_phi)
            set_vanishing!(reg, :G5)
            set_vanishing!(reg, :G5_X)
            set_vanishing!(reg, :G5_phi)
            set_vanishing!(reg, :G5_XX)
            set_vanishing!(reg, :G4_X)

            # L3 = -G3 * Box(phi) => vanishes since G3 = 0
            L3 = horndeski_L3(ht; registry=reg)
            L3_s = simplify(L3; registry=reg)
            @test L3_s == TScalar(0 // 1)

            # L5 = G5 * ... => vanishes since G5 = G5_X = 0
            L5 = horndeski_L5(ht; registry=reg)
            L5_s = simplify(L5; registry=reg)
            @test L5_s == TScalar(0 // 1)

            # L4 = G4*R + G4_X*[...] => G4*R since G4_X = 0
            L4 = horndeski_L4(ht; registry=reg)
            L4_s = simplify(L4; registry=reg)
            @test isempty(free_indices(L4_s))  # scalar density
            # L4 should be G4 * RicScalar (a product of two scalar tensors)
            @test L4_s isa TProduct || L4_s isa Tensor
            expr_str = string(L4_s)
            @test occursin("G4", expr_str)
            @test occursin("RicScalar", expr_str)

            # L2 = G2 (just the abstract function, representing f - Rf')
            L2 = horndeski_L2(ht; registry=reg)
            @test L2 isa Tensor
            @test L2.name == :G2
        end
    end

    # ── Test 2: GR limit f(R) = R ──────────────────────────────────

    @testset "GR limit: f(R)=R gives G4=1/2, G2=0, standard Einstein eqs" begin
        reg = _fR_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            # f(R) = R => f' = 1, f'' = 0
            # G2 = f - R f' = R - R = 0
            # G4 = f'/2 = 1/2  (constant => G4_phi = 0)
            # G3 = G5 = 0, all X-derivatives = 0

            # Set everything to zero except G4 (kept as abstract symbol)
            for s in [:G2, :G2_X, :G2_phi, :G3, :G3_X, :G3_phi,
                      :G5, :G5_X, :G5_phi, :G5_XX,
                      :G4_X, :G4_XX, :G4_phi]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end
            for s in [:G5_phiX, :G4_phiX]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end

            # Metric EOM should reduce to G4 * Ein_{ab}
            E_ab = horndeski_metric_eom(ht; registry=reg)
            E_s = simplify(E_ab; registry=reg)
            @test E_s isa TProduct
            tensor_names = Set(f isa Tensor ? f.name : nothing for f in E_s.factors)
            @test :G4 in tensor_names
            @test :Ein in tensor_names

            # Scalar EOM should vanish (no scalar dynamics when f''=0)
            E_phi = horndeski_scalar_eom(ht; registry=reg)
            E_phi_s = simplify(E_phi; registry=reg)
            @test E_phi_s == TScalar(0 // 1)

            # Numerical check: G4 = 1/2, M_*^2 = 2*G4 = 1.0
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)
            gvals = Dict{Symbol,Float64}(
                :G4 => 0.5,
                :G4_X => 0.0, :G4_XX => 0.0, :G4_XXX => 0.0,
                :G4_phi => 0.0, :G4_phiX => 0.0,
                :G2_X => 0.0, :G2_XX => 0.0,
                :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                :G5_X => 0.0, :G5_XX => 0.0, :G5_XXX => 0.0,
                :G5_phi => 0.0, :G5_phiX => 0.0,
                :H => 1.0, :phi_dot => 0.0, :phi_ddot => 0.0,
            )
            result = compute_alphas_numerical(ht, bg, gvals; registry=reg)
            @test result[:M_star_sq] ≈ 1.0     # 2*G4 = 2*(1/2) = 1
            @test result[:alpha_T] ≈ 0.0 atol=1e-14
            @test result[:alpha_K] ≈ 0.0 atol=1e-14
            @test result[:alpha_B] ≈ 0.0 atol=1e-14
        end
    end

    # ── Test 3: alpha_T = 0 (GW speed = light speed) ───────────────

    @testset "alpha_T = 0 for generic f(R) (GW170817 constraint)" begin
        # f(R) has G4_X = 0 and G5 = G5_X = 0
        # => alpha_T = (2X/M*^2)[2G4X - 2G5phi - (pdd-H pd)G5X] = 0
        # This is the critical GW170817 constraint.
        reg = _fR_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            # Test for several different f(R) background values
            for (fR_val, pd, H, pdd) in [
                (1.5, 0.3, 1.0, -0.1),   # generic f(R)
                (2.0, 0.5, 0.7, 0.2),    # another background
                (0.8, 1.0, 2.0, -0.5),   # yet another
            ]
                gvals = Dict{Symbol,Float64}(
                    :G4 => fR_val / 2,     # G4 = f'/2
                    :G4_X => 0.0,          # no X dependence
                    :G4_XX => 0.0, :G4_XXX => 0.0,
                    :G4_phi => 0.5,        # G4_phi = d(phi/2)/dphi = 1/2
                    :G4_phiX => 0.0,
                    :G2_X => 0.0, :G2_XX => 0.0,    # no X in G2
                    :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                    :G5_X => 0.0, :G5_XX => 0.0, :G5_XXX => 0.0,
                    :G5_phi => 0.0, :G5_phiX => 0.0,
                    :H => H, :phi_dot => pd, :phi_ddot => pdd,
                )
                result = compute_alphas_numerical(ht, bg, gvals; registry=reg)
                @test result[:alpha_T] ≈ 0.0 atol=1e-14
            end
        end
    end

    # ── Test 4: Scalar EOM structure (trace equation) ───────────────

    @testset "Scalar EOM reduces to trace equation structure" begin
        # For f(R) Horndeski:
        #   J^(2) = -G2_phi + G2_X * Box(phi) = -G2_phi (since G2_X = 0)
        #   J^(3) = 0 (since G3 = 0)
        #   J^(4) = -G4_phi * R - G4_phiX * [...] = -(1/2)*R (since G4_phi=1/2, G4_phiX=0)
        #   J^(5) = 0 (since G5 = 0)
        # Total: E_phi = -G2_phi - (1/2)*R = 0
        #
        # With G2 = f - Rf', we get G2_phi = d(f-Rf')/dphi.
        # Since phi = f'(R), we have dR/dphi = 1/f'' and
        #   G2_phi = (f'(R) - f'(R) - R f''(R)) * (1/f'') = -R
        # So E_phi = R - (1/2)*R = (1/2)*R ... but this is the abstract EOM.
        #
        # The actual scalar EOM in the Horndeski form is:
        #   -G2_phi - (1/2) R = 0
        # which with G2_phi as an abstract symbol gives the correct structure.

        reg = _fR_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            # f(R) mapping: set all X derivatives and G3, G5 to zero
            for s in [:G2_X, :G3, :G3_X, :G3_phi,
                      :G5, :G5_X, :G5_phi, :G5_XX,
                      :G4_X, :G4_XX]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end
            for s in [:G5_phiX, :G4_phiX]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end

            E_phi = horndeski_scalar_eom(ht; registry=reg)
            E_phi_s = simplify(E_phi; registry=reg)

            # Should be a scalar (no free indices)
            @test isempty(free_indices(E_phi_s))

            # The simplified EOM should contain RicScalar and G2_phi
            # Structure: -G2_phi - G4_phi * R
            expr_str = string(E_phi_s)
            @test occursin("RicScalar", expr_str) || occursin("G4_phi", expr_str)
            @test occursin("G2_phi", expr_str) || occursin("G4_phi", expr_str)

            # Non-trivial: the EOM is not zero for generic f(R) != R
            @test E_phi_s != TScalar(0 // 1)
        end
    end

    # ── Test 5: M_*^2 = f'(R) = phi (effective Planck mass) ────────

    @testset "M_star_sq = 2*G4 = f'(R) for f(R)" begin
        # M_*^2 = 2(G4 - 2X G4X + X G5phi - H pd X G5X)
        # For f(R): G4_X = 0, G5_phi = 0, G5_X = 0
        # => M_*^2 = 2 * G4 = 2 * (f'/2) = f' = phi
        reg = _fR_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            for fR_val in [0.5, 1.0, 1.5, 3.0]
                gvals = Dict{Symbol,Float64}(
                    :G4 => fR_val / 2,
                    :G4_X => 0.0, :G5_phi => 0.0, :G5_X => 0.0,
                    :H => 1.0, :phi_dot => 0.3,
                )
                Mstar2 = sym_eval(alphas.M_star_sq, gvals)
                @test Mstar2 ≈ fR_val  # M_*^2 = 2*(f'/2) = f'
            end
        end
    end

    # ── Test 6: alpha_B and alpha_K for f(R) ────────────────────────

    @testset "alpha_B and alpha_K for f(R) gravity" begin
        # For f(R): G3_X = 0, G4_X = 0, G4_XX = 0, G5 terms = 0
        # alpha_B = (pd / (H M*^2)) * [pd*G3X + 2H(G4X + ... - G5phi - ...) - ...]
        # With all G3_X, G4_X, G5_phi, G5_X, G5_XX, G4_XX = 0:
        #   alpha_B = 0 (all terms in bracket vanish)
        #
        # alpha_K: G2_X = 0, G2_XX = 0, G3_X = G3_XX = 0, G4_X = G4_XX = G4_XXX = 0,
        #          G5_X = G5_XX = G5_XXX = 0
        #   alpha_K = 0 (no kinetic term for the scalaron in Jordan frame)
        reg = _fR_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            gvals = Dict{Symbol,Float64}(
                :G4 => 0.75,   # f'/2 for f' = 1.5
                :G4_X => 0.0, :G4_XX => 0.0, :G4_XXX => 0.0,
                :G4_phi => 0.5, :G4_phiX => 0.0,
                :G2_X => 0.0, :G2_XX => 0.0,
                :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                :G5_X => 0.0, :G5_XX => 0.0, :G5_XXX => 0.0,
                :G5_phi => 0.0, :G5_phiX => 0.0,
                :H => 1.0, :phi_dot => 0.3, :phi_ddot => -0.1,
            )
            result = compute_alphas_numerical(ht, bg, gvals; registry=reg)

            # alpha_T = 0 (no tensor speed excess)
            @test result[:alpha_T] ≈ 0.0 atol=1e-14

            # alpha_K = 0 (no kinetic term in Jordan frame)
            @test result[:alpha_K] ≈ 0.0 atol=1e-14

            # alpha_B = 0 (all terms in bracket vanish for f(R))
            @test result[:alpha_B] ≈ 0.0 atol=1e-14

            # M_*^2 = f' = 1.5
            @test result[:M_star_sq] ≈ 1.5
        end
    end

    # ── Test 7: Stability: M_*^2 > 0 requires f' > 0 ──────────────

    @testset "Stability: no tensor ghost requires f'(R) > 0" begin
        # The effective Planck mass M_*^2 = f'(R) must be positive
        # to avoid a tensor ghost (negative-norm graviton).
        # Also c_T^2 = 1 (no superluminal gravitons) since alpha_T = 0.
        reg = _fR_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            # Viable f(R): f' > 0 => M_*^2 > 0
            for fR_val in [0.1, 0.5, 1.0, 2.0]
                gvals = Dict{Symbol,Float64}(
                    :G4 => fR_val / 2,
                    :G4_X => 0.0, :G5_phi => 0.0, :G5_X => 0.0,
                    :H => 1.0, :phi_dot => 0.5,
                )
                Mstar2 = sym_eval(alphas.M_star_sq, gvals)
                @test Mstar2 > 0  # no tensor ghost

                # c_T^2 = 1 + alpha_T = 1 (since alpha_T = 0)
                full_gvals = Dict{Symbol,Float64}(
                    :G4 => fR_val / 2,
                    :G4_X => 0.0, :G4_XX => 0.0, :G4_XXX => 0.0,
                    :G4_phi => 0.5, :G4_phiX => 0.0,
                    :G2_X => 0.0, :G2_XX => 0.0,
                    :G3_X => 0.0, :G3_XX => 0.0, :G3_phiX => 0.0,
                    :G5_X => 0.0, :G5_XX => 0.0, :G5_XXX => 0.0,
                    :G5_phi => 0.0, :G5_phiX => 0.0,
                    :H => 1.0, :phi_dot => 0.5, :phi_ddot => 0.0,
                )
                result = compute_alphas_numerical(ht, bg, full_gvals; registry=reg)
                cT_sq = 1.0 + result[:alpha_T]
                @test cT_sq ≈ 1.0 atol=1e-14  # graviton speed = speed of light
            end
        end
    end

    # ── Test 8: Metric EOM for f(R) Horndeski ───────────────────────

    @testset "f(R) metric EOM contains Einstein tensor and scalar coupling" begin
        # For f(R) Horndeski, the metric EOM should have:
        # E^(2)_{ab} = (1/2) G2 g_{ab}  (since G2_X = 0, no dphi terms)
        # E^(3)_{ab} = 0  (G3 = 0)
        # E^(4)_{ab} = G4 Ein_{ab} - G4_phi (dd_ab phi - g_ab Box phi)
        # E^(5)_{ab} = 0  (G5 = 0)
        # Total: G4 Ein_{ab} - G4_phi (dd_ab phi - g_ab Box phi) + (1/2) G2 g_{ab}
        reg = _fR_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            # f(R) specialization
            for s in [:G2_X, :G3, :G3_X, :G3_phi,
                      :G5, :G5_X, :G5_phi, :G5_XX,
                      :G4_X, :G4_XX]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end
            for s in [:G5_phiX, :G4_phiX]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end

            E_ab = horndeski_metric_eom(ht; registry=reg)
            E_s = simplify(E_ab; registry=reg)

            # Should have rank-2 free indices
            @test length(free_indices(E_s)) == 2

            # The simplified expression should contain Ein, G4, and metric terms
            expr_str = string(E_s)
            @test occursin("Ein", expr_str)
            @test occursin("G4", expr_str) || occursin("G2", expr_str)
        end
    end

end
