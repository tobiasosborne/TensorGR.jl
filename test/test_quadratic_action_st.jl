@testset "Quadratic action for scalar-tensor perturbations on FRW" begin

    # Helper: standard Horndeski registry (same as test_alpha_params.jl)
    function _qa_registry()
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

    # Helper: construct symbolic alphas for building the quadratic action.
    # Uses Symbols for non-integer values to avoid _sym_div(Float,Float).
    function _make_symbolic_alphas()
        bg = define_frw_background()
        BelliniSawickiAlphas(:alpha_M, :alpha_K, :alpha_B, :alpha_T,
                             0, :M_star_sq, bg)
    end

    # Helper: construct alphas with integer/zero values (safe for _sym_* arithmetic)
    function _make_int_alphas(; aK=0, aB=0, aT=0, aM=0, M2=1, H=1)
        bg = define_frw_background(H=H)
        BelliniSawickiAlphas(aM, aK, aB, aT, 0, M2, bg)
    end

    # ── tensor_sound_speed ──────────────────────────────────────────

    @testset "tensor_sound_speed: c_T^2 = 1 + alpha_T" begin
        # GR: alpha_T = 0 => c_T^2 = 1
        alphas_gr = _make_int_alphas(aT=0)
        @test tensor_sound_speed(alphas_gr) == 1

        # Symbolic alpha_T
        alphas_sym = _make_symbolic_alphas()
        ct2 = tensor_sound_speed(alphas_sym)
        # Evaluate numerically
        @test sym_eval(ct2, Dict(:alpha_T => 0.1)) ≈ 1.1
        @test sym_eval(ct2, Dict(:alpha_T => -0.5)) ≈ 0.5
        @test sym_eval(ct2, Dict(:alpha_T => 0.0)) ≈ 1.0
    end

    @testset "GW170817: alpha_T = 0 => c_T = 1" begin
        alphas = _make_int_alphas(aK=1, aB=0, aT=0)
        @test tensor_sound_speed(alphas) == 1
    end

    # ── scalar_sound_speed ──────────────────────────────────────────

    @testset "scalar_sound_speed: zero braiding" begin
        # With alpha_M = 0, alpha_T = 0, alpha_B = 0:
        # c_s^2 = -(1/D) [0 + 0 + 0] = 0
        alphas = _make_int_alphas(aK=2, aB=0, aT=0, aM=0)
        cs2 = scalar_sound_speed(alphas)
        @test cs2 == 0
    end

    @testset "scalar_sound_speed: numeric via sym_eval" begin
        # Non-trivial braiding: alpha_B != 0, alpha_T = alpha_M = 0
        # c_s^2 = -(1/D) [(2 - aB) aB + 0 + (aB - 0)(1 + 0)]
        #       = -(1/D) [2 aB - aB^2 + aB]  = -(3 aB - aB^2) / D
        aK = 1.0
        aB = 0.5
        D = aK + 1.5 * aB^2
        expected = -(3 * aB - aB^2) / D

        alphas = _make_symbolic_alphas()
        cs2_expr = scalar_sound_speed(alphas)
        vals = Dict{Symbol,Float64}(
            :alpha_K => aK, :alpha_B => aB, :alpha_T => 0.0, :alpha_M => 0.0,
        )
        cs2 = sym_eval(cs2_expr, vals)
        @test cs2 ≈ expected atol=1e-14
    end

    # ── ScalarTensorQuadraticAction struct ──────────────────────────

    @testset "ScalarTensorQuadraticAction struct fields" begin
        alphas = _make_int_alphas(aK=1, aB=0, aT=0, M2=2)
        qa = quadratic_action_horndeski(alphas)

        @test qa isa ScalarTensorQuadraticAction
        @test qa.alphas === alphas
        # tensor_kinetic = (1/8) * M_*^2 = (1/8) * 2 = 1/4
        @test qa.tensor_kinetic == 1//4
    end

    # ── GR limit ────────────────────────────────────────────────────

    @testset "GR limit: no propagating scalar" begin
        # GR: alpha_K = alpha_B = alpha_T = alpha_M = 0, M_*^2 = 1
        # => D = 0, Q_S = 0 (no scalar degree of freedom)
        # => tensor_kinetic = 1/8, tensor_gradient = -1/8
        alphas = _make_int_alphas(aK=0, aB=0, aT=0, aM=0, M2=1, H=1)
        qa = quadratic_action_horndeski(alphas)

        @test qa.tensor_kinetic == 1//8
        @test tensor_sound_speed(alphas) == 1

        # D = 0, so Q_S = 0/1 = 0
        # scalar_kinetic should be zero
        @test qa.scalar_kinetic == 0

        # check_stability numeric path
        params = Dict{Symbol,Float64}(
            :alpha_K => 0.0, :alpha_B => 0.0, :alpha_T => 0.0,
            :alpha_M => 0.0, :M_star_sq => 1.0, :H => 1.0,
        )
        sc = check_stability(qa, params)
        @test sc.Q_S ≈ 0.0 atol=1e-14
        @test sc.c_T_sq ≈ 1.0 atol=1e-14
        @test sc.no_tensor_ghost == true
        @test sc.no_tensor_gradient == true
        # Q_S = 0 means no scalar ghost condition is not satisfied (degenerate)
        @test sc.no_scalar_ghost == false
    end

    # ── Quintessence ────────────────────────────────────────────────

    @testset "Quintessence: alpha_T = 0, alpha_B = 0, alpha_K > 0" begin
        # Pure kinetic term: c_T = 1, D = alpha_K > 0
        # Use symbolic alphas and check_stability for numeric evaluation
        alphas = _make_symbolic_alphas()
        qa = quadratic_action_horndeski(alphas)

        aK = 2.0
        params = Dict{Symbol,Float64}(
            :alpha_K => aK, :alpha_B => 0.0, :alpha_T => 0.0,
            :alpha_M => 0.0, :M_star_sq => 1.0, :H => 1.0,
        )
        sc = check_stability(qa, params)

        # Q_S = M_*^2 * alpha_K / (1 * 1) = alpha_K
        @test sc.Q_S ≈ aK atol=1e-14
        @test sc.no_scalar_ghost == true
        @test sc.no_tensor_ghost == true
        @test sc.no_tensor_gradient == true
        # c_s^2 = 0 for pure k-essence with aM=aB=aT=0
        @test sc.c_s_sq ≈ 0.0 atol=1e-14
    end

    # ── Cubic galileon ──────────────────────────────────────────────

    @testset "Cubic galileon: nonzero braiding" begin
        # alpha_B != 0 => scalar-metric mixing generates effective c_s^2
        aK = 1.0
        aB = 0.3
        H = 1.0
        M2 = 1.0

        alphas = _make_symbolic_alphas()
        qa = quadratic_action_horndeski(alphas)

        D = aK + 1.5 * aB^2
        Q_S_expected = M2 * D / (1.0 * H^2)

        params = Dict{Symbol,Float64}(
            :alpha_K => aK, :alpha_B => aB, :alpha_T => 0.0,
            :alpha_M => 0.0, :M_star_sq => M2, :H => H,
        )
        sc = check_stability(qa, params)

        @test sc.Q_S ≈ Q_S_expected atol=1e-14
        @test sc.no_scalar_ghost == true

        # c_s^2 = -(1/D)[(2 - aB)aB + 0 + (aB - 0)(1 + 0)]
        #       = -(3aB - aB^2)/D
        cs2_expected = -(3 * aB - aB^2) / D
        @test sc.c_s_sq ≈ cs2_expected atol=1e-14
    end

    # ── Stability violation detection ───────────────────────────────

    @testset "Stability violation: c_s^2 < 0 detected" begin
        # With aB = 0.3, aT = aM = 0, aK = 1:
        # c_s^2 = -(3*0.3 - 0.09)/(1 + 1.5*0.09) = -0.81/1.135 < 0
        alphas = _make_symbolic_alphas()
        qa = quadratic_action_horndeski(alphas)

        params = Dict{Symbol,Float64}(
            :alpha_K => 1.0, :alpha_B => 0.3, :alpha_T => 0.0,
            :alpha_M => 0.0, :M_star_sq => 1.0, :H => 1.0,
        )
        sc = check_stability(qa, params)
        @test sc.no_scalar_gradient == false  # c_s^2 < 0
        @test sc.c_s_sq < 0
    end

    @testset "Stability violation: tensor ghost (M_*^2 < 0)" begin
        alphas = _make_symbolic_alphas()
        qa = quadratic_action_horndeski(alphas)

        params = Dict{Symbol,Float64}(
            :alpha_K => 1.0, :alpha_B => 0.0, :alpha_T => 0.0,
            :alpha_M => 0.0, :M_star_sq => -1.0, :H => 1.0,
        )
        sc = check_stability(qa, params)
        @test sc.no_tensor_ghost == false
    end

    @testset "Stability violation: tensor gradient (alpha_T < -1)" begin
        alphas = _make_symbolic_alphas()
        qa = quadratic_action_horndeski(alphas)

        params = Dict{Symbol,Float64}(
            :alpha_K => 1.0, :alpha_B => 0.0, :alpha_T => -1.5,
            :alpha_M => 0.0, :M_star_sq => 1.0, :H => 1.0,
        )
        sc = check_stability(qa, params)
        @test sc.no_tensor_gradient == false
        @test sc.c_T_sq < 0
    end

    # ── QuadraticForm integration ───────────────────────────────────

    @testset "to_quadratic_form: block-diagonal structure" begin
        alphas = _make_symbolic_alphas()
        qa = quadratic_action_horndeski(alphas)
        qf = to_quadratic_form(qa)

        @test qf isa QuadraticForm
        @test qf.fields == [:h_TT, :zeta]
        @test size(qf) == (2, 2)
        # Off-diagonal entries are zero (decoupled sectors)
        @test qf.matrix[1, 2] == 0
        @test qf.matrix[2, 1] == 0
    end

    # ── stability_conditions (symbolic) ─────────────────────────────

    @testset "stability_conditions returns StabilityConditions" begin
        alphas = _make_symbolic_alphas()
        qa = quadratic_action_horndeski(alphas)
        sc = stability_conditions(qa)
        @test sc isa StabilityConditions
    end

    # ── Full pipeline: compute_alphas -> quadratic_action ───────────

    @testset "Full pipeline: Horndeski to quadratic action" begin
        reg = _qa_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bg = define_frw_background()
            alphas = compute_alphas(ht, bg; registry=reg)

            # The quadratic action should build without error
            qa = quadratic_action_horndeski(alphas)
            @test qa isa ScalarTensorQuadraticAction
            @test qa.alphas === alphas

            # QuadraticForm should build without error
            qf = to_quadratic_form(qa)
            @test qf isa QuadraticForm
            @test length(qf.fields) == 2
        end
    end

    # ── GR numerical: tensor action = (M_Pl^2/8)(dot{h}^2 - (nabla h)^2)

    @testset "GR numerical: tensor action coefficient" begin
        # GR: all alphas zero, M_*^2 = 1
        alphas = _make_int_alphas(aK=0, aB=0, aT=0, aM=0, M2=1, H=1)
        qa = quadratic_action_horndeski(alphas)

        # tensor_kinetic = (1/8) M_Pl^2 = 1/8
        @test qa.tensor_kinetic == 1//8
        # tensor_gradient = -(1/8) M_Pl^2 * 1 (since c_T^2 = 1)
        @test qa.tensor_gradient == -1//8
    end

    # ── All stable configuration ────────────────────────────────────

    @testset "All stable: positive Q_S and c_s^2" begin
        # Choose parameters where all conditions are satisfied.
        # Need c_s^2 > 0. The formula:
        # c_s^2 = -(1/D)[(2-aB)aB + aT + (aB-aM)(1+aT)]
        # With aM large enough (aM > aB + something), we can make c_s^2 > 0.
        # Let aK=2, aB=0.1, aT=0.01, aM=5 =>
        #   term1 = (2-0.1)*0.1 = 0.19
        #   term2 = 0.01
        #   term3 = (0.1-5)*(1.01) = -4.949
        #   numerator = -(0.19 + 0.01 + (-4.949)) = -(-4.749) = 4.749
        #   D = 2 + 1.5*0.01 = 2.015
        #   c_s^2 = 4.749/2.015 > 0
        alphas = _make_symbolic_alphas()
        qa = quadratic_action_horndeski(alphas)

        params = Dict{Symbol,Float64}(
            :alpha_K => 2.0, :alpha_B => 0.1, :alpha_T => 0.01,
            :alpha_M => 5.0, :M_star_sq => 1.0, :H => 1.0,
        )
        sc = check_stability(qa, params)

        @test sc.no_tensor_ghost == true
        @test sc.no_tensor_gradient == true
        @test sc.no_scalar_ghost == true
        @test sc.no_scalar_gradient == true
        @test sc.c_T_sq > 0
        @test sc.c_s_sq > 0
        @test sc.Q_S > 0
    end

end
