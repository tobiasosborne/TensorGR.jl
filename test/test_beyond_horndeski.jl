@testset "Beyond-Horndeski (GLPV) extensions" begin

    # Helper to set up a standard registry for beyond-Horndeski tests
    function _bh_registry()
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

    # ── Type hierarchy ────────────────────────────────────────────────

    @testset "BeyondHorndeskiTheory contains HorndeskiTheory" begin
        reg = _bh_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht)

            @test bht isa BeyondHorndeskiTheory
            @test bht.horndeski === ht
            @test bht.F4 isa ScalarTensorFunction
            @test bht.F5 isa ScalarTensorFunction
            @test g_tensor_name(bht.F4) == :F4
            @test g_tensor_name(bht.F5) == :F5
        end
    end

    # ── Registration ──────────────────────────────────────────────────

    @testset "define_beyond_horndeski! registers F4, F5, and derivatives" begin
        reg = _bh_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht)

            @test has_tensor(reg, :F4)
            @test has_tensor(reg, :F5)
            @test has_tensor(reg, :F4_X)
            @test has_tensor(reg, :F5_X)

            # Check rank-0
            for name in [:F4, :F5, :F4_X, :F5_X]
                props = get_tensor(reg, name)
                @test props.rank == (0, 0)
            end
        end
    end

    @testset "Custom F4/F5 names" begin
        reg = _bh_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht; F4=:A4, F5=:A5)

            @test has_tensor(reg, :A4)
            @test has_tensor(reg, :A5)
            @test has_tensor(reg, :A4_X)
            @test has_tensor(reg, :A5_X)
            @test g_tensor_name(bht.F4) == :A4
            @test g_tensor_name(bht.F5) == :A5
        end
    end

    # ── Index structure: Lagrangians are scalar (no free indices) ─────

    @testset "L_4^bH is a scalar density (no free indices)" begin
        reg = _bh_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht)

            L4bh = beyond_horndeski_L4(bht; registry=reg)
            @test isempty(free_indices(L4bh))
        end
    end

    @testset "L_5^bH is a scalar density (no free indices)" begin
        reg = _bh_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht)

            L5bh = beyond_horndeski_L5(bht; registry=reg)
            @test isempty(free_indices(L5bh))
        end
    end

    # ── Vanishing limit: F4 = F5 = 0 recovers Horndeski ─────────────

    @testset "F4 = F5 = 0 recovers Horndeski Lagrangian" begin
        reg = _bh_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht)

            # Set F4 = F5 = 0
            set_vanishing!(reg, :F4)
            set_vanishing!(reg, :F5)

            L4bh = beyond_horndeski_L4(bht; registry=reg)
            L4bh_simplified = simplify(L4bh; registry=reg)
            @test L4bh_simplified == TScalar(0 // 1)

            L5bh = beyond_horndeski_L5(bht; registry=reg)
            L5bh_simplified = simplify(L5bh; registry=reg)
            @test L5bh_simplified == TScalar(0 // 1)
        end
    end

    # ── Full beyond-Horndeski Lagrangian ──────────────────────────────

    @testset "beyond_horndeski_lagrangian includes all pieces" begin
        reg = _bh_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht)

            L_full = beyond_horndeski_lagrangian(bht; registry=reg)
            @test isempty(free_indices(L_full))
            # Should be a sum of Horndeski + beyond terms
            @test L_full isa TSum
        end
    end

    # ── alpha_H is zero for pure Horndeski ───────────────────────────

    @testset "alpha_H vanishes when F4 = F5 = 0" begin
        reg = _bh_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht)
            bg = define_frw_background()

            # Evaluate with F4 = F5 = 0
            pd = 0.5
            H = 1.0
            X = 0.5 * pd^2

            gvals = Dict{Symbol,Float64}(
                :G4 => 0.5, :G4_X => 0.0,
                :G5_phi => 0.0, :G5_X => 0.0,
                :F4 => 0.0, :F5 => 0.0,
                :H => H, :phi_dot => pd,
            )

            aH = alpha_H(bht, bg; registry=reg)
            val = sym_eval(aH, gvals)
            @test val ≈ 0.0 atol=1e-14
        end
    end

    # ── alpha_H nonzero for beyond-Horndeski ──────────────────────────

    @testset "alpha_H nonzero with F4 != 0" begin
        reg = _bh_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht)
            bg = define_frw_background()

            pd = 0.5
            H = 1.0
            X = 0.5 * pd^2
            F4_val = 0.3
            F5_val = 0.0

            gvals = Dict{Symbol,Float64}(
                :G4 => 0.5, :G4_X => 0.0,
                :G5_phi => 0.0, :G5_X => 0.0,
                :F4 => F4_val, :F5 => F5_val,
                :H => H, :phi_dot => pd,
            )

            aH = alpha_H(bht, bg; registry=reg)
            val = sym_eval(aH, gvals)

            # Expected: alpha_H = (2X / M_*^2) [2 F4 pd^2]
            # M_*^2 = 2 G4 + 2X * 4 F4 X = 1.0 + 2*X*4*F4*X
            M_star_sq = 2 * 0.5 + 2 * X * 4 * F4_val * X
            expected = (2 * X / M_star_sq) * (2 * F4_val * pd^2)
            @test val ≈ expected atol=1e-12
            @test abs(val) > 0.0
        end
    end

    @testset "alpha_H nonzero with F5 != 0" begin
        reg = _bh_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            bht = define_beyond_horndeski!(reg, ht)
            bg = define_frw_background()

            pd = 0.6
            H = 0.8
            X = 0.5 * pd^2
            F4_val = 0.0
            F5_val = 0.2

            gvals = Dict{Symbol,Float64}(
                :G4 => 0.5, :G4_X => 0.0,
                :G5_phi => 0.0, :G5_X => 0.0,
                :F4 => F4_val, :F5 => F5_val,
                :H => H, :phi_dot => pd,
            )

            aH = alpha_H(bht, bg; registry=reg)
            val = sym_eval(aH, gvals)

            # Expected: alpha_H = (2X / M_*^2) [F5 pd^3 H]
            # M_*^2 = 2*G4 + 2X*(-F5 pd H X)
            M_star_sq = 2 * 0.5 + 2 * X * (-F5_val * pd * H * X)
            expected = (2 * X / M_star_sq) * (F5_val * pd^3 * H)
            @test val ≈ expected atol=1e-12
            @test abs(val) > 0.0
        end
    end

end
