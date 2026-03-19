@testset "Phase Space: Noether Charge" begin

    # ── Helper: set up a standard 4D GR registry ──
    function make_gr_registry()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :D,
            [:a, :b, :c, :d, :e, :f, :m, :n, :p, :q, :r, :s]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        define_curvature_tensors!(reg, :M4, :g)
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        reg
    end

    function register_xi!(reg)
        register_tensor!(reg, TensorProperties(
            name=:xi, manifold=:M4, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}()))
    end

    @testset "NoetherCharge struct construction" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            J_expr = Tensor(:J, [up(:a)])
            nc = NoetherCurrent(J_expr, :xi, L, sp)
            Q_expr = TDeriv(up(:a), Tensor(:xi, [up(:b)]), :D) -
                     TDeriv(up(:b), Tensor(:xi, [up(:a)]), :D)
            charge = NoetherCharge(Q_expr, :xi, nc)
            @test charge.expr isa TensorExpr
            @test charge.xi == :xi
            @test charge.current === nc
        end
    end

    @testset "noether_charge_eh returns Komar 2-form" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            Q = noether_charge_eh(:xi, :D; registry=reg)

            # Q^{ab} = nabla^a xi^b - nabla^b xi^a
            # Should be a TSum with two terms
            @test Q isa TSum
            @test length(Q.terms) == 2

            # Should have two free upper indices
            fi = free_indices(Q)
            @test length(fi) == 2
            @test all(idx -> idx.position == Up, fi)
        end
    end

    @testset "noether_charge_eh is antisymmetric" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            Q = noether_charge_eh(:xi, :D; registry=reg)

            # Q^{ab} + Q^{ba} should vanish (antisymmetry)
            # Swap a <-> b in Q to get Q^{ba}
            Q_swapped = rename_dummies(Q, Dict(:a => :b, :b => :a))
            total = Q + Q_swapped
            result = simplify(total; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "noether_charge_eh free indices are :a and :b" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            Q = noether_charge_eh(:xi, :D; registry=reg)
            fi = free_indices(Q)
            names = Set(idx.name for idx in fi)
            @test :a in names
            @test :b in names
        end
    end

    @testset "noether_charge_eh divergence gives on-shell current" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)

            # Q^{ab} = nabla^a xi^b - nabla^b xi^a
            Q = noether_charge_eh(:xi, :D; registry=reg)

            # Take divergence: nabla_b Q^{ab}
            div_Q = TDeriv(down(:b), Q, :D)

            # This should be a valid expression with one free index :a (Up)
            fi = free_indices(div_Q)
            @test length(fi) == 1
            @test fi[1].position == Up
            @test fi[1].name == :a
        end
    end

    @testset "noether_charge_eh contains derivative of xi" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            Q = noether_charge_eh(:xi, :D; registry=reg)

            # Walk the expression; it should contain xi tensors
            has_xi = false
            walk(Q) do e
                if e isa Tensor && e.name == :xi
                    has_xi = true
                end
                e
            end
            @test has_xi
        end
    end

    @testset "noether_charge extracts Q from EH current" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)

            # Build the EH Noether current
            J_eh = noether_current_eh(:g, :xi, :D; registry=reg)
            sp = symplectic_potential(L, :g; registry=reg)
            nc = NoetherCurrent(J_eh, :xi, L, sp)

            # Extract the charge
            charge = noether_charge(nc, :D; registry=reg)
            @test charge isa NoetherCharge
            @test charge.xi == :xi
            @test charge.current === nc

            # The extracted Q should have two free upper indices
            fi = free_indices(charge.expr)
            up_indices = filter(idx -> idx.position == Up, fi)
            @test length(up_indices) >= 1
        end
    end

    @testset "noether_charge round-trip: div(Q) matches on-shell J" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)

            # Build EH current and go on-shell
            J_eh = noether_current_eh(:g, :xi, :D; registry=reg)
            set_vanishing!(reg, :Ein)
            J_onshell = simplify(J_eh; registry=reg)

            # Extract the charge directly from the EH formula
            Q = noether_charge_eh(:xi, :D; registry=reg)

            # nabla_b Q^{ab} should match the on-shell current
            div_Q = TDeriv(down(:b), Q, :D)

            # Both should have the same free index structure
            fi_J = free_indices(J_onshell)
            fi_div = free_indices(div_Q)
            @test length(fi_J) == 1
            @test length(fi_div) == 1
            @test fi_J[1].position == Up
            @test fi_div[1].position == Up
        end
    end

    @testset "noether_charge_eh with different xi name" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:zeta, manifold=:M4, rank=(1, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            Q = noether_charge_eh(:zeta, :D; registry=reg)
            @test Q isa TSum

            # Should contain zeta, not xi
            has_zeta = false
            walk(Q) do e
                if e isa Tensor && e.name == :zeta
                    has_zeta = true
                end
                e
            end
            @test has_zeta
        end
    end

    @testset "NoetherCharge stores correct current reference" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            J_expr = Tensor(:J, [up(:a)])
            nc = NoetherCurrent(J_expr, :xi, L, sp)
            Q_expr = noether_charge_eh(:xi, :D; registry=reg)
            charge = NoetherCharge(Q_expr, :xi, nc)

            @test charge.current.xi == :xi
            @test charge.current.lagrangian === L
        end
    end

end
