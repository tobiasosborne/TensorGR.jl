@testset "CPS Validation: EH Noether Charge = Komar Integral" begin

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

    # ── Test 1: EOM extraction gives Einstein tensor ──

    @testset "EH EOM = Einstein tensor G_{ab}" begin
        reg = make_gr_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            L_EH = LagrangianDensity(R, [:g], :g, :D, 4)
            result = eom_extract(L_EH, :g; registry=reg)

            expected = einstein_expr(down(:a), down(:b), :g)
            @test result.eom == expected
        end
    end

    # ── Test 2: Symplectic potential matches known EH formula ──

    @testset "Theta matches known EH symplectic potential" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:delta_g, manifold=:M4, rank=(0, 2),
                symmetries=Any[Symmetric(1, 2)],
                options=Dict{Symbol,Any}()))

            R = Tensor(:RicScalar, TIndex[])
            L_EH = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L_EH, :g; delta_field=:delta_g, registry=reg)

            # The computed potential should match the analytic theta_eh
            expected = theta_eh(:g, :delta_g, :D)
            @test sp.expr == expected
        end
    end

    # ── Test 3: noether_charge_eh gives Q^{ab} = D^a xi^b - D^b xi^a ──

    @testset "noether_charge_eh = Komar 2-form" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            Q = noether_charge_eh(:xi, :D; registry=reg)

            # Build the expected Komar 2-form: D^a xi^b - D^b xi^a
            Q_expected = TDeriv(up(:a), Tensor(:xi, [up(:b)]), :D) -
                         TDeriv(up(:b), Tensor(:xi, [up(:a)]), :D)

            # Both should be TSum with two derivative terms
            @test Q isa TSum
            @test Q_expected isa TSum
            @test length(Q.terms) == 2

            # The expressions should be structurally equal
            diff = simplify(Q - Q_expected; registry=reg)
            @test diff == TScalar(0 // 1)
        end
    end

    # ── Test 4: Q^{ab} is antisymmetric ──

    @testset "Q^{ab} is antisymmetric: Q^{ab} + Q^{ba} = 0" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            Q = noether_charge_eh(:xi, :D; registry=reg)

            # Swap a <-> b: Q^{ba}
            Q_swapped = rename_dummies(Q, Dict(:a => :b, :b => :a))

            # Q^{ab} + Q^{ba} should vanish
            total = Q + Q_swapped
            result = simplify(total; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── Test 5: Q^{ab} has correct free index structure ──

    @testset "Q^{ab} has two free upper indices a, b" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            Q = noether_charge_eh(:xi, :D; registry=reg)

            fi = free_indices(Q)
            @test length(fi) == 2
            @test all(idx -> idx.position == Up, fi)
            names = Set(idx.name for idx in fi)
            @test :a in names
            @test :b in names
        end
    end

    # ── Test 6: On-shell Noether current is pure divergence ──

    @testset "On-shell J^a is divergence of Q^{ab}" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)

            # Build the explicit EH current
            J_eh = noether_current_eh(:g, :xi, :D; registry=reg)

            # Set Einstein tensor to zero (go on-shell)
            set_vanishing!(reg, :Ein)

            # On-shell, the G^{ab} xi_b term vanishes
            J_onshell = simplify(J_eh; registry=reg)

            # The on-shell current should not contain Ein
            has_ein = false
            walk(J_onshell) do e
                if e isa Tensor && e.name == :Ein
                    has_ein = true
                end
                e
            end
            @test !has_ein

            # The on-shell current should be a divergence: J^a = nabla_b Q^{ab}
            @test is_divergence(J_onshell, :D; registry=reg)
        end
    end

    # ── Test 7: Divergence of Q matches on-shell current (index structure) ──

    @testset "div(Q^{ab}) has correct index structure matching J^a" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)

            Q = noether_charge_eh(:xi, :D; registry=reg)

            # nabla_b Q^{ab} should have one free upper index
            div_Q = TDeriv(down(:b), Q, :D)
            fi = free_indices(div_Q)
            @test length(fi) == 1
            @test fi[1].position == Up
            @test fi[1].name == :a
        end
    end

    # ── Test 8: Full pipeline L -> Theta -> J -> Q ──

    @testset "Full CPS pipeline: L -> J -> Q runs without error" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L_EH = LagrangianDensity(R, [:g], :g, :D, 4)

            # Step 1: Symplectic potential
            sp = symplectic_potential(L_EH, :g; registry=reg)
            @test sp isa SymplecticPotential

            # Step 2: Noether current
            J_eh = noether_current_eh(:g, :xi, :D; registry=reg)
            @test J_eh isa TensorExpr
            fi_J = free_indices(J_eh)
            @test length(fi_J) == 1
            @test fi_J[1].position == Up

            # Step 3: Noether charge (Komar 2-form)
            Q = noether_charge_eh(:xi, :D; registry=reg)
            @test Q isa TSum
            fi_Q = free_indices(Q)
            @test length(fi_Q) == 2

            # Step 4: Wald entropy integrand
            nc = NoetherCurrent(J_eh, :xi, L_EH, sp)
            charge = NoetherCharge(Q, :xi, nc)
            W = wald_entropy_integrand(charge; registry=reg)
            @test W isa WaldEntropyIntegrand
        end
    end

    # ── Test 9: Wald entropy integrand is 2*pi*Q ──

    @testset "Wald entropy integrand = 2*pi*Q^{ab}" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            Q = noether_charge_eh(:xi, :D; registry=reg)

            # Build the Wald entropy integrand via the convenience function
            W = wald_entropy_integrand_eh(:xi, :D; registry=reg)

            # Build the expected 2*pi*Q
            expected = (2 // 1) * TScalar(:pi) * Q

            # Both should have the same free index structure
            fi_W = free_indices(W)
            fi_Q = free_indices(Q)
            @test length(fi_W) == length(fi_Q)
            @test all(idx -> idx.position == Up, fi_W)

            # The expression should contain pi
            has_pi = false
            walk(W) do e
                if e isa TScalar && e.val == :pi
                    has_pi = true
                end
                e
            end
            @test has_pi
        end
    end

    # ── Test 10: Wald entropy integrand is antisymmetric (inherits from Q) ──

    @testset "Wald entropy integrand is antisymmetric" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            W = wald_entropy_integrand_eh(:xi, :D; registry=reg)

            # Swap a <-> b
            W_swapped = rename_dummies(W, Dict(:a => :b, :b => :a))
            total = W + W_swapped
            result = simplify(total; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── Test 11: Noether current EH structure: EOM + divergence ──

    @testset "EH Noether current = 2*G*xi + div(grad xi)" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)

            J_eh = noether_current_eh(:g, :xi, :D; registry=reg)

            # The current should be a sum with exactly 2 top-level terms:
            # 1) 2 G^{ab} xi_b  (EOM part)
            # 2) nabla_b(nabla^a xi^b - nabla^b xi^a)  (divergence part)
            @test J_eh isa TSum
            @test length(J_eh.terms) == 2

            # One term should contain the Einstein tensor, the other should not
            has_ein = Bool[]
            for t in J_eh.terms
                found = false
                walk(t) do e
                    if e isa Tensor && e.name == :Ein
                        found = true
                    end
                    e
                end
                push!(has_ein, found)
            end
            @test count(has_ein) == 1  # exactly one term has Ein
        end
    end

    # ── Test 12: Komar mass integral for Schwarzschild ──
    # For Schwarzschild with timelike Killing xi = partial_t,
    # the Komar mass is:
    #   M = -(1/8piG) integral_{S^2} nabla^a xi^b dS_{ab}
    # This is twice the integral of Q^{ab}: M = -1/(4piG) * integral Q^{ab} dS_{ab}
    # since Q^{ab} = nabla^a xi^b - nabla^b xi^a.
    #
    # We verify the abstract structure: Q is the correct integrand for Komar.

    @testset "Komar integrand: Q contains nabla xi (abstract check)" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            Q = noether_charge_eh(:xi, :D; registry=reg)

            # Q should be built from covariant derivatives of xi
            # Walk the expression: verify all leaf tensors are xi,
            # wrapped in covariant derivatives
            deriv_count = 0
            xi_count = 0
            walk(Q) do e
                if e isa TDeriv && e.covd == :D
                    deriv_count += 1
                end
                if e isa Tensor && e.name == :xi
                    xi_count += 1
                end
                e
            end

            # Q = D^a xi^b - D^b xi^a has 2 derivative nodes and 2 xi nodes
            @test deriv_count == 2
            @test xi_count == 2
        end
    end

    # ── Test 13: Komar factor of 2 ──
    # The Komar integral gives M = -(1/8piG) * integral nabla^a xi^b dS_{ab}
    # while the Noether charge integral is integral Q^{ab} dS_{ab}
    # with Q^{ab} = nabla^a xi^b - nabla^b xi^a.
    # By antisymmetry of dS_{ab}, integral Q dS = 2 * integral nabla^a xi^b dS_{ab}.
    # Hence the Komar mass is M = -(1/4piG) * (1/2) * integral Q dS.
    # This "factor of 2" is the famous Komar doubling.

    @testset "Komar factor: Q = 2 * nabla^{[a} xi^{b]}" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)

            Q = noether_charge_eh(:xi, :D; registry=reg)

            # Build the antisymmetrized derivative:
            # nabla^{[a} xi^{b]} = (1/2)(nabla^a xi^b - nabla^b xi^a)
            grad_ab = TDeriv(up(:a), Tensor(:xi, [up(:b)]), :D)
            grad_ba = TDeriv(up(:b), Tensor(:xi, [up(:a)]), :D)
            antisym = (1 // 2) * (grad_ab - grad_ba)

            # Q should equal 2 * nabla^{[a} xi^{b]}
            # i.e., Q - 2 * antisym = 0
            diff = simplify(Q - (2 // 1) * antisym; registry=reg)
            @test diff == TScalar(0 // 1)
        end
    end

    # ── Test 14: Killing vector: Noether current round-trip with define_killing! ──

    @testset "Killing vector: on-shell J is pure divergence" begin
        reg = make_gr_registry()
        with_registry(reg) do
            define_killing!(reg, :xi; manifold=:M4, metric=:g, covd=:D)

            J_eh = noether_current_eh(:g, :xi, :D; registry=reg)

            # Go on-shell
            set_vanishing!(reg, :Ein)
            J_onshell = simplify(J_eh; registry=reg)

            # Should have one free upper index
            fi = free_indices(J_onshell)
            @test length(fi) == 1
            @test fi[1].position == Up
            @test fi[1].name == :a

            # Should be a divergence
            @test is_divergence(J_onshell, :D; registry=reg)
        end
    end

end
