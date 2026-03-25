# Tests for Iyer-Wald first law / Wald entropy (src/phase_space/first_law.jl)
#
# Ground truth:
#   Iyer & Wald (1994), PRD 50, 846, Eqs 3.5, 4.1
#   Wald (1993), PRD 48, R3427
#   EH entropy: S = A/4G (Bekenstein-Hawking)

@testset "First Law / Wald Entropy" begin

    # ── Helper: standard GR registry with xi ──
    function make_firstlaw_registry()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :D,
            [:a, :b, :c, :d, :e, :f, :m, :n, :p, :q, :r, :s]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        define_curvature_tensors!(reg, :M4, :g)
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        register_tensor!(reg, TensorProperties(
            name=:xi, manifold=:M4, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}()))
        reg
    end

    # ── WaldEntropyIntegrand struct ──

    @testset "WaldEntropyIntegrand construction" begin
        reg = make_firstlaw_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            J_expr = Tensor(:J, [up(:a)])
            nc = NoetherCurrent(J_expr, :xi, L, sp)
            Q_expr = noether_charge_eh(:xi, :D; registry=reg)
            charge = NoetherCharge(Q_expr, :xi, nc)

            W = wald_entropy_integrand(charge; registry=reg)
            @test W isa WaldEntropyIntegrand
            @test W.xi == :xi
            @test W.charge === charge
            @test W.expr isa TensorExpr
        end
    end

    @testset "Wald entropy integrand = 2pi * Q" begin
        reg = make_firstlaw_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            J_expr = Tensor(:J, [up(:a)])
            nc = NoetherCurrent(J_expr, :xi, L, sp)
            Q_expr = noether_charge_eh(:xi, :D; registry=reg)
            charge = NoetherCharge(Q_expr, :xi, nc)

            W = wald_entropy_integrand(charge; registry=reg)

            # W.expr should contain TScalar(:pi) as a factor
            has_pi = false
            walk(W.expr) do e
                if e isa TScalar && e.val == :pi
                    has_pi = true
                end
                e
            end
            @test has_pi

            # W.expr should contain derivative of xi
            has_deriv_xi = false
            walk(W.expr) do e
                if e isa TDeriv && e.arg isa Tensor && e.arg.name == :xi
                    has_deriv_xi = true
                end
                e
            end
            @test has_deriv_xi
        end
    end

    # ── EH specialization ──

    @testset "wald_entropy_integrand_eh structure" begin
        reg = make_firstlaw_registry()
        with_registry(reg) do
            W_eh = wald_entropy_integrand_eh(:xi, :D; registry=reg)

            # Should be a valid tensor expression
            @test W_eh isa TensorExpr

            # Should have two free upper indices :a, :b
            fi = free_indices(W_eh)
            @test length(fi) == 2
            @test all(idx -> idx.position == Up, fi)
            names = Set(idx.name for idx in fi)
            @test :a in names
            @test :b in names
        end
    end

    @testset "wald_entropy_integrand_eh antisymmetry" begin
        reg = make_firstlaw_registry()
        with_registry(reg) do
            W = wald_entropy_integrand_eh(:xi, :D; registry=reg)

            # W^{ab} + W^{ba} should be zero (inherited from Q antisymmetry)
            W_swapped = rename_dummies(W, Dict(:a => :b, :b => :a))
            total = W + W_swapped
            result = simplify(total; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── Hamiltonian variation ──

    @testset "hamiltonian_variation_eh structure" begin
        reg = make_firstlaw_registry()
        with_registry(reg) do
            # Need delta_g registered
            register_tensor!(reg, TensorProperties(
                name=:delta_g, manifold=:M4, rank=(0, 2),
                symmetries=Any[Symmetric(1, 2)],
                options=Dict{Symbol,Any}()))

            H = hamiltonian_variation_eh(:xi, :D; registry=reg)

            # Should be a valid tensor expression
            @test H isa TensorExpr

            # Should have two free upper indices
            fi = free_indices(H)
            up_fi = filter(idx -> idx.position == Up, fi)
            @test length(up_fi) >= 2
        end
    end

    @testset "HamiltonianVariation struct" begin
        reg = make_firstlaw_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            J_expr = Tensor(:J, [up(:a)])
            nc = NoetherCurrent(J_expr, :xi, L, sp)
            Q_expr = noether_charge_eh(:xi, :D; registry=reg)
            charge = NoetherCharge(Q_expr, :xi, nc)

            H = hamiltonian_variation(charge, sp, :xi; registry=reg)
            @test H isa HamiltonianVariation
            @test H.xi == :xi
            @test H.charge === charge
            @test H.potential === sp
            @test H.expr isa TensorExpr
        end
    end

end
