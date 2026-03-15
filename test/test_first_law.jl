@testset "Phase Space: First Law / Hamiltonian Variation" begin

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

    function register_delta_g!(reg)
        register_tensor!(reg, TensorProperties(
            name=:delta_g, manifold=:M4, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}()))
    end

    # ── HamiltonianVariation struct ──

    @testset "HamiltonianVariation struct construction" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            register_delta_g!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            nc = NoetherCurrent(Tensor(:J, [up(:a)]), :xi, L, sp)
            Q_expr = noether_charge_eh(:xi, :D; registry=reg)
            charge = NoetherCharge(Q_expr, :xi, nc)
            H = hamiltonian_variation(charge, sp, :xi; registry=reg)
            @test H isa HamiltonianVariation
            @test H.xi == :xi
            @test H.charge === charge
            @test H.potential === sp
        end
    end

    @testset "HamiltonianVariation expr has two free upper indices" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            register_delta_g!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            nc = NoetherCurrent(Tensor(:J, [up(:a)]), :xi, L, sp)
            Q_expr = noether_charge_eh(:xi, :D; registry=reg)
            charge = NoetherCharge(Q_expr, :xi, nc)
            H = hamiltonian_variation(charge, sp, :xi; registry=reg)
            fi = free_indices(H.expr)
            up_indices = filter(idx -> idx.position == Up, fi)
            @test length(up_indices) >= 2
            names = Set(idx.name for idx in up_indices)
            @test :a in names
            @test :b in names
        end
    end

    @testset "HamiltonianVariation contains xi" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            register_delta_g!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            nc = NoetherCurrent(Tensor(:J, [up(:a)]), :xi, L, sp)
            Q_expr = noether_charge_eh(:xi, :D; registry=reg)
            charge = NoetherCharge(Q_expr, :xi, nc)
            H = hamiltonian_variation(charge, sp, :xi; registry=reg)

            has_xi = false
            walk(H.expr) do e
                if e isa Tensor && e.name == :xi
                    has_xi = true
                end
                e
            end
            @test has_xi
        end
    end

    @testset "hamiltonian_variation_eh returns expression" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            register_delta_g!(reg)
            H = hamiltonian_variation_eh(:xi, :D; registry=reg)
            @test H isa TensorExpr
            fi = free_indices(H)
            up_indices = filter(idx -> idx.position == Up, fi)
            names = Set(idx.name for idx in up_indices)
            @test :a in names
            @test :b in names
        end
    end

    @testset "hamiltonian_variation_eh contains delta_Q and xi" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            register_delta_g!(reg)
            H = hamiltonian_variation_eh(:xi, :D; registry=reg)

            has_xi = false
            has_dQ = false
            walk(H) do e
                if e isa Tensor && e.name == :xi
                    has_xi = true
                end
                if e isa Tensor && e.name == :delta_Q_delta_g
                    has_dQ = true
                end
                e
            end
            @test has_xi
            @test has_dQ
        end
    end

    # ── WaldEntropyIntegrand ──

    @testset "WaldEntropyIntegrand struct construction" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            nc = NoetherCurrent(Tensor(:J, [up(:a)]), :xi, L, sp)
            Q_expr = noether_charge_eh(:xi, :D; registry=reg)
            charge = NoetherCharge(Q_expr, :xi, nc)
            W = wald_entropy_integrand(charge; registry=reg)
            @test W isa WaldEntropyIntegrand
            @test W.xi == :xi
            @test W.charge === charge
        end
    end

    @testset "Wald entropy integrand is 2*pi*Q" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            nc = NoetherCurrent(Tensor(:J, [up(:a)]), :xi, L, sp)
            Q_expr = noether_charge_eh(:xi, :D; registry=reg)
            charge = NoetherCharge(Q_expr, :xi, nc)
            W = wald_entropy_integrand(charge; registry=reg)

            # The integrand should contain pi as a scalar factor
            has_pi = false
            walk(W.expr) do e
                if e isa TScalar && e.val == :pi
                    has_pi = true
                end
                e
            end
            @test has_pi

            # The integrand should have the same free index structure as Q
            fi_W = free_indices(W.expr)
            fi_Q = free_indices(Q_expr)
            @test length(fi_W) == length(fi_Q)
        end
    end

    @testset "wald_entropy_integrand_eh returns expression with pi" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            W = wald_entropy_integrand_eh(:xi, :D; registry=reg)
            @test W isa TensorExpr

            # Should contain pi
            has_pi = false
            walk(W) do e
                if e isa TScalar && e.val == :pi
                    has_pi = true
                end
                e
            end
            @test has_pi

            # Should contain xi
            has_xi = false
            walk(W) do e
                if e isa Tensor && e.name == :xi
                    has_xi = true
                end
                e
            end
            @test has_xi
        end
    end

    @testset "wald_entropy_integrand_eh has two free upper indices" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            W = wald_entropy_integrand_eh(:xi, :D; registry=reg)
            fi = free_indices(W)
            up_indices = filter(idx -> idx.position == Up, fi)
            @test length(up_indices) == 2
        end
    end

    @testset "Wald entropy integrand is antisymmetric" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            W = wald_entropy_integrand_eh(:xi, :D; registry=reg)

            # Swap a <-> b and add: should vanish (antisymmetry of Q)
            W_swapped = rename_dummies(W, Dict(:a => :b, :b => :a))
            total = W + W_swapped
            result = simplify(total; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── Round-trip consistency ──

    @testset "hamiltonian_variation stores correct references" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            register_delta_g!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            nc = NoetherCurrent(Tensor(:J, [up(:a)]), :xi, L, sp)
            Q_expr = noether_charge_eh(:xi, :D; registry=reg)
            charge = NoetherCharge(Q_expr, :xi, nc)
            H = hamiltonian_variation(charge, sp, :xi; registry=reg)

            @test H.charge.xi == :xi
            @test H.potential.field == :g
            @test H.potential.lagrangian === L
        end
    end

end
