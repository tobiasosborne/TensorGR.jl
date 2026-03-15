@testset "Phase Space: Symplectic Current" begin

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

    function register_variation!(reg, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=:M4, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}()))
    end

    @testset "SymplecticCurrent struct construction" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_variation!(reg, :h1)
            register_variation!(reg, :h2)

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            Theta = symplectic_potential(L, :g; delta_field=:h1, registry=reg)

            omega_expr = Tensor(:omega_test, [up(:a)])
            sc = SymplecticCurrent(omega_expr, :h1, :h2, Theta)
            @test sc.expr == omega_expr
            @test sc.delta1_field == :h1
            @test sc.delta2_field == :h2
            @test sc.potential === Theta
        end
    end

    @testset "EH symplectic current: free index structure" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_variation!(reg, :h1)
            register_variation!(reg, :h2)

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            Theta = symplectic_potential(L, :g; delta_field=:h1, registry=reg)

            omega = symplectic_current(Theta, :h1, :h2; registry=reg)

            @test omega isa SymplecticCurrent
            @test omega.delta1_field == :h1
            @test omega.delta2_field == :h2

            # omega should have exactly one free upper index :a
            fi = free_indices(omega.expr)
            @test length(fi) == 1
            @test fi[1].position == Up
            @test fi[1].name == :a
        end
    end

    @testset "EH omega antisymmetry: swap delta1 <-> delta2" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_variation!(reg, :h1)
            register_variation!(reg, :h2)

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            Theta = symplectic_potential(L, :g; delta_field=:h1, registry=reg)

            omega_12 = symplectic_current(Theta, :h1, :h2; registry=reg)
            omega_21 = symplectic_current(Theta, :h2, :h1; registry=reg)

            # Antisymmetry: omega(delta1, delta2) = -omega(delta2, delta1)
            # Verify by simplifying omega_12 + omega_21 = 0
            sum_expr = omega_12.expr + omega_21.expr
            result = simplify(sum_expr; registry=reg)
            @test result isa TScalar && result.val == 0 // 1
        end
    end

    @testset "EH omega bilinearity: scalar multiple" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_variation!(reg, :h1)
            register_variation!(reg, :h2)

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            Theta = symplectic_potential(L, :g; delta_field=:h1, registry=reg)

            omega = symplectic_current(Theta, :h1, :h2; registry=reg)

            # omega is bilinear: structurally, each term has exactly one
            # factor of h1 and one factor of h2 (or their derivatives).
            expanded = expand_products(omega.expr)
            function count_field(expr::TensorExpr, name::Symbol)
                n = 0
                walk(expr) do e
                    if e isa Tensor && e.name == name
                        n += 1
                    end
                    e
                end
                n
            end

            if expanded isa TSum
                for t in expanded.terms
                    n1 = count_field(t, :h1)
                    n2 = count_field(t, :h2)
                    @test n1 == 1
                    @test n2 == 1
                end
            else
                @test count_field(expanded, :h1) == 1
                @test count_field(expanded, :h2) == 1
            end
        end
    end

    @testset "EH omega contains derivative terms" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_variation!(reg, :h1)
            register_variation!(reg, :h2)

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            Theta = symplectic_potential(L, :g; delta_field=:h1, registry=reg)

            omega = symplectic_current(Theta, :h1, :h2; registry=reg)

            # The EH symplectic current contains covariant derivatives
            has_deriv = false
            walk(omega.expr) do e
                if e isa TDeriv
                    has_deriv = true
                end
                e
            end
            @test has_deriv
        end
    end

    @testset "EH omega: structure is TSum with multiple terms" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_variation!(reg, :h1)
            register_variation!(reg, :h2)

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            Theta = symplectic_potential(L, :g; delta_field=:h1, registry=reg)

            omega = symplectic_current(Theta, :h1, :h2; registry=reg)

            # The EH omega expression should be a TSum (difference of two brackets)
            @test omega.expr isa TSum
        end
    end

    @testset "General Lagrangian: formal symplectic current" begin
        reg = make_gr_registry()
        with_registry(reg) do
            # L = Ric_{ab}Ric^{ab}, which gives a formal Theta
            Ric_down = Tensor(:Ric, [down(:c), down(:d)])
            Ric_up = Tensor(:Ric, [up(:c), up(:d)])
            L_Ric2 = Ric_down * Ric_up
            L = LagrangianDensity(L_Ric2, [:g], :g, :D, 4)

            register_variation!(reg, :h1)
            register_variation!(reg, :h2)

            Theta = symplectic_potential(L, :g; delta_field=:delta_g, registry=reg)
            omega = symplectic_current(Theta, :h1, :h2; registry=reg)

            @test omega isa SymplecticCurrent
            @test omega.delta1_field == :h1
            @test omega.delta2_field == :h2

            # For a formal potential, omega should be a TSum
            # (Theta(h2) - Theta(h1)), i.e., the antisymmetrization
            @test omega.expr isa TSum
        end
    end

    @testset "omega stores reference to potential" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_variation!(reg, :h1)
            register_variation!(reg, :h2)

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            Theta = symplectic_potential(L, :g; delta_field=:h1, registry=reg)

            omega = symplectic_current(Theta, :h1, :h2; registry=reg)
            @test omega.potential === Theta
            @test omega.potential.lagrangian === L
        end
    end

    @testset "EH omega: identical variations give zero" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_variation!(reg, :h1)

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            Theta = symplectic_potential(L, :g; delta_field=:h1, registry=reg)

            # omega(delta, delta) should vanish by antisymmetry
            omega = symplectic_current(Theta, :h1, :h1; registry=reg)
            result = simplify(omega.expr; registry=reg)
            @test result isa TScalar && result.val == 0 // 1
        end
    end

end
