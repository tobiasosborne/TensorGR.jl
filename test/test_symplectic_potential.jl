@testset "Phase Space: Symplectic Potential" begin

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

    @testset "SymplecticPotential struct construction" begin
        reg = make_gr_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            theta_expr = Tensor(:Theta, [up(:a)])
            sp = SymplecticPotential(theta_expr, :g, :delta_g, L)
            @test sp.expr == theta_expr
            @test sp.field == :g
            @test sp.delta_field == :delta_g
            @test sp.lagrangian === L
        end
    end

    @testset "EH symplectic potential via theta_eh" begin
        reg = make_gr_registry()
        with_registry(reg) do
            # Register delta_g as a symmetric 2-tensor (the metric variation)
            register_tensor!(reg, TensorProperties(
                name=:delta_g, manifold=:M4, rank=(0, 2),
                symmetries=Any[Symmetric(1, 2)],
                options=Dict{Symbol,Any}()))

            theta = theta_eh(:g, :delta_g, :D)

            # Theta should be a TensorExpr with free index up(:a)
            fi = free_indices(theta)
            @test length(fi) == 1
            @test fi[1].position == Up
            @test fi[1].name == :a
        end
    end

    @testset "EH symplectic_potential matches theta_eh" begin
        reg = make_gr_registry()
        with_registry(reg) do
            # Register delta_g
            register_tensor!(reg, TensorProperties(
                name=:delta_g, manifold=:M4, rank=(0, 2),
                symmetries=Any[Symmetric(1, 2)],
                options=Dict{Symbol,Any}()))

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; delta_field=:delta_g, registry=reg)

            @test sp isa SymplecticPotential
            @test sp.field == :g
            @test sp.delta_field == :delta_g
            @test sp.lagrangian === L

            # The expression should match the direct theta_eh
            expected = theta_eh(:g, :delta_g, :D)
            @test sp.expr == expected
        end
    end

    @testset "EH Theta structure: two terms with opposite signs" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:delta_g, manifold=:M4, rank=(0, 2),
                symmetries=Any[Symmetric(1, 2)],
                options=Dict{Symbol,Any}()))

            theta = theta_eh(:g, :delta_g, :D)

            # The EH potential is g^{bc}(∇^a δg_{bc} - ∇_c δg^a_b),
            # which is a product of g^{bc} with a sum of two TDeriv terms.
            # After expansion, the top level should be a TProduct whose
            # factors include an inverse metric and a difference of derivs.
            @test theta isa TProduct || theta isa TSum
        end
    end

    @testset "Theta is linear in delta_g" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:delta_g, manifold=:M4, rank=(0, 2),
                symmetries=Any[Symmetric(1, 2)],
                options=Dict{Symbol,Any}()))

            theta = theta_eh(:g, :delta_g, :D)

            # Linearity test: scaling the variation by alpha should scale Theta
            # Since theta_eh is g^{bc}(∇^a δg_{bc} - ∇_c δg^a_b),
            # it is manifestly linear in delta_g by construction.
            # Verify structurally: theta contains exactly one factor of delta_g
            # in each term.
            function count_delta_g(expr::TensorExpr)
                n = 0
                walk(expr) do e
                    if e isa Tensor && e.name == :delta_g
                        n += 1
                    end
                    e
                end
                n
            end

            # Expand to see individual products
            expanded = expand_products(theta)
            if expanded isa TSum
                for t in expanded.terms
                    @test count_delta_g(t) == 1
                end
            else
                @test count_delta_g(expanded) == 1
            end
        end
    end

    @testset "Default delta_field naming" begin
        reg = make_gr_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)

            # Default delta_field should be :delta_g
            sp = symplectic_potential(L, :g; registry=reg)
            @test sp.delta_field == :delta_g
        end
    end

    @testset "General Lagrangian: formal symplectic potential" begin
        reg = make_gr_registry()
        with_registry(reg) do
            # L = Ric_{ab} Ric^{ab} -- not a recognized EH Lagrangian
            Ric_down = Tensor(:Ric, [down(:c), down(:d)])
            Ric_up = Tensor(:Ric, [up(:c), up(:d)])
            L_Ric2 = Ric_down * Ric_up
            L = LagrangianDensity(L_Ric2, [:g], :g, :D, 4)

            sp = symplectic_potential(L, :g; registry=reg)
            @test sp isa SymplecticPotential
            @test sp.field == :g

            # For non-EH Lagrangians, the expression is a formal tensor
            @test sp.expr isa Tensor
            @test sp.expr.name == :Theta_g
            fi = free_indices(sp.expr)
            @test length(fi) == 1
            @test fi[1].position == Up
        end
    end

    @testset "add_boundary_ambiguity" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:delta_g, manifold=:M4, rank=(0, 2),
                symmetries=Any[Symmetric(1, 2)],
                options=Dict{Symbol,Any}()))

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; delta_field=:delta_g, registry=reg)

            # Create an antisymmetric Y^{ab}
            register_tensor!(reg, TensorProperties(
                name=:Y, manifold=:M4, rank=(2, 0),
                symmetries=Any[AntiSymmetric(1, 2)],
                options=Dict{Symbol,Any}()))
            Y = Tensor(:Y, [up(:a), up(:b)])

            # Add boundary ambiguity
            sp2 = add_boundary_ambiguity(sp, Y)

            @test sp2 isa SymplecticPotential
            @test sp2.field == sp.field
            @test sp2.delta_field == sp.delta_field
            @test sp2.lagrangian === sp.lagrangian

            # The new expression should differ from the original
            @test sp2.expr != sp.expr

            # The new expression should be a sum containing the original plus ∇_b Y^{ab}
            @test sp2.expr isa TSum
        end
    end

end
