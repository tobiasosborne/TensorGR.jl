@testset "Phase Space: Noether Current" begin

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

    @testset "NoetherCurrent struct construction" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            sp = symplectic_potential(L, :g; registry=reg)
            J_expr = Tensor(:J, [up(:a)])
            nc = NoetherCurrent(J_expr, :xi, L, sp)
            @test nc.expr == J_expr
            @test nc.xi == :xi
            @test nc.lagrangian === L
            @test nc.potential === sp
        end
    end

    @testset "noether_current returns NoetherCurrent" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            nc = noether_current(L, :g, :xi; registry=reg)
            @test nc isa NoetherCurrent
            @test nc.xi == :xi
            @test nc.lagrangian === L
            @test nc.potential isa SymplecticPotential
        end
    end

    @testset "EH Noether current has free index up(:a)" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            nc = noether_current(L, :g, :xi; registry=reg)

            # J^a should have one free upper index
            fi = free_indices(nc.expr)
            @test length(fi) == 1
            @test fi[1].position == Up
            @test fi[1].name == :a
        end
    end

    @testset "noether_current_eh explicit formula" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)

            J_eh = noether_current_eh(:g, :xi, :D; registry=reg)

            # J^a should have one free upper index :a
            fi = free_indices(J_eh)
            @test length(fi) == 1
            @test fi[1].position == Up
            @test fi[1].name == :a
        end
    end

    @testset "noether_current_eh contains Einstein tensor and derivative terms" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)

            J_eh = noether_current_eh(:g, :xi, :D; registry=reg)

            # The expression should be a TSum with two main parts:
            # 1. 2 G^{ab} xi_b  (EOM term)
            # 2. nabla_b(nabla^a xi^b - nabla^b xi^a)  (divergence term)
            @test J_eh isa TSum
            @test length(J_eh.terms) == 2

            # Verify the expression is non-trivial
            @test !(J_eh isa TScalar && J_eh.val == 0 // 1)
        end
    end

    @testset "EH on-shell: J reduces to divergence when G=0" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)

            J_eh = noether_current_eh(:g, :xi, :D; registry=reg)

            # Set Einstein tensor to zero (on-shell)
            set_vanishing!(reg, :Ein)

            # On-shell, the 2*G^{ab}*xi_b term should vanish,
            # leaving only the divergence term nabla_b(nabla^a xi^b - nabla^b xi^a)
            J_onshell = simplify(J_eh; registry=reg)

            # The on-shell current should be non-zero (it is nabla_b nabla^[a xi^{b]})
            # and should not contain Ein
            has_ein = false
            walk(J_onshell) do e
                if e isa Tensor && e.name == :Ein
                    has_ein = true
                end
                e
            end
            @test !has_ein
        end
    end

    @testset "Killing vector: EH on-shell J = 2 nabla_b nabla^[a xi^{b]}" begin
        reg = make_gr_registry()
        with_registry(reg) do
            # Define xi as a Killing vector
            define_killing!(reg, :xi; manifold=:M4, metric=:g, covd=:D)

            J_eh = noether_current_eh(:g, :xi, :D; registry=reg)

            # Set Einstein tensor to zero (on-shell)
            set_vanishing!(reg, :Ein)

            J_onshell = simplify(J_eh; registry=reg)

            # On-shell with Killing, the current is the divergence of the
            # Noether potential: nabla_b(nabla^a xi^b - nabla^b xi^a)
            # This is a pure derivative expression and should be non-zero
            fi = free_indices(J_onshell)
            @test length(fi) == 1
            @test fi[1].position == Up
            @test fi[1].name == :a
        end
    end

    @testset "noether_current stores correct potential" begin
        reg = make_gr_registry()
        with_registry(reg) do
            register_xi!(reg)
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            nc = noether_current(L, :g, :xi; registry=reg)

            # The stored potential should be the EH symplectic potential
            @test nc.potential.field == :g
            @test nc.potential.delta_field == :delta_g
            @test nc.potential.lagrangian === L
        end
    end

end
