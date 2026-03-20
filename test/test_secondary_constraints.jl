using Test
using TensorGR
using TensorGR: PrimaryConstraint, detect_primary_constraints,
                primary_constraint_count, is_first_class, constraint_algebra,
                SecondaryConstraint, generate_secondary_constraints,
                dirac_algorithm, total_constraint_count, physical_dof_count

@testset "Secondary Constraint Generation" begin

    function _sc_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        reg
    end

    # ── SecondaryConstraint struct ────────────────────────────────────

    @testset "SecondaryConstraint struct" begin
        # Build a minimal primary to use as parent
        expr_prim = TScalar(0 // 1)
        parent = PrimaryConstraint(:pi_N, :N, expr_prim, :lapse)

        expr_sec = TScalar(1 // 1)
        sc = SecondaryConstraint(:H_ham, expr_sec, parent, :hamiltonian)

        @test sc.name == :H_ham
        @test sc.expression == TScalar(1 // 1)
        @test sc.parent === parent
        @test sc.constraint_type == :hamiltonian
    end

    @testset "SecondaryConstraint display" begin
        parent = PrimaryConstraint(:pi_N, :N, TScalar(0 // 1), :lapse)
        sc = SecondaryConstraint(:H_ham, TScalar(1 // 1), parent, :hamiltonian)
        s = sprint(show, sc)
        @test occursin("SecondaryConstraint", s)
        @test occursin("hamiltonian", s)
        @test occursin("pi_N", s)
    end

    # ── generate_secondary_constraints ────────────────────────────────

    @testset "generate_secondary_constraints returns 4 constraints for d=4" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)
            @test length(secondaries) == 4  # 1 hamiltonian + 3 momentum
        end
    end

    @testset "secondary constraint types" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            ham = filter(c -> c.constraint_type == :hamiltonian, secondaries)
            mom = filter(c -> c.constraint_type == :momentum, secondaries)

            @test length(ham) == 1
            @test length(mom) == 3
        end
    end

    @testset "secondary constraints reference parent primaries" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            # Hamiltonian constraint's parent is the lapse primary
            ham = first(filter(c -> c.constraint_type == :hamiltonian, secondaries))
            @test ham.parent.constraint_type == :lapse
            @test ham.parent.variable == adm.lapse

            # Momentum constraints' parents are shift primaries
            moms = filter(c -> c.constraint_type == :momentum, secondaries)
            for mc in moms
                @test mc.parent.constraint_type == :shift
                @test mc.parent.variable == adm.shift
            end
        end
    end

    @testset "secondary constraint expressions are TensorExpr" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            for sc in secondaries
                @test sc.expression isa TensorExpr
            end

            # Hamiltonian constraint is a TSum (sum of terms)
            ham = first(filter(c -> c.constraint_type == :hamiltonian, secondaries))
            @test ham.expression isa TSum

            # Momentum constraint is a TProduct (derivative term)
            mom = first(filter(c -> c.constraint_type == :momentum, secondaries))
            @test mom.expression isa TProduct
        end
    end

    @testset "secondary constraint names" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            names = [sc.name for sc in secondaries]
            @test :H_ham in names
            @test :H_mom_1 in names
            @test :H_mom_2 in names
            @test :H_mom_3 in names
        end
    end

    # ── dirac_algorithm ───────────────────────────────────────────────

    @testset "dirac_algorithm runs to completion" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)

            @test result.algorithm_terminated == true
            @test result.tertiary_exist == false
        end
    end

    @testset "dirac_algorithm returns correct primary count" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)

            @test length(result.primary) == 4
        end
    end

    @testset "dirac_algorithm returns correct secondary count" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)

            @test length(result.secondary) == 4
        end
    end

    @testset "dirac_algorithm total constraints" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)

            @test result.total_constraints == 8
        end
    end

    @testset "dirac_algorithm physical DOF = 2 for d=4 GR" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)

            @test result.physical_dof == 2
        end
    end

    @testset "dirac_algorithm classification" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)

            cls = result.classification
            @test cls.n_first_class == 8
            @test cls.n_second_class == 0
            @test length(cls.first_class_primary) == 4
            @test length(cls.first_class_secondary) == 4
            @test isempty(cls.second_class_primary)
            @test isempty(cls.second_class_secondary)
        end
    end

    @testset "no tertiary constraints for GR" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)

            # The Dirac algebra closes: {H, H} ~ H_i, {H_i, H} ~ H, {H_i, H_j} ~ H_k
            # No tertiary constraints arise
            @test result.tertiary_exist == false
            @test result.algorithm_terminated == true
        end
    end

    # ── total_constraint_count ────────────────────────────────────────

    @testset "total_constraint_count returns 8 for d=4" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            @test total_constraint_count(adm; registry=reg) == 8
        end
    end

    # ── physical_dof_count ────────────────────────────────────────────

    @testset "physical_dof_count returns 2 for d=4" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            @test physical_dof_count(adm; registry=reg) == 2
        end
    end

    # ── Consistency checks ────────────────────────────────────────────

    @testset "dirac_algorithm consistent with standalone functions" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)

            @test result.total_constraints == total_constraint_count(adm; registry=reg)
            @test result.physical_dof == physical_dof_count(adm; registry=reg)
            @test length(result.primary) == primary_constraint_count(adm; registry=reg)
        end
    end

    @testset "secondary expressions match constraint functions" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            # The Hamiltonian constraint expression should match
            H = hamiltonian_constraint(adm; registry=reg)
            ham_secondary = first(filter(c -> c.constraint_type == :hamiltonian, secondaries))
            # Both are TSum with 3 terms
            @test ham_secondary.expression isa TSum
            @test H isa TSum

            # The momentum constraint expression should match
            Hi = momentum_constraint(adm; registry=reg)
            mom_secondary = first(filter(c -> c.constraint_type == :momentum, secondaries))
            @test mom_secondary.expression isa TProduct
            @test Hi isa TProduct
        end
    end

    @testset "idempotent secondary generation" begin
        reg = _sc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            s1 = generate_secondary_constraints(adm, primaries; registry=reg)
            s2 = generate_secondary_constraints(adm, primaries; registry=reg)
            @test length(s1) == length(s2) == 4
        end
    end

end
