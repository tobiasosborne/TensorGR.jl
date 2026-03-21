using Test
using TensorGR
using TensorGR: PrimaryConstraint, detect_primary_constraints,
                SecondaryConstraint, generate_secondary_constraints,
                dirac_algorithm

@testset "Constraint Classification" begin

    function _cc_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        reg
    end

    # ── ConstraintClassification struct ────────────────────────────────

    @testset "ConstraintClassification struct fields" begin
        # Build a minimal classification by hand
        expr_prim = TScalar(0 // 1)
        parent = PrimaryConstraint(:pi_N, :N, expr_prim, :lapse)
        sec = SecondaryConstraint(:H_ham, TScalar(1 // 1), parent, :hamiltonian)

        fc = Union{PrimaryConstraint, SecondaryConstraint}[parent, sec]
        sc = Union{PrimaryConstraint, SecondaryConstraint}[]
        cc = ConstraintClassification(fc, sc, 2, 0, 1)

        @test cc.n_first_class == 2
        @test cc.n_second_class == 0
        @test cc.dof == 1
        @test length(cc.first_class) == 2
        @test isempty(cc.second_class)
    end

    @testset "ConstraintClassification display" begin
        fc = Union{PrimaryConstraint, SecondaryConstraint}[]
        sc = Union{PrimaryConstraint, SecondaryConstraint}[]
        cc = ConstraintClassification(fc, sc, 0, 0, 0)
        s = sprint(show, cc)
        @test occursin("ConstraintClassification", s)
        @test occursin("first_class=0", s)
        @test occursin("second_class=0", s)
        @test occursin("dof=0", s)
    end

    # ── classify_constraints for GR (d=4) ─────────────────────────────

    @testset "GR: all 8 constraints are first-class" begin
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            cc = classify_constraints(primaries, secondaries; registry=reg)

            @test cc.n_first_class == 8
            @test cc.n_second_class == 0
        end
    end

    @testset "GR: second_class list is empty" begin
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            cc = classify_constraints(primaries, secondaries; registry=reg)

            @test isempty(cc.second_class)
        end
    end

    @testset "GR: DOF = 2 (graviton polarizations)" begin
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            cc = classify_constraints(primaries, secondaries; registry=reg)

            # DOF = (20 - 2*8 - 0) / 2 = 2
            @test cc.dof == 2
        end
    end

    @testset "GR: explicit phase_space_dim=20 gives DOF=2" begin
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            cc = classify_constraints(primaries, secondaries;
                                       phase_space_dim=20, registry=reg)

            @test cc.dof == 2
        end
    end

    @testset "GR: first_class contains both primary and secondary" begin
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            cc = classify_constraints(primaries, secondaries; registry=reg)

            # Count primary vs secondary in the first_class list
            n_prim_fc = count(c -> c isa PrimaryConstraint, cc.first_class)
            n_sec_fc = count(c -> c isa SecondaryConstraint, cc.first_class)

            @test n_prim_fc == 4
            @test n_sec_fc == 4
        end
    end

    # ── count_dof ─────────────────────────────────────────────────────

    @testset "count_dof with phase_space_dim=20 gives DOF=2" begin
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            cc = classify_constraints(primaries, secondaries; registry=reg)

            # Recompute DOF with explicit phase space dim
            dof = count_dof(cc; phase_space_dim=20)
            @test dof == 2
        end
    end

    @testset "count_dof formula: (phase_space_dim - 2*n_fc - n_sc) / 2" begin
        # Build a mock classification with known values
        fc = Union{PrimaryConstraint, SecondaryConstraint}[]
        sc = Union{PrimaryConstraint, SecondaryConstraint}[]

        # 3 first-class, 2 second-class, phase_space_dim = 12
        # DOF = (12 - 6 - 2) / 2 = 2
        cc = ConstraintClassification(fc, sc, 3, 2, 0)
        @test count_dof(cc; phase_space_dim=12) == 2

        # 0 constraints, phase_space_dim = 10
        # DOF = 10/2 = 5
        cc2 = ConstraintClassification(fc, sc, 0, 0, 0)
        @test count_dof(cc2; phase_space_dim=10) == 5
    end

    # ── gauge_generators ──────────────────────────────────────────────

    @testset "gauge_generators returns first-class constraints" begin
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            cc = classify_constraints(primaries, secondaries; registry=reg)
            gg = gauge_generators(cc)

            @test length(gg) == 8
            @test gg === cc.first_class
        end
    end

    @testset "gauge generators include lapse and shift primaries" begin
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            cc = classify_constraints(primaries, secondaries; registry=reg)
            gg = gauge_generators(cc)

            # Primary constraints in the gauge generators
            primary_gg = filter(c -> c isa PrimaryConstraint, gg)
            types = [c.constraint_type for c in primary_gg]
            @test count(==(  :lapse), types) == 1
            @test count(==(:shift), types) == 3
        end
    end

    @testset "gauge generators include Hamiltonian and momentum secondaries" begin
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            cc = classify_constraints(primaries, secondaries; registry=reg)
            gg = gauge_generators(cc)

            # Secondary constraints in the gauge generators
            secondary_gg = filter(c -> c isa SecondaryConstraint, gg)
            types = [c.constraint_type for c in secondary_gg]
            @test count(==(  :hamiltonian), types) == 1
            @test count(==(:momentum), types) == 3
        end
    end

    # ── Consistency with dirac_algorithm ──────────────────────────────

    @testset "classify_constraints consistent with dirac_algorithm" begin
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)

            cc = classify_constraints(result.primary, result.secondary; registry=reg)

            # Both should agree on DOF
            @test cc.dof == result.physical_dof

            # Both should agree on total first-class count
            @test cc.n_first_class == result.classification.n_first_class

            # Both should agree on total constraint count
            @test length(cc.first_class) + length(cc.second_class) == result.total_constraints
        end
    end

    # ── DOF formula verification ──────────────────────────────────────

    @testset "DOF formula: phase space counting" begin
        # Standard GR: 10 metric components, 10 momenta = 20 phase space DOF
        # 8 first-class constraints, 0 second-class
        # DOF = (20 - 2*8 - 0) / 2 = 2
        reg = _cc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            cc = classify_constraints(primaries, secondaries;
                                       phase_space_dim=20, registry=reg)

            phase_space_dim = 20
            expected_dof = (phase_space_dim - 2 * cc.n_first_class - cc.n_second_class) ÷ 2
            @test cc.dof == expected_dof
            @test cc.dof == 2
        end
    end

end
