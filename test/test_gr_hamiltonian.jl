using Test
using TensorGR
using TensorGR: PrimaryConstraint, SecondaryConstraint,
                detect_primary_constraints, generate_secondary_constraints,
                dirac_algorithm

@testset "GR Hamiltonian Analysis: 2 DOF" begin

    function _gr_ham_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        reg
    end

    # ── Full pipeline via dof_summary ──────────────────────────────────

    @testset "dof_summary gives config_dof=2, phase_dof=4" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            summary = dof_summary(adm; registry=reg)

            @test summary.config_dof == 2
            @test summary.phase_dof == 4
        end
    end

    # ── Canonical pairs ────────────────────────────────────────────────

    @testset "10 canonical pairs (metric components in 4D)" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            summary = dof_summary(adm; registry=reg)

            # d(d+1)/2 = 4*5/2 = 10 independent metric components
            @test summary.n_canonical_pairs == 10
        end
    end

    # ── Primary constraints ────────────────────────────────────────────

    @testset "4 primary constraints (pi_N + 3 pi_{N^i})" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)

            @test length(primaries) == 4

            # 1 lapse + 3 shift
            n_lapse = count(c -> c.constraint_type == :lapse, primaries)
            n_shift = count(c -> c.constraint_type == :shift, primaries)
            @test n_lapse == 1
            @test n_shift == 3
        end
    end

    # ── Secondary constraints ──────────────────────────────────────────

    @testset "4 secondary constraints (H + 3 H_i)" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)

            @test length(secondaries) == 4

            # 1 Hamiltonian + 3 momentum
            n_ham = count(c -> c.constraint_type == :hamiltonian, secondaries)
            n_mom = count(c -> c.constraint_type == :momentum, secondaries)
            @test n_ham == 1
            @test n_mom == 3
        end
    end

    # ── Constraint classification ──────────────────────────────────────

    @testset "all 8 constraints are first-class" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            summary = dof_summary(adm; registry=reg)

            @test summary.n_first_class == 8
        end
    end

    @testset "0 second-class constraints" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            summary = dof_summary(adm; registry=reg)

            @test summary.n_second_class == 0
        end
    end

    # ── DOF formula verification ───────────────────────────────────────

    @testset "DOF formula: (20 - 2*8 - 0)/2 = 2" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            summary = dof_summary(adm; registry=reg)

            # Verify the Dirac formula explicitly
            phase_space_dim = 2 * summary.n_canonical_pairs
            expected = (phase_space_dim - 2 * summary.n_first_class - summary.n_second_class) ÷ 2
            @test expected == 2
            @test summary.config_dof == expected
        end
    end

    # ── Gauge generators ───────────────────────────────────────────────

    @testset "8 gauge generators (4 diffeomorphisms)" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            summary = dof_summary(adm; registry=reg)

            # n_gauge = n_first_class for first-class constraints
            @test summary.n_gauge == 8
        end
    end

    @testset "gauge generators include lapse and shift" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            primaries = detect_primary_constraints(adm; registry=reg)
            secondaries = generate_secondary_constraints(adm, primaries; registry=reg)
            cc = classify_constraints(primaries, secondaries; registry=reg)
            gg = gauge_generators(cc)

            # Primary generators: 1 lapse (time reparam) + 3 shift (spatial diffeo)
            primary_gg = filter(c -> c isa PrimaryConstraint, gg)
            types = [c.constraint_type for c in primary_gg]
            @test count(==(  :lapse), types) == 1
            @test count(==(:shift), types) == 3

            # Secondary generators: 1 Hamiltonian + 3 momentum
            secondary_gg = filter(c -> c isa SecondaryConstraint, gg)
            sec_types = [c.constraint_type for c in secondary_gg]
            @test count(==(  :hamiltonian), sec_types) == 1
            @test count(==(:momentum), sec_types) == 3
        end
    end

    # ── Consistency with dirac_algorithm ───────────────────────────────

    @testset "dirac_algorithm consistent with dof_summary" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)
            summary = dof_summary(adm; registry=reg)

            # Both pipelines agree on physical DOF
            @test result.physical_dof == summary.config_dof

            # dirac_algorithm confirms algorithm terminates (no tertiary constraints)
            @test result.algorithm_terminated == true
            @test result.tertiary_exist == false

            # Total constraint count = 8
            @test result.total_constraints == 8
        end
    end

    # ── Description string ─────────────────────────────────────────────

    @testset "description mentions 2 physical DOF" begin
        reg = _gr_ham_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            summary = dof_summary(adm; registry=reg)

            @test occursin("2", summary.description)
            @test occursin("gauge", summary.description) || occursin("first-class", summary.description)
        end
    end

end
