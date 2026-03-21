using Test
using TensorGR
using TensorGR: PrimaryConstraint, detect_primary_constraints,
                SecondaryConstraint, generate_secondary_constraints,
                dirac_algorithm

@testset "DOF Counting (Dirac formula)" begin

    function _dof_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        reg
    end

    # ── DOFSummary struct ─────────────────────────────────────────────

    @testset "DOFSummary struct construction" begin
        ds = DOFSummary(2, 4, 10, 8, 0, 8, "test")
        @test ds.config_dof == 2
        @test ds.phase_dof == 4
        @test ds.n_canonical_pairs == 10
        @test ds.n_first_class == 8
        @test ds.n_second_class == 0
        @test ds.n_gauge == 8
        @test ds.description == "test"
    end

    @testset "DOFSummary display" begin
        ds = DOFSummary(2, 4, 10, 8, 0, 8, "test")
        s = sprint(show, ds)
        @test occursin("DOFSummary", s)
        @test occursin("config_dof=2", s)
        @test occursin("phase_dof=4", s)
        @test occursin("n_gauge=8", s)
    end

    # ── dof_count: GR ground truth (Henneaux & Teitelboim 1992) ──────

    @testset "GR (4D): 10 pairs, 8 first-class, 0 second-class -> config=2" begin
        ds = dof_count(10, 8, 0)
        @test ds.config_dof == 2
        @test ds.phase_dof == 4
        @test ds.n_canonical_pairs == 10
        @test ds.n_first_class == 8
        @test ds.n_second_class == 0
        @test ds.n_gauge == 8
    end

    # ── dof_count: Maxwell ground truth ──────────────────────────────

    @testset "Maxwell (4D): 4 pairs, 2 first-class, 0 second-class -> config=2" begin
        ds = dof_count(4, 2, 0)
        @test ds.config_dof == 2
        @test ds.phase_dof == 4
        @test ds.n_gauge == 2
    end

    # ── dof_count: Proca ground truth ────────────────────────────────

    @testset "Proca (massive vector): 4 pairs, 0 first-class, 2 second-class -> config=3" begin
        # Proca: mass term breaks gauge invariance, so NO first-class constraints.
        # pi^0 ~ 0 and Gauss law are both second-class (their bracket is non-zero).
        # Ground truth: Henneaux & Teitelboim (1992) Sec 1.2.
        ds = dof_count(4, 0, 2)
        @test ds.config_dof == 3
        @test ds.phase_dof == 6
        @test ds.n_gauge == 0
    end

    # ── dof_count: gauge freedom = n_first_class ─────────────────────

    @testset "gauge freedom count equals n_first_class" begin
        ds1 = dof_count(10, 8, 0)
        @test ds1.n_gauge == ds1.n_first_class

        ds2 = dof_count(4, 2, 0)
        @test ds2.n_gauge == ds2.n_first_class

        ds3 = dof_count(4, 0, 2)
        @test ds3.n_gauge == ds3.n_first_class
    end

    # ── dof_count: description string ────────────────────────────────

    @testset "description string is informative" begin
        ds = dof_count(10, 8, 0)
        @test occursin("10 canonical pairs", ds.description)
        @test occursin("8 first-class", ds.description)
        @test occursin("0 second-class", ds.description)
        @test occursin("2 config-space", ds.description)
        @test occursin("4 phase-space", ds.description)
    end

    # ── dof_count: unconstrained system ──────────────────────────────

    @testset "unconstrained system: all pairs are physical" begin
        ds = dof_count(5, 0, 0)
        @test ds.config_dof == 5
        @test ds.phase_dof == 10
        @test ds.n_gauge == 0
    end

    # ── dof_count: error cases ───────────────────────────────────────

    @testset "error: negative DOF (too many constraints)" begin
        # 2 canonical pairs but 3 first-class constraints -> phase_dof = 4 - 6 = -2
        @test_throws ErrorException dof_count(2, 3, 0)
    end

    @testset "error: odd second-class count" begin
        @test_throws ErrorException dof_count(5, 0, 3)
    end

    @testset "error: negative inputs" begin
        @test_throws ErrorException dof_count(-1, 0, 0)
        @test_throws ErrorException dof_count(5, -1, 0)
        @test_throws ErrorException dof_count(5, 0, -2)
    end

    # ── dof_from_classification ──────────────────────────────────────

    @testset "dof_from_classification matches dof_count for GR" begin
        # Build a ConstraintClassification with GR values
        fc = Union{PrimaryConstraint, SecondaryConstraint}[]
        sc = Union{PrimaryConstraint, SecondaryConstraint}[]
        cc = ConstraintClassification(fc, sc, 8, 0, 2)

        ds = dof_from_classification(cc; n_canonical_pairs=10)
        @test ds.config_dof == 2
        @test ds.phase_dof == 4
        @test ds.n_first_class == 8
        @test ds.n_second_class == 0
        @test ds.n_gauge == 8
    end

    @testset "dof_from_classification with second-class constraints (Proca)" begin
        # Proca: 0 first-class, 2 second-class
        fc = Union{PrimaryConstraint, SecondaryConstraint}[]
        sc = Union{PrimaryConstraint, SecondaryConstraint}[]
        cc = ConstraintClassification(fc, sc, 0, 2, 0)

        ds = dof_from_classification(cc; n_canonical_pairs=4)
        @test ds.config_dof == 3
        @test ds.phase_dof == 6
    end

    # ── dof_summary (full pipeline) ──────────────────────────────────

    @testset "dof_summary(adm) for GR gives config_dof=2" begin
        reg = _dof_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            ds = dof_summary(adm; registry=reg)

            @test ds.config_dof == 2
            @test ds.phase_dof == 4
            @test ds.n_canonical_pairs == 10
            @test ds.n_first_class == 8
            @test ds.n_second_class == 0
            @test ds.n_gauge == 8
        end
    end

    @testset "dof_summary consistent with dirac_algorithm" begin
        reg = _dof_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            result = dirac_algorithm(adm; registry=reg)
            ds = dof_summary(adm; registry=reg)

            # Both should agree on physical DOF
            @test ds.config_dof == result.physical_dof
        end
    end

    @testset "dof_summary description mentions gauge generators" begin
        reg = _dof_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            ds = dof_summary(adm; registry=reg)

            @test occursin("gauge generators", ds.description)
        end
    end

end
