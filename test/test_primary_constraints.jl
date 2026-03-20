using Test
using TensorGR
using TensorGR: PrimaryConstraint, detect_primary_constraints,
                primary_constraint_count, is_first_class, constraint_algebra

@testset "Primary Constraint Detection" begin

    function _pc_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        reg
    end

    @testset "PrimaryConstraint struct" begin
        expr = TScalar(0 // 1)
        pc = PrimaryConstraint(:pi_N, :N, expr, :lapse)
        @test pc.name == :pi_N
        @test pc.variable == :N
        @test pc.expression == TScalar(0 // 1)
        @test pc.constraint_type == :lapse
    end

    @testset "PrimaryConstraint display" begin
        expr = TScalar(0 // 1)
        pc = PrimaryConstraint(:pi_N, :N, expr, :lapse)
        s = sprint(show, pc)
        @test occursin("PrimaryConstraint", s)
        @test occursin("lapse", s)
    end

    @testset "detect_primary_constraints returns d constraints for d=4" begin
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            constraints = detect_primary_constraints(adm; registry=reg)
            @test length(constraints) == 4  # 1 lapse + 3 shift
        end
    end

    @testset "constraint types" begin
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            constraints = detect_primary_constraints(adm; registry=reg)

            lapse_constraints = filter(c -> c.constraint_type == :lapse, constraints)
            shift_constraints = filter(c -> c.constraint_type == :shift, constraints)

            @test length(lapse_constraints) == 1
            @test length(shift_constraints) == 3
        end
    end

    @testset "constraint variables" begin
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            constraints = detect_primary_constraints(adm; registry=reg)

            lapse_c = first(filter(c -> c.constraint_type == :lapse, constraints))
            @test lapse_c.variable == :N_adm

            shift_cs = filter(c -> c.constraint_type == :shift, constraints)
            for sc in shift_cs
                @test sc.variable == :Ni_adm
            end
        end
    end

    @testset "constraint momentum tensors registered as vanishing" begin
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            constraints = detect_primary_constraints(adm; registry=reg)

            # Lapse momentum π_N registered and vanishing
            @test has_tensor(reg, :pi_N_adm)
            @test get_tensor(reg, :pi_N_adm).vanishing

            # Shift momentum π_{N^i} registered and vanishing
            @test has_tensor(reg, :pi_Ni_adm)
            @test get_tensor(reg, :pi_Ni_adm).vanishing
        end
    end

    @testset "constraint expressions" begin
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            constraints = detect_primary_constraints(adm; registry=reg)

            # Lapse constraint expression is a scalar tensor (rank-0)
            lapse_c = first(filter(c -> c.constraint_type == :lapse, constraints))
            @test lapse_c.expression isa Tensor
            @test lapse_c.expression.name == :pi_N_adm
            @test isempty(lapse_c.expression.indices)

            # Shift constraint expressions are rank-1 tensors (covariant)
            shift_cs = filter(c -> c.constraint_type == :shift, constraints)
            for sc in shift_cs
                @test sc.expression isa Tensor
                @test sc.expression.name == :pi_Ni_adm
                @test length(sc.expression.indices) == 1
                @test sc.expression.indices[1].position == Down
            end
        end
    end

    @testset "primary_constraint_count" begin
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            @test primary_constraint_count(adm; registry=reg) == 4
        end
    end

    @testset "is_first_class" begin
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            constraints = detect_primary_constraints(adm; registry=reg)

            # All GR primary constraints are first-class
            for c in constraints
                @test is_first_class(c, constraints; registry=reg)
            end
        end
    end

    @testset "constraint_algebra classification" begin
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            constraints = detect_primary_constraints(adm; registry=reg)
            alg = constraint_algebra(constraints; registry=reg)

            # All 4 primary constraints are first-class
            @test alg.n_first_class == 4
            @test alg.n_second_class == 0
            @test alg.n_primary == 4

            # 4 secondary constraints (1 Hamiltonian + 3 momentum)
            @test alg.n_secondary == 4

            # Total first-class = 4 primary + 4 secondary = 8
            @test alg.n_total_first_class == 8

            # DOF = 10 - 2*4 = 2 (graviton has 2 polarizations)
            @test alg.physical_dof == 2
        end
    end

    @testset "constraint_algebra vectors" begin
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            constraints = detect_primary_constraints(adm; registry=reg)
            alg = constraint_algebra(constraints; registry=reg)

            @test length(alg.first_class) == 4
            @test isempty(alg.second_class)

            # Verify we can identify lapse/shift in first-class list
            types = [c.constraint_type for c in alg.first_class]
            @test count(==(  :lapse), types) == 1
            @test count(==(:shift), types) == 3
        end
    end

    @testset "idempotent detection" begin
        # Calling detect_primary_constraints twice should not error or
        # duplicate constraints
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            c1 = detect_primary_constraints(adm; registry=reg)
            c2 = detect_primary_constraints(adm; registry=reg)
            @test length(c1) == length(c2) == 4
        end
    end

    @testset "custom ADM names" begin
        reg = _pc_reg()
        with_registry(reg) do
            adm = define_adm!(reg; lapse=:alpha, shift=:beta_i, spatial_metric=:h)
            constraints = detect_primary_constraints(adm; registry=reg)

            @test length(constraints) == 4

            lapse_c = first(filter(c -> c.constraint_type == :lapse, constraints))
            @test lapse_c.variable == :alpha
            @test lapse_c.name == :pi_alpha

            shift_cs = filter(c -> c.constraint_type == :shift, constraints)
            @test all(c -> c.variable == :beta_i, shift_cs)
        end
    end

end
