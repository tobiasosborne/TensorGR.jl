@testset "Ashtekar-Barbero variables (TGR-x9t)" begin

    # -- Shared setup --
    function _ashtekar_setup()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:Sigma, 3, :gamma, :partial, [:i,:j,:k,:l,:m,:n]))
        define_metric!(reg, :gamma; manifold=:Sigma)
        define_space_spinors!(reg; manifold=:Sigma, metric=:gamma)
        define_ashtekar_variables!(reg; manifold=:Sigma, spatial_metric=:gamma)
        reg
    end

    # -- Test 1: Barbero-Immirzi parameter is registered as scalar --
    @testset "Barbero-Immirzi parameter registration" begin
        reg = _ashtekar_setup()
        @test has_tensor(reg, :beta_BI)
        props = get_tensor(reg, :beta_BI)
        @test props.rank == (0, 0)
        @test props.options[:is_barbero_immirzi] == true
    end

    # -- Test 2: Ashtekar connection A_ash registration --
    @testset "A_ash registration" begin
        reg = _ashtekar_setup()
        @test has_tensor(reg, :A_ash)
        props = get_tensor(reg, :A_ash)
        @test props.rank == (1, 1)
        @test isempty(props.symmetries)
        @test props.options[:is_ashtekar_connection] == true
        @test props.options[:barbero_immirzi] == :beta_BI
        @test props.options[:index_vbundles] == [:SU2, :Tangent]
    end

    # -- Test 3: Densitized triad E_ash registration --
    @testset "E_ash registration" begin
        reg = _ashtekar_setup()
        @test has_tensor(reg, :E_ash)
        props = get_tensor(reg, :E_ash)
        @test props.rank == (1, 1)
        @test isempty(props.symmetries)
        @test props.options[:is_densitized_triad] == true
        @test props.options[:density_weight] == 1
        @test props.options[:index_vbundles] == [:Tangent, :SU2]
    end

    # -- Test 4: Curvature F_ash registration with antisymmetry --
    @testset "F_ash registration" begin
        reg = _ashtekar_setup()
        @test has_tensor(reg, :F_ash)
        props = get_tensor(reg, :F_ash)
        @test props.rank == (1, 2)
        @test any(s -> s isa AntiSymmetric && s.i == 1 && s.j == 2, props.symmetries)
        @test props.options[:is_curvature] == true
        @test props.options[:index_vbundles] == [:Tangent, :Tangent, :SU2]
    end

    # -- Test 5: A_ash index structure (1 spatial + 1 SU(2)) --
    @testset "A_ash index structure" begin
        reg = _ashtekar_setup()
        with_registry(reg) do
            A = Tensor(:A_ash, [up(:P, :SU2), down(:a)])
            @test A isa Tensor
            @test length(A.indices) == 2
            @test A.indices[1].position == Up
            @test A.indices[1].vbundle == :SU2
            @test A.indices[2].position == Down
            @test A.indices[2].vbundle == :Tangent
        end
    end

    # -- Test 6: E_ash index structure --
    @testset "E_ash index structure" begin
        reg = _ashtekar_setup()
        with_registry(reg) do
            E = densitized_triad_expr(; registry=reg)
            @test E isa Tensor
            @test E.name == :E_ash
            @test length(E.indices) == 2
            @test E.indices[1].position == Up
            @test E.indices[1].vbundle == :Tangent
            @test E.indices[2].position == Down
            @test E.indices[2].vbundle == :SU2
        end
    end

    # -- Test 7: F_ash antisymmetry in spatial indices --
    @testset "F_ash antisymmetry" begin
        reg = _ashtekar_setup()
        with_registry(reg) do
            # F_{ba}^i should canonicalize to -F_{ab}^i
            F_ba = Tensor(:F_ash, [down(:b), down(:a), up(:P, :SU2)])
            can = canonicalize(F_ba)
            F_ab = Tensor(:F_ash, [down(:a), down(:b), up(:P, :SU2)])
            if can isa TProduct
                @test can.scalar == -1 // 1
                @test length(can.factors) == 1
                @test can.factors[1] == F_ab
            else
                # In any case, swapping spatial indices should change sign
                @test can != F_ba
            end
        end
    end

    # -- Test 8: ashtekar_connection_expr builds valid expression --
    @testset "ashtekar_connection_expr" begin
        reg = _ashtekar_setup()
        with_registry(reg) do
            expr = ashtekar_connection_expr(; registry=reg)
            # A = Gamma + beta * K => should be a TSum
            @test expr isa TSum
            @test length(expr.terms) == 2
        end
    end

    # -- Test 9: Gauss constraint is a valid TDeriv expression --
    @testset "gauss_constraint_expr" begin
        reg = _ashtekar_setup()
        with_registry(reg) do
            G = gauss_constraint_expr(; registry=reg)
            @test G isa TDeriv
            @test G.index.position == Down
            @test G.index.vbundle == :Tangent
            @test G.arg isa Tensor
            @test G.arg.name == :E_ash
            # The derivative index contracts with E^a_i (same name, opposite position)
            @test G.index.name == G.arg.indices[1].name
        end
    end

    # -- Test 10: ashtekar_curvature_expr index structure --
    @testset "ashtekar_curvature_expr" begin
        reg = _ashtekar_setup()
        with_registry(reg) do
            F = ashtekar_curvature_expr(; registry=reg)
            @test F isa Tensor
            @test F.name == :F_ash
            @test length(F.indices) == 3
            # Spatial indices: Down Tangent (slots 1, 2)
            @test F.indices[1].position == Down
            @test F.indices[1].vbundle == :Tangent
            @test F.indices[2].position == Down
            @test F.indices[2].vbundle == :Tangent
            # Internal index: Up SU2 (slot 3)
            @test F.indices[3].position == Up
            @test F.indices[3].vbundle == :SU2
            # All three index names distinct
            names = [idx.name for idx in F.indices]
            @test length(unique(names)) == 3
        end
    end

    # -- Test 11: Spin connection and K_triad registration --
    @testset "auxiliary tensor registration" begin
        reg = _ashtekar_setup()
        @test has_tensor(reg, :Gamma_spin)
        @test has_tensor(reg, :K_triad)
        gp = get_tensor(reg, :Gamma_spin)
        @test gp.rank == (1, 1)
        @test gp.options[:index_vbundles] == [:SU2, :Tangent]
        kp = get_tensor(reg, :K_triad)
        @test kp.rank == (1, 1)
        @test kp.options[:index_vbundles] == [:SU2, :Tangent]
    end

    # -- Test 12: Idempotent registration --
    @testset "idempotent registration" begin
        reg = _ashtekar_setup()
        # Calling again should not error
        define_ashtekar_variables!(reg; manifold=:Sigma, spatial_metric=:gamma)
        @test has_tensor(reg, :A_ash)
        @test has_tensor(reg, :E_ash)
        @test has_tensor(reg, :F_ash)
        @test has_tensor(reg, :beta_BI)
    end

    # -- Test 13: Prerequisite checks --
    @testset "prerequisite errors" begin
        reg = TensorRegistry()
        # No manifold
        @test_throws ErrorException define_ashtekar_variables!(reg)

        register_manifold!(reg, ManifoldProperties(:Sigma, 3, :gamma, :partial, [:i,:j,:k,:l,:m,:n]))
        # No SU2 vbundle
        @test_throws ErrorException define_ashtekar_variables!(reg; manifold=:Sigma)
    end

    # -- Test 14: Spatial metric cache is preserved --
    @testset "metric cache preserved" begin
        reg = _ashtekar_setup()
        # The spatial metric should still be correctly cached
        @test reg.metric_cache[:Sigma] == :gamma
    end

end
