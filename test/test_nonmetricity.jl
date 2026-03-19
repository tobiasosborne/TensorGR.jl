# Ground truth: Hehl, McCrea, Mielke & Ne'eman, Phys. Rep. 258 (1995), Sec 2.4.

@testset "Non-metricity decomposition" begin

    function _nm_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        ac = define_affine_connection!(reg, :Gamma; manifold=:M4, metric=:g)
        reg, ac
    end

    @testset "decompose_nonmetricity! registration" begin
        reg, ac = _nm_reg()
        nd = decompose_nonmetricity!(reg, ac.nonmetricity_name; manifold=:M4)

        @test nd isa NonmetricityDecomposition
        @test has_tensor(reg, nd.weyl_name)
        @test has_tensor(reg, nd.second_trace_name)
        @test has_tensor(reg, nd.traceless_name)
    end

    @testset "Weyl vector Q_a properties" begin
        reg, ac = _nm_reg()
        nd = decompose_nonmetricity!(reg, ac.nonmetricity_name; manifold=:M4)

        props = get_tensor(reg, nd.weyl_name)
        @test props.rank == (0, 1)
        @test props.options[:is_weyl_vector]
    end

    @testset "Second trace tilde_Q_a properties" begin
        reg, ac = _nm_reg()
        nd = decompose_nonmetricity!(reg, ac.nonmetricity_name; manifold=:M4)

        props = get_tensor(reg, nd.second_trace_name)
        @test props.rank == (0, 1)
        @test props.options[:is_second_trace]
    end

    @testset "Traceless part Omega_{abc} properties" begin
        reg, ac = _nm_reg()
        nd = decompose_nonmetricity!(reg, ac.nonmetricity_name; manifold=:M4)

        props = get_tensor(reg, nd.traceless_name)
        @test props.rank == (0, 3)
        # Symmetric in last two indices
        @test !isempty(props.symmetries)
        @test props.options[:traces_vanish]
    end

    @testset "weyl_vector_expr structure" begin
        reg, ac = _nm_reg()
        with_registry(reg) do
            Q_name = ac.nonmetricity_name
            expr = weyl_vector_expr(Q_name)
            @test expr isa TProduct
            # Should have g^{bc} Q_{abc} — one free index a
            free = free_indices(expr)
            @test length(free) == 1
        end
    end

    @testset "second_trace_expr structure" begin
        reg, ac = _nm_reg()
        with_registry(reg) do
            Q_name = ac.nonmetricity_name
            expr = second_trace_expr(Q_name)
            @test expr isa TProduct
            # Should have g^{ab} Q_{abc} — one free index c
            free = free_indices(expr)
            @test length(free) == 1
        end
    end

    @testset "Idempotent registration" begin
        reg, ac = _nm_reg()
        nd1 = decompose_nonmetricity!(reg, ac.nonmetricity_name; manifold=:M4)
        nd2 = decompose_nonmetricity!(reg, ac.nonmetricity_name; manifold=:M4)
        @test nd1.weyl_name == nd2.weyl_name
    end

    @testset "Display" begin
        reg, ac = _nm_reg()
        nd = decompose_nonmetricity!(reg, ac.nonmetricity_name; manifold=:M4)
        s = sprint(show, nd)
        @test occursin("NonmetricityDecomposition", s)
    end
end
