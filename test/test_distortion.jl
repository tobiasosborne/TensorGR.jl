# Ground truth: Hehl et al, Phys. Rep. 258 (1995), Eq 2.8.

@testset "Distortion tensor decomposition" begin

    function _dist_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        ac = define_affine_connection!(reg, :Gamma; manifold=:M4, metric=:g)
        reg, ac
    end

    @testset "decompose_distortion! registration" begin
        reg, ac = _dist_reg()
        dd = decompose_distortion!(reg, ac)

        @test dd isa DistortionDecomposition
        @test has_tensor(reg, dd.contortion_name)
        @test has_tensor(reg, dd.disformation_name)
    end

    @testset "Contortion K properties" begin
        reg, ac = _dist_reg()
        dd = decompose_distortion!(reg, ac)
        props = get_tensor(reg, dd.contortion_name)
        @test props.rank == (1, 2)
        @test props.options[:is_contortion]
        @test props.options[:torsion] == ac.torsion_name
    end

    @testset "Disformation L properties" begin
        reg, ac = _dist_reg()
        dd = decompose_distortion!(reg, ac)
        props = get_tensor(reg, dd.disformation_name)
        @test props.rank == (1, 2)
        @test props.options[:is_disformation]
        # Symmetric in last two indices
        @test !isempty(props.symmetries)
    end

    @testset "contortion_from_torsion structure" begin
        reg, ac = _dist_reg()
        with_registry(reg) do
            K_expr = contortion_from_torsion(ac.torsion_name)
            @test K_expr isa TSum
            @test length(K_expr.terms) == 3  # three torsion terms

            # Free indices: a (Up), b (Down), c (Down)
            free = free_indices(K_expr)
            @test length(free) == 3
        end
    end

    @testset "disformation_from_nonmetricity structure" begin
        reg, ac = _dist_reg()
        with_registry(reg) do
            L_expr = disformation_from_nonmetricity(ac.nonmetricity_name)
            @test L_expr isa TSum
            @test length(L_expr.terms) == 3  # Q+ - Q- - Q-

            free = free_indices(L_expr)
            @test length(free) == 3
        end
    end

    @testset "Idempotent registration" begin
        reg, ac = _dist_reg()
        dd1 = decompose_distortion!(reg, ac)
        dd2 = decompose_distortion!(reg, ac)
        @test dd1.contortion_name == dd2.contortion_name
    end

    @testset "Display" begin
        reg, ac = _dist_reg()
        dd = decompose_distortion!(reg, ac)
        s = sprint(show, dd)
        @test occursin("DistortionDecomp", s)
    end
end
