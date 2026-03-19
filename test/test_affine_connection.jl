@testset "General affine connection" begin

    function _ma_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        define_curvature_tensors!(reg, :M4, :g)
        reg
    end

    @testset "define_affine_connection!" begin
        reg = _ma_reg()
        with_registry(reg) do
            ac = define_affine_connection!(reg, :Gamma_gen)
            @test ac isa AffineConnection
            @test ac.name == :Gamma_gen
            @test ac.metric == :g
        end
    end

    @testset "All tensors registered" begin
        reg = _ma_reg()
        with_registry(reg) do
            ac = define_affine_connection!(reg, :Gamma_gen)

            # Connection (no lower symmetry)
            @test has_tensor(reg, :Gamma_gen)
            @test get_tensor(reg, :Gamma_gen).rank == (1, 2)
            @test isempty(get_tensor(reg, :Gamma_gen).symmetries)

            # Levi-Civita part (symmetric lower)
            @test has_tensor(reg, :LC_Gamma_gen)
            @test get_tensor(reg, :LC_Gamma_gen).rank == (1, 2)
            @test !isempty(get_tensor(reg, :LC_Gamma_gen).symmetries)

            # Distortion
            @test has_tensor(reg, :N_Gamma_gen)

            # Torsion (antisymmetric lower)
            @test has_tensor(reg, :T_Gamma_gen)
            @test !isempty(get_tensor(reg, :T_Gamma_gen).symmetries)

            # Non-metricity (symmetric in last two)
            @test has_tensor(reg, :Q_Gamma_gen)
            @test get_tensor(reg, :Q_Gamma_gen).rank == (0, 3)
        end
    end

    @testset "Torsion antisymmetry" begin
        reg = _ma_reg()
        with_registry(reg) do
            ac = define_affine_connection!(reg, :Gamma_gen)
            tp = get_tensor(reg, :T_Gamma_gen)
            @test any(s -> s isa AntiSymmetric, tp.symmetries)
        end
    end

    @testset "is_metric_compatible: initially false" begin
        reg = _ma_reg()
        with_registry(reg) do
            ac = define_affine_connection!(reg, :Gamma_gen)
            @test !is_metric_compatible(ac, reg)
        end
    end

    @testset "set_metric_compatible!" begin
        reg = _ma_reg()
        with_registry(reg) do
            ac = define_affine_connection!(reg, :Gamma_gen)
            set_metric_compatible!(reg, ac)
            @test is_metric_compatible(ac, reg)
        end
    end

    @testset "is_torsion_free: initially false" begin
        reg = _ma_reg()
        with_registry(reg) do
            ac = define_affine_connection!(reg, :Gamma_gen)
            @test !is_torsion_free(ac, reg)
        end
    end

    @testset "set_torsion_free!" begin
        reg = _ma_reg()
        with_registry(reg) do
            ac = define_affine_connection!(reg, :Gamma_gen)
            set_torsion_free!(reg, ac)
            @test is_torsion_free(ac, reg)
        end
    end

    @testset "LC = metric-compatible + torsion-free" begin
        reg = _ma_reg()
        with_registry(reg) do
            ac = define_affine_connection!(reg, :Gamma_gen)
            set_metric_compatible!(reg, ac)
            set_torsion_free!(reg, ac)
            @test is_metric_compatible(ac, reg)
            @test is_torsion_free(ac, reg)
        end
    end

    @testset "display" begin
        reg = _ma_reg()
        ac = define_affine_connection!(reg, :Gamma_gen)
        s = sprint(show, ac)
        @test occursin("AffineConnection", s)
        @test occursin("Gamma_gen", s)
    end

end
