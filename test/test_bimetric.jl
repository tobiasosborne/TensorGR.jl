@testset "Bimetric gravity" begin

    function _bimetric_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        reg
    end

    @testset "define_bimetric!" begin
        reg = _bimetric_reg()
        with_registry(reg) do
            bs = define_bimetric!(reg, :g_phys, :f_phys)
            @test bs isa BimetricSetup
            @test bs.manifold == :M4
            @test bs.metric_g == :g_phys
            @test bs.metric_f == :f_phys
        end
    end

    @testset "Both metrics registered" begin
        reg = _bimetric_reg()
        with_registry(reg) do
            bs = define_bimetric!(reg, :g_phys, :f_phys)
            @test has_tensor(reg, :g_phys)
            @test has_tensor(reg, :f_phys)

            # Both symmetric rank-2
            @test get_tensor(reg, :g_phys).rank == (0, 2)
            @test get_tensor(reg, :f_phys).rank == (0, 2)
        end
    end

    @testset "Curvature tensors for g" begin
        reg = _bimetric_reg()
        with_registry(reg) do
            bs = define_bimetric!(reg, :g_phys, :f_phys)
            @test has_tensor(reg, :Riem_g_phys)
            @test has_tensor(reg, :Ric_g_phys)
            @test has_tensor(reg, :RicScalar_g_phys)
            @test has_tensor(reg, :Ein_g_phys)
            @test has_tensor(reg, :Weyl_g_phys)
            @test has_tensor(reg, :Chris_g_phys)
        end
    end

    @testset "Curvature tensors for f" begin
        reg = _bimetric_reg()
        with_registry(reg) do
            bs = define_bimetric!(reg, :g_phys, :f_phys)
            @test has_tensor(reg, :Riem_f_phys)
            @test has_tensor(reg, :Ric_f_phys)
            @test has_tensor(reg, :RicScalar_f_phys)
            @test has_tensor(reg, :Ein_f_phys)
            @test has_tensor(reg, :Weyl_f_phys)
            @test has_tensor(reg, :Chris_f_phys)
        end
    end

    @testset "Curvature tensors independent" begin
        reg = _bimetric_reg()
        with_registry(reg) do
            bs = define_bimetric!(reg, :g_phys, :f_phys)

            # Riemann tensors for g and f should be different
            Riem_g = Tensor(:Riem_g_phys, [down(:a), down(:b), down(:c), down(:d)])
            Riem_f = Tensor(:Riem_f_phys, [down(:a), down(:b), down(:c), down(:d)])
            @test Riem_g != Riem_f
        end
    end

    @testset "Matrix square root tensor S" begin
        reg = _bimetric_reg()
        with_registry(reg) do
            bs = define_bimetric!(reg, :g_phys, :f_phys)
            S_name = Symbol(:S_g_phys_f_phys)
            @test has_tensor(reg, S_name)
            @test get_tensor(reg, S_name).rank == (1, 1)
        end
    end

    @testset "bimetric_field_equations" begin
        reg = _bimetric_reg()
        with_registry(reg) do
            bs = define_bimetric!(reg, :g_phys, :f_phys)
            G_g, G_f = bimetric_field_equations(bs; registry=reg)

            @test G_g isa Tensor
            @test G_g.name == :Ein_g_phys
            @test G_f isa Tensor
            @test G_f.name == :Ein_f_phys
        end
    end

    @testset "BimetricSetup display" begin
        reg = _bimetric_reg()
        bs = define_bimetric!(reg, :g_phys, :f_phys)
        s = sprint(show, bs)
        @test occursin("BimetricSetup", s)
        @test occursin("g_phys", s)
    end

end
