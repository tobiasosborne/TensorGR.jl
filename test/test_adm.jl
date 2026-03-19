@testset "ADM decomposition" begin

    function _adm_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        reg
    end

    @testset "define_adm!" begin
        reg = _adm_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            @test adm isa ADMDecomposition
            @test adm.lapse == :N_adm
            @test adm.shift == :Ni_adm
            @test adm.spatial_metric == :gamma_adm
            @test adm.manifold == :M4
        end
    end

    @testset "ADM tensors registered" begin
        reg = _adm_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            @test has_tensor(reg, :N_adm)
            @test has_tensor(reg, :Ni_adm)
            @test has_tensor(reg, :gamma_adm)
            @test has_tensor(reg, :K_gamma_adm)
            @test has_tensor(reg, :K_trace_gamma_adm)
            @test has_tensor(reg, :pi_gamma_adm)
        end
    end

    @testset "ADM foliation" begin
        reg = _adm_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            @test has_foliation(reg, :adm)
            fol = get_foliation(reg, :adm)
            @test fol.temporal_component == 0
            @test fol.spatial_components == [1, 2, 3]
        end
    end

    @testset "ADM custom names" begin
        reg = _adm_reg()
        with_registry(reg) do
            adm = define_adm!(reg; lapse=:alpha, shift=:beta_i, spatial_metric=:h)
            @test adm.lapse == :alpha
            @test has_tensor(reg, :alpha)
            @test has_tensor(reg, :beta_i)
            @test has_tensor(reg, :h)
            @test has_tensor(reg, :K_h)
            @test has_tensor(reg, :pi_h)
        end
    end

    @testset "hamiltonian_constraint structure" begin
        reg = _adm_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            H = hamiltonian_constraint(adm; registry=reg)
            @test H isa TensorExpr
            # Should be a sum of 3 terms: π²_ij - (1/2)π² - R³
            @test H isa TSum
        end
    end

    @testset "momentum_constraint structure" begin
        reg = _adm_reg()
        with_registry(reg) do
            adm = define_adm!(reg)
            Hi = momentum_constraint(adm; registry=reg)
            @test Hi isa TensorExpr
            # Should contain a derivative
            @test Hi isa TProduct
        end
    end

    @testset "ADMDecomposition display" begin
        reg = _adm_reg()
        adm = define_adm!(reg)
        s = sprint(show, adm)
        @test occursin("ADM", s)
        @test occursin("N_adm", s)
    end

end
