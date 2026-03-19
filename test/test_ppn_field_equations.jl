@testset "PPN field equations" begin

    function _ppn_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l]))
        reg
    end

    @testset "ppn_solve_gr" begin
        reg = _ppn_reg()
        with_registry(reg) do
            result = ppn_solve_gr(reg; order=2)
            @test result isa PPNFieldEquationResult
            @test result.order == 2
            @test result.params == ppn_gr()
        end
    end

    @testset "GR PPN parameters: gamma=1, beta=1" begin
        reg = _ppn_reg()
        with_registry(reg) do
            result = ppn_solve_gr(reg)
            p = extract_ppn_parameters(result)
            @test p.gamma == 1
            @test p.beta == 1
            @test p.xi == 0
            @test p.alpha1 == 0
            @test p.alpha2 == 0
            @test p.alpha3 == 0
            @test p.zeta1 == 0
            @test p.zeta2 == 0
            @test p.zeta3 == 0
            @test p.zeta4 == 0
        end
    end

    @testset "ppn_solve_scalar_tensor: Brans-Dicke" begin
        reg = _ppn_reg()
        with_registry(reg) do
            omega = 40  # Cassini bound: omega > 40000, but any value works
            result = ppn_solve_scalar_tensor(reg, omega; order=2)
            p = result.params
            @test p.gamma == (1 + omega) / (2 + omega)
            @test p.beta == 1
        end
    end

    @testset "Brans-Dicke gamma: omega -> infinity gives GR" begin
        reg = _ppn_reg()
        with_registry(reg) do
            # Large omega: gamma -> 1
            result = ppn_solve_scalar_tensor(reg, 10000)
            p = result.params
            @test abs(p.gamma - 1) < 1e-3
        end
    end

    @testset "ppn_solve: dispatch" begin
        reg = _ppn_reg()
        with_registry(reg) do
            r_gr = ppn_solve(:GR, reg)
            @test r_gr.params.gamma == 1

            r_bd = ppn_solve(:BransDicke, reg; omega=100)
            @test r_bd.params.gamma ≈ 101 / 102

            r_rosen = ppn_solve(:Rosen, reg)
            @test r_rosen.params.alpha1 == -2
        end
    end

    @testset "ppn_solve: unknown theory" begin
        reg = _ppn_reg()
        with_registry(reg) do
            @test_throws ErrorException ppn_solve(:unknown_theory, reg)
        end
    end

    @testset "ppn_parameter_table" begin
        reg = _ppn_reg()
        with_registry(reg) do
            result = ppn_solve_gr(reg)
            table = ppn_parameter_table(result)
            @test table isa Dict
            @test table[:gamma] == 1
            @test table[:beta] == 1
            @test length(table) == 10
        end
    end

    @testset "PPNFieldEquationResult display" begin
        reg = _ppn_reg()
        with_registry(reg) do
            result = ppn_solve_gr(reg)
            s = sprint(show, result)
            @test occursin("PPNFieldEquationResult", s)
            @test occursin("γ=1", s)
        end
    end

end
