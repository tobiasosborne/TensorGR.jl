@testset "PPN observables" begin

    @testset "GR perihelion factor = 1" begin
        @test ppn_perihelion_factor(ppn_gr()) == 1
    end

    @testset "GR deflection factor = 1" begin
        @test ppn_deflection_factor(ppn_gr()) == 1
    end

    @testset "GR Shapiro factor = 1" begin
        @test ppn_shapiro_factor(ppn_gr()) == 1
    end

    @testset "GR geodetic factor = 3/2" begin
        @test ppn_geodetic_factor(ppn_gr()) == 3 // 2
    end

    @testset "GR Nordtvedt eta = 0" begin
        @test ppn_nordtvedt_eta(ppn_gr()) == 0
    end

    @testset "Brans-Dicke: deflection < GR" begin
        bd = PPNParameters(:BransDicke; omega=40)
        @test ppn_deflection_factor(bd) < 1
    end

    @testset "Brans-Dicke: Nordtvedt eta nonzero" begin
        bd = PPNParameters(:BransDicke; omega=40)
        # BD has β=1 but γ≠1, so η = 4(1) - γ - 3 = 1 - γ ≠ 0
        eta = ppn_nordtvedt_eta(bd)
        @test eta != 0
    end

    @testset "Rosen theory: preferred-frame effects" begin
        rosen = PPNParameters(:Rosen)
        # α₁ = -2 for Rosen
        @test rosen.alpha1 == -2
        # Nordtvedt eta should be nonzero
        @test ppn_nordtvedt_eta(rosen) != 0
    end

    @testset "Observational bounds table" begin
        bounds = ppn_observational_bounds()
        @test bounds isa Dict
        @test length(bounds) == 10
        @test haskey(bounds, :gamma)
        @test haskey(bounds, :beta)
        @test haskey(bounds, :alpha1)

        # Cassini bound on gamma
        val, exp_name = bounds[:gamma]
        @test val < 1e-4
        @test occursin("Cassini", exp_name)
    end

    @testset "Large omega limit: BD → GR" begin
        bd_large = PPNParameters(:BransDicke; omega=100000)
        @test abs(ppn_deflection_factor(bd_large) - 1.0) < 1e-4
        @test abs(ppn_perihelion_factor(bd_large) - 1.0) < 1e-4
        @test abs(ppn_nordtvedt_eta(bd_large)) < 1e-4
    end

end
