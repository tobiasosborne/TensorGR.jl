@testset "PN potential extraction" begin

    @testset "FourierEntry construction" begin
        fe = FourierEntry(:inv_k2, "1/k²", "1/r", 1 // 4)
        @test fe.name == :inv_k2
        @test fe.coefficient == 1 // 4
    end

    @testset "FOURIER_TABLE" begin
        @test haskey(FOURIER_TABLE, :inv_k2)
        @test haskey(FOURIER_TABLE, :inv_k4)
        @test haskey(FOURIER_TABLE, :contact)

        # 1/k² → 1/(4πr), coefficient = 1/4 (π absorbed)
        @test FOURIER_TABLE[:inv_k2].coefficient == 1 // 4
    end

    @testset "fourier_transform_potential: 1/k² (Coulomb)" begin
        coeff, type = fourier_transform_potential(1, 2)
        @test type == :coulomb
        @test coeff == 1 // 4
    end

    @testset "fourier_transform_potential: 1/k⁴ (linear)" begin
        coeff, type = fourier_transform_potential(1, 4)
        @test type == :linear
        @test coeff == 1 // 8
    end

    @testset "fourier_transform_potential: contact (k⁰)" begin
        coeff, type = fourier_transform_potential(1, 0)
        @test type == :contact
    end

    @testset "fourier_transform_potential: negative power" begin
        @test_throws ErrorException fourier_transform_potential(1, -2)
    end

    @testset "PNPotentialTerm" begin
        pt = PNPotentialTerm(:coeff, 1, 0, :newtonian)
        @test pt.r_power == 1
        @test pt.pn_order == 0
        @test pt.type == :newtonian

        s = sprint(show, pt)
        @test occursin("newtonian", s)
        @test occursin("0PN", s)
    end

    @testset "classify_pn_order" begin
        # Newtonian: 1/k² × m² -> 0PN
        @test classify_pn_order(2, 0) == 0

        # 1PN: 1/k² × v² or 1/k⁴ × m²
        @test classify_pn_order(2, 2) == 1
        @test classify_pn_order(4, 0) == 1

        # 2PN: 1/k² × v⁴
        @test classify_pn_order(2, 4) == 2
    end

    @testset "newton_potential_coeff" begin
        result = newton_potential_coeff(:m1, :m2, :G)
        @test result[1] == :m1
        @test result[2] == :m2
        @test result[3] == :G
        @test result[4] == :coulomb
    end

end
