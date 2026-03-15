@testset "Energy Conditions" begin

    # Helper: build a diagonal matrix
    _diag(v) = Float64[i == j ? v[i] : 0.0 for i in 1:length(v), j in 1:length(v)]

    @testset "EnergyConditionResult display" begin
        ec = EnergyConditionResult(true, true, false, true, 1.0, [0.5, 0.5, 0.5])
        s = sprint(show, ec)
        @test occursin("NEC=true", s)
        @test occursin("SEC=false", s)
    end

    @testset "Minkowski (vacuum): all conditions satisfied" begin
        dim = 4
        Ric = zeros(dim, dim)
        R = 0.0
        g = _diag([-1.0, 1.0, 1.0, 1.0])
        ginv = _diag([-1.0, 1.0, 1.0, 1.0])

        ec = check_energy_conditions(Ric, R, g, ginv; dim=dim)
        @test ec.NEC == true
        @test ec.WEC == true
        @test ec.SEC == true
        @test ec.DEC == true
        @test abs(ec.rho) < 1e-10
    end

    @testset "Schwarzschild (vacuum): all conditions satisfied" begin
        # Schwarzschild is vacuum: R_{ab} = 0, R = 0
        dim = 4
        Ric = zeros(dim, dim)
        R = 0.0
        # Metric at r=4, M=1: f = 1 - 2/4 = 0.5
        f = 0.5
        r = 4.0
        g = _diag([-f, 1.0/f, r^2, r^2])
        ginv = _diag([-1.0/f, f, 1.0/r^2, 1.0/r^2])

        ec = check_energy_conditions(Ric, R, g, ginv; dim=dim)
        @test ec.NEC == true
        @test ec.WEC == true
        @test ec.SEC == true
        @test ec.DEC == true
        @test abs(ec.rho) < 1e-10
    end

    @testset "de Sitter (Lambda > 0): NEC satisfied, SEC violated" begin
        # de Sitter: R_{ab} = Lambda * g_{ab}, R = 4 * Lambda (dim=4)
        # G_{ab} = R_{ab} - (1/2) R g_{ab} = Lambda g_{ab} - 2 Lambda g_{ab} = -Lambda g_{ab}
        # T_{ab} = -Lambda g_{ab} => rho = Lambda, p = -Lambda
        # NEC: rho + p = 0 >= 0 (marginally satisfied)
        # SEC: rho + 3p = Lambda - 3Lambda = -2Lambda < 0 (violated)
        dim = 4
        Lambda = 1.0
        g = _diag([-1.0, 1.0, 1.0, 1.0])
        ginv = _diag([-1.0, 1.0, 1.0, 1.0])
        Ric = Lambda .* g
        R = dim * Lambda

        ec = check_energy_conditions(Ric, R, g, ginv; dim=dim)
        @test ec.NEC == true
        @test ec.SEC == false
        # WEC: rho = Lambda > 0 and NEC => satisfied
        @test ec.WEC == true
        # DEC: rho = Lambda >= |p_i| = Lambda => marginally satisfied
        @test ec.DEC == true
        @test abs(ec.rho - Lambda) < 1e-8
    end

    @testset "de Sitter with small Lambda" begin
        dim = 4
        Lambda = 0.1
        g = _diag([-1.0, 1.0, 1.0, 1.0])
        ginv = _diag([-1.0, 1.0, 1.0, 1.0])
        Ric = Lambda .* g
        R = dim * Lambda

        ec = check_energy_conditions(Ric, R, g, ginv; dim=dim)
        @test ec.NEC == true
        @test ec.SEC == false
        @test abs(ec.rho - Lambda) < 1e-8
    end

    @testset "Anti-de Sitter (Lambda < 0): SEC satisfied, WEC violated" begin
        # AdS: rho = Lambda < 0, p = -Lambda > 0
        # NEC: rho + p = 0 (marginally satisfied)
        # WEC: rho < 0 (violated)
        # SEC: rho + 3p = -2Lambda > 0 (satisfied), and rho + p = 0 (satisfied)
        dim = 4
        Lambda = -1.0
        g = _diag([-1.0, 1.0, 1.0, 1.0])
        ginv = _diag([-1.0, 1.0, 1.0, 1.0])
        Ric = Lambda .* g
        R = dim * Lambda

        ec = check_energy_conditions(Ric, R, g, ginv; dim=dim)
        @test ec.NEC == true
        @test ec.SEC == true
        @test ec.WEC == false
        @test ec.DEC == false
    end

    @testset "Perfect fluid: dust (rho > 0, p = 0)" begin
        # T^a_b = diag(-rho, p, p, p) = diag(-rho, 0, 0, 0)
        rho = 1.0
        dim = 4
        T_mixed = _diag([-rho, 0.0, 0.0, 0.0])
        g = _diag([-1.0, 1.0, 1.0, 1.0])

        ec = check_energy_conditions(T_mixed, g; dim=dim)
        @test ec.NEC == true
        @test ec.WEC == true
        @test ec.SEC == true
        @test ec.DEC == true
        @test abs(ec.rho - rho) < 1e-10
    end

    @testset "Perfect fluid: radiation (rho > 0, p = rho/3)" begin
        # T^a_b = diag(-rho, p, p, p)
        rho = 3.0
        p = 1.0  # rho/3
        dim = 4
        T_mixed = _diag([-rho, p, p, p])
        g = _diag([-1.0, 1.0, 1.0, 1.0])

        ec = check_energy_conditions(T_mixed, g; dim=dim)
        @test ec.NEC == true     # rho + p = 4 > 0
        @test ec.WEC == true     # rho > 0 and NEC
        @test ec.SEC == true     # rho + 3p = 6 > 0
        @test ec.DEC == true     # rho = 3 >= |p| = 1
    end

    @testset "Phantom energy (rho > 0, p < -rho): NEC violated" begin
        # Phantom dark energy: w = p/rho < -1
        rho = 1.0
        p = -2.0  # w = -2
        dim = 4
        T_mixed = _diag([-rho, p, p, p])
        g = _diag([-1.0, 1.0, 1.0, 1.0])

        ec = check_energy_conditions(T_mixed, g; dim=dim)
        @test ec.NEC == false    # rho + p = -1 < 0
        @test ec.WEC == false    # NEC violated => WEC violated
        @test ec.SEC == false    # rho + 3p = -5 < 0
        @test ec.DEC == false    # |p| = 2 > rho = 1
    end

    @testset "check_energy_conditions from T_mixed interface" begin
        dim = 4
        T_mixed = zeros(dim, dim)
        g = _diag([-1.0, 1.0, 1.0, 1.0])

        ec = check_energy_conditions(T_mixed, g; dim=dim)
        @test ec.NEC == true
        @test ec.WEC == true
        @test ec.SEC == true
        @test ec.DEC == true
    end

    @testset "3D spacetime" begin
        # 2+1 gravity with cosmological constant
        dim = 3
        Lambda = 1.0
        g = _diag([-1.0, 1.0, 1.0])
        ginv = _diag([-1.0, 1.0, 1.0])
        Ric = Lambda .* g
        R = dim * Lambda

        ec = check_energy_conditions(Ric, R, g, ginv; dim=dim)
        # G_{ab} = Lambda g_{ab} - (3/2) Lambda g_{ab} = -(1/2) Lambda g_{ab}
        # T^a_b = -(1/2) Lambda delta^a_b
        # rho = (1/2) Lambda, p = -(1/2) Lambda
        @test ec.NEC == true
        @test abs(ec.rho - Lambda / 2) < 1e-8
    end

    @testset "Anisotropic pressures" begin
        # T^a_b = diag(-rho, p1, p2, p3) with different pressures
        rho = 2.0
        dim = 4
        T_mixed = _diag([-rho, 1.0, 0.5, -0.5])
        g = _diag([-1.0, 1.0, 1.0, 1.0])

        ec = check_energy_conditions(T_mixed, g; dim=dim)
        # NEC: rho + p_i >= 0 => 3, 2.5, 1.5 all >= 0
        @test ec.NEC == true
        @test ec.WEC == true
        # SEC: rho + sum(p) = 2 + 1 = 3 >= 0
        @test ec.SEC == true
        # DEC: rho >= |p_i| => 2 >= 1, 2 >= 0.5, 2 >= 0.5
        @test ec.DEC == true
    end

    @testset "DEC violation with large pressure" begin
        rho = 1.0
        dim = 4
        T_mixed = _diag([-rho, 1.5, 0.0, 0.0])
        g = _diag([-1.0, 1.0, 1.0, 1.0])

        ec = check_energy_conditions(T_mixed, g; dim=dim)
        # NEC: rho + p_1 = 2.5 >= 0
        @test ec.NEC == true
        @test ec.WEC == true
        # DEC: rho = 1 < |p_1| = 1.5 => violated
        @test ec.DEC == false
    end

end
