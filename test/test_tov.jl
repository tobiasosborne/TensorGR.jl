@testset "TOV Solver" begin
    # Tests that don't need DiffEq first (struct tests etc)
    @testset "TOVSolution struct" begin
        # Test that the struct exists and can be constructed
        sol = TOVSolution(1.0, 0.5, [0.1, 0.5, 1.0], [0.01, 0.1, 0.5],
                          [1.0, 0.5, 0.0], [1.0, 0.5, 0.0], nothing)
        @test sol.r_surface == 1.0
        @test sol.M_total == 0.5
    end

    _diffeq_available = try
        @eval using DifferentialEquations
        true
    catch
        false
    end

    if _diffeq_available

    @testset "Constant density (dust): m(r) ~ (4/3)pi*rho*r^3" begin
        # Use a polytropic EOS with high K (stiff) and check mass growth.
        # Near the center, m(r) should grow as (4/3)pi*rho_c*r^3
        eos = PolytropicEOS(100//1, 2//1)
        rho_c = 0.001  # low density for near-Newtonian regime
        tov = setup_tov(eos, rho_c)
        sol = solve_tov(tov, 50.0)

        # Near the center, m(r) should grow as (4/3)pi*rho_c*r^3
        # Check at small r (first few points after r0)
        for i in 1:min(5, length(sol.r))
            r = sol.r[i]
            m_expected = (4.0/3.0) * pi * rho_c * r^3
            if r < 0.01  # very near center
                @test abs(sol.m[i] - m_expected) / max(m_expected, 1e-20) < 0.1
            end
        end

        @test sol.r_surface > 0
        @test sol.M_total > 0
    end

    @testset "Polytropic EOS (K=100, Gamma=2): finite star" begin
        eos = PolytropicEOS(100//1, 2//1)
        rho_c = 0.01
        tov = setup_tov(eos, rho_c)
        sol = solve_tov(tov, 100.0)

        @test sol.r_surface > 0
        @test sol.r_surface < 100.0  # should terminate before r_max
        @test sol.M_total > 0
        @test length(sol.r) > 2

        # Pressure should be ~0 at surface
        @test abs(sol.p[end]) < 1e-6
    end

    @testset "Buchdahl limit: 2M/R < 8/9" begin
        eos = PolytropicEOS(100//1, 2//1)
        for rho_c in [0.005, 0.01, 0.05]
            tov = setup_tov(eos, rho_c)
            sol = solve_tov(tov, 100.0)
            compactness = 2.0 * sol.M_total / sol.r_surface
            @test compactness < 8.0/9.0
        end
    end

    @testset "Mass-radius curve" begin
        eos = PolytropicEOS(100//1, 2//1)
        rho_range = range(0.005, 0.05; length=5)
        result = mass_radius_curve(eos, rho_range; r_max=100.0)

        @test length(result.R) == 5
        @test length(result.M) == 5
        @test all(result.R .> 0)
        @test all(result.M .> 0)
    end

    @testset "Surface pressure ~ 0" begin
        eos = PolytropicEOS(100//1, 2//1)
        rho_c = 0.02
        tov = setup_tov(eos, rho_c)
        sol = solve_tov(tov, 100.0)

        # Pressure at surface should be approximately 0
        @test abs(sol.p[end]) < 1e-4

        # Density at surface should also be ~0
        @test sol.rho[end] < 1e-4
    end

    else
        @info "Skipping TOV integration tests: DifferentialEquations.jl not available"
    end  # _diffeq_available
end
