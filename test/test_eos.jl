@testset "Equation of State" begin

    # --- BarotropicEOS ---
    @testset "BarotropicEOS" begin
        dust = BarotropicEOS(0)
        rad  = BarotropicEOS(1//3)
        de   = BarotropicEOS(-1)

        @test dust.w == 0//1
        @test rad.w  == 1//3
        @test de.w   == -1//1

        # pressure
        @test pressure(dust, 10) == 0
        @test pressure(rad, 3)   == 1//1
        @test pressure(de, 5)    == -5//1

        # sound speed (cs^2 = w for barotropic)
        @test sound_speed(dust, 1) == 0//1
        @test sound_speed(rad, 1)  == 1//3
        @test sound_speed(de, 1)   == -1//1

        # Int constructor convenience
        @test BarotropicEOS(0).w == 0//1
    end

    # --- PolytropicEOS ---
    @testset "PolytropicEOS" begin
        eos = PolytropicEOS(1//10, 5//3)
        @test eos.K     == 1//10
        @test eos.gamma == 5//3

        # pressure: K * rho^gamma  (rho^Rational returns Float64)
        @test pressure(eos, 1) ≈ 1/10
        @test pressure(eos, 8) ≈ (1//10) * 8^(5//3)

        # sound speed: K * gamma * rho^(gamma-1)
        @test sound_speed(eos, 1) ≈ (1/10) * (5/3)

        # gamma=1 reduces to barotropic
        linear = PolytropicEOS(1//3, 1)
        @test pressure(linear, 6) ≈ 2.0   # (1//3)*6^1
        @test sound_speed(linear, 99) ≈ 1/3  # K*gamma*rho^0 = K

        # Int convenience constructors
        @test PolytropicEOS(2, 3//2).K == 2//1
        @test PolytropicEOS(1//2, 2).gamma == 2//1
        @test PolytropicEOS(1, 1).K == 1//1
    end

    # --- TabularEOS ---
    @testset "TabularEOS" begin
        eos = TabularEOS([0.0, 1.0, 2.0, 4.0], [0.0, 0.5, 1.5, 3.5])

        # Exact table points
        @test pressure(eos, 0.0) == 0.0
        @test pressure(eos, 1.0) == 0.5
        @test pressure(eos, 4.0) == 3.5

        # Linear interpolation in first interval
        @test pressure(eos, 0.5) == 0.25

        # Linear interpolation in second interval
        @test pressure(eos, 1.5) == 1.0

        # Clamping at bounds
        @test pressure(eos, -1.0) == 0.0
        @test pressure(eos, 10.0) == 3.5

        # Sound speed (finite difference)
        @test sound_speed(eos, 0.5) == 0.5   # (0.5-0.0)/(1.0-0.0)
        @test sound_speed(eos, 1.5) == 1.0   # (1.5-0.5)/(2.0-1.0)

        # Validation errors
        @test_throws ErrorException TabularEOS([1.0], [1.0])              # too few points
        @test_throws ErrorException TabularEOS([1.0, 2.0], [1.0])        # mismatched lengths
        @test_throws ErrorException TabularEOS([2.0, 1.0], [1.0, 2.0])   # unsorted
    end

    # --- PerfectFluid with EOS ---
    @testset "PerfectFluid" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            fp = define_perfect_fluid!(reg, :T; manifold=:M4, metric=:g)
            fluid = PerfectFluid(BarotropicEOS(1//3), fp)

            @test fluid.eos isa BarotropicEOS
            @test fluid.properties === fp
            @test fluid.properties.name == :T

            # Can build the stress-energy expression from properties
            expr = perfect_fluid_expr(up(:a), up(:b), fluid.properties)
            @test expr isa TSum
        end
    end

    # --- Type hierarchy ---
    @testset "type hierarchy" begin
        @test BarotropicEOS <: EquationOfState
        @test PolytropicEOS <: EquationOfState
        @test TabularEOS <: EquationOfState
    end

end
