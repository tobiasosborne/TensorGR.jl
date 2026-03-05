@testset "Phase 4: Component Calculations" begin

    @testset "4.1: Define chart" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M2, 2, :g, :∂, [:a,:b]))

        with_registry(reg) do
            chart = define_chart!(reg, :polar; manifold=:M2, coords=[:r, :θ])
            @test chart.name == :polar
            @test chart.coords == [:r, :θ]
            @test chart.manifold == :M2
        end
    end

    @testset "4.2: CTensor basics" begin
        g = CTensor(Float64[1 0; 0 1], :cart)
        @test size(g) == (2, 2)
        @test g[1, 1] == 1.0
        @test g[1, 2] == 0.0

        # Scalar multiplication
        g2 = 2.0 * g
        @test g2[1, 1] == 2.0

        # Addition
        g3 = g + g
        @test g3[1, 1] == 2.0
    end

    @testset "4.2: CTensor trace" begin
        # Identity matrix trace = 2
        id = CTensor(Float64[1 0; 0 1], :cart, [Up, Down])
        result = ctensor_trace(id, 1, 2)
        @test result.data[1] == 2.0
    end

    @testset "4.4: Flat metric Christoffel = 0" begin
        # Flat 2D metric: g = diag(1,1), all Christoffels vanish
        g = Float64[1 0; 0 1]
        ginv = Float64[1 0; 0 1]
        coords = [:x, :y]

        # deriv_fn returns 0 for constant metric
        deriv_fn(expr, coord) = 0.0

        Gamma = metric_christoffel(g, ginv, coords; deriv_fn=deriv_fn)
        for a in 1:2, b in 1:2, c in 1:2
            @test Gamma[a, b, c] == 0
        end
    end

    @testset "4.4: Polar coordinates Christoffel" begin
        # 2D polar: ds² = dr² + r²dθ²
        # g = [1 0; 0 r²], g^{-1} = [1 0; 0 1/r²]
        # Non-zero Christoffels: Γ^r_{θθ} = -r, Γ^θ_{rθ} = Γ^θ_{θr} = 1/r
        # Using r=1 for simplicity: Γ^1_{22} = -1, Γ^2_{12} = Γ^2_{21} = 1

        r_val = 2.0
        g = [1.0 0.0; 0.0 r_val^2]
        ginv = [1.0 0.0; 0.0 1.0/r_val^2]
        coords = [:r, :θ]

        # Derivatives: ∂_r(g_{θθ}) = ∂_r(r²) = 2r, all others = 0
        function polar_deriv(expr, coord)
            if expr == r_val^2 && coord == :r
                return 2 * r_val
            end
            return 0.0
        end

        Gamma = metric_christoffel(g, ginv, coords; deriv_fn=polar_deriv)

        # Γ^r_{θθ} = Gamma[1,2,2] = -r
        @test Gamma[1, 2, 2] ≈ -r_val

        # Γ^θ_{rθ} = Gamma[2,1,2] = 1/r
        @test Gamma[2, 1, 2] ≈ 1.0/r_val

        # Γ^θ_{θr} = Gamma[2,2,1] = 1/r
        @test Gamma[2, 2, 1] ≈ 1.0/r_val

        # All other Christoffels should be zero
        @test Gamma[1, 1, 1] ≈ 0.0
        @test Gamma[1, 1, 2] ≈ 0.0
        @test Gamma[2, 1, 1] ≈ 0.0
    end
end
