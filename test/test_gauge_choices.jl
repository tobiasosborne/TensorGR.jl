@testset "Cosmological gauge choices" begin

    @testset "gauge_choice lookup" begin
        for name in [:synchronous, :newtonian, :flat_slicing, :comoving, :uniform_density]
            gc = gauge_choice(name)
            @test gc isa GaugeChoice
            @test gc.name == name
        end
    end

    @testset "gauge_choice: unknown" begin
        @test_throws ErrorException gauge_choice(:unknown_gauge)
    end

    @testset "synchronous_gauge" begin
        gc = synchronous_gauge()
        @test gc.name == :synchronous
        @test length(gc.vanishing_fields) == 3  # ϕ, B, S
    end

    @testset "newtonian_gauge" begin
        gc = newtonian_gauge()
        @test gc.name == :newtonian
        @test length(gc.vanishing_fields) == 2  # B, E
    end

    @testset "flat_slicing_gauge" begin
        gc = flat_slicing_gauge()
        @test gc.name == :flat_slicing
        @test length(gc.vanishing_fields) == 2  # ψ, E
    end

    @testset "comoving_gauge" begin
        gc = comoving_gauge()
        @test gc.name == :comoving
        @test length(gc.vanishing_fields) == 1  # B
    end

    @testset "uniform_density_gauge" begin
        gc = uniform_density_gauge()
        @test gc.name == :uniform_density
        @test isempty(gc.vanishing_fields)  # δρ is matter, not SVT
    end

    @testset "GaugeChoice display" begin
        gc = newtonian_gauge()
        s = sprint(show, gc)
        @test occursin("newtonian", s)
    end

    @testset "apply_gauge!" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))

        with_registry(reg) do
            # Register B and E as tensors
            register_tensor!(reg, TensorProperties(
                name=:B_svt, manifold=:M4, rank=(0,0), symmetries=SymmetrySpec[]))
            register_tensor!(reg, TensorProperties(
                name=:E_svt, manifold=:M4, rank=(0,0), symmetries=SymmetrySpec[]))

            # Apply Newtonian gauge (sets B, E to vanish)
            # But our default SVT field names may differ from :B_svt
            # Test with custom fields
            fields = SVTFields(:phi, :B_svt, :psi, :E_svt, :S, :F, :hTT)
            gc = newtonian_gauge(; fields=fields)
            apply_gauge!(reg, gc)

            @test get_tensor(reg, :B_svt).vanishing
            @test get_tensor(reg, :E_svt).vanishing
        end
    end

end
