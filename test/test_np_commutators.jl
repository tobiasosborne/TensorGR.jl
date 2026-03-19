@testset "NP commutator relations and directional derivatives" begin

    function _npcomm_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
            define_spinor_structure!(reg; manifold=:M4, metric=:g)
            define_null_tetrad!(reg; manifold=:M4, metric=:g)
        end
        reg
    end

    @testset "Directional derivative D = l^a nabla_a" begin
        reg = _npcomm_reg()
        with_registry(reg) do
            # Apply D to a scalar tensor
            f = Tensor(:RicScalar, TIndex[])
            Df = np_directional_derivative(:np_l, f; covd_name=:D)
            @test Df isa TProduct

            # Should have l^a and nabla_a(f)
            tensors = filter(x -> x isa Tensor, Df.factors)
            derivs = filter(x -> x isa TDeriv, Df.factors)
            @test length(tensors) == 1
            @test length(derivs) == 1
            @test tensors[1].name == :np_l
            @test tensors[1].indices[1].position == Up
            @test derivs[1].covd == :D

            # Result should be a scalar (contracted dummy)
            free = free_indices(Df)
            @test isempty(free)
        end
    end

    @testset "Directional derivative delta = m^a nabla_a" begin
        reg = _npcomm_reg()
        with_registry(reg) do
            f = Tensor(:RicScalar, TIndex[])
            delta_f = np_directional_derivative(:np_m, f; covd_name=:D)
            @test delta_f isa TProduct
            tensors = filter(x -> x isa Tensor, delta_f.factors)
            @test tensors[1].name == :np_m
        end
    end

    @testset "Directional derivative on vector" begin
        reg = _npcomm_reg()
        with_registry(reg) do
            # Apply D to a vector T^a
            T = Tensor(:np_l, [up(:a)])
            DT = np_directional_derivative(:np_l, T; covd_name=:D)
            @test DT isa TProduct
            # Should have free index a
            free = free_indices(DT)
            @test length(free) == 1
        end
    end

    @testset "Commutator table structure" begin
        table = np_commutator_table()
        @test length(table) == 4

        # First commutator: [D, Delta] = [l, n]
        @test table[1].d1 == :np_l
        @test table[1].d2 == :np_n

        # Second: [D, delta] = [l, m]
        @test table[2].d1 == :np_l
        @test table[2].d2 == :np_m

        # Third: [Delta, delta] = [n, m]
        @test table[3].d1 == :np_n
        @test table[3].d2 == :np_m

        # Fourth: [delta, deltabar] = [m, mbar]
        @test table[4].d1 == :np_m
        @test table[4].d2 == :np_mbar

        # Each should have coefficient lists for 4 directions
        for comm in table
            @test haskey(comm, :D_coeffs)
            @test haskey(comm, :Delta_coeffs)
            @test haskey(comm, :delta_coeffs)
            @test haskey(comm, :deltabar_coeffs)
        end
    end

    @testset "Commutator 1: [D,Delta] coefficients" begin
        table = np_commutator_table()
        comm1 = table[1]

        # D coefficient contains gamma terms
        gamma_present = any(p -> p[1] == :gamma_np, comm1.D_coeffs)
        @test gamma_present

        # Delta coefficient contains epsilon terms
        eps_present = any(p -> p[1] == :epsilon_np, comm1.Delta_coeffs)
        @test eps_present
    end

    @testset "Fresh indices in directional derivatives" begin
        reg = _npcomm_reg()
        with_registry(reg) do
            # Two applications should use different dummy indices
            f = Tensor(:RicScalar, TIndex[])
            Df = np_directional_derivative(:np_l, f; covd_name=:D)
            DDf = np_directional_derivative(:np_l, Df; covd_name=:D)

            @test DDf isa TProduct
            # Should still be a scalar (all indices contracted)
            free = free_indices(DDf)
            @test isempty(free)
        end
    end
end
