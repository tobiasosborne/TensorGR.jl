@testset "NP spin coefficients" begin

    function _sc_reg()
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

    simple_names = [:kappa, :sigma_np, :lambda_np, :nu_np,
                    :rho_np, :mu_np, :tau_np, :pi_np]
    compound_names = [:epsilon_np, :gamma_np, :alpha_np, :beta_np]

    @testset "Simple spin coefficients" begin
        reg = _sc_reg()
        with_registry(reg) do
            for name in simple_names
                sc = spin_coefficient(name; covd_name=:D)
                @test sc isa TProduct

                # Should be a scalar (no free indices)
                free = free_indices(sc)
                @test isempty(free)

                # Should have 3 factors: v1^a, v2^b, nabla_b(v3_a)
                @test length(sc.factors) == 3

                # Two tensors and one derivative
                tensors = filter(f -> f isa Tensor, sc.factors)
                derivs = filter(f -> f isa TDeriv, sc.factors)
                @test length(tensors) == 2
                @test length(derivs) == 1

                # The derivative should be of a tetrad vector
                @test derivs[1].arg isa Tensor
                @test derivs[1].arg.name in (:np_l, :np_n, :np_m, :np_mbar)
                @test derivs[1].covd == :D
            end
        end
    end

    @testset "Compound spin coefficients (epsilon, gamma, alpha, beta)" begin
        reg = _sc_reg()
        with_registry(reg) do
            for name in compound_names
                sc = spin_coefficient(name; covd_name=:D)
                @test sc isa TSum
                @test length(sc.terms) == 2

                # Each term should be a scalar
                for term in sc.terms
                    @test term isa TProduct
                    free = free_indices(term)
                    @test isempty(free)
                end
            end
        end
    end

    @testset "Specific coefficient: kappa = m^a l^b nabla_b l_a" begin
        reg = _sc_reg()
        with_registry(reg) do
            kappa = spin_coefficient(:kappa; covd_name=:D)
            tensors = filter(f -> f isa Tensor, kappa.factors)
            # v1 = m, v2 = l
            @test tensors[1].name == :np_m
            @test tensors[2].name == :np_l
            # Derivative of l
            deriv = first(f for f in kappa.factors if f isa TDeriv)
            @test deriv.arg.name == :np_l
        end
    end

    @testset "Specific coefficient: nu = mbar^a n^b nabla_b n_a" begin
        reg = _sc_reg()
        with_registry(reg) do
            nu = spin_coefficient(:nu_np; covd_name=:D)
            tensors = filter(f -> f isa Tensor, nu.factors)
            @test tensors[1].name == :np_mbar
            @test tensors[2].name == :np_n
            deriv = first(f for f in nu.factors if f isa TDeriv)
            @test deriv.arg.name == :np_n
        end
    end

    @testset "all_spin_coefficients returns all 12" begin
        reg = _sc_reg()
        with_registry(reg) do
            all_sc = all_spin_coefficients(; covd_name=:D)
            @test length(all_sc) == 12
            for name in vcat(simple_names, compound_names)
                @test haskey(all_sc, name)
            end
        end
    end

    @testset "Invalid coefficient name" begin
        reg = _sc_reg()
        with_registry(reg) do
            @test_throws ErrorException spin_coefficient(:invalid_name)
        end
    end
end
