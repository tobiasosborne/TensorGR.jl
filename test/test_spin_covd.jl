@testset "Spin covariant derivative" begin

    function _scovd_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
            define_spinor_structure!(reg; manifold=:M4, metric=:g)

            # Register a test spinor phi_A
            register_tensor!(reg, TensorProperties(
                name=:phi_spin, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle => :SL2C,
                                          :index_vbundles => [:SL2C])))
        end
        reg
    end

    @testset "spin_covd produces sigma * nabla product" begin
        reg = _scovd_reg()
        with_registry(reg) do
            phi = Tensor(:phi_spin, [spin_down(:B)])
            result = spin_covd(phi, :A, :Ap; covd_name=:D)

            # Should be a TProduct of sigma and TDeriv
            @test result isa TProduct
            @test length(result.factors) == 2

            # One factor should be sigma, the other TDeriv
            has_sigma = any(f -> f isa Tensor && f.name == :sigma, result.factors)
            has_deriv = any(f -> f isa TDeriv, result.factors)
            @test has_sigma
            @test has_deriv

            # The sigma should have three indices: Tangent, SL2C, SL2C_dot
            sigma_factor = first(f for f in result.factors if f isa Tensor && f.name == :sigma)
            @test length(sigma_factor.indices) == 3
            @test sigma_factor.indices[2].name == :A
            @test sigma_factor.indices[2].vbundle == :SL2C
            @test sigma_factor.indices[3].name == :Ap
            @test sigma_factor.indices[3].vbundle == :SL2C_dot

            # The TDeriv should differentiate phi
            deriv_factor = first(f for f in result.factors if f isa TDeriv)
            @test deriv_factor.arg isa Tensor
            @test deriv_factor.arg.name == :phi_spin
            @test deriv_factor.covd == :D
        end
    end

    @testset "spin_covd_expr generates fresh indices" begin
        reg = _scovd_reg()
        with_registry(reg) do
            phi = Tensor(:phi_spin, [spin_down(:B)])
            result = spin_covd_expr(phi; covd_name=:D)

            @test result isa TProduct
            sigma_factor = first(f for f in result.factors if f isa Tensor && f.name == :sigma)

            # Indices should be fresh (not :B which is already used)
            sl2c_idx = sigma_factor.indices[2]
            @test sl2c_idx.vbundle == :SL2C
            @test sl2c_idx.name != :B
        end
    end

    @testset "spin_covd free index structure" begin
        reg = _scovd_reg()
        with_registry(reg) do
            phi = Tensor(:phi_spin, [spin_down(:B)])
            result = spin_covd(phi, :A, :Ap; covd_name=:D)

            # Free indices should be: B (from phi), A (from sigma), Ap (from sigma)
            free = free_indices(result)
            free_names = Set(idx.name for idx in free)
            @test :B in free_names
            @test :A in free_names
            @test :Ap in free_names
            @test length(free) == 3
        end
    end

    @testset "Tangent index is dummy (contracted)" begin
        reg = _scovd_reg()
        with_registry(reg) do
            phi = Tensor(:phi_spin, [spin_down(:B)])
            result = spin_covd(phi, :A, :Ap; covd_name=:D)

            # The Tangent index should be a dummy pair (Up in sigma, Down in deriv)
            dummies = dummy_pairs(result)
            tangent_dummies = filter(p -> p[1].vbundle == :Tangent, dummies)
            @test length(tangent_dummies) == 1
        end
    end
end
