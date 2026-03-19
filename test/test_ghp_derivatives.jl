@testset "GHP derivative operators" begin

    function _ghpd_reg()
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

    @testset "Weight shifts" begin
        @test ghp_weight_shift(:thorn) == GHPWeight(1, 1)
        @test ghp_weight_shift(:thorn_prime) == GHPWeight(-1, -1)
        @test ghp_weight_shift(:edth) == GHPWeight(1, -1)
        @test ghp_weight_shift(:edth_prime) == GHPWeight(-1, 1)
    end

    @testset "GHP_DERIVATIVES table" begin
        @test length(GHP_DERIVATIVES) == 4
        @test GHP_DERIVATIVES[:thorn].tetrad_vec == :np_l
        @test GHP_DERIVATIVES[:thorn_prime].tetrad_vec == :np_n
        @test GHP_DERIVATIVES[:edth].tetrad_vec == :np_m
        @test GHP_DERIVATIVES[:edth_prime].tetrad_vec == :np_mbar
    end

    @testset "thorn on scalar (weight {0,0})" begin
        reg = _ghpd_reg()
        with_registry(reg) do
            # For weight {0,0}, thorn = D (no connection terms)
            f = Tensor(:RicScalar, TIndex[])
            result = ghp_derivative(:thorn, f, GHPWeight(0, 0); covd_name=:D)
            # Should be just the directional derivative (no sum)
            @test result isa TProduct
            # Should be scalar
            @test isempty(free_indices(result))
        end
    end

    @testset "thorn on weight {1,1} quantity" begin
        reg = _ghpd_reg()
        with_registry(reg) do
            # For weight {1,1}, thorn = D - epsilon*eta - epsilon_bar*eta
            f = Tensor(:RicScalar, TIndex[])
            result = ghp_derivative(:thorn, f, GHPWeight(1, 1); covd_name=:D)
            # Should be a sum: D(f) - 1*epsilon*f - 1*epsilon_bar*f
            @test result isa TSum
            @test length(result.terms) == 3
        end
    end

    @testset "edth on weight {2,0} quantity" begin
        reg = _ghpd_reg()
        with_registry(reg) do
            f = Tensor(:RicScalar, TIndex[])
            result = ghp_derivative(:edth, f, GHPWeight(2, 0); covd_name=:D)
            # p=2, q=0: delta(f) - 2*beta*f (no q term)
            @test result isa TSum
            @test length(result.terms) == 2
        end
    end

    @testset "Result weight = input weight + shift" begin
        # Just verify the algebra
        input_weight = GHPWeight(3, -1)  # sigma_np weight
        for op in [:thorn, :thorn_prime, :edth, :edth_prime]
            shift = ghp_weight_shift(op)
            output_weight = input_weight + shift
            @test output_weight isa GHPWeight
        end

        # thorn on sigma {3,-1} -> {4,0} (same as Psi_0)
        @test GHPWeight(3, -1) + ghp_weight_shift(:thorn) == GHPWeight(4, 0)

        # edth on rho {1,1} -> {2,0} (same as Phi_01 weight, sort of)
        @test GHPWeight(1, 1) + ghp_weight_shift(:edth) == GHPWeight(2, 0)
    end

    @testset "Invalid operator name" begin
        reg = _ghpd_reg()
        with_registry(reg) do
            f = Tensor(:RicScalar, TIndex[])
            @test_throws ErrorException ghp_derivative(:invalid, f, GHPWeight(0, 0))
            @test_throws ErrorException ghp_weight_shift(:invalid)
        end
    end

    @testset "All operators produce scalar output from scalar input" begin
        reg = _ghpd_reg()
        with_registry(reg) do
            f = Tensor(:RicScalar, TIndex[])
            for op in [:thorn, :thorn_prime, :edth, :edth_prime]
                result = ghp_derivative(op, f, GHPWeight(0, 0); covd_name=:D)
                free = free_indices(result)
                @test isempty(free)
            end
        end
    end
end
