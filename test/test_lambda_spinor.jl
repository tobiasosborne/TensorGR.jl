@testset "Scalar curvature spinor Lambda" begin

    function _spin_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f]))
        define_curvature_tensors!(reg, :M4, :g)
        reg
    end

    @testset "define_lambda_spinor!" begin
        reg = _spin_reg()
        with_registry(reg) do
            define_lambda_spinor!(reg)
            @test has_tensor(reg, :Lambda_spin)

            tp = get_tensor(reg, :Lambda_spin)
            @test tp.rank == (0, 0)
            @test tp.options[:is_curvature_spinor] == true
        end
    end

    @testset "lambda_spinor_expr" begin
        reg = _spin_reg()
        with_registry(reg) do
            define_lambda_spinor!(reg)
            expr = lambda_spinor_expr(; registry=reg)
            @test expr isa TensorExpr
            # Should be (1/24) * RicScalar
            @test expr isa TProduct
            @test expr.scalar == 1 // 24
        end
    end

    @testset "Lambda = R/24 definition" begin
        reg = _spin_reg()
        with_registry(reg) do
            define_lambda_spinor!(reg)
            # The expression Lambda = (1/24) R should be correctly formed
            expr = lambda_spinor_expr(; registry=reg)
            # It's (1/24) * RicScalar
            @test expr isa TProduct
            @test expr.scalar == 1 // 24
            @test length(expr.factors) == 1
            @test expr.factors[1] isa Tensor
            @test expr.factors[1].name == :RicScalar
        end
    end

    @testset "Lambda display" begin
        Lambda = Tensor(:Lambda_spin, TIndex[])
        s = sprint(show, Lambda)
        @test occursin("Lambda_spin", s)
    end

end
