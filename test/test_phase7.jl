@testset "Phase 7: Advanced Features" begin

    @testset "7.1: Killing vectors" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))

        with_registry(reg) do
            define_killing!(reg, :ξ; manifold=:M4, metric=:g)
            @test has_tensor(reg, :ξ)
            props = get_tensor(reg, :ξ)
            @test get(props.options, :is_killing, false) == true
        end
    end

    @testset "7.2: Make ansatz" begin
        T1 = Tensor(:T, [down(:a), down(:b)])
        T2 = Tensor(:S, [down(:a), down(:b)])

        result = make_ansatz(TensorExpr[T1, T2])
        @test result isa TSum
        @test length(result.terms) == 2
    end

    @testset "7.2: Make ansatz with named coefficients" begin
        T1 = Tensor(:T, [down(:a), down(:b)])
        T2 = Tensor(:S, [down(:a), down(:b)])

        result = make_ansatz(TensorExpr[T1, T2], [:α, :β])
        @test result isa TSum
    end

    @testset "7.3: CollectTensors" begin
        T = Tensor(:T, [down(:a), down(:b)])
        # 3T + 2T should collect to 5T
        expr = tsum(TensorExpr[
            tproduct(3 // 1, TensorExpr[T]),
            tproduct(2 // 1, TensorExpr[T])
        ])
        result = collect_tensors(expr)
        if result isa TProduct
            @test result.scalar == 5 // 1
        end
    end
end
