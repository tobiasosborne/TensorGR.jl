@testset "Rank-3 spin projectors" begin

    @testset "antisym3_identity construction" begin
        A = antisym3_identity(down(:a), down(:b), down(:c),
                              down(:d), down(:e), down(:f))
        @test A isa TensorExpr
    end

    @testset "antisym3_spin0_projector construction" begin
        P0 = antisym3_spin0_projector(down(:a), down(:b), down(:c),
                                       down(:d), down(:e), down(:f))
        @test P0 isa TensorExpr
    end

    @testset "antisym3_spin1_projector construction" begin
        P1 = antisym3_spin1_projector(down(:a), down(:b), down(:c),
                                       down(:d), down(:e), down(:f))
        @test P1 isa TensorExpr
    end

    @testset "completeness: P1 + P0 = A" begin
        P1 = antisym3_spin1_projector(down(:a), down(:b), down(:c),
                                       down(:d), down(:e), down(:f))
        P0 = antisym3_spin0_projector(down(:a), down(:b), down(:c),
                                       down(:d), down(:e), down(:f))
        A = antisym3_identity(down(:a), down(:b), down(:c),
                              down(:d), down(:e), down(:f))

        # By construction: P1 = A - P0, so P1 + P0 = A
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        with_registry(reg) do
            total = simplify(P1 + P0; registry=reg)
            A_simp = simplify(A; registry=reg)
            @test total == A_simp
        end
    end

    @testset "custom kwargs" begin
        P1 = antisym3_spin1_projector(down(:a), down(:b), down(:c),
                                       down(:d), down(:e), down(:f);
                                       metric=:eta, k_name=:p, k_sq=:p2)
        @test P1 isa TensorExpr
    end

end
