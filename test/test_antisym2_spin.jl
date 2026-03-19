@testset "Antisymmetric rank-2 spin projectors" begin

    @testset "antisym2_spin1_projector construction" begin
        P1 = antisym2_spin1_projector(down(:a), down(:b), down(:c), down(:d))
        @test P1 isa TensorExpr
    end

    @testset "antisym2_spin0_projector construction" begin
        P0 = antisym2_spin0_projector(down(:a), down(:b), down(:c), down(:d))
        @test P0 isa TensorExpr
    end

    @testset "antisym2_identity construction" begin
        A = antisym2_identity(down(:a), down(:b), down(:c), down(:d))
        @test A isa TensorExpr
    end

    @testset "completeness: P1 + P0 = A" begin
        P1 = antisym2_spin1_projector(down(:a), down(:b), down(:c), down(:d))
        P0 = antisym2_spin0_projector(down(:a), down(:b), down(:c), down(:d))
        A = antisym2_identity(down(:a), down(:b), down(:c), down(:d))

        # By construction: P0 = A - P1, so P1 + P0 = A
        total = P1 + P0

        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h]))
        with_registry(reg) do
            simp_total = simplify(total; registry=reg)
            simp_A = simplify(A; registry=reg)
            @test simp_total == simp_A
        end
    end

    @testset "custom metric" begin
        P1 = antisym2_spin1_projector(down(:a), down(:b), down(:c), down(:d);
                                       metric=:eta, k_name=:p, k_sq=:p2)
        @test P1 isa TensorExpr
    end

end
