@testset "Vector field spin projectors" begin

    @testset "vector_spin1_projector construction" begin
        P1 = vector_spin1_projector(down(:a), down(:b))
        @test P1 isa TensorExpr
        # θ_{ab} = η_{ab} - k_a k_b / k²
        # Should be a TSum of two terms
        @test P1 isa TSum
        @test length(P1.terms) == 2
    end

    @testset "vector_spin0_projector construction" begin
        P0 = vector_spin0_projector(down(:a), down(:b))
        @test P0 isa TensorExpr
        # ω_{ab} = k_a k_b / k²
        @test P0 isa TProduct
    end

    @testset "completeness: P1 + P0 = eta" begin
        P1 = vector_spin1_projector(down(:a), down(:b))
        P0 = vector_spin0_projector(down(:a), down(:b))

        # P1 + P0 should equal η_{ab}
        total = P1 + P0
        @test total isa TSum

        # The sum should simplify: θ + ω = η - k·k/k² + k·k/k² = η
        # This requires simplification, so just verify structure
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f]))
        with_registry(reg) do
            simplified = simplify(total; registry=reg)
            # After simplification, should be a single metric tensor
            @test simplified isa Tensor
            @test simplified.name == :g
        end
    end

    @testset "custom kwargs" begin
        P1 = vector_spin1_projector(down(:a), down(:b);
                                    metric=:eta, k_name=:p, k_sq=:p2)
        @test P1 isa TensorExpr

        P0 = vector_spin0_projector(down(:a), down(:b);
                                    k_name=:p, k_sq=:p2)
        @test P0 isa TensorExpr
    end

    @testset "vector_spin_project" begin
        # Build a simple vector kernel K_{ab} = η_{ab}
        K = Tensor(:g, [down(:a), down(:b)])

        proj1 = vector_spin_project(K, :spin1)
        @test proj1 isa TensorExpr

        proj0 = vector_spin_project(K, :spin0)
        @test proj0 isa TensorExpr
    end

    @testset "vector_spin_project: invalid spin" begin
        K = Tensor(:g, [down(:a), down(:b)])
        @test_throws ErrorException vector_spin_project(K, :spin2)
    end

end
