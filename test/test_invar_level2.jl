@testset "Invar Level 2: Cyclic symmetry (Bianchi identity)" begin

    function _gr_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l,:m,:n]))
        define_curvature_tensors!(reg, :M4, :g)
        reg
    end

    @testset "bianchi_relation: structure" begin
        reg = _gr_reg()
        with_registry(reg) do
            expr = bianchi_relation(down(:a), down(:b), down(:c), down(:d))
            # Three Riemann terms with cycled indices
            @test expr isa TSum
            @test length(expr.terms) == 3

            # After canonicalization, terms should have canonical index ordering
            # but NOT cancel (Bianchi is algebraic, not a permutation symmetry)
            canon = simplify_level1(expr; registry=reg)
            @test canon isa TensorExpr
        end
    end

    @testset "bianchi_relation structure" begin
        expr = bianchi_relation(down(:a), down(:b), down(:c), down(:d))
        @test expr isa TSum
        @test length(expr.terms) == 3
    end

    @testset "simplify_level2 delegates to simplify" begin
        reg = _gr_reg()
        with_registry(reg) do
            # A simple Riemann expression should be canonical after level 2
            R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify_level2(R; registry=reg)
            @test result isa TensorExpr
        end
    end

    @testset "Bianchi contracted: structure check" begin
        reg = _gr_reg()
        with_registry(reg) do
            # (R_{abcd} + R_{acdb} + R_{adbc}) R^{abcd}
            # This creates a sum of 3 Kretschner-like terms
            bianchi = bianchi_relation(down(:a), down(:b), down(:c), down(:d))
            R_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            contracted = bianchi * R_up

            @test contracted isa TensorExpr
            # After simplification, the 3 terms should be scalar invariants
            result = simplify(contracted; registry=reg)
            @test isempty(free_indices(result))
        end
    end

    @testset "degree-2 invariants: 3 independent in d=4" begin
        reg = _gr_reg()
        with_registry(reg) do
            # The three independent degree-2 invariants in d=4:
            # 1. R² (Ricci scalar squared)
            R = Tensor(:RicScalar, TIndex[])
            R_sq = R * R

            # 2. R_{ab}R^{ab} (Ricci squared)
            Ric_down = Tensor(:Ric, [down(:a), down(:b)])
            Ric_up = Tensor(:Ric, [up(:a), up(:b)])
            Ric_sq = Ric_down * Ric_up

            # 3. R_{abcd}R^{abcd} (Kretschner scalar)
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            Kretschner = Riem_down * Riem_up

            # All three should be non-zero after simplification
            @test simplify(R_sq; registry=reg) != TScalar(0 // 1)
            @test simplify(Ric_sq; registry=reg) != TScalar(0 // 1)
            @test simplify(Kretschner; registry=reg) != TScalar(0 // 1)

            # All three should be different after simplification
            s1 = simplify(R_sq; registry=reg)
            s2 = simplify(Ric_sq; registry=reg)
            s3 = simplify(Kretschner; registry=reg)
            @test s1 != s2
            @test s2 != s3
            @test s1 != s3
        end
    end

end
