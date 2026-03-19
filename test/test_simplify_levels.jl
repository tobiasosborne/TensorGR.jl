@testset "Invar Level 1: Permutation symmetries" begin

    function _gr_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l,:m,:n]))
        define_curvature_tensors!(reg, :M4, :g)
        reg
    end

    @testset "simplify_level1: single Riemann" begin
        reg = _gr_reg()
        with_registry(reg) do
            # R_{abcd} should be canonical already
            R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            canon = simplify_level1(R; registry=reg)
            @test canon isa TensorExpr
        end
    end

    @testset "simplify_level1: Riemann pair symmetry" begin
        reg = _gr_reg()
        with_registry(reg) do
            # R_{cdab} = R_{abcd} (pair exchange symmetry)
            R1 = Tensor(:Riem, [down(:c), down(:d), down(:a), down(:b)])
            R2 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])

            c1 = simplify_level1(R1; registry=reg)
            c2 = simplify_level1(R2; registry=reg)
            @test c1 == c2
        end
    end

    @testset "simplify_level1: Riemann antisymmetry" begin
        reg = _gr_reg()
        with_registry(reg) do
            # R_{bacd} = -R_{abcd}
            R_ba = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])
            R_ab = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])

            c_ba = simplify_level1(R_ba; registry=reg)
            c_ab = simplify_level1(R_ab; registry=reg)

            # Should differ by a sign (or be canonical forms that show this)
            result = simplify(c_ba + c_ab; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "simplify_level1: Ricci symmetry" begin
        reg = _gr_reg()
        with_registry(reg) do
            # R_{ba} = R_{ab}
            Ric_ba = Tensor(:Ric, [down(:b), down(:a)])
            Ric_ab = Tensor(:Ric, [down(:a), down(:b)])

            c_ba = simplify_level1(Ric_ba; registry=reg)
            c_ab = simplify_level1(Ric_ab; registry=reg)
            @test c_ba == c_ab
        end
    end

    @testset "simplify_level1: Riemann product" begin
        reg = _gr_reg()
        with_registry(reg) do
            # Product of two Riemanns should canonicalize
            R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R2 = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            prod = R1 * R2

            canon = simplify_level1(prod; registry=reg)
            @test canon isa TensorExpr
        end
    end

    @testset "is_riemann_monomial" begin
        reg = _gr_reg()
        with_registry(reg) do
            # Kretschner scalar: R_{abcd} R^{abcd}
            R_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = R_down * R_up
            @test is_riemann_monomial(kretschner)

            # Ricci scalar squared
            R = Tensor(:RicScalar, TIndex[])
            @test is_riemann_monomial(R * R)

            # Not a monomial: has free indices
            @test !is_riemann_monomial(R_down)

            # Not a monomial: has derivatives
            @test !is_riemann_monomial(TDeriv(down(:e), R))
        end
    end

    @testset "count_riemann_degree" begin
        R_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
        R_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
        R = Tensor(:RicScalar, TIndex[])

        @test count_riemann_degree(R) == 1
        @test count_riemann_degree(R_down * R_up) == 2
        @test count_riemann_degree(R * R * R) == 3
        @test count_riemann_degree(TScalar(42)) == 0
        @test count_riemann_degree(Tensor(:g, [down(:a), down(:b)])) == 0
    end

end
