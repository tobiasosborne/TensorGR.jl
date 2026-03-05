@testset "Integration V2: Full Pipeline Tests" begin

    @testset "Full GR setup via macros" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @define_tensor V on=M4 rank=(1,0)
            @define_tensor W on=M4 rank=(0,1)
            @covd D on=M4 metric=g

            @test has_manifold(reg, :M4)
            @test has_tensor(reg, :g)
            @test has_tensor(reg, :δ)
            @test has_tensor(reg, :Riem)
            @test has_tensor(reg, :Ric)
            @test has_tensor(reg, :V)
            @test has_tensor(reg, :W)
            @test has_tensor(reg, :ΓD)
        end
    end

    @testset "g^{ab} g_{bc} = δ^a_c via simplify" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            # δ already registered by @manifold

            result = simplify(Tensor(:g, [up(:a), up(:b)]) *
                             Tensor(:g, [down(:b), down(:c)]))
            @test result isa Tensor
            @test result.name == :δ
            @test result.indices == [up(:a), down(:c)]
        end
    end

    @testset "δ^a_a = 4" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g

            result = simplify(Tensor(:δ, [up(:a), down(:a)]))
            @test result == TScalar(4 // 1)
        end
    end

    @testset "Riemann antisymmetry: R_{abcd} + R_{bacd} = 0" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)

            R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R2 = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])
            result = simplify(R1 + R2)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Riemann pair symmetry: R_{abcd} = R_{cdab}" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)

            R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R2 = Tensor(:Riem, [down(:c), down(:d), down(:a), down(:b)])
            result = simplify(R1 - R2)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Symmetric tensor T_{ab} + T_{ba} = 2T_{ab}" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor T on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            T1 = Tensor(:T, [down(:a), down(:b)])
            T2 = Tensor(:T, [down(:b), down(:a)])
            result = simplify(T1 + T2)
            @test result isa TProduct
            @test result.scalar == 2 // 1
        end
    end

    @testset "Antisymmetric tensor A_{ab} + A_{ba} = 0" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor A on=M4 rank=(0,2) symmetry=AntiSymmetric(1,2)

            A1 = Tensor(:A, [down(:a), down(:b)])
            A2 = Tensor(:A, [down(:b), down(:a)])
            result = simplify(A1 + A2)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Perturbation + simplify pipeline" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g

            mp = define_metric_perturbation!(reg, :g, :h)

            # δ(g^{ab}) at order 1 should give -g^{ac}g^{bd}h_{cd}
            result = δinverse_metric(mp, up(:a), up(:b), 1)
            @test result isa TProduct
            @test result.scalar == -1 // 1
            @test length(result.factors) == 3
        end
    end

    @testset "Derivative canonicalization in product" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor T on=M4 rank=(0,2) symmetry=Symmetric(1,2)
            @define_tensor V on=M4 rank=(1,0)

            # ∂_a(T_{cb}) * V^b should canonicalize T's indices
            d_T = TDeriv(down(:a), Tensor(:T, [down(:c), down(:b)]))
            prod = d_T * Tensor(:V, [up(:b)])
            result = canonicalize(prod)

            # T should have indices sorted: T[-b,-c]
            @test result isa TProduct
            has_sorted_T = any(result.factors) do f
                if f isa TDeriv && f.arg isa Tensor && f.arg.name == :T
                    return f.arg.indices[1].name == :b && f.arg.indices[2].name == :c
                end
                false
            end
            @test has_sorted_T
        end
    end

    @testset "CovD expansion + Bianchi" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            for r in bianchi_rules()
                register_rule!(reg, r)
            end

            # ∇^a G_{ab} = 0
            expr = TDeriv(up(:a), Tensor(:Ein, [down(:a), down(:b)]))
            result = simplify(expr)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Lie derivative Leibniz on product" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor V on=M4 rank=(1,0)
            @define_tensor W on=M4 rank=(0,1)
            @define_tensor ξ on=M4 rank=(1,0)

            xi = Tensor(:ξ, [up(:a)])
            product = Tensor(:V, [up(:b)]) * Tensor(:W, [down(:c)])
            result = lie_derivative(xi, product)

            # Leibniz: £_ξ(V*W) = (£_ξ V)*W + V*(£_ξ W)
            @test result isa TSum
            @test length(result.terms) == 2
        end
    end

    @testset "Exterior form antisymmetry via canonicalize" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_form!(reg, :F; manifold=:M4, degree=2)

            # F_{ba} = -F_{ab} for a 2-form
            F1 = Tensor(:F, [down(:a), down(:b)])
            F2 = Tensor(:F, [down(:b), down(:a)])
            result = simplify(F1 + F2)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Weyl decomposition structure" begin
        a, b, c, d = down(:a), down(:b), down(:c), down(:d)
        expr = riemann_to_weyl(a, b, c, d, :g; dim=4)
        # Should contain Weyl, Ricci, and RicciScalar terms
        @test expr isa TSum

        # And the inverse
        expr2 = weyl_to_riemann(a, b, c, d, :g; dim=4)
        @test expr2 isa TSum
    end

    @testset "Make ansatz and verify structure" begin
        T1 = Tensor(:Ric, [down(:a), down(:b)])
        T2 = Tensor(:g, [down(:a), down(:b)]) * Tensor(:RicScalar, TIndex[])

        ansatz = make_ansatz(TensorExpr[T1, T2], [:α, :β])
        @test ansatz isa TSum
        @test length(ansatz.terms) == 2
    end
end
