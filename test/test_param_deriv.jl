@testset "Parametric Derivatives (TParamDeriv)" begin
    using TensorGR: TensorExpr, Tensor, TProduct, TSum, TDeriv, TScalar,
                    TParamDeriv, TIndex, up, down, tproduct, tsum,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    ManifoldProperties, register_manifold!, register_tensor!,
                    define_parameter!, is_parameter, param_deriv,
                    expand_param_deriv, with_registry, indices,
                    free_indices, derivative_order, walk, children,
                    to_latex, to_unicode, dagger

    @testset "Smart constructor" begin
        reg = TensorRegistry()
        with_registry(reg) do
            register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))
            register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,2)))

            T_ab = Tensor(:T, [down(:a), down(:b)])

            # Single parameter
            d = param_deriv(:t, T_ab)
            @test d isa TParamDeriv
            @test d.params == [:t]
            @test d.arg == T_ab

            # Parameters sorted
            d2 = param_deriv([:s, :t], T_ab)
            @test d2.params == [:s, :t]

            d3 = param_deriv([:t, :s], T_ab)
            @test d3.params == [:s, :t]

            # Flatten nested
            inner = param_deriv(:t, T_ab)
            outer = param_deriv(:s, inner)
            @test outer isa TParamDeriv
            @test outer.params == [:s, :t]
            @test outer.arg == T_ab

            # Zero propagation
            z = param_deriv(:t, TScalar(0 // 1))
            @test z == TScalar(0 // 1)

            # Empty params returns arg
            @test param_deriv(Symbol[], T_ab) === T_ab
        end
    end

    @testset "Indices (index-free)" begin
        T_ab = Tensor(:T, [down(:a), down(:b)])
        d = TParamDeriv([:t], T_ab)

        # ParamDeriv carries no indices of its own
        @test indices(d) == [down(:a), down(:b)]
        @test free_indices(d) == [down(:a), down(:b)]
    end

    @testset "AST traversal" begin
        T_ab = Tensor(:T, [down(:a), down(:b)])
        d = TParamDeriv([:t], T_ab)

        # children
        @test children(d) == TensorExpr[T_ab]

        # walk
        walked = walk(d) do node
            if node isa Tensor && node.name == :T
                Tensor(:S, node.indices)
            else
                node
            end
        end
        @test walked isa TParamDeriv
        @test walked.arg.name == :S

        # derivative_order
        @test derivative_order(d) == 1
        d2 = TParamDeriv([:s, :t], T_ab)
        @test derivative_order(d2) == 2
    end

    @testset "Display" begin
        T_ab = Tensor(:T, [down(:a), down(:b)])
        d = TParamDeriv([:t], T_ab)

        # Base.show
        s = sprint(show, d)
        @test occursin("D[t]", s)
        @test occursin("T", s)

        # to_unicode
        u = to_unicode(d)
        @test occursin("d/d(t)", u)

        # to_latex
        l = to_latex(d)
        @test occursin("\\dot{", l)
    end

    @testset "Equality and hashing" begin
        T_ab = Tensor(:T, [down(:a), down(:b)])
        d1 = TParamDeriv([:t], T_ab)
        d2 = TParamDeriv([:t], T_ab)
        d3 = TParamDeriv([:s], T_ab)

        @test d1 == d2
        @test d1 != d3
        @test hash(d1) == hash(d2)
        @test hash(d1) != hash(d3)
    end

    @testset "Dagger" begin
        T_ab = Tensor(:T, [down(:a), down(:b)])
        d = TParamDeriv([:t], T_ab)
        dd = dagger(d)
        @test dd isa TParamDeriv
        @test dd.params == [:t]
        @test dd.arg.name == :T_dag
    end

    @testset "define_parameter!" begin
        reg = TensorRegistry()
        with_registry(reg) do
            register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))
            define_parameter!(reg, :t)
            @test is_parameter(reg, :t)
            @test !is_parameter(reg, :x)

            # Double registration errors
            @test_throws ErrorException define_parameter!(reg, :t)
        end
    end

    @testset "expand_param_deriv" begin
        reg = TensorRegistry()
        with_registry(reg) do
            register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))
            register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,2)))
            register_tensor!(reg, TensorProperties(name=:V, manifold=:M4, rank=(1,0)))
            define_parameter!(reg, :t)

            T_ab = Tensor(:T, [down(:a), down(:b)])
            V_c = Tensor(:V, [up(:c)])

            # Linearity over sums
            s = TSum([T_ab, T_ab])
            ds = expand_param_deriv(param_deriv(:t, s); registry=reg)
            @test ds isa TSum
            @test length(ds.terms) == 2

            # Leibniz on product
            prod = tproduct(1 // 1, TensorExpr[T_ab, V_c])
            dprod = expand_param_deriv(param_deriv(:t, prod); registry=reg)
            @test dprod isa TSum
            @test length(dprod.terms) == 2

            # Zero on rational constant
            c = TScalar(3 // 1)
            dc = expand_param_deriv(param_deriv(:t, c); registry=reg)
            @test dc == TScalar(0 // 1)

            # Parameter self-derivative: d/dt(t) = 1
            t_param = Tensor(:t, TIndex[])
            dt = expand_param_deriv(param_deriv(:t, t_param); registry=reg)
            @test dt == TScalar(1 // 1)

            # Commutation with partial derivative
            dtd = TDeriv(down(:a), T_ab)
            result = expand_param_deriv(param_deriv(:t, dtd); registry=reg)
            @test result isa TDeriv
            @test result.arg isa TParamDeriv
        end
    end
end
