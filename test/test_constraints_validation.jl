using TensorGR: define_curvature_tensors!, enforce_tracefree, enforce_divfree

@testset "Constraints validation: Weyl identities" begin
    # Shared setup: 4D manifold with metric, curvature tensors, and covariant derivative
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)
        @covd D on=M4 metric=g

        # Register Weyl as trace-free on all cross-pairs (Riemann symmetry
        # makes these equivalent: any contraction across the two antisymmetric
        # pairs vanishes)
        set_tracefree!(reg, :Weyl; pairs=[(1,3), (1,4), (2,3), (2,4)])

        # In vacuum, Weyl is divergence-free: D^a C_{abcd} = 0
        set_divfree!(reg, :Weyl; covd=:D, index=1)

        @testset "Trace-free: g^{ac} C_{abcd} = 0" begin
            g_up = Tensor(:g, [up(:a), up(:c)])
            C = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify(g_up * C; registry=reg)
            @test result isa TScalar
            @test result.val == 0
        end

        @testset "Trace-free: g^{bd} C_{abcd} = 0" begin
            g_up = Tensor(:g, [up(:b), up(:d)])
            C = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify(g_up * C; registry=reg)
            @test result isa TScalar
            @test result.val == 0
        end

        @testset "Trace-free: g^{ad} C_{abcd} = 0 (cross-pair 1,4)" begin
            g_up = Tensor(:g, [up(:a), up(:d)])
            C = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify(g_up * C; registry=reg)
            @test result isa TScalar
            @test result.val == 0
        end

        @testset "Self-trace: C^a_{ba}_d = 0 via enforce_tracefree" begin
            C_traced = Tensor(:Weyl, [up(:a), down(:b), down(:a), down(:d)])
            result = enforce_tracefree(C_traced; registry=reg)
            @test result isa TScalar
            @test result.val == 0
        end

        @testset "Divergence-free (vacuum): D_a C^a_{bcd} = 0" begin
            expr = TDeriv(down(:a),
                Tensor(:Weyl, [up(:a), down(:b), down(:c), down(:d)]), :D)
            result = simplify(expr; registry=reg)
            @test result isa TScalar
            @test result.val == 0
        end

        @testset "Divergence-free via enforce_divfree directly" begin
            expr = TDeriv(down(:a),
                Tensor(:Weyl, [up(:a), down(:b), down(:c), down(:d)]), :D)
            result = enforce_divfree(expr; registry=reg)
            @test result isa TScalar
            @test result.val == 0
        end

        @testset "Negative: Ricci trace is NOT zero" begin
            g_up = Tensor(:g, [up(:a), up(:b)])
            Ric = Tensor(:Ric, [down(:a), down(:b)])
            result = simplify(g_up * Ric; registry=reg)
            @test !(result isa TScalar && result.val == 0)
        end

        @testset "Double trace: g^{ac} g^{bd} C_{abcd} = 0" begin
            g_ac = Tensor(:g, [up(:a), up(:c)])
            g_bd = Tensor(:g, [up(:b), up(:d)])
            C = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify(g_ac * g_bd * C; registry=reg)
            @test result isa TScalar
            @test result.val == 0
        end

        @testset "Weyl antisymmetry: C_{bacd} = -C_{abcd}" begin
            C_abcd = TProduct(1//1, TensorExpr[
                Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])])
            C_bacd = TProduct(1//1, TensorExpr[
                Tensor(:Weyl, [down(:b), down(:a), down(:c), down(:d)])])
            r = canonicalize(C_bacd)
            @test r isa TProduct
            @test r.scalar == -1//1
        end

        @testset "Weyl pair symmetry: C_{cdab} = C_{abcd}" begin
            C_abcd = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            C_cdab = TProduct(1//1, TensorExpr[
                Tensor(:Weyl, [down(:c), down(:d), down(:a), down(:b)])])
            r = canonicalize(C_cdab)
            # Pair swap has sign +1, so canonical form collapses to bare Tensor
            @test r isa Tensor
            @test r.name == :Weyl
        end
    end
end
