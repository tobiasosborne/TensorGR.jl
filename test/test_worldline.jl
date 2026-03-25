# Tests for src/worldline/worldline.jl
# EFTofPNG worldline formalism: point-particle trajectories, PN order counting

@testset "Worldline" begin

    @testset "Worldline construction" begin
        wl = Worldline(:A)
        @test wl.label == :A
        @test wl.parameter == :τ
        @test wl.manifold == :M4
        @test wl.velocity == :vA
        @test wl.position == :xA

        wl2 = Worldline(:B; parameter=:s, manifold=:M3)
        @test wl2.parameter == :s
        @test wl2.manifold == :M3
        @test wl2.velocity == :vB
    end

    @testset "define_worldline! registers velocity" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a, :b, :c, :d]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))

        wl = Worldline(:A)
        with_registry(reg) do
            result = define_worldline!(reg, wl)
            @test result === wl
            @test has_tensor(reg, :vA)
            props = get_tensor(reg, :vA)
            @test props.rank == (1, 0)
            @test props.options[:is_velocity] == true
        end
    end

    @testset "pn_order counting" begin
        v = Tensor(:v, [up(:a)])
        T = Tensor(:T, [down(:a), down(:b)])
        s = TScalar(2)

        @test pn_order(v, :v) == 1
        @test pn_order(T, :v) == 0
        @test pn_order(s, :v) == 0

        # Product: v * v = O(v^2), so pn_order = 2
        vv = tproduct(1 // 1, TensorExpr[
            Tensor(:v, [up(:a)]), Tensor(:v, [up(:b)])])
        @test pn_order(vv, :v) == 2

        # Product: T * v = O(v^1), so pn_order = 1
        Tv = tproduct(1 // 1, TensorExpr[T, v])
        @test pn_order(Tv, :v) == 1

        # Sum: max of term orders
        expr = vv + Tv
        @test pn_order(expr, :v) == 2

        # Derivative: passes through
        dv = TDeriv(down(:c), v)
        @test pn_order(dv, :v) == 1
    end

    @testset "truncate_pn" begin
        v = Tensor(:v, [up(:a)])
        T = Tensor(:T, [down(:a)])

        # v * v = O(v^2) = 1PN, T = 0PN
        vv = tproduct(1 // 1, TensorExpr[
            Tensor(:v, [up(:a)]), Tensor(:v, [up(:b)])])

        # Sum of 0PN and 1PN terms
        expr = T + vv

        # Truncate at 0PN (max velocity power = 0): should drop vv
        result0 = truncate_pn(expr, 0, :v)
        @test result0 isa TensorExpr
        # Only T should remain
        @test pn_order(result0, :v) == 0

        # Truncate at 1PN (max velocity power = 2): should keep both
        result1 = truncate_pn(expr, 1, :v)
        @test pn_order(result1, :v) <= 2
    end

end
