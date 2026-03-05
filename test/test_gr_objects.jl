using TensorGR: define_curvature_tensors!, expand_products, collect_terms,
                einstein_expr, ricci_from_riemann

@testset "Define curvature tensors" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing,
        [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[Symmetric(1,2)],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))

    with_registry(reg) do
        define_curvature_tensors!(reg, :M4, :g)

        @test has_tensor(reg, :Riem)
        @test has_tensor(reg, :Ric)
        @test has_tensor(reg, :RicScalar)
        @test has_tensor(reg, :Ein)
        @test has_tensor(reg, :Weyl)
        @test has_tensor(reg, :Sch)
        @test has_tensor(reg, :δ)

        # Check symmetries
        @test get_tensor(reg, :Riem).symmetries[1] isa RiemannSymmetry
        @test get_tensor(reg, :Ric).symmetries[1] isa Symmetric
    end
end

@testset "Riemann symmetries via canonicalize" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing,
        [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[Symmetric(1,2)],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    define_curvature_tensors!(reg, :M4, :g)

    with_registry(reg) do
        R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])

        # Antisymmetry in first pair: R_{bacd} = -R_{abcd}
        R_bacd = TProduct(1//1, TensorExpr[
            Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])])
        r1 = canonicalize(R_bacd)
        @test r1 isa TProduct
        @test r1.scalar == -1//1

        # Pair symmetry: R_{cdab} = R_{abcd}
        R_cdab = TProduct(1//1, TensorExpr[
            Tensor(:Riem, [down(:c), down(:d), down(:a), down(:b)])])
        r2 = canonicalize(R_cdab)
        @test r2 == R
    end
end

@testset "Einstein tensor expression" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing,
        [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[Symmetric(1,2)],
        options=Dict{Symbol,Any}(:is_metric => true, :metric_inverse => :g)))
    define_curvature_tensors!(reg, :M4, :g)

    with_registry(reg) do
        G = einstein_expr(down(:a), down(:b), :g)
        @test G isa TSum
        @test length(G.terms) == 2
    end
end

@testset "expand_products and collect_terms" begin
    # (A + B) * (C + D) should expand to 4 terms
    A = Tensor(:A, [down(:a)])
    B = Tensor(:B, [down(:a)])
    C = Tensor(:C, [up(:b)])
    D = Tensor(:D, [up(:b)])

    expr = (A + B) * (C + D)
    expanded = expand_products(expr)
    # Should have 4 terms
    @test expanded isa TSum
    @test length(expanded.terms) == 4

    # collect_terms: 2R + 3R = 5R
    R = Tensor(:R, [down(:a), down(:b)])
    s = (2 * R) + (3 * R)
    collected = collect_terms(s)
    @test collected isa TProduct
    @test collected.scalar == 5//1
end

@testset "collect_terms cancellation" begin
    R = Tensor(:R, [down(:a), down(:b)])
    s = R + (-1 * R)
    collected = collect_terms(s)
    @test collected == TScalar(0//1)
end
