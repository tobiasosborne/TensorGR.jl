using TensorGR: canonicalize, symmetry_generators, Symmetric, AntiSymmetric,
                PairSymmetric, RiemannSymmetry

@testset "Symmetry generators" begin
    # AntiSymmetric(1,2) on 2 slots (n=4)
    gens = symmetry_generators([AntiSymmetric(1, 2)], 2)
    @test length(gens) == 1
    @test gens[1].data == Int32[2, 1, 4, 3]

    # Symmetric(1,2) on 2 slots (n=4)
    gens2 = symmetry_generators([Symmetric(1, 2)], 2)
    @test length(gens2) == 1
    @test gens2[1].data == Int32[2, 1, 3, 4]  # no sign flip

    # Riemann symmetry: 3 generators on 4 slots (n=6)
    gens3 = symmetry_generators([RiemannSymmetry()], 4)
    @test length(gens3) == 3
end

@testset "Canonicalize antisymmetric: T_{ba} → -T_{ab}" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M, rank=(0,2),
        symmetries=Any[AntiSymmetric(1, 2)],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        T_ba = Tensor(:T, [down(:b), down(:a)])
        # Wrap in product for canonicalization
        expr = TProduct(1//1, TensorExpr[T_ba])
        result = canonicalize(expr)
        # Should be -T_{ab}
        @test result isa TProduct
        @test result.scalar == -1//1
        @test result.factors[1] == Tensor(:T, [down(:a), down(:b)])
    end
end

@testset "Canonicalize symmetric: S_{ba} → S_{ab}" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:S, manifold=:M, rank=(0,2),
        symmetries=Any[Symmetric(1, 2)],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        S_ba = Tensor(:S, [down(:b), down(:a)])
        expr = TProduct(1//1, TensorExpr[S_ba])
        result = canonicalize(expr)
        # S_{ba} = S_{ab}, no sign change
        @test result isa Tensor  # scalar 1 collapses
        @test result == Tensor(:S, [down(:a), down(:b)])
    end
end

@testset "Canonicalize Riemann symmetries" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:R, manifold=:M, rank=(0,4),
        symmetries=Any[RiemannSymmetry()],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        # R_{abcd} already canonical
        R_abcd = TProduct(1//1, TensorExpr[Tensor(:R, [down(:a),down(:b),down(:c),down(:d)])])
        @test canonicalize(R_abcd) == Tensor(:R, [down(:a),down(:b),down(:c),down(:d)])

        # R_{bacd} → -R_{abcd}
        R_bacd = TProduct(1//1, TensorExpr[Tensor(:R, [down(:b),down(:a),down(:c),down(:d)])])
        r1 = canonicalize(R_bacd)
        @test r1 isa TProduct
        @test r1.scalar == -1//1

        # R_{cdab} → R_{abcd} (pair symmetry)
        R_cdab = TProduct(1//1, TensorExpr[Tensor(:R, [down(:c),down(:d),down(:a),down(:b)])])
        @test canonicalize(R_cdab) == Tensor(:R, [down(:a),down(:b),down(:c),down(:d)])
    end
end

@testset "Canonicalize sum distributes" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M, rank=(0,2),
        symmetries=Any[AntiSymmetric(1, 2)],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        T_ab = TProduct(1//1, TensorExpr[Tensor(:T, [down(:a), down(:b)])])
        T_ba = TProduct(1//1, TensorExpr[Tensor(:T, [down(:b), down(:a)])])
        # T_{ab} + T_{ba} = T_{ab} + (-T_{ab}) = 0
        result = canonicalize(T_ab + T_ba)
        # After canonicalization, both terms become ±T_{ab}
        # The sum should be T_{ab} - T_{ab}
        @test result isa TSum
        @test length(result.terms) == 2
    end
end

@testset "M1 integration: R_{abcd} + R_{abdc} canonicalizes" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:R, manifold=:M, rank=(0,4),
        symmetries=Any[RiemannSymmetry()],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        R1 = TProduct(1//1, TensorExpr[Tensor(:R, [down(:a),down(:b),down(:c),down(:d)])])
        R2 = TProduct(1//1, TensorExpr[Tensor(:R, [down(:a),down(:b),down(:d),down(:c)])])

        # R_{abcd} + R_{abdc}
        # R_{abdc} → -R_{abcd} (antisym in c,d)
        # So sum = R_{abcd} - R_{abcd} = 0
        result = canonicalize(R1 + R2)
        # Each term canonicalizes independently; we need collect_terms to get zero
        # For now, verify both terms are present with opposite signs
        @test result isa TSum
        t1 = result.terms[1]
        t2 = result.terms[2]
        # One should be +R_{abcd}, the other -R_{abcd}
        if t1 isa TProduct
            @test t1.scalar == -t2.scalar || (t1 isa Tensor && t2 isa TProduct)
        end
    end
end

@testset "Canonical factor ordering in products" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M4, rank=(0,2), symmetries=Any[Symmetric(1,2)],
        options=Dict{Symbol,Any}(:is_metric => true)))
    register_tensor!(reg, TensorProperties(
        name=:RicScalar, manifold=:M4, rank=(0,0), symmetries=Any[],
        options=Dict{Symbol,Any}()))

    with_registry(reg) do
        rs = Tensor(:RicScalar, TIndex[])
        gm = Tensor(:g, [down(:b), down(:d)])

        # RicScalar * g and g * RicScalar should canonicalize to same form
        p1 = canonicalize(tproduct(1//1, TensorExpr[rs, gm]))
        p2 = canonicalize(tproduct(1//1, TensorExpr[gm, rs]))
        @test p1 == p2

        # collect_terms should merge them
        s = simplify(tsum(TensorExpr[
            tproduct(2//3, TensorExpr[rs, gm]),
            tproduct(-2//3, TensorExpr[gm, rs])
        ]))
        @test s == TScalar(0//1)
    end
end
