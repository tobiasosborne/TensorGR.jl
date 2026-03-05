using TensorGR: indices, free_indices, dummy_pairs, rename_dummy, fresh_index,
                ensure_no_dummy_clash

@testset "indices on Tensor" begin
    t = Tensor(:R, [down(:a), down(:b), down(:c), down(:d)])
    @test indices(t) == [down(:a), down(:b), down(:c), down(:d)]

    s = Tensor(:RicciScalar, TIndex[])
    @test isempty(indices(s))
end

@testset "indices on TProduct" begin
    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:a), down(:c), down(:b), down(:d)])
    p = TProduct(1//1, TensorExpr[g, R])
    idxs = indices(p)
    @test length(idxs) == 6
    @test idxs[1] == up(:a)
    @test idxs[3] == down(:a)
end

@testset "indices on TSum" begin
    t1 = Tensor(:R, [down(:a), down(:b)])
    t2 = Tensor(:g, [down(:a), down(:b)])
    s = TSum(TensorExpr[t1, t2])
    # indices of a sum: indices of first term (all terms should have same free indices)
    @test indices(s) == [down(:a), down(:b)]
end

@testset "indices on TDeriv" begin
    t = Tensor(:T, [up(:b), down(:c)])
    d = TDeriv(down(:a), t)
    idxs = indices(d)
    @test length(idxs) == 3
    @test idxs[1] == down(:a)
    @test idxs[2] == up(:b)
    @test idxs[3] == down(:c)
end

@testset "indices on TScalar" begin
    @test isempty(indices(TScalar(42)))
end

@testset "free_indices" begin
    # All free: R_{abcd}
    t = Tensor(:R, [down(:a), down(:b), down(:c), down(:d)])
    @test Set(free_indices(t)) == Set([down(:a), down(:b), down(:c), down(:d)])

    # Product with contraction: g^{ab} R_{acbd}
    # a appears as up and down → dummy
    # b appears as up and down → dummy
    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:a), down(:c), down(:b), down(:d)])
    p = TProduct(1//1, TensorExpr[g, R])
    fi = free_indices(p)
    @test length(fi) == 2
    @test down(:c) in fi
    @test down(:d) in fi

    # No free indices (fully contracted)
    g2 = Tensor(:g, [up(:c), up(:d)])
    p2 = TProduct(1//1, TensorExpr[g, R, g2])
    @test isempty(free_indices(p2))
end

@testset "dummy_pairs" begin
    # g^{ab} R_{acbd}: dummies are a and b
    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:a), down(:c), down(:b), down(:d)])
    p = TProduct(1//1, TensorExpr[g, R])
    dp = dummy_pairs(p)
    @test length(dp) == 2

    # Each pair should have one Up and one Down with same name
    names = Set(pair[1].name for pair in dp)
    @test :a in names
    @test :b in names
    for (u, d) in dp
        @test u.position != d.position
        @test u.name == d.name
    end
end

@testset "fresh_index" begin
    used = Set([:a, :b, :c, :d])
    f = fresh_index(used)
    @test f isa Symbol
    @test !(f in used)

    # Requesting multiple fresh indices
    used2 = Set(Symbol.('a':'z'))
    f2 = fresh_index(used2)
    @test !(f2 in used2)
end

@testset "rename_dummy" begin
    # Rename dummy :a to :z in a product
    g = Tensor(:g, [up(:a), up(:b)])
    R = Tensor(:R, [down(:a), down(:c), down(:b), down(:d)])
    p = TProduct(1//1, TensorExpr[g, R])

    p2 = rename_dummy(p, :a, :z)
    all_idxs = indices(p2)
    names = [idx.name for idx in all_idxs]
    @test :z in names
    @test count(==(:a), names) == 0  # :a should be gone (was dummy, both up and down renamed)

    # Rename in TSum
    t1 = Tensor(:R, [down(:a), down(:b)])
    t2 = Tensor(:g, [down(:a), down(:b)])
    s = TSum(TensorExpr[t1, t2])
    s2 = rename_dummy(s, :a, :z)
    @test indices(s2.terms[1]) == [down(:z), down(:b)]
end

@testset "ensure_no_dummy_clash" begin
    # Two products with the same dummy index :a
    g1 = Tensor(:g, [up(:a), up(:b)])
    t1 = Tensor(:T, [down(:a), down(:c)])
    p1 = TProduct(1//1, TensorExpr[g1, t1])  # dummies: :a

    g2 = Tensor(:g, [up(:a), up(:d)])
    t2 = Tensor(:S, [down(:a), down(:e)])
    p2 = TProduct(1//1, TensorExpr[g2, t2])  # dummies: :a (clash!)

    # After ensuring no clash, the second product's :a should be renamed
    p2_fixed = ensure_no_dummy_clash(p1, p2)
    dp1 = Set(pair[1].name for pair in dummy_pairs(p1))
    dp2 = Set(pair[1].name for pair in dummy_pairs(p2_fixed))
    @test isempty(intersect(dp1, dp2))  # no shared dummy names

    # No clash case: should return unchanged
    g3 = Tensor(:g, [up(:x), up(:y)])
    t3 = Tensor(:U, [down(:x), down(:f)])
    p3 = TProduct(1//1, TensorExpr[g3, t3])  # dummies: :x
    p3_fixed = ensure_no_dummy_clash(p1, p3)
    @test indices(p3_fixed) == indices(p3)
end
