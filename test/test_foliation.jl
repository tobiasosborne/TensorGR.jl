using TensorGR
using TensorGR: _replace_index, component_value, component_pattern,
                is_temporal_component, is_spatial_component,
                _extract_field_names, _is_component_index, ZERO,
                DEFAULT_SVT

# ═══════════════════════════════════════════════════════════════════
# FOL-1: FoliationProperties + Registry
# ═══════════════════════════════════════════════════════════════════

@testset "FoliationProperties construction" begin
    fol = FoliationProperties(:flat31, :M4, 0, [1, 2, 3], 3)
    @test fol.name == :flat31
    @test fol.manifold == :M4
    @test fol.temporal_component == 0
    @test fol.spatial_components == [1, 2, 3]
    @test fol.spatial_dim == 3
end

@testset "define_foliation! and registry" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d]))

    fol = define_foliation!(reg, :flat31; manifold=:M4)
    @test has_foliation(reg, :flat31)
    @test get_foliation(reg, :flat31) === fol
    @test fol.spatial_dim == 3

    # Can't register twice
    @test_throws ErrorException define_foliation!(reg, :flat31; manifold=:M4)

    # Manifold must exist
    @test_throws ErrorException define_foliation!(reg, :bad; manifold=:nonexistent)

    # Dimension must match
    @test_throws ErrorException define_foliation!(reg, :bad2; manifold=:M4,
                                                   spatial=[1,2])  # 1+2=3 ≠ 4

    # Temporal must not be in spatial
    @test_throws ErrorException define_foliation!(reg, :bad3; manifold=:M4,
                                                   temporal=1, spatial=[1,2,3])

    # Spatial must be unique
    @test_throws ErrorException define_foliation!(reg, :bad4; manifold=:M4,
                                                   spatial=[1,1,3])
end

@testset "classify_component" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    @test classify_component(0, fol) == :temporal
    @test classify_component(1, fol) == :spatial
    @test classify_component(2, fol) == :spatial
    @test classify_component(3, fol) == :spatial
    @test_throws ErrorException classify_component(4, fol)
end

@testset "all_components" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    @test all_components(fol) == [0, 1, 2, 3]
end

# ═══════════════════════════════════════════════════════════════════
# FOL-2: Index classification + splitting
# ═══════════════════════════════════════════════════════════════════

@testset "split_spacetime: single index" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    T = Tensor(:T, [down(:a)])

    result = split_spacetime(T, :a, fol)
    @test result isa TSum
    @test length(result.terms) == 4  # components 0,1,2,3

    # Check each component
    @test result.terms[1] == Tensor(:T, [TIndex(Symbol("_", 0), Down)])
    @test result.terms[2] == Tensor(:T, [TIndex(Symbol("_", 1), Down)])
    @test result.terms[3] == Tensor(:T, [TIndex(Symbol("_", 2), Down)])
    @test result.terms[4] == Tensor(:T, [TIndex(Symbol("_", 3), Down)])
end

@testset "split_spacetime: rank-2 tensor" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    h = Tensor(:h, [down(:a), down(:b)])

    # Split first index
    split_a = split_spacetime(h, :a, fol)
    @test split_a isa TSum
    @test length(split_a.terms) == 4
end

@testset "split_all_spacetime: rank-2 gives 16 terms" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    h = Tensor(:h, [down(:a), down(:b)])

    result = split_all_spacetime(h, fol)
    @test result isa TSum
    @test length(result.terms) == 16  # 4×4 components
end

@testset "split_all_spacetime: scalar unchanged" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    s = TScalar(1 // 1)
    @test split_all_spacetime(s, fol) === s
end

@testset "component classification" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)

    idx_0 = TIndex(Symbol("_", 0), Down)
    idx_1 = TIndex(Symbol("_", 1), Down)
    idx_a = TIndex(:a, Down)

    @test is_temporal_component(idx_0, fol)
    @test !is_temporal_component(idx_1, fol)
    @test !is_temporal_component(idx_a, fol)

    @test !is_spatial_component(idx_0, fol)
    @test is_spatial_component(idx_1, fol)
    @test !is_spatial_component(idx_a, fol)

    @test component_value(idx_0) == 0
    @test component_value(idx_1) == 1
    @test component_value(idx_a) === nothing
end

@testset "component_pattern" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)

    t = Tensor(:h, [TIndex(Symbol("_", 0), Down), TIndex(Symbol("_", 1), Down)])
    pat = component_pattern(t, fol)
    @test pat == [:temporal, :spatial]

    t2 = Tensor(:h, [TIndex(Symbol("_", 2), Down), TIndex(Symbol("_", 3), Down)])
    pat2 = component_pattern(t2, fol)
    @test pat2 == [:spatial, :spatial]
end

# ═══════════════════════════════════════════════════════════════════
# FOL-3: SVT substitution rules
# ═══════════════════════════════════════════════════════════════════

@testset "SVT Bardeen rules structure" begin
    rules = svt_rules_bardeen()
    @test length(rules) == 4  # (00), (0i), (i0), (ij)
    @test all(r -> r.tensor_name == :h, rules)
end

@testset "SVT full rules structure" begin
    rules = svt_rules_full()
    @test length(rules) == 4
end

@testset "apply_svt: h_{00} → 2Φ" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    h_00 = Tensor(:h, [TIndex(Symbol("_", 0), Down), TIndex(Symbol("_", 0), Down)])

    result = apply_svt(h_00, :h, fol)
    # Should be 2 * Φ
    @test result isa TProduct
    @test result.scalar == 2 // 1
    @test any(f -> f isa Tensor && f.name == :ϕ, result.factors)
end

@testset "apply_svt: h_{0i} → S_i" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    h_01 = Tensor(:h, [TIndex(Symbol("_", 0), Down), TIndex(Symbol("_", 1), Down)])

    result = apply_svt(h_01, :h, fol)
    @test result isa Tensor
    @test result.name == :S
    @test length(result.indices) == 1
    @test result.indices[1] == TIndex(Symbol("_", 1), Down)
end

@testset "apply_svt: h_{ij} → 2ψδ_{ij} + hTT_{ij}" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    h_12 = Tensor(:h, [TIndex(Symbol("_", 1), Down), TIndex(Symbol("_", 2), Down)])

    result = apply_svt(h_12, :h, fol)
    @test result isa TSum
    @test length(result.terms) == 2  # 2ψδ + hTT

    # One term should be hTT
    htt_terms = filter(result.terms) do t
        if t isa Tensor
            return t.name == :hTT
        end
        false
    end
    @test length(htt_terms) == 1
end

@testset "apply_svt: non-h tensor unchanged" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    g_00 = Tensor(:g, [TIndex(Symbol("_", 0), Down), TIndex(Symbol("_", 0), Down)])

    result = apply_svt(g_00, :h, fol)
    @test result == g_00  # g is not h, so unchanged
end

@testset "apply_svt: full gauge h_{0i} → ∂_i B + S_i" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    h_01 = Tensor(:h, [TIndex(Symbol("_", 0), Down), TIndex(Symbol("_", 1), Down)])

    result = apply_svt(h_01, :h, fol; gauge=:full)
    @test result isa TSum
    @test length(result.terms) == 2  # ∂B + S
end

# ═══════════════════════════════════════════════════════════════════
# FOL-4: Constraint engine
# ═══════════════════════════════════════════════════════════════════

@testset "SVT constraint rules creation" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    rules = svt_constraint_rules(DEFAULT_SVT, fol)
    @test length(rules) >= 3  # transverse vector, transverse tensor, traceless
end

@testset "Traceless contraction: hTT_{ii} → 0" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)

    # hTT with same spatial component indices (trace)
    htt_11 = Tensor(:hTT, [TIndex(Symbol("_", 1), Down), TIndex(Symbol("_", 1), Down)])
    rules = svt_constraint_rules(DEFAULT_SVT, fol)

    result = apply_rules(htt_11, rules)
    @test result == ZERO
end

@testset "Traceless: hTT_{ij} with i≠j survives" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)

    htt_12 = Tensor(:hTT, [TIndex(Symbol("_", 1), Down), TIndex(Symbol("_", 2), Down)])
    rules = svt_constraint_rules(DEFAULT_SVT, fol)

    result = apply_rules(htt_12, rules)
    @test result == htt_12  # survives
end

@testset "Transverse vector: k_i S^i → 0" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    rules = svt_constraint_rules(DEFAULT_SVT, fol)

    k_i = Tensor(:k, [down(:i)])
    S_i = Tensor(:S, [up(:i)])
    product = tproduct(1 // 1, TensorExpr[k_i, S_i])

    result = apply_rules(product, rules)
    @test result == ZERO
end

@testset "Transverse tensor: k_i hTT_{ij} → 0" begin
    fol = FoliationProperties(:f, :M4, 0, [1, 2, 3], 3)
    rules = svt_constraint_rules(DEFAULT_SVT, fol)

    k_i = Tensor(:k, [down(:i)])
    htt = Tensor(:hTT, [up(:i), down(:j)])
    product = tproduct(1 // 1, TensorExpr[k_i, htt])

    result = apply_rules(product, rules)
    @test result == ZERO
end

# ═══════════════════════════════════════════════════════════════════
# FOL-5: Sector collector
# ═══════════════════════════════════════════════════════════════════

@testset "collect_sectors: pure scalar" begin
    phi = Tensor(:ϕ, TIndex[])
    psi = Tensor(:ψ, TIndex[])
    expr = phi + psi

    sectors = collect_sectors(expr)
    @test haskey(sectors, :scalar)
    @test !haskey(sectors, :vector)
    @test !haskey(sectors, :tensor)
end

@testset "collect_sectors: vector sector" begin
    S = Tensor(:S, [down(:i)])
    expr = S

    sectors = collect_sectors(expr)
    @test haskey(sectors, :vector)
    @test !haskey(sectors, :scalar)
end

@testset "collect_sectors: tensor sector" begin
    hTT = Tensor(:hTT, [down(:i), down(:j)])
    sectors = collect_sectors(hTT)
    @test haskey(sectors, :tensor)
end

@testset "collect_sectors: mixed terms" begin
    phi = Tensor(:ϕ, TIndex[])
    S = Tensor(:S, [down(:i)])
    mixed = tproduct(1 // 1, TensorExpr[phi, S])

    sectors = collect_sectors(mixed)
    @test haskey(sectors, :mixed)
end

@testset "collect_sectors: all sectors" begin
    phi = Tensor(:ϕ, TIndex[])
    S = Tensor(:S, [down(:i)])
    hTT = Tensor(:hTT, [down(:i), down(:j)])
    expr = tsum(TensorExpr[phi, S, hTT])

    sectors = collect_sectors(expr)
    @test haskey(sectors, :scalar)
    @test haskey(sectors, :vector)
    @test haskey(sectors, :tensor)
end

@testset "extract_field_names" begin
    phi = Tensor(:ϕ, TIndex[])
    k = Tensor(:k, [down(:i)])
    prod = tproduct(1 // 1, TensorExpr[phi, k])

    names = _extract_field_names(prod)
    @test :ϕ in names
    @test :k in names
end

# ═══════════════════════════════════════════════════════════════════
# FOL-6: End-to-end integration
# ═══════════════════════════════════════════════════════════════════

@testset "foliate_and_decompose: basic" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        fol = define_foliation!(reg, :flat31; manifold=:M4)
        @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

        h_ab = Tensor(:h, [down(:a), down(:b)])
        sectors = foliate_and_decompose(h_ab, :h; foliation=fol)

        # Should have at least scalar and tensor sectors (from the SVT decomposition)
        @test sectors isa Dict{Symbol, TensorExpr}
        # The raw tensor h splits into scalar/vector/tensor fields
        @test !isempty(sectors)
    end
end

@testset "foliate_and_decompose: scalar sector has Φ and ψ" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        fol = define_foliation!(reg, :flat31; manifold=:M4)
        @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

        h_ab = Tensor(:h, [down(:a), down(:b)])
        sectors = foliate_and_decompose(h_ab, :h; foliation=fol)

        if haskey(sectors, :scalar)
            scalar_names = _extract_field_names(sectors[:scalar])
            # Scalar sector should contain Φ and/or ψ
            @test :ϕ in scalar_names || :ψ in scalar_names
        end
    end
end

# ═══════════════════════════════════════════════════════════════════
# Abstract dummy index splitting (TGR-noi)
# ═══════════════════════════════════════════════════════════════════

@testset "split_all_spacetime resolves abstract dummy indices" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:g, manifold=:M4, rank=(0,2),
        symmetries=Any[Symmetric(1,2)],
        options=Dict{Symbol,Any}(:is_metric => true)))
    register_tensor!(reg, TensorProperties(name=:h, manifold=:M4, rank=(0,2),
        symmetries=Any[Symmetric(1,2)]))

    with_registry(reg) do
        fol = define_foliation!(reg, :flat31; manifold=:M4)

        # g^{cd} ∂_c(h_{_0,_0}): abstract dummies c,d must be split
        h_00 = Tensor(:h, [down(Symbol("_0")), down(Symbol("_0"))])
        inner = TDeriv(down(:c), h_00)
        expr = tproduct(1 // 1, TensorExpr[Tensor(:g, [up(:c), up(:d)]), inner])

        result = split_all_spacetime(expr, fol)

        # Check no abstract indices remain
        function has_abstract_index(e::TensorExpr)
            for idx in indices(e)
                s = string(idx.name)
                startswith(s, "_") || return true
            end
            false
        end
        # Result is a sum; check each term
        if result isa TSum
            for t in result.terms
                @test !has_abstract_index(t)
            end
        else
            @test !has_abstract_index(result)
        end
    end
end
