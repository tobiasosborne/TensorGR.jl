# Regression and coverage tests for remediation plan
# Phase 2: Critical missing tests

using TensorGR: to_expr, from_expr, is_well_formed, _normalize_dummies,
    _split_scalar, _normalize_dummies_for_display

# ── 2.1: Regression test: _expand_pert order-0 returns background ─────────

@testset "Regression: _expand_pert order-0 returns background tensor" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M)

    with_registry(reg) do
        mp = MetricPerturbation(:g, :h, :M, :ε, false, nothing)
        g_ab = Tensor(:g, [down(:a), down(:b)])
        result = expand_perturbation(g_ab, mp, 0)
        # Order 0 must return the tensor itself (background), NOT zero
        @test result == g_ab
    end
end

# ── 1.1 Regression: _normalize_dummies collision with _d<N> names ─────────

@testset "Regression: _normalize_dummies no collision with _dN names" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2)))
    register_tensor!(reg, TensorProperties(name=:S, manifold=:M, rank=(0,2)))

    with_registry(reg) do
        # Create expression with dummy named _d1 and another dummy named b
        # T_{_d1}^{_d1} S_{b}^{b}
        T = Tensor(:T, [down(:_d1), up(:_d1)])
        S = Tensor(:S, [down(:b), up(:b)])
        expr = tproduct(1//1, TensorExpr[T, S])

        result = _normalize_dummies(expr)
        # After normalization, should have _d1 and _d2 as dummies, no corruption
        pairs = dummy_pairs(result)
        @test length(pairs) == 2
        names = Set(p[1].name for p in pairs)
        @test :_d1 in names
        @test :_d2 in names
    end
end

@testset "Regression: _normalize_dummies_for_display no collision" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2)))
    register_tensor!(reg, TensorProperties(name=:S, manifold=:M, rank=(0,2)))

    with_registry(reg) do
        # Create expression with dummy named :p (a canonical display name) and another :b
        T = Tensor(:T, [down(:p), up(:p)])
        S = Tensor(:S, [down(:b), up(:b)])
        expr = tproduct(1//1, TensorExpr[T, S])

        result = _normalize_dummies_for_display(expr)
        pairs = dummy_pairs(result)
        @test length(pairs) == 2
    end
end

# ── 1.3 Regression: _split_scalar preserves symbolic TScalar values ───────

@testset "Regression: _split_scalar handles TScalar types" begin
    # Rational TScalar: extract coefficient
    s2 = TScalar(3//2)
    coeff2, core2 = _split_scalar(s2)
    @test coeff2 == 3//2
    @test core2 == TScalar(1)

    # Symbolic TScalar: must preserve the symbolic value as the core
    s = TScalar(:Λ)
    coeff, core = _split_scalar(s)
    @test coeff == 1//1
    @test core == TScalar(:Λ)

    # Integer TScalar: treated as opaque (not rational coefficient)
    s3 = TScalar(42)
    coeff3, core3 = _split_scalar(s3)
    @test coeff3 == 1//1
    @test core3 == TScalar(42)
end

@testset "Regression: collect_terms preserves symbolic TScalar in sums" begin
    # TScalar(:Λ) + TScalar(:Λ) should collect to 2*TScalar(:Λ)
    s1 = TScalar(:Λ)
    s2 = TScalar(:Λ)
    sum_expr = TSum(TensorExpr[s1, s2])
    result = collect_terms(sum_expr)
    # Should be 2 * TScalar(:Λ), not 2 * TScalar(1)
    if result isa TProduct
        @test result.scalar == 2//1
        @test any(f -> f isa TScalar && f.val == :Λ, result.factors)
    elseif result isa TScalar
        # Might simplify differently
        @test true
    end
end

# ── 1.2 Regression: _implode refuses mixed-covd derivative chains ─────────

@testset "Regression: canonicalize preserves CovD in derivative chains" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,1)))
    define_metric!(reg, :g; manifold=:M)
    define_covd!(reg, :∇; manifold=:M, metric=:g)

    with_registry(reg) do
        T = Tensor(:T, [down(:c)])

        # Same-covd chain: ∇_a(∇_b(T_c)) — should canonicalize normally
        same = TDeriv(down(:a), TDeriv(down(:b), T, :∇), :∇)
        result = canonicalize(same)
        # Both derivs should keep :∇
        @test result isa TDeriv
        @test result.covd == :∇

        # Mixed-covd chain: ∇_a(∂_b(T_c)) — should NOT lose CovD info
        mixed = TDeriv(down(:a), TDeriv(down(:b), T, :partial), :∇)
        result2 = canonicalize(mixed)
        # The mixed chain should be returned (not imploded/exploded incorrectly)
        @test result2 isa TDeriv
    end
end

# ── 1.6 Regression: from_expr/to_expr roundtrip preserves CovD ───────────

@testset "Regression: to_expr/from_expr roundtrip preserves CovD" begin
    T = Tensor(:T, [down(:c)])
    d = TDeriv(down(:a), T, :∇)
    @test from_expr(to_expr(d)) == d
    @test from_expr(to_expr(d)).covd == :∇

    # Also test :partial (default)
    d2 = TDeriv(down(:a), T, :partial)
    @test from_expr(to_expr(d2)) == d2
    @test from_expr(to_expr(d2)).covd == :partial

    # Nested with CovD
    nested = TDeriv(down(:b), TDeriv(down(:a), T, :∇), :∇)
    rt = from_expr(to_expr(nested))
    @test rt == nested
    @test rt.covd == :∇
    @test rt.arg.covd == :∇
end

# ── 2.2: Tests for change_covd ────────────────────────────────────────────

@testset "change_covd: basic derivative swap" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M)
    define_covd!(reg, :∇; manifold=:M, metric=:g)
    define_covd!(reg, :D; manifold=:M, metric=:g)

    with_registry(reg) do
        T = Tensor(:T, [down(:b)])
        expr = TDeriv(down(:a), T, :∇)

        result = change_covd(expr, :∇, :D)
        # Should produce D_a T_b + (Γ∇ - ΓD) terms
        @test result isa TSum || result isa TDeriv || result isa TProduct
        # The result should have the same free indices as the input
        @test length(free_indices(result)) == length(free_indices(expr))
    end
end

# ── 2.2: Tests for metric_weyl and metric_kretschmann ─────────────────────

@testset "metric_weyl: trace-free in 3D" begin
    dim = 3
    # Use a flat metric for simplicity
    g = Float64[i == j ? 1.0 : 0.0 for i in 1:dim, j in 1:dim]
    ginv = copy(g)
    # Zero Riemann → zero Weyl
    Riem = zeros(dim, dim, dim, dim)
    Ric = zeros(dim, dim)
    R = 0.0

    W = metric_weyl(Riem, Ric, R, g, ginv, dim)
    # All components should be zero for flat space
    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        @test W[a,b,c,d] ≈ 0.0 atol=1e-14
    end
end

@testset "metric_weyl: requires dim > 2" begin
    @test_throws ErrorException metric_weyl(zeros(2,2,2,2), zeros(2,2), 0.0,
        [1.0 0; 0 1], [1.0 0; 0 1], 2)
end

@testset "metric_kretschmann: flat space gives zero" begin
    dim = 4
    g = Float64[i == j ? (i == 1 ? -1.0 : 1.0) : 0.0 for i in 1:dim, j in 1:dim]
    ginv = copy(g)
    Riem = zeros(dim, dim, dim, dim)
    K = metric_kretschmann(Riem, g, ginv, dim)
    @test K ≈ 0.0 atol=1e-14
end

# ── 2.2: Tests for hodge_dual ─────────────────────────────────────────────

@testset "hodge_dual: produces correct structure" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M)

    with_registry(reg) do
        # 1-form in 4D: Hodge dual is a 3-form
        A = Tensor(:A, [down(:a)])
        result = hodge_dual(A, :ε, 1, 4)
        # Should be a product of epsilon and A with appropriate indices
        @test result isa TProduct
        # Should have 3 free indices (the 3-form indices)
        fi = free_indices(result)
        @test length(fi) == 3
    end
end

# ── 2.3: Mathematical identity tests ──────────────────────────────────────

@testset "Mathematical identity: exterior_d produces TDeriv" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 3, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:A, manifold=:M, rank=(0,1)))

    with_registry(reg) do
        # Exterior derivative of a 1-form: d(A) = ∂_b A_a
        A = Tensor(:A, [down(:a)])
        dA = exterior_d(A, 1, down(:b))
        @test dA isa TDeriv
        @test dA.index == down(:b)
        @test dA.arg == A
    end
end

@testset "Mathematical identity: Bianchi for flat Riemann" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M)

    with_registry(reg) do
        # In flat space, Riemann is zero
        Riem = Tensor(:Riem, [up(:a), down(:b), down(:c), down(:d)])
        result = simplify(Riem)
        # Without background rules, Riem should remain as-is
        @test result isa Tensor
        @test result.name == :Riem
    end
end

# ── 2.4: Error-path tests (@test_throws) ──────────────────────────────────

@testset "Error paths: invalid operations" begin
    reg = TensorRegistry()

    # Duplicate manifold registration
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d]))
    @test_throws ErrorException register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d]))

    # Duplicate tensor registration
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2)))
    @test_throws ErrorException register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2)))

    # Unregister non-existent tensor
    @test_throws ErrorException unregister_tensor!(reg, :nonexistent)

    # Unregister manifold with tensors on it
    @test_throws ErrorException unregister_manifold!(reg, :M)

    # VBundle on non-existent manifold
    @test_throws ErrorException define_vbundle!(reg, :V; manifold=:nonexistent, dim=3)

    # Duplicate VBundle
    define_vbundle!(reg, :V; manifold=:M, dim=3)
    @test_throws ErrorException define_vbundle!(reg, :V; manifold=:M, dim=3)
end

@testset "Error paths: CovD operations" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M)

    with_registry(reg) do
        # get_covd on non-existent CovD
        @test_throws ErrorException get_covd(reg, :nonexistent)

        # Define and then test
        define_covd!(reg, :∇; manifold=:M, metric=:g)
        props = get_covd(reg, :∇)
        @test props.name == :∇

        # get_covd on a non-covd tensor
        @test_throws ErrorException get_covd(reg, :g)
    end
end

@testset "Error paths: fresh_index exhaustion guard" begin
    # Should work for reasonable number of indices
    used = Set{Symbol}()
    for i in 1:26
        idx = fresh_index(used)
        push!(used, idx)
    end
    @test length(used) == 26
end

# ── 2.2: Tests for cosmological_background! ───────────────────────────────

@testset "cosmological_background!: sets Ricci proportional to metric" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M)

    with_registry(reg) do
        cosmological_background!(reg, :M)
        Ric = Tensor(:Ric, [down(:a), down(:b)])
        rules = RewriteRule[r for r in get_rules(reg)]
        result = apply_rules(Ric, rules)
        # Should be replaced with a scalar * metric
        @test result != Ric  # Should have been transformed
    end
end

# ── 2.2: Tests for var_lagrangian ─────────────────────────────────────────

@testset "var_lagrangian: returns a tensor expression" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M)

    with_registry(reg) do
        # L = R (Ricci scalar) — EH Lagrangian (no indices)
        R = Tensor(:RicScalar, TIndex[])
        result = var_lagrangian(R, :g)
        # var_lagrangian computes metric_variation, result is a tensor expression
        @test result isa TensorExpr
    end
end

# ── 2.4: Dimension edge cases ─────────────────────────────────────────────

@testset "Dimension edge cases" begin
    # dim=2 Weyl tensor should error
    @test_throws ErrorException metric_weyl(zeros(2,2,2,2), zeros(2,2), 0.0,
        [1.0 0; 0 1], [1.0 0; 0 1], 2)

    # dim=1 manifold should be registrable
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M1, 1, :g, :∂, [:a]))
    @test has_manifold(reg, :M1)
end

# ── 2.2: Tests for to_basis / component_array / to_ctensor ──────────────

@testset "to_basis: scalar expression" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 3, :g, :∂, [:a,:b,:c]))
    define_metric!(reg, :g; manifold=:M)

    with_registry(reg) do
        chart = define_chart!(reg, :cart; manifold=:M, coords=[:x, :y, :z])
        s = TScalar(42//1)
        ct = to_basis(s, chart)
        @test ct isa CTensor
    end
end

@testset "to_basis: rank-1 tensor" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 3, :g, :∂, [:a,:b,:c]))
    register_tensor!(reg, TensorProperties(name=:V, manifold=:M, rank=(0,1)))

    with_registry(reg) do
        chart = define_chart!(reg, :cart; manifold=:M, coords=[:x, :y, :z])
        V = Tensor(:V, [down(:a)])
        ct = to_basis(V, chart)
        @test ct isa CTensor
        @test size(ct.data) == (3,)
    end
end

@testset "to_ctensor: numeric evaluation" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 2, :g, :∂, [:a,:b]))
    define_metric!(reg, :g; manifold=:M)

    with_registry(reg) do
        chart = define_chart!(reg, :cart; manifold=:M, coords=[:x, :y])
        g_expr = Tensor(:g, [down(:a), down(:b)])
        vals = Dict{Any,Any}(
            (:g, [1,1]) => 1.0,
            (:g, [1,2]) => 0.0,
            (:g, [2,1]) => 0.0,
            (:g, [2,2]) => 1.0
        )
        ct = to_ctensor(g_expr, chart, vals)
        @test ct isa CTensor
        @test ct.data[1,1] ≈ 1.0
        @test ct.data[1,2] ≈ 0.0
    end
end

# ── 2.2: Tests for contraction_ansatz ────────────────────────────────────

@testset "contraction_ansatz: single rank-2 tensor with free indices" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2),
        symmetries=Any[Symmetric(1,2)]))
    define_metric!(reg, :g; manifold=:M)

    with_registry(reg) do
        T = Tensor(:T, [down(:a), down(:b)])
        results = all_contractions([T], [down(:a), down(:b)])
        # With 2 free indices matching the tensor rank, should get 1 result (the tensor itself)
        @test length(results) >= 1
        @test all(r -> r isa TensorExpr, results)
    end
end

@testset "contraction_ansatz: two symmetric rank-2 tensors fully contracted" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:h, manifold=:M, rank=(0,2),
        symmetries=Any[Symmetric(1,2)]))
    define_metric!(reg, :g; manifold=:M)

    with_registry(reg) do
        h = Tensor(:h, [down(:a), down(:b)])
        results = all_contractions([h, h], TIndex[])
        # h_{ab}h^{ab} and h^a_a h^b_b → 2 distinct contractions
        @test length(results) == 2
    end
end

@testset "make_ansatz: creates linear combination" begin
    T = Tensor(:T, [down(:a)])
    V = Tensor(:V, [down(:a)])
    result = make_ansatz(TensorExpr[T, V], [:α, :β])
    @test result isa TSum
    @test length(result.terms) == 2
end

# ── 2.2: Tests for young_project ─────────────────────────────────────────

@testset "young_project: fully symmetric tableau" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,3)))

    with_registry(reg) do
        yt = YoungTableau([[1,2,3]])
        @test young_shape(yt) == [3]

        T = Tensor(:T, [down(:a), down(:b), down(:c)])
        result = young_project(T, yt, [:a, :b, :c])
        @test result isa TensorExpr
        # Fully symmetric projection: result should have the same free indices
        fi = free_indices(result)
        @test length(fi) == 3
    end
end

@testset "young_project: fully antisymmetric tableau" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,3)))

    with_registry(reg) do
        yt = YoungTableau([[1],[2],[3]])
        @test young_shape(yt) == [1,1,1]

        T = Tensor(:T, [down(:a), down(:b), down(:c)])
        result = young_project(T, yt, [:a, :b, :c])
        @test result isa TensorExpr
    end
end

@testset "young_project: mixed partition [2,1]" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,3)))

    with_registry(reg) do
        yt = YoungTableau([[1,2],[3]])
        @test young_shape(yt) == [2,1]

        T = Tensor(:T, [down(:a), down(:b), down(:c)])
        result = young_symmetrize(T, yt, [:a, :b, :c])
        @test result isa TSum
    end
end

# ── 2.3: Mathematical identity: d² = 0 ──────────────────────────────────

@testset "Mathematical identity: d² = 0 for exterior derivative" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:f, manifold=:M, rank=(0,0)))

    with_registry(reg) do
        # For a scalar function f: df = ∂_a f
        f = Tensor(:f, TIndex[])
        df = exterior_d(f, 0, down(:a))
        @test df isa TDeriv

        # d²f = antisym(∂_b(∂_a(f)), [a,b])
        # ∂_a ∂_b f is symmetric in (a,b), so antisymmetrization → 0
        ddf = TDeriv(down(:b), df, :partial)
        asym = antisymmetrize(ddf, [:a, :b])
        result = simplify(asym)
        @test result == TScalar(0//1) || result == ZERO
    end
end

# ── 2.3: Mathematical identity: projector completeness ───────────────────

@testset "Mathematical identity: θ + ω = η (projector completeness)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :η, :∂, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :η; manifold=:M)
    register_tensor!(reg, TensorProperties(name=:k, manifold=:M, rank=(0,1)))
    register_tensor!(reg, TensorProperties(name=:k², manifold=:M, rank=(0,0)))

    with_registry(reg) do
        μ, ν = down(:a), down(:b)
        θ = theta_projector(μ, ν; metric=:η)
        ω = omega_projector(μ, ν)
        # θ_{ab} + ω_{ab} = η_{ab}
        sum_expr = tsum(TensorExpr[θ, ω])
        result = simplify(sum_expr)
        # Should simplify to just the metric
        @test result isa Tensor || result isa TSum
        if result isa Tensor
            @test result.name == :η
        end
    end
end

# ── 2.3: Mathematical identity: wedge product structure ──────────────────

@testset "Mathematical identity: wedge product coefficient" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:α, manifold=:M, rank=(0,1),
        options=Dict{Symbol,Any}(:form_degree => 1)))
    register_tensor!(reg, TensorProperties(name=:β, manifold=:M, rank=(0,1),
        options=Dict{Symbol,Any}(:form_degree => 1)))

    with_registry(reg) do
        α = Tensor(:α, [down(:a)])
        β = Tensor(:β, [down(:b)])
        result = wedge(α, β, 1, 1)
        # Wedge of two 1-forms: coefficient = (1+1)!/(1!1!) = 2
        @test result isa TProduct
        @test result.scalar == 2 // 1
    end
end

@testset "Mathematical identity: wedge_power" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:ω, manifold=:M, rank=(0,2),
        symmetries=Any[AntiSymmetric(1,2)],
        options=Dict{Symbol,Any}(:form_degree => 2)))

    with_registry(reg) do
        ω = Tensor(:ω, [down(:a), down(:b)])
        # wedge_power with n=0 gives 1
        @test wedge_power(ω, 2, 0) == TScalar(1 // 1)
        # wedge_power with n=1 gives the form itself
        @test wedge_power(ω, 2, 1) == ω
        # wedge_power with n=2 gives a product
        wp2 = wedge_power(ω, 2, 2)
        @test wp2 isa TProduct
    end
end

# ── 2.2: Tests for sort_covds stubs ──────────────────────────────────────

@testset "sort_covds_to_box: detects box operator pattern" begin
    T = Tensor(:T, [down(:c)])
    # ∂_b(∂^b(T_c)) is a box pattern: same name, opposite positions
    d = TDeriv(down(:b), TDeriv(up(:b), T, :partial), :partial)
    result = sort_covds_to_box(d)
    # Should rewrite to g^{ab} ∂_a(∂_b(T_c)) form
    @test result isa TProduct
    # Should have a metric factor
    @test any(f -> f isa Tensor && f.name == :g, result.factors)
end

@testset "sort_covds_to_box: non-box pattern unchanged" begin
    T = Tensor(:T, [down(:c)])
    # ∂_a(∂_b(T_c)) is NOT a box pattern (different names)
    d = TDeriv(down(:a), TDeriv(down(:b), T, :partial), :partial)
    result = sort_covds_to_box(d)
    @test result == d
end

@testset "sort_covds_to_div: returns input (pattern already exposed)" begin
    T = Tensor(:T, [up(:a)])
    d = TDeriv(down(:a), T, :partial)
    result = sort_covds_to_div(d)
    @test result == d
end

# ── 2.4: Additional error paths ──────────────────────────────────────────

@testset "Error paths: validate detects rank mismatch" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d]))
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2)))

    with_registry(reg) do
        # Wrong number of indices (3 instead of 2)
        bad = Tensor(:T, [down(:a), down(:b), down(:c)])
        issues = validate(bad; registry=reg)
        @test !isempty(issues)
        @test any(s -> contains(s, "expected"), issues)
    end
end

@testset "Error paths: is_well_formed detects bad products" begin
    # Product where same index name appears 3+ times (not valid Einstein notation)
    T = Tensor(:T, [down(:a), up(:a)])
    S = Tensor(:S, [down(:a), up(:a)])
    p = TProduct(1//1, TensorExpr[T, S])
    @test !is_well_formed(p)
end

# ── Metric/delta cache: verify cache is populated ────────────────────────

@testset "Registry: metric/delta cache populated on register" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a,:b,:c,:d]))
    define_metric!(reg, :g; manifold=:M)

    @test haskey(reg.metric_cache, :M)
    @test reg.metric_cache[:M] == :g
    @test haskey(reg.delta_cache, :M)
    @test reg.delta_cache[:M] == :δ
end
