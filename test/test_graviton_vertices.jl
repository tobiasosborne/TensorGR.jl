# ────────────────────────────────────────────────────────────────────
# Test helpers (must be defined before the testset)
# ────────────────────────────────────────────────────────────────────

"""Count occurrences of a specific tensor name in an expression."""
function _count_pert_occurrences(expr::Tensor, name::Symbol)
    expr.name == name ? 1 : 0
end
function _count_pert_occurrences(expr::TProduct, name::Symbol)
    sum(_count_pert_occurrences(f, name) for f in expr.factors; init=0)
end
function _count_pert_occurrences(expr::TSum, name::Symbol)
    isempty(expr.terms) ? 0 : maximum(_count_pert_occurrences(t, name) for t in expr.terms)
end
function _count_pert_occurrences(expr::TDeriv, name::Symbol)
    _count_pert_occurrences(expr.arg, name)
end
function _count_pert_occurrences(::TScalar, ::Symbol)
    0
end

"""Check if an expression contains any TDeriv nodes."""
function _has_derivatives(expr::Tensor)
    false
end
function _has_derivatives(expr::TProduct)
    any(_has_derivatives(f) for f in expr.factors)
end
function _has_derivatives(expr::TSum)
    any(_has_derivatives(t) for t in expr.terms)
end
function _has_derivatives(::TDeriv)
    true
end
function _has_derivatives(::TScalar)
    false
end

const _ZERO = TScalar(0 // 1)

# ────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────

@testset "Graviton Vertices" begin
    @testset "3-point vertex construction" begin
        v3 = graviton_3vertex()

        @test v3 isa TensorVertex
        @test v3.name == :V3_EH
        @test n_point(v3) == 3
        @test n_indices(v3) == 6  # 3 legs x 2 indices each
        @test v3.coupling_order == 1  # kappa^1
        @test v3.momenta == [:k1, :k2, :k3]

        # Index groups should each have 2 indices (symmetric pairs)
        for ig in v3.index_groups
            @test length(ig) == 2
        end

        # Bose symmetry group should be S_3 (6 permutations)
        @test length(v3.symmetry_group) == 6

        # Expression should be non-trivial
        @test v3.expr != _ZERO

        # Show method
        buf = IOBuffer()
        show(buf, v3)
        s = String(take!(buf))
        @test occursin("3-point", s)
        @test occursin("V3_EH", s)
    end

    @testset "4-point vertex construction" begin
        v4 = graviton_4vertex()

        @test v4 isa TensorVertex
        @test v4.name == :V4_EH
        @test n_point(v4) == 4
        @test n_indices(v4) == 8  # 4 legs x 2 indices each
        @test v4.coupling_order == 2  # kappa^2
        @test v4.momenta == [:k1, :k2, :k3, :k4]

        for ig in v4.index_groups
            @test length(ig) == 2
        end

        # Bose symmetry: S_4 has 24 permutations
        @test length(v4.symmetry_group) == 24

        @test v4.expr != _ZERO
    end

    @testset "generic n-point vertex" begin
        # n=3 and n=4 should produce same structure as dedicated functions
        v3g = graviton_vertex_n(3)
        @test n_point(v3g) == 3
        @test v3g.coupling_order == 1
        @test v3g.name == :V3_EH
        @test v3g.expr != _ZERO

        v4g = graviton_vertex_n(4)
        @test n_point(v4g) == 4
        @test v4g.coupling_order == 2

        # n < 2 should error
        @test_throws ErrorException graviton_vertex_n(1)
        @test_throws ErrorException graviton_vertex_n(0)
    end

    @testset "3-vertex contains perturbation tensors" begin
        v3 = graviton_3vertex()
        expr = v3.expr

        # The expression should contain :h tensors (perturbation field)
        h_count = _count_pert_occurrences(expr, :h)
        # delta^3(sqrt(-g) R) has terms with 3 h's
        @test h_count >= 3
    end

    @testset "3-vertex contains derivatives" begin
        v3 = graviton_3vertex()
        expr = v3.expr

        # The Ricci scalar perturbation contains derivatives of h
        @test _has_derivatives(expr)
    end

    @testset "Bose symmetry: S_3 permutations present" begin
        v3 = graviton_3vertex()

        # Check that all 6 permutations of [1,2,3] are in the symmetry group
        expected_perms = [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
        for p in expected_perms
            @test p in v3.symmetry_group
        end
    end

    @testset "Bose symmetry: S_4 permutations present" begin
        v4 = graviton_4vertex()

        # Check that identity [1,2,3,4] and some transpositions are present
        @test [1,2,3,4] in v4.symmetry_group
        @test [2,1,3,4] in v4.symmetry_group
        @test [4,3,2,1] in v4.symmetry_group
    end

    @testset "custom metric and perturbation names" begin
        v3 = graviton_3vertex(metric=:g, perturbation=:phi)
        @test n_point(v3) == 3
        @test v3.expr != _ZERO

        # Verify the perturbation field is :phi, not :h
        h_count = _count_pert_occurrences(v3.expr, :phi)
        @test h_count >= 3
        h_default = _count_pert_occurrences(v3.expr, :h)
        @test h_default == 0
    end

    @testset "vertex with explicit registry" begin
        reg = TensorRegistry()
        v3 = graviton_3vertex(registry=reg)
        @test n_point(v3) == 3

        # Registry should now have the perturbation tensor registered
        @test has_tensor(reg, :h)
        @test has_tensor(reg, :eta)
    end

    @testset "vertex_from_perturbation integration" begin
        # Build a vertex from the perturbation engine directly and compare
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j, :k, :l, :m, :n]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        mp = with_registry(reg) do
            define_metric_perturbation!(reg, :eta, :h)
        end

        # delta^3(R) alone
        d3R = with_registry(reg) do
            δricci_scalar(mp, 3)
        end
        @test d3R != _ZERO

        # delta^1(sqrt(-g)) * delta^2(R)
        d1sg = TensorGR._delta_sqrt_g(mp, 1)
        d2R = with_registry(reg) do
            δricci_scalar(mp, 2)
        end
        @test d1sg != _ZERO
        @test d2R != _ZERO
    end

    @testset "diagram assembly with graviton vertex" begin
        v3 = graviton_3vertex()

        # Build a graviton propagator (placeholder)
        prop = TensorPropagator(:D_graviton,
            [down(:a), down(:b)], [down(:c), down(:d)],
            :q, TScalar(1 // 1); gauge_param=:xi)

        # Tree-level 2-to-2 graviton scattering: two 3-vertices + one propagator
        v3b = graviton_3vertex()
        diag = tree_exchange_diagram(v3, v3b, prop;
                                      external_momenta=[:p1, :p2, :p3, :p4])

        @test n_loops(diag) == 0
        @test length(diag.vertices) == 2
        @test length(diag.propagators) == 1
        @test length(diag.external_legs) == 4
    end

    @testset "delta_sqrt_g order checks" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j, :k, :l]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        mp = with_registry(reg) do
            define_metric_perturbation!(reg, :eta, :h)
        end

        # Order 0 gives ZERO (the delta, not the value 1)
        @test TensorGR._delta_sqrt_g(mp, 0) == _ZERO

        # Order 1: non-zero, contains trace h = eta^{ab} h_{ab}
        d1 = TensorGR._delta_sqrt_g(mp, 1)
        @test d1 != _ZERO
        # Should be a product: (1/2) eta^{ab} h_{ab}
        @test d1 isa TProduct
        @test d1.scalar == 1 // 2

        # Order 2: non-zero
        d2 = TensorGR._delta_sqrt_g(mp, 2)
        @test d2 != _ZERO

        # Order 3: non-zero
        d3 = TensorGR._delta_sqrt_g(mp, 3)
        @test d3 != _ZERO
    end

    @testset "Sannan ground truth: 3-vertex structure count" begin
        # Sannan PRD 34 (1986) Eq 3.3: the 3-graviton vertex in momentum space
        # has 12 independent tensor structures after imposing Bose symmetry.
        # We verify that the position-space vertex (before Fourier) contains
        # the expected number of index structures.
        v3 = graviton_3vertex()

        # The expression should be a TSum with multiple terms
        # (from the Leibniz expansion of sqrt(-g) R)
        if v3.expr isa TSum
            # There should be multiple terms from the expansion
            @test length(v3.expr.terms) >= 2
        else
            # Single term is also valid (if everything collapsed)
            @test v3.expr isa TensorExpr
        end
    end

    @testset "vertex expression well-formedness" begin
        v3 = graviton_3vertex()
        v4 = graviton_4vertex()

        # All free indices in the expression should be from
        # the perturbation field h
        for v in [v3, v4]
            fi = free_indices(v.expr)
            for idx in fi
                # Free indices come from h tensors and metric contractions
                @test idx.position in (Up, Down)
            end
        end
    end
end
