# ────────────────────────────────────────────────────────────────────
# Tests for matter-graviton coupling vertices
# ────────────────────────────────────────────────────────────────────

"""Count occurrences of a specific tensor name in an expression (local helper)."""
function _mv_count_tensor(expr::Tensor, name::Symbol)
    expr.name == name ? 1 : 0
end
function _mv_count_tensor(expr::TProduct, name::Symbol)
    sum(_mv_count_tensor(f, name) for f in expr.factors; init=0)
end
function _mv_count_tensor(expr::TSum, name::Symbol)
    isempty(expr.terms) ? 0 : maximum(_mv_count_tensor(t, name) for t in expr.terms)
end
function _mv_count_tensor(expr::TDeriv, name::Symbol)
    _mv_count_tensor(expr.arg, name)
end
function _mv_count_tensor(::TScalar, ::Symbol)
    0
end

"""Check if an expression contains TDeriv nodes (local helper)."""
function _mv_has_derivs(expr::Tensor)
    false
end
function _mv_has_derivs(expr::TProduct)
    any(_mv_has_derivs(f) for f in expr.factors)
end
function _mv_has_derivs(expr::TSum)
    any(_mv_has_derivs(t) for t in expr.terms)
end
function _mv_has_derivs(::TDeriv)
    true
end
function _mv_has_derivs(::TScalar)
    false
end

const _MV_ZERO = TScalar(0 // 1)

@testset "Matter-Graviton Vertices" begin

    @testset "1-graviton vertex construction" begin
        reg = TensorRegistry()
        v1 = matter_graviton_vertex(1, reg)

        @test v1 isa TensorVertex
        @test v1.name == :V1_pp_m
        @test n_point(v1) == 1
        @test n_indices(v1) == 2  # 1 leg with 2 symmetric indices
        @test v1.coupling_order == 1
        @test v1.momenta == [:k1]

        # Index group should have 2 down indices (symmetric pair)
        @test length(v1.index_groups[1]) == 2
        @test all(idx -> idx.position == Down, v1.index_groups[1])

        # Expression should be non-trivial
        @test v1.expr != _MV_ZERO

        # Should contain velocity tensor v_m
        vel_count = _mv_count_tensor(v1.expr, :v_m)
        @test vel_count >= 1

        # Should contain the metric eta
        eta_count = _mv_count_tensor(v1.expr, :eta)
        @test eta_count >= 1

        # Show method
        buf = IOBuffer()
        show(buf, v1)
        s = String(take!(buf))
        @test occursin("1-point", s)
        @test occursin("V1_pp_m", s)
    end

    @testset "1-graviton vertex index structure (symmetric)" begin
        reg = TensorRegistry()
        v1 = matter_graviton_vertex(1, reg)

        # The vertex has one pair of indices -- both must be Down
        idx_a = v1.index_groups[1][1]
        idx_b = v1.index_groups[1][2]
        @test idx_a.position == Down
        @test idx_b.position == Down

        # Free indices of the expression should include these
        fi = free_indices(v1.expr)
        idx_names = Set(idx.name for idx in fi)
        @test idx_a.name in idx_names
        @test idx_b.name in idx_names
    end

    @testset "1-graviton vertex has correct structure" begin
        reg = TensorRegistry()
        v1 = matter_graviton_vertex(1, reg)
        expr = v1.expr

        # The vertex should be a TSum with two terms:
        # term 1: m * v_a v_b (velocity-velocity)
        # term 2: -(m/2) * eta_{ab} (metric trace)
        @test expr isa TSum
        @test length(expr.terms) == 2
    end

    @testset "2-graviton vertex construction" begin
        reg = TensorRegistry()
        v2 = matter_graviton_vertex(2, reg)

        @test v2 isa TensorVertex
        @test v2.name == :V2_pp_m
        @test n_point(v2) == 2
        @test n_indices(v2) == 4  # 2 legs with 2 indices each
        @test v2.coupling_order == 2
        @test v2.momenta == [:k1, :k2]

        # Both index groups should have 2 down indices
        for ig in v2.index_groups
            @test length(ig) == 2
            @test all(idx -> idx.position == Down, ig)
        end

        # Bose symmetry: S_2 (2 permutations)
        @test length(v2.symmetry_group) == 2
        @test [1, 2] in v2.symmetry_group
        @test [2, 1] in v2.symmetry_group

        @test v2.expr != _MV_ZERO
    end

    @testset "matter vertex with custom particle name" begin
        reg = TensorRegistry()
        v1 = matter_graviton_vertex(1, reg; particle=:M)

        @test v1.name == :V1_pp_M
        @test n_point(v1) == 1

        # Should contain velocity tensor v_M
        vel_count = _mv_count_tensor(v1.expr, :v_M)
        @test vel_count >= 1
    end

    @testset "matter vertex with explicit registry" begin
        reg = TensorRegistry()
        v1 = matter_graviton_vertex(1, reg)

        # Registry should now have velocity and metric tensors
        @test has_tensor(reg, :v_m)
        @test has_tensor(reg, :eta)
    end

    @testset "matter vertex n < 1 errors" begin
        reg = TensorRegistry()
        @test_throws ErrorException matter_graviton_vertex(0, reg)
    end

    @testset "matter vertex n > 2 errors" begin
        reg = TensorRegistry()
        @test_throws ErrorException matter_graviton_vertex(3, reg)
    end

    @testset "scalar matter 1-graviton vertex construction" begin
        reg = TensorRegistry()
        v1 = scalar_matter_vertex(1, reg)

        @test v1 isa TensorVertex
        @test v1.name == :V1_scalar_phi
        @test n_point(v1) == 1
        @test n_indices(v1) == 2
        @test v1.coupling_order == 1
        @test v1.momenta == [:k1]

        # Should have 2 symmetric down indices
        @test length(v1.index_groups[1]) == 2
        @test all(idx -> idx.position == Down, v1.index_groups[1])

        @test v1.expr != _MV_ZERO
    end

    @testset "scalar matter vertex contains derivatives" begin
        reg = TensorRegistry()
        v1 = scalar_matter_vertex(1, reg)

        # The scalar field stress-energy involves partial_a phi partial_b phi
        @test _mv_has_derivs(v1.expr)
    end

    @testset "scalar matter vertex has correct structure" begin
        reg = TensorRegistry()
        v1 = scalar_matter_vertex(1, reg)
        expr = v1.expr

        # T_{ab} = partial_a phi partial_b phi - (1/2) eta_{ab} (partial phi)^2
        # Should be a TSum with two terms
        @test expr isa TSum
        @test length(expr.terms) == 2
    end

    @testset "scalar matter 2-graviton vertex" begin
        reg = TensorRegistry()
        v2 = scalar_matter_vertex(2, reg)

        @test v2 isa TensorVertex
        @test v2.name == :V2_scalar_phi
        @test n_point(v2) == 2
        @test n_indices(v2) == 4
        @test v2.coupling_order == 2
        @test v2.momenta == [:k1, :k2]

        # Bose symmetry: S_2
        @test length(v2.symmetry_group) == 2
        @test [1, 2] in v2.symmetry_group
        @test [2, 1] in v2.symmetry_group

        # Should contain derivatives of the scalar field
        @test _mv_has_derivs(v2.expr)

        @test v2.expr != _MV_ZERO
    end

    @testset "scalar matter vertex custom field name" begin
        reg = TensorRegistry()
        v1 = scalar_matter_vertex(1, reg; field=:psi)

        @test v1.name == :V1_scalar_psi

        # Should have registered the field
        @test has_tensor(reg, :psi)
    end

    @testset "scalar matter vertex n < 1 errors" begin
        reg = TensorRegistry()
        @test_throws ErrorException scalar_matter_vertex(0, reg)
    end

    @testset "scalar matter vertex n > 2 errors" begin
        reg = TensorRegistry()
        @test_throws ErrorException scalar_matter_vertex(3, reg)
    end

    @testset "vertex expression well-formedness" begin
        reg = TensorRegistry()
        v1_pp = matter_graviton_vertex(1, reg)
        v1_sc = scalar_matter_vertex(1, reg)

        for v in [v1_pp, v1_sc]
            fi = free_indices(v.expr)
            for idx in fi
                @test idx.position in (Up, Down)
            end
        end
    end

    @testset "diagram assembly with matter vertex" begin
        reg = TensorRegistry()
        v_matter = matter_graviton_vertex(1, reg)
        v_gravity = graviton_3vertex(registry=reg)

        # The matter vertex has 1 leg (rank 2), the graviton vertex has 3 legs (rank 2)
        # Build a propagator connecting them
        prop = TensorPropagator(:D_graviton,
            [down(:a), down(:b)], [down(:c), down(:d)],
            :q, TScalar(1 // 1); gauge_param=:xi)

        # Tree-level diagram: matter vertex emitting a graviton into a 3-point vertex
        diag = tree_exchange_diagram(v_matter, v_gravity, prop;
                                      leg1=1, leg2=1,
                                      external_momenta=[:p1, :p2])

        @test n_loops(diag) == 0
        @test length(diag.vertices) == 2
        @test length(diag.propagators) == 1
        # matter has 1 leg (used), gravity has 3 legs (1 used + 2 external)
        @test length(diag.external_legs) == 2
    end
end
