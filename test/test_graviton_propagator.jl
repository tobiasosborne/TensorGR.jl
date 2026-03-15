@testset "Graviton Propagator" begin
    @testset "graviton_propagator construction" begin
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

        prop = graviton_propagator(reg)

        @test prop isa TensorPropagator
        @test prop.name == :D_graviton
        @test prop.momentum == :k
        @test prop.gauge_param == :harmonic
        @test length(prop.indices_left) == 2
        @test length(prop.indices_right) == 2

        # All propagator indices should be Up
        for idx in prop.indices_left
            @test idx.position == Up
        end
        for idx in prop.indices_right
            @test idx.position == Up
        end

        # Show method
        buf = IOBuffer()
        show(buf, prop)
        s = String(take!(buf))
        @test occursin("D_graviton", s)
        @test occursin("gauge=harmonic", s)
    end

    @testset "graviton_propagator unsupported gauge" begin
        reg = TensorRegistry()
        @test_throws ErrorException graviton_propagator(reg; gauge=:axial)
    end

    @testset "propagator_numerator construction" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        idx = [up(:a), up(:b), up(:c), up(:d)]
        P = propagator_numerator(idx, reg)

        @test P isa TensorExpr

        # The numerator has free indices a, b, c, d (all up)
        free = free_indices(P)
        free_names = Set(idx.name for idx in free)
        @test :a in free_names
        @test :b in free_names
        @test :c in free_names
        @test :d in free_names
        @test length(free) == 4
        @test all(idx.position == Up for idx in free)
    end

    @testset "propagator_numerator wrong index count" begin
        reg = TensorRegistry()
        @test_throws ErrorException propagator_numerator([up(:a), up(:b), up(:c)], reg)
        @test_throws ErrorException propagator_numerator(
            [up(:a), up(:b), up(:c), up(:d), up(:e)], reg)
    end

    @testset "propagator_numerator symmetry: P^{abcd} = P^{bacd}" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        # P^{abcd}
        P_abcd = propagator_numerator([up(:a), up(:b), up(:c), up(:d)], reg)
        # P^{bacd}  (swap first pair)
        P_bacd = propagator_numerator([up(:b), up(:a), up(:c), up(:d)], reg)

        # Simplify the difference; should be zero
        diff = with_registry(reg) do
            simplify(P_abcd - P_bacd; registry=reg)
        end
        @test diff == TScalar(0 // 1)
    end

    @testset "propagator_numerator symmetry: P^{abcd} = P^{abdc}" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        # P^{abcd}
        P_abcd = propagator_numerator([up(:a), up(:b), up(:c), up(:d)], reg)
        # P^{abdc}  (swap second pair)
        P_abdc = propagator_numerator([up(:a), up(:b), up(:d), up(:c)], reg)

        diff = with_registry(reg) do
            simplify(P_abcd - P_abdc; registry=reg)
        end
        @test diff == TScalar(0 // 1)
    end

    @testset "propagator_numerator symmetry: P^{abcd} = P^{cdab}" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        # P^{abcd}
        P_abcd = propagator_numerator([up(:a), up(:b), up(:c), up(:d)], reg)
        # P^{cdab}  (swap pairs)
        P_cdab = propagator_numerator([up(:c), up(:d), up(:a), up(:b)], reg)

        diff = with_registry(reg) do
            simplify(P_abcd - P_cdab; registry=reg)
        end
        @test diff == TScalar(0 // 1)
    end

    @testset "propagator_numerator trace: eta_{ac} P^{abcd}" begin
        # Trace on (a,c): eta_{ac} P^{abcd}
        # = (1/2)(eta_{ac} eta^{ac} eta^{bd} + eta_{ac} eta^{ad} eta^{bc}
        #         - eta_{ac} eta^{ab} eta^{cd})
        # = (1/2)(d * eta^{bd} + delta^d_c eta^{bc} - delta^b_c eta^{cd})
        # = (1/2)(d * eta^{bd} + eta^{bd} - eta^{bd})
        # = (d/2) eta^{bd}
        # In d=4: 2 * eta^{bd}
        #
        # We verify the structure by contracting with eta_{ac} and simplifying.
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

        P = propagator_numerator([up(:a), up(:b), up(:c), up(:d)], reg)
        eta_ac = Tensor(:eta, [down(:a), down(:c)])

        traced = tproduct(1 // 1, TensorExpr[eta_ac, P])

        result = with_registry(reg) do
            simplify(traced; registry=reg)
        end

        # Result should have free indices b (up) and d (up)
        free = free_indices(result)
        free_names = Set(idx.name for idx in free)
        @test length(free) == 2
        # The traced result should be proportional to eta^{bd}
        # In d=4: eta_{ac} P^{abcd} = 2 eta^{bd}
        # Check it is non-zero
        @test result != TScalar(0 // 1)
    end

    @testset "propagator_numerator with down indices" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        # P_{abcd} with all-down indices
        P_down = propagator_numerator([down(:a), down(:b), down(:c), down(:d)], reg)
        @test P_down isa TensorExpr

        free = free_indices(P_down)
        @test length(free) == 4
        @test all(idx.position == Down for idx in free)
    end

    @testset "propagator expression structure" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        prop = graviton_propagator(reg)
        expr = prop.expr

        # The expression should be a product containing TScalar(:inv_k2)
        # and the numerator tensor structure
        @test expr isa TProduct

        # Check for the 1/k^2 factor
        has_inv_k2 = false
        for f in expr.factors
            if f isa TScalar && f.val == :inv_k2
                has_inv_k2 = true
            elseif f isa TProduct
                for ff in f.factors
                    if ff isa TScalar && ff.val == :inv_k2
                        has_inv_k2 = true
                    end
                end
            end
        end
        @test has_inv_k2
    end

    @testset "propagator has correct free indices" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        prop = graviton_propagator(reg)

        # The propagator expression should have 4 free indices (all Up)
        free = free_indices(prop.expr)
        @test length(free) == 4
        @test all(idx.position == Up for idx in free)

        # The free indices should match the left+right index names
        all_prop_names = Set(idx.name for idx in vcat(prop.indices_left, prop.indices_right))
        expr_free_names = Set(idx.name for idx in free)
        @test all_prop_names == expr_free_names
    end

    @testset "propagator custom momentum and k_sq" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        prop = graviton_propagator(reg; momentum=:q, k_sq=:q2)

        @test prop.momentum == :q

        # Check for inv_q2 factor
        has_inv_q2 = false
        function _check_inv(e::TScalar)
            e.val == :inv_q2
        end
        function _check_inv(e::TProduct)
            any(_check_inv(f) for f in e.factors)
        end
        function _check_inv(::TensorExpr)
            false
        end
        @test _check_inv(prop.expr)
    end

    @testset "propagator with custom metric" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :g, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        prop = graviton_propagator(reg; metric=:g)
        @test prop isa TensorPropagator

        # Expression should contain :g metric tensors, not :eta
        function _has_tensor_name(e::Tensor, name::Symbol)
            e.name == name
        end
        function _has_tensor_name(e::TProduct, name::Symbol)
            any(_has_tensor_name(f, name) for f in e.factors)
        end
        function _has_tensor_name(e::TSum, name::Symbol)
            any(_has_tensor_name(t, name) for t in e.terms)
        end
        function _has_tensor_name(::TScalar, ::Symbol)
            false
        end
        function _has_tensor_name(e::TDeriv, name::Symbol)
            _has_tensor_name(e.arg, name)
        end

        @test _has_tensor_name(prop.expr, :g)
    end

    @testset "idempotency: P^{ab}_{ef} P^{efcd} structure" begin
        # P^{abef} eta_{eg} eta_{fh} P^{ghcd} should give a result
        # related to P^{abcd} (with dimension-dependent coefficient).
        # In d dimensions: P^{ab}_{ef} P^{efcd} = ((d+1)/2) P^{abcd}
        # ... but this is only in d=4 if the indices are fully contracted.
        # Actually: P^2 = (d/2) P  (the projection onto spin-2 + spin-0s).
        #
        # We verify structurally: the contraction should produce an expression
        # with 4 free indices (a, b, c, d), all Up.
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

        P1 = propagator_numerator([up(:a), up(:b), up(:e), up(:f)], reg)
        P2 = propagator_numerator([up(:g), up(:h), up(:c), up(:d)], reg)

        # Lower e,f on P1 and g,h on P2, then contract
        eta_eg = Tensor(:eta, [down(:e), down(:g)])
        eta_fh = Tensor(:eta, [down(:f), down(:h)])

        contraction = tproduct(1 // 1, TensorExpr[P1, eta_eg, eta_fh, P2])

        result = with_registry(reg) do
            simplify(contraction; registry=reg)
        end

        # Should have free indices a, b, c, d (all Up)
        free = free_indices(result)
        free_names = Set(idx.name for idx in free)
        @test :a in free_names
        @test :b in free_names
        @test :c in free_names
        @test :d in free_names
        @test length(free) == 4
        @test result != TScalar(0 // 1)
    end

    @testset "diagram assembly with graviton propagator" begin
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

        # Build a graviton propagator and use it in a diagram
        prop = graviton_propagator(reg)

        # Create simple test vertices with matching index structure
        ig = [[down(:a), down(:b)], [down(:c), down(:d)], [down(:e), down(:f)]]
        v1 = TensorVertex(:V1, ig, [:k1, :k2, :k3], TScalar(1 // 1))
        v2 = TensorVertex(:V2, ig, [:k4, :k5, :k6], TScalar(1 // 1))

        # Build tree-level exchange diagram
        diag = tree_exchange_diagram(v1, v2, prop)
        @test n_loops(diag) == 0
        @test length(diag.vertices) == 2
        @test length(diag.propagators) == 1
    end

    @testset "propagator numerator contains eta tensors" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(
            :M4, 4, :eta, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        P = propagator_numerator([up(:a), up(:b), up(:c), up(:d)], reg)

        # Count eta tensors in the expression
        function _count_eta(e::Tensor)
            e.name == :eta ? 1 : 0
        end
        function _count_eta(e::TProduct)
            sum(_count_eta(f) for f in e.factors; init=0)
        end
        function _count_eta(e::TSum)
            sum(_count_eta(t) for t in e.terms; init=0)
        end
        function _count_eta(::TScalar)
            0
        end

        # P = (1/2)(eta*eta + eta*eta - eta*eta) -> 6 eta tensors total
        @test _count_eta(P) == 6
    end
end
