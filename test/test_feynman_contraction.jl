@testset "Feynman contraction engine" begin

    # Helper: set up a flat-space registry with metric eta
    function _flat_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :eta, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l,:m,:n]))
        reg
    end

    @testset "contract_line: basic rank check" begin
        reg = _flat_reg()
        with_registry(reg) do
            prop = graviton_propagator(reg)

            # Build two simple vertex-like expressions
            v1 = Tensor(:T1, [down(:a), down(:b)])
            v2 = Tensor(:T2, [down(:c), down(:d)])

            result = contract_line(prop, v1, v2,
                                   [down(:a), down(:b)],
                                   [down(:c), down(:d)];
                                   registry=reg)
            @test result isa TensorExpr

            # Result should be a product of 3 expressions
            @test result isa TProduct
        end
    end

    @testset "contract_line: rank mismatch" begin
        reg = _flat_reg()
        with_registry(reg) do
            prop = graviton_propagator(reg)
            v1 = Tensor(:T1, [down(:a)])
            v2 = Tensor(:T2, [down(:c), down(:d)])

            @test_throws ErrorException contract_line(prop, v1, v2,
                [down(:a)], [down(:c), down(:d)]; registry=reg)
        end
    end

    @testset "find_loop_momenta: tree diagram" begin
        reg = _flat_reg()
        with_registry(reg) do
            # Tree-level: 2 vertices, 1 propagator -> 0 loops
            v1 = TensorVertex(:V1, [[down(:a), down(:b)], [down(:c), down(:d)]],
                              [:k1, :k2], TScalar(1))
            v2 = TensorVertex(:V2, [[down(:e), down(:f)], [down(:g), down(:h)]],
                              [:k3, :k4], TScalar(1))

            prop = graviton_propagator(reg)
            diag = build_diagram([v1, v2], [prop], [(1, 1, 2, 1)])

            loops = find_loop_momenta(diag)
            @test isempty(loops)
            @test n_loops(diag) == 0
        end
    end

    @testset "find_loop_momenta: bubble diagram" begin
        reg = _flat_reg()
        with_registry(reg) do
            # Bubble: 2 vertices, 2 propagators -> 1 loop
            v1 = TensorVertex(:V1,
                [[down(:a), down(:b)], [down(:c), down(:d)], [down(:e), down(:f)]],
                [:k1, :k2, :k3], TScalar(1))
            v2 = TensorVertex(:V2,
                [[down(:g), down(:h)], [down(:i), down(:j)], [down(:k), down(:l)]],
                [:k4, :k5, :k6], TScalar(1))

            p1 = graviton_propagator(reg)
            p2 = graviton_propagator(reg)
            diag = build_diagram([v1, v2], [p1, p2], [(1, 1, 2, 1), (1, 2, 2, 2)])

            loops = find_loop_momenta(diag)
            @test length(loops) == 1
            @test n_loops(diag) == 1
        end
    end

    @testset "find_loop_momenta: triangle diagram" begin
        reg = _flat_reg()
        with_registry(reg) do
            # Triangle: 3 vertices, 3 propagators -> 1 loop
            v1 = TensorVertex(:V1,
                [[down(:a), down(:b)], [down(:c), down(:d)], [down(:e), down(:f)]],
                [:k1, :k2, :k3], TScalar(1))
            v2 = TensorVertex(:V2,
                [[down(:g), down(:h)], [down(:i), down(:j)], [down(:k), down(:l)]],
                [:k4, :k5, :k6], TScalar(1))
            v3 = TensorVertex(:V3,
                [[down(:m), down(:n)], [up(:a), up(:b)], [up(:c), up(:d)]],
                [:k7, :k8, :k9], TScalar(1))

            p1 = graviton_propagator(reg)
            p2 = graviton_propagator(reg)
            p3 = graviton_propagator(reg)

            diag = build_diagram([v1, v2, v3], [p1, p2, p3],
                [(1, 1, 2, 1), (2, 2, 3, 1), (3, 2, 1, 2)])

            loops = find_loop_momenta(diag)
            @test length(loops) == 1
            @test n_loops(diag) == 1
        end
    end

    @testset "momentum_constraints" begin
        reg = _flat_reg()
        with_registry(reg) do
            v1 = TensorVertex(:V1, [[down(:a), down(:b)], [down(:c), down(:d)]],
                              [:k1, :k2], TScalar(1))
            v2 = TensorVertex(:V2, [[down(:e), down(:f)], [down(:g), down(:h)]],
                              [:k3, :k4], TScalar(1))
            prop = graviton_propagator(reg)
            diag = build_diagram([v1, v2], [prop], [(1, 1, 2, 1)])

            constraints = momentum_constraints(diag)
            @test length(constraints) == 2

            # Each vertex should have momenta from internal line + external legs
            for c in constraints
                @test length(c.momenta) >= 1
                @test length(c.signs) == length(c.momenta)
            end
        end
    end

    @testset "impose_momentum_conservation: tree diagram" begin
        reg = _flat_reg()
        with_registry(reg) do
            v1 = TensorVertex(:V1, [[down(:a), down(:b)], [down(:c), down(:d)]],
                              [:k1, :k2], TScalar(1))
            v2 = TensorVertex(:V2, [[down(:e), down(:f)], [down(:g), down(:h)]],
                              [:k3, :k4], TScalar(1))
            prop = graviton_propagator(reg)
            diag = build_diagram([v1, v2], [prop], [(1, 1, 2, 1)])

            routing = impose_momentum_conservation(diag)
            # For tree diagram, internal momentum should be expressible
            # in terms of external momenta
            @test routing isa Dict
        end
    end

    @testset "symmetry_factor: tree diagram" begin
        reg = _flat_reg()
        with_registry(reg) do
            v1 = TensorVertex(:V1, [[down(:a), down(:b)], [down(:c), down(:d)]],
                              [:k1, :k2], TScalar(1))
            v2 = TensorVertex(:V2, [[down(:e), down(:f)], [down(:g), down(:h)]],
                              [:k3, :k4], TScalar(1))
            prop = graviton_propagator(reg)
            diag = build_diagram([v1, v2], [prop], [(1, 1, 2, 1)])

            # Tree diagram with single propagator: symmetry factor = 1
            sf = symmetry_factor(diag)
            @test sf == 1 // 1
        end
    end

    @testset "symmetry_factor: bubble diagram = 1/2" begin
        reg = _flat_reg()
        with_registry(reg) do
            v1 = TensorVertex(:V1,
                [[down(:a), down(:b)], [down(:c), down(:d)], [down(:e), down(:f)]],
                [:k1, :k2, :k3], TScalar(1))
            v2 = TensorVertex(:V2,
                [[down(:g), down(:h)], [down(:i), down(:j)], [down(:k), down(:l)]],
                [:k4, :k5, :k6], TScalar(1))

            p1 = graviton_propagator(reg)
            p2 = graviton_propagator(reg)
            diag = build_diagram([v1, v2], [p1, p2], [(1, 1, 2, 1), (1, 2, 2, 2)])

            # Bubble with 2 propagators between same vertices: 1/2!
            sf = symmetry_factor(diag)
            @test sf == 1 // 2
        end
    end

    @testset "symmetry_factor: sunset diagram = 1/6" begin
        reg = _flat_reg()
        with_registry(reg) do
            # Sunset: 2 vertices, 3 propagators between them -> 1/3! = 1/6
            v1 = TensorVertex(:V1,
                [[down(:a), down(:b)], [down(:c), down(:d)],
                 [down(:e), down(:f)], [down(:g), down(:h)]],
                [:k1, :k2, :k3, :k4], TScalar(1))
            v2 = TensorVertex(:V2,
                [[down(:i), down(:j)], [down(:k), down(:l)],
                 [down(:m), down(:n)], [up(:a), up(:b)]],
                [:k5, :k6, :k7, :k8], TScalar(1))

            p1 = graviton_propagator(reg)
            p2 = graviton_propagator(reg)
            p3 = graviton_propagator(reg)

            diag = build_diagram([v1, v2], [p1, p2, p3],
                [(1, 1, 2, 1), (1, 2, 2, 2), (1, 3, 2, 3)])

            sf = symmetry_factor(diag)
            @test sf == 1 // 6
        end
    end

    @testset "contract_diagram with contraction engine" begin
        reg = _flat_reg()
        with_registry(reg) do
            # Basic tree-level exchange: should produce a DiagramAmplitude
            T1 = Tensor(:T1, [down(:a), down(:b)])
            T2 = Tensor(:T2, [down(:c), down(:d)])

            v1 = TensorVertex(:V1, [[down(:a), down(:b)]],
                              [:k1], T1; coupling_order=0)
            v2 = TensorVertex(:V2, [[down(:c), down(:d)]],
                              [:k2], T2; coupling_order=0)

            prop = graviton_propagator(reg)
            diag = tree_exchange_diagram(v1, v2, prop)

            amp = contract_diagram(diag; registry=reg)
            @test amp isa DiagramAmplitude
            @test amp.expr isa TensorExpr
        end
    end

end
