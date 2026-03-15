@testset "Feynman Diagram Types" begin
    @testset "TensorVertex construction" begin
        # 3-graviton vertex with 3 index groups of 2 indices each
        ig = [
            [down(:a), down(:b)],
            [down(:c), down(:d)],
            [down(:e), down(:f)],
        ]
        momenta = [:k1, :k2, :k3]
        expr = TScalar(1 // 1)  # placeholder
        v3 = TensorVertex(:V3_EH, ig, momenta, expr)

        @test v3.name == :V3_EH
        @test n_point(v3) == 3
        @test n_indices(v3) == 6
        @test v3.coupling_order == 1  # 3-2 = 1
        @test v3.momenta == [:k1, :k2, :k3]
        @test isempty(v3.symmetry_group)

        # Custom coupling order
        v4 = TensorVertex(:V4_EH,
            [[down(:a), down(:b)], [down(:c), down(:d)],
             [down(:e), down(:f)], [down(:g), down(:h)]],
            [:k1, :k2, :k3, :k4], expr;
            coupling_order=2)
        @test n_point(v4) == 4
        @test v4.coupling_order == 2
        @test n_indices(v4) == 8

        # Show method
        buf = IOBuffer()
        show(buf, v3)
        s = String(take!(buf))
        @test occursin("3-point", s)
        @test occursin("V3_EH", s)
    end

    @testset "TensorVertex mismatched legs/momenta" begin
        ig = [[down(:a), down(:b)], [down(:c), down(:d)]]
        @test_throws ErrorException TensorVertex(:bad, ig, [:k1], TScalar(1 // 1))
    end

    @testset "TensorPropagator construction" begin
        il = [down(:a), down(:b)]
        ir = [down(:c), down(:d)]
        expr = TScalar(1 // 1)
        prop = TensorPropagator(:D_graviton, il, ir, :k, expr;
                                gauge_param=:xi)

        @test prop.name == :D_graviton
        @test length(prop.indices_left) == 2
        @test length(prop.indices_right) == 2
        @test prop.momentum == :k
        @test prop.gauge_param == :xi

        # Show method
        buf = IOBuffer()
        show(buf, prop)
        s = String(take!(buf))
        @test occursin("D_graviton", s)
        @test occursin("gauge=xi", s)

        # Without gauge
        prop2 = TensorPropagator(:D_scalar, [down(:a)], [down(:b)], :q, expr)
        @test prop2.gauge_param === nothing
    end

    @testset "FeynmanDiagram and n_loops" begin
        # Two 3-point vertices
        ig3 = [[down(:a), down(:b)], [down(:c), down(:d)], [down(:e), down(:f)]]
        v1 = TensorVertex(:V1, ig3, [:k1, :k2, :k3], TScalar(1 // 1))
        v2 = TensorVertex(:V2, ig3, [:k4, :k5, :k6], TScalar(1 // 1))

        # Propagator
        prop = TensorPropagator(:D, [down(:a), down(:b)], [down(:c), down(:d)],
                                :q, TScalar(1 // 1))

        # Tree-level: 2 vertices, 1 propagator -> L = 1 - 2 + 1 = 0
        tree = build_diagram([v1, v2], [prop],
                             [(1, 1, 2, 1)])
        @test n_loops(tree) == 0
        @test length(tree.external_legs) == 4  # legs 2,3 of v1 + legs 2,3 of v2

        # Bubble: 2 vertices, 2 propagators -> L = 2 - 2 + 1 = 1
        prop2 = TensorPropagator(:D2, [down(:a), down(:b)], [down(:c), down(:d)],
                                 :q2, TScalar(1 // 1))
        bubble = build_diagram([v1, v2], [prop, prop2],
                               [(1, 1, 2, 1), (1, 2, 2, 2)])
        @test n_loops(bubble) == 1
        @test length(bubble.external_legs) == 2  # legs 3 of v1 + leg 3 of v2

        # Show method
        buf = IOBuffer()
        show(buf, tree)
        s = String(take!(buf))
        @test occursin("0 loops", s)
    end

    @testset "build_diagram validation" begin
        ig2 = [[down(:a), down(:b)], [down(:c), down(:d)]]
        v = TensorVertex(:V, ig2, [:k1, :k2], TScalar(1 // 1))

        prop = TensorPropagator(:D, [down(:a), down(:b)], [down(:c), down(:d)],
                                :q, TScalar(1 // 1))

        # Vertex index out of range
        @test_throws ErrorException build_diagram([v], [prop], [(1, 1, 3, 1)])

        # Leg index out of range
        @test_throws ErrorException build_diagram([v, v], [prop], [(1, 5, 2, 1)])

        # Rank mismatch: vertex with rank-2 legs vs rank-1 legs
        v_scalar = TensorVertex(:Vs, [[down(:a)], [down(:b)]], [:k1, :k2],
                                TScalar(1 // 1))
        @test_throws ErrorException build_diagram([v, v_scalar], [prop], [(1, 1, 2, 1)])

        # Mismatched propagator count
        @test_throws ErrorException build_diagram([v, v], TensorPropagator[],
                                                  [(1, 1, 2, 1)])
    end

    @testset "build_diagram external momenta" begin
        ig3 = [[down(:a), down(:b)], [down(:c), down(:d)], [down(:e), down(:f)]]
        v1 = TensorVertex(:V1, ig3, [:k1, :k2, :k3], TScalar(1 // 1))
        v2 = TensorVertex(:V2, ig3, [:k4, :k5, :k6], TScalar(1 // 1))
        prop = TensorPropagator(:D, [down(:a), down(:b)], [down(:c), down(:d)],
                                :q, TScalar(1 // 1))

        # With custom external momenta
        diag = build_diagram([v1, v2], [prop], [(1, 1, 2, 1)];
                             external_momenta=[:p1, :p2, :p3, :p4])
        @test length(diag.external_legs) == 4
        @test diag.external_legs[1].momentum == :p1
        @test diag.external_legs[4].momentum == :p4

        # Auto-generated momenta (default :p1, :p2, ...)
        diag2 = build_diagram([v1, v2], [prop], [(1, 1, 2, 1)])
        @test diag2.external_legs[1].momentum == :p1
    end

    @testset "tree_exchange_diagram" begin
        ig3 = [[down(:a), down(:b)], [down(:c), down(:d)], [down(:e), down(:f)]]
        v1 = TensorVertex(:V1, ig3, [:k1, :k2, :k3], TScalar(1 // 1))
        v2 = TensorVertex(:V2, ig3, [:k4, :k5, :k6], TScalar(1 // 1))
        prop = TensorPropagator(:D, [down(:a), down(:b)], [down(:c), down(:d)],
                                :q, TScalar(1 // 1))

        diag = tree_exchange_diagram(v1, v2, prop)
        @test n_loops(diag) == 0
        @test length(diag.vertices) == 2
        @test length(diag.propagators) == 1
        @test length(diag.external_legs) == 4

        # Custom leg assignment
        diag2 = tree_exchange_diagram(v1, v2, prop; leg1=2, leg2=3)
        @test n_loops(diag2) == 0
        @test length(diag2.external_legs) == 4
    end

    @testset "vertex_from_perturbation" begin
        # Build a simple expression containing two field occurrences
        h_ab = Tensor(:h, [down(:a), down(:b)])
        h_cd = Tensor(:h, [down(:c), down(:d)])
        g_ac = Tensor(:g, [up(:a), up(:c)])
        bilinear = tproduct(1 // 2, TensorExpr[g_ac, h_ab, h_cd])

        v = vertex_from_perturbation(bilinear, 2, :h)
        @test n_point(v) == 2
        @test v.coupling_order == 0  # 2-2 = 0
        @test v.name == :V2_h
        @test length(v.momenta) == 2

        # Index groups should match the h indices
        @test length(v.index_groups[1]) == 2
        @test length(v.index_groups[2]) == 2

        # With custom name and momenta
        v2 = vertex_from_perturbation(bilinear, 2, :h;
                                       name=:V2_graviton,
                                       momenta=[:p, :q])
        @test v2.name == :V2_graviton
        @test v2.momenta == [:p, :q]
    end

    @testset "vertex_from_perturbation with TSum" begin
        # A sum of bilinear terms (typical perturbation expansion output)
        h1 = Tensor(:h, [down(:a), down(:b)])
        h2 = Tensor(:h, [down(:c), down(:d)])
        term1 = tproduct(1 // 2, TensorExpr[h1, h2])
        term2 = tproduct(-1 // 4, TensorExpr[h1, h2])
        s = TSum([term1, term2])

        v = vertex_from_perturbation(s, 2, :h)
        @test n_point(v) == 2
        @test v.expr isa TSum
    end

    @testset "vertex_from_perturbation with TDeriv" begin
        # Expression with derivatives on the field
        h_ab = Tensor(:h, [down(:a), down(:b)])
        dh = TDeriv(down(:c), h_ab)
        h_de = Tensor(:h, [down(:d), down(:e)])
        expr = tproduct(1 // 1, TensorExpr[dh, h_de])

        v = vertex_from_perturbation(expr, 2, :h)
        @test n_point(v) == 2
    end

    @testset "contract_diagram basic" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
                           [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))
        register_tensor!(reg, TensorProperties(name=:delta, manifold=:M4, rank=(1, 1),
            is_delta=true))

        with_registry(reg) do
            # Simple vertex and propagator expressions
            v_expr = Tensor(:V, [down(:a), down(:b)])
            p_expr = Tensor(:D, [up(:a), up(:b), up(:c), up(:d)])
            v_expr2 = Tensor(:V, [down(:e), down(:f)])

            v1 = TensorVertex(:V1, [[down(:a), down(:b)]], [:k1], v_expr;
                              coupling_order=0)
            v2 = TensorVertex(:V2, [[down(:e), down(:f)]], [:k2], v_expr2;
                              coupling_order=0)
            prop = TensorPropagator(:D, [down(:a), down(:b)], [down(:c), down(:d)],
                                    :q, p_expr)

            diag = tree_exchange_diagram(v1, v2, prop)
            amp = contract_diagram(diag; registry=reg)

            @test amp isa DiagramAmplitude
            @test amp.expr isa TensorExpr
            @test !isempty(amp.external_momenta) || isempty(diag.external_legs)

            # Show
            buf = IOBuffer()
            show(buf, amp)
            s = String(take!(buf))
            @test occursin("DiagramAmplitude", s)
        end
    end

    @testset "DiagramAmplitude show" begin
        da = DiagramAmplitude(
            [[down(:a), down(:b)], [down(:c), down(:d)]],
            [:k1, :k2],
            TScalar(1 // 1)
        )
        buf = IOBuffer()
        show(buf, da)
        s = String(take!(buf))
        @test occursin("4 external indices", s)
        @test occursin("2 momenta", s)
    end

    @testset "n_loops edge cases" begin
        # Empty diagram
        empty_diag = FeynmanDiagram(TensorVertex[], TensorPropagator[],
            Tuple{Int,Int,Int,Int}[],
            @NamedTuple{vertex::Int, leg::Int, momentum::Symbol}[])
        @test n_loops(empty_diag) == 0

        # Single vertex, no propagators: L = 0 - 1 + 1 = 0
        ig = [[down(:a), down(:b)]]
        v = TensorVertex(:V, ig, [:k], TScalar(1 // 1))
        single_v = FeynmanDiagram([v], TensorPropagator[],
            Tuple{Int,Int,Int,Int}[],
            [(vertex=1, leg=1, momentum=:k)])
        @test n_loops(single_v) == 0

        # Triangle: 3 vertices, 3 propagators -> L = 3 - 3 + 1 = 1
        ig3 = [[down(:a), down(:b)], [down(:c), down(:d)], [down(:e), down(:f)]]
        v1 = TensorVertex(:V1, ig3, [:k1, :k2, :k3], TScalar(1 // 1))
        v2 = TensorVertex(:V2, ig3, [:k4, :k5, :k6], TScalar(1 // 1))
        v3 = TensorVertex(:V3, ig3, [:k7, :k8, :k9], TScalar(1 // 1))
        prop = TensorPropagator(:D, [down(:a), down(:b)], [down(:c), down(:d)],
                                :q, TScalar(1 // 1))
        triangle = build_diagram([v1, v2, v3], [prop, prop, prop],
            [(1, 1, 2, 1), (2, 2, 3, 1), (3, 2, 1, 2)])
        @test n_loops(triangle) == 1
    end
end
