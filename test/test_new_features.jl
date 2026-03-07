@testset "New Features" begin

    # ── G1: Undefine functions ──
    @testset "Undefine" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2)))
        @test has_tensor(reg, :T)
        unregister_tensor!(reg, :T)
        @test !has_tensor(reg, :T)

        register_tensor!(reg, TensorProperties(name=:T2, manifold=:M, rank=(0,2)))
        unregister_tensor!(reg, :T2)
        unregister_manifold!(reg, :M)
        @test !has_manifold(reg, :M)
    end

    # ── G2: VanishingQ ──
    @testset "VanishingQ" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:V, manifold=:M, rank=(0,2)))
        set_vanishing!(reg, :V)

        v = Tensor(:V, [down(:a), down(:b)])
        rules = RewriteRule[r for r in get_rules(reg)]
        result = apply_rules(v, rules)
        @test result == TScalar(0//1)
    end

    # ── C11: DerivativeOrder ──
    @testset "DerivativeOrder" begin
        T = Tensor(:T, [down(:a)])
        @test derivative_order(T) == 0
        @test derivative_order(TDeriv(down(:b), T)) == 1
        @test derivative_order(TDeriv(down(:c), TDeriv(down(:b), T))) == 2
        @test derivative_order(TScalar(1//1)) == 0
    end

    # ── H2: ConstantExprQ / SortedCovDsQ ──
    @testset "ConstantExprQ" begin
        @test is_constant(TScalar(42//1))
        @test is_constant(Tensor(:R, TIndex[]))
        @test !is_constant(Tensor(:T, [down(:a)]))
    end

    @testset "SortedCovDsQ" begin
        T = Tensor(:T, [down(:c)])
        # ∂_a(∂_b(T_c)) — sorted (a < b)
        sorted = TDeriv(down(:a), TDeriv(down(:b), T))
        @test is_sorted_covds(sorted)
        # ∂_b(∂_a(T_c)) — NOT sorted (b > a)
        unsorted = TDeriv(down(:b), TDeriv(down(:a), T))
        @test !is_sorted_covds(unsorted)
    end

    # ── H1: RemoveConstants / RemoveTensors ──
    @testset "RemoveConstants" begin
        T = Tensor(:T, [down(:a), down(:b)])
        expr = tproduct(3//1, TensorExpr[T])
        @test remove_constants(expr) == T

        s = TScalar(5//1)
        @test remove_constants(s) == TScalar(1//1)
    end

    @testset "RemoveTensors" begin
        T = Tensor(:T, [down(:a)])
        @test remove_tensors(T) == TScalar(1//1)
    end

    # ── C6: Lie bracket ──
    @testset "LieBracket" begin
        v = Tensor(:v, [up(:a)])
        w = Tensor(:w, [up(:a)])
        bracket = lie_bracket(v, w)
        @test bracket isa TSum
        # Should have 2 terms: v^b ∂_b w^a - w^b ∂_b v^a
        @test length(bracket.terms) == 2
    end

    # ── A2: Multi-slot symmetry ──
    @testset "FullySymmetric" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d,:e,:f]))

        # Fully symmetric rank-3 tensor
        fs = FullySymmetric(1, 2, 3)
        gens = symmetry_generators([fs], 3)
        # Should generate 2 adjacent transposition generators
        @test length(gens) == 2
    end

    @testset "FullyAntiSymmetric" begin
        fas = FullyAntiSymmetric(1, 2, 3)
        gens = symmetry_generators([fas], 3)
        @test length(gens) == 2
    end

    # ── A5: IndexSort ──
    @testset "IndexSort" begin
        idxs = [down(:c), down(:a), down(:b)]
        sorted = index_sort(idxs)
        @test sorted[1].name == :a
        @test sorted[2].name == :b
        @test sorted[3].name == :c
    end

    # ── A6: SameDummies ──
    @testset "SameDummies" begin
        T1 = tproduct(1//1, TensorExpr[Tensor(:T, [down(:x), up(:x)])])
        T2 = tproduct(1//1, TensorExpr[Tensor(:T, [down(:y), up(:y)])])
        s = TSum([T1, T2])
        result = same_dummies(s)
        @test result isa TSum
    end

    # ── B6: FlatMetricQ ──
    @testset "FlatMetric" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(name=:δ, manifold=:M, rank=(1,1),
            options=Dict{Symbol,Any}(:is_delta => true)))
        define_curvature_tensors!(reg, :M, :g)

        @test !is_flat(reg, :g)
        set_flat!(reg, :g)
        @test is_flat(reg, :g)

        # Rules should vanish curvature tensors
        riem = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
        rules = RewriteRule[r for r in get_rules(reg)]
        result = apply_rules(riem, rules)
        @test result == TScalar(0//1)
    end

    # ── B7: Frozen metrics ──
    @testset "FrozenMetric" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))

        @test !is_frozen(reg, :g)
        freeze_metric!(reg, :g)
        @test is_frozen(reg, :g)
        unfreeze_metric!(reg, :g)
        @test !is_frozen(reg, :g)
    end

    # ── B4: SeparateMetric ──
    @testset "SeparateMetric" begin
        T = Tensor(:T, [up(:a), down(:b)])
        result = separate_metric(T, :a, :g)
        # Should insert metric to separate index a
        @test result isa TProduct || result isa TSum
    end

    # ── B8: Generalized Kronecker delta ──
    @testset "GeneralizedDelta" begin
        # 1-index case: just ordinary delta
        result = gdelta([up(:a)], [down(:b)])
        @test result isa Tensor
        @test result.name == :δ

        # 2-index case: δ^{ab}_{cd} = δ^a_c δ^b_d - δ^a_d δ^b_c
        result2 = gdelta([up(:a), up(:b)], [down(:c), down(:d)])
        @test result2 isa TSum
        @test length(result2.terms) == 2
    end

    # ── F7: CTensor inverse and determinant ──
    @testset "CTensor inverse/det" begin
        g_data = Float64[1 0; 0 1]
        ct = CTensor(g_data, :cart, [Down, Down])
        inv_ct = ctensor_inverse(ct)
        @test inv_ct.data ≈ g_data
        @test inv_ct.positions == [Up, Up]

        d = ctensor_det(ct)
        @test d ≈ 1.0
    end

    # ── F8: MetricCompute Einstein ──
    @testset "MetricCompute Einstein" begin
        # 2D metric: g = diag(1, 1)
        dim = 2
        g = Float64[1 0; 0 1]
        ginv = Float64[1 0; 0 1]
        coords = [:x, :y]
        deriv_fn(expr, coord) = 0.0  # constant metric

        Gamma = metric_christoffel(g, ginv, coords; deriv_fn=deriv_fn)
        Riem = metric_riemann(Gamma, dim; coords=coords, deriv_fn=deriv_fn)
        Ric = metric_ricci(Riem, dim)
        R = metric_ricci_scalar(Ric, ginv, dim)
        G = metric_einstein(Ric, R, g, dim)

        # Flat metric: G should be zero
        for a in 1:dim, b in 1:dim
            @test G[a,b] ≈ 0.0 atol=1e-10
        end
    end

    # ── J1: MakeRule with symmetries ──
    @testset "MakeRule" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2),
            symmetries=Any[Symmetric(1,2)]))

        lhs = Tensor(:T, [down(:a), down(:b)])
        rhs = TScalar(0//1)
        rules = with_registry(reg) do
            make_rule(lhs, rhs; use_symmetries=true, registry=reg)
        end
        # Should generate at least the original + symmetry variant
        @test length(rules) >= 1
    end

    # ── J3: FoldedRule ──
    @testset "FoldedRule" begin
        r1 = RewriteRule[RewriteRule(Tensor(:A, TIndex[]), TScalar(1//1))]
        r2 = RewriteRule[RewriteRule(TScalar(1//1), TScalar(2//1))]
        f = folded_rule(r1, r2)
        result = f(Tensor(:A, TIndex[]))
        @test result == TScalar(2//1)
    end

    # ── E4: DefTensorPerturbation ──
    @testset "DefTensorPerturbation" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2),
            symmetries=Any[Symmetric(1,2)]))
        define_tensor_perturbation!(reg, :T, :δT)
        @test has_tensor(reg, :δT)
        tp = get_tensor(reg, :δT)
        @test tp.rank == (0, 2)
    end

    # ── E5: PerturbationOrder ──
    @testset "PerturbationOrder" begin
        perts = Set([:h])
        @test perturbation_order(Tensor(:h, [down(:a), down(:b)]), perts) == 1
        @test perturbation_order(Tensor(:g, [down(:a), down(:b)]), perts) == 0

        p = tproduct(1//1, TensorExpr[Tensor(:h, [down(:a), down(:b)]),
                                       Tensor(:h, [down(:c), down(:d)])])
        @test perturbation_order(p, perts) == 2
    end

    # ── E9: BackgroundSolution ──
    @testset "BackgroundSolution" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:Ric, manifold=:M, rank=(0,2)))
        background_solution!(reg, [:Ric])

        ric = Tensor(:Ric, [down(:a), down(:b)])
        rules = RewriteRule[r for r in get_rules(reg)]
        result = apply_rules(ric, rules)
        @test result == TScalar(0//1)
    end

    # ── G10: ScalarFunction ──
    @testset "ScalarFunction" begin
        reg = TensorRegistry()
        define_scalar_function!(reg, :sin; derivative=:cos)
        @test scalar_function_derivative(reg, :sin) == :cos
    end

    # ── C2: ChristoffelToGradMetric ──
    @testset "ChristoffelToGradMetric" begin
        expr = christoffel_to_grad_metric(:g, up(:a), down(:b), down(:c))
        @test expr isa TProduct || expr isa TSum
    end

    # ── I1: Codifferential ──
    @testset "Codifferential" begin
        α = Tensor(:ω, [down(:a)])
        result = codifferential(α, :ε, :g, 1, 4)
        @test result isa TProduct || result isa TDeriv || result isa TSum
    end

    # ── I2: Cartan formula ──
    @testset "CartanFormula" begin
        v = Tensor(:v, [up(:a)])
        α = Tensor(:ω, [down(:b)])
        result = cartan_lie_d(v, α, 1, down(:c))
        @test result isa TSum
    end

    # ── B1: define_metric! ──
    @testset "DefineMetric" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d,:e,:f]))
        with_registry(reg) do
            define_metric!(reg, :g; manifold=:M)
        end
        # Should have metric, delta, epsilon, Riemann, Ricci, etc.
        @test has_tensor(reg, :g)
        @test has_tensor(reg, :δ)
        @test has_tensor(reg, :εg)  # epsilon
        @test has_tensor(reg, :Riem)
        @test has_tensor(reg, :Ric)
        @test has_tensor(reg, :RicScalar)
        @test has_tensor(reg, :Ein)
        @test has_tensor(reg, :Weyl)
    end

    # ── B2: Signature ──
    @testset "MetricSignature" begin
        sig = lorentzian(4)
        @test sig.signs == [-1, 1, 1, 1]
        @test sign_det(sig) == -1

        sig_e = euclidean(3)
        @test sig_e.signs == [1, 1, 1]
        @test sign_det(sig_e) == 1
    end

    # ── B5: Metric determinant ──
    @testset "MetricDeterminant" begin
        d = metric_det_expr(:g)
        @test d isa TScalar
        @test d.val == :det_g

        sd = sqrt_det_expr(:g)
        @test sd isa TScalar
    end

    # ── A3: ImposeSymmetry ──
    @testset "ImposeSymmetry" begin
        T = Tensor(:T, [down(:a), down(:b)])
        result = impose_symmetry(T, :symmetric, [:a, :b])
        @test result isa TSum
    end

    # ── A4: Young tableaux ──
    @testset "YoungTableaux" begin
        yt = YoungTableau([[1, 2], [3]])
        @test young_shape(yt) == [2, 1]

        T = Tensor(:T, [down(:a), down(:b), down(:c)])
        result = young_symmetrize(T, yt, [:a, :b, :c])
        @test result isa TensorExpr
    end

    # ── C10: SortCovDsToBox ──
    @testset "SortCovDsToBox" begin
        T = Tensor(:T, TIndex[])
        expr = TDeriv(down(:a), TDeriv(up(:a), T))
        result = sort_covds_to_box(expr)
        @test result isa TDeriv
    end

    # ── G9: Validate ──
    @testset "Validate" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2)))

        T = Tensor(:T, [down(:a), down(:b)])
        issues = with_registry(reg) do
            validate(T; registry=reg)
        end
        @test isempty(issues)

        T_bad = Tensor(:T, [down(:a)])
        issues_bad = with_registry(reg) do
            validate(T_bad; registry=reg)
        end
        @test !isempty(issues_bad)
    end

    # ── D2: RiemannToChristoffel ──
    @testset "RiemannToChristoffel" begin
        expr = riemann_to_christoffel(up(:a), down(:b), down(:c), down(:d), :Gamma)
        @test expr isa TSum
    end

    # ── D4: Kretschmann ──
    @testset "Kretschmann" begin
        K = kretschmann_expr(:g)
        @test K isa TProduct
    end

    # ── F5: ComponentValue storage ──
    @testset "ComponentStore" begin
        store = ComponentStore(:g, :cart, 2; symmetries=Any[Symmetric(1,2)])
        set_component!(store, [1, 1], 1.0)
        set_component!(store, [1, 2], 0.0)
        set_component!(store, [2, 2], 1.0)

        @test get_component(store, [1, 1]) == 1.0
        @test get_component(store, [2, 1]) == 0.0  # symmetric

        indep = independent_components(store)
        @test length(indep) == 3
    end

    # ── E7: MetricVariation ──
    @testset "MetricVariation" begin
        g_up = Tensor(:g, [up(:a), up(:b)])
        result = metric_variation(g_up, :g, down(:c), down(:d))
        @test result != TScalar(0//1)  # non-zero variation
    end

    # ── C3: Torsion tensor ──
    @testset "TorsionTensor" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        covd = define_covd!(reg, :D; manifold=:M, metric=:g, torsion_free=false)
        torsion_name = Symbol(:T, :D)
        @test has_tensor(reg, torsion_name)
        tp = get_tensor(reg, torsion_name)
        @test any(s -> s isa AntiSymmetric, tp.symmetries)
    end

    # ── C9: SymmetrizeCovDs ──
    @testset "SymmetrizeCovDs" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(name=:δ, manifold=:M, rank=(1,1),
            options=Dict{Symbol,Any}(:is_delta => true)))
        define_curvature_tensors!(reg, :M, :g)
        define_covd!(reg, :D; manifold=:M, metric=:g)

        T = Tensor(:T, [down(:c)])
        expr = TDeriv(down(:a), TDeriv(down(:b), T))
        result = with_registry(reg) do
            symmetrize_covds(expr, :D; registry=reg)
        end
        @test result isa TSum
    end

    # ── H5: IndexCollect ──
    @testset "IndexCollect" begin
        T1 = tproduct(3//1, TensorExpr[Tensor(:T, [down(:a), down(:b)])])
        T2 = tproduct(2//1, TensorExpr[Tensor(:T, [down(:a), down(:b)])])
        s = TSum([T1, T2])
        result = index_collect(s, :T)
        @test !isempty(result)
    end

    # ── D1: ContractCurvature ──
    @testset "ContractCurvature" begin
        riem = Tensor(:Riem, [up(:a), down(:b), down(:a), down(:d)])
        result = contract_curvature(riem)
        @test result isa Tensor || result isa TProduct
        if result isa Tensor
            @test result.name == :Ric
        end

        ric = Tensor(:Ric, [up(:a), down(:a)])
        result2 = contract_curvature(ric)
        @test result2 isa Tensor
        @test result2.name == :RicScalar
    end

    # ── D5: Schouten ──
    @testset "Schouten" begin
        expr = schouten_to_ricci(down(:a), down(:b), :g; dim=4)
        @test expr isa TProduct || expr isa TSum
    end

    # ── D6: ToRiemann/ToRicci ──
    @testset "ToRiemann" begin
        ein = Tensor(:Ein, [down(:a), down(:b)])
        result = to_riemann(ein; metric=:g, dim=4)
        @test result != ein
    end

    @testset "ToRicci" begin
        sch = Tensor(:Sch, [down(:a), down(:b)])
        result = to_ricci(sch; metric=:g, dim=4)
        @test result != sch
    end

    # ── G8: Dagger ──
    @testset "Dagger" begin
        T = Tensor(:T, [up(:a), down(:b)])
        Td = dagger(T)
        @test Td isa Tensor
        @test Td.name == Symbol("T_dag")
        @test Td.indices[1].position == Down  # swapped
        @test Td.indices[2].position == Up    # swapped

        s = TScalar(3//1)
        @test dagger(s) == s  # scalars unchanged
    end

    # ── E6: Higher-order gauge ──
    @testset "GaugeOrder2" begin
        h = Tensor(:h, [down(:a), down(:b)])
        ξ = Tensor(:ξ, [up(:c)])
        result = gauge_transformation(h, ξ, :g; order=2)
        @test result isa TSum
    end

    # ── B9: Conformal ──
    @testset "Conformal" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(name=:g2, manifold=:M, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        set_conformal_to!(reg, :g, :g2, :f)
        @test get_tensor(reg, :g).options[:conformal_to] == :g2
    end

    # ── ExpandPerturbation ──
    @testset "ExpandPerturbation" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(name=:δ, manifold=:M, rank=(1,1),
            options=Dict{Symbol,Any}(:is_delta => true)))
        mp = define_metric_perturbation!(reg, :g, :h)

        # δΓ at order 1
        result = with_registry(reg) do
            δchristoffel(mp, up(:a), down(:b), down(:c), 1)
        end
        @test result isa TensorExpr
        @test result != TScalar(0//1)
    end

    # ── Connection forms ──
    @testset "ConnectionForms" begin
        cf = connection_form(:Γ, up(:a), down(:b), down(:c))
        @test cf isa Tensor

        # Curvature form
        Ω = curvature_form(:Γ, up(:a), down(:b), down(:c), down(:d))
        @test Ω isa TensorExpr
    end

    # ── Cartan structure equations ──
    @testset "CartanStructure" begin
        T = cartan_first_structure(:T, :Γ, :θ, up(:a), down(:b), down(:c))
        @test T isa TensorExpr

        Ω = cartan_second_structure(:Ω, :Γ, up(:a), down(:b), down(:c), down(:d))
        @test Ω isa TensorExpr
    end

    # ── A8: SplitIndex ──
    @testset "SplitIndex" begin
        T = Tensor(:T, [down(:a)])
        result = split_index(T, :a, 3)
        @test result isa TSum
        @test length(result.terms) == 3
    end

    # ── F6: BasisChange ──
    @testset "BasisChange" begin
        v = CTensor([1.0, 0.0, 0.0], :cart, [Up])
        J = Float64[0 1 0; 1 0 0; 0 0 1]  # swap x,y
        v2 = basis_change(v, J)
        @test v2.data ≈ [0.0, 1.0, 0.0]

        # Rank-2 test
        g = CTensor(Float64[1 0; 0 1], :cart, [Down, Down])
        J2 = Float64[2 0; 0 1]
        g2 = basis_change(g, J2)
        @test g2.data[1,1] ≈ 0.25  # g'_11 = J^{-1 a}_1 J^{-1 b}_1 g_{ab} = (1/2)^2
    end

    # ── Cotton tensor (3D) ──
    @testset "Cotton tensor" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M3 dim=3 metric=g
            define_curvature_tensors!(reg, :M3, :g)
            ct = cotton_expr(down(:a), down(:b), :g; epsilon=:εg, dim=3)
            @test ct isa TProduct
        end
    end

    # ── Tensor norm ──
    @testset "Tensor norm" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            R = Tensor(:Ric, [down(:a), down(:b)])
            n = tensor_norm(R, :g; registry=reg)
            @test n isa TProduct
            @test length(n.factors) == 2
        end
    end

    # ── Hypersurface ──
    @testset "Hypersurface geometry" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=G
            hs = define_hypersurface!(reg, :Σ; ambient=:M4, metric=:G,
                                       normal_name=:n, signature=-1)
            @test hs.dim_surface == 3
            @test hs.signature == -1
            @test has_tensor(reg, :n)
            @test has_tensor(reg, :K)
            @test has_tensor(reg, :γ)

            # Induced metric: γ_{ab} = G_{ab} + n_a n_b (timelike normal, σ=1)
            γ = induced_metric_expr(down(:a), down(:b), :G, :n; signature=-1)
            @test γ isa TSum
            @test length(γ.terms) == 2

            # Projector: P^a_b = δ^a_b + n^a n_b
            P = projector_expr(up(:a), down(:b), :n; signature=-1)
            @test P isa TSum
        end
    end

    # ── wedge_power ──
    @testset "Wedge power" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            register_tensor!(reg, TensorProperties(name=:ω, manifold=:M4, rank=(0,2),
                symmetries=Any[AntiSymmetric(1,2)]))
            ω = Tensor(:ω, [down(:a), down(:b)])
            wp0 = wedge_power(ω, 2, 0)
            @test wp0 == TScalar(1//1)
            wp1 = wedge_power(ω, 2, 1)
            @test wp1 == ω
            wp2 = wedge_power(ω, 2, 2)
            @test wp2 isa TProduct
        end
    end

    # ── Topological densities ──
    @testset "Topological densities" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)

            # Pontryagin: ε^{abcd} R_{ab}^{ef} R_{cdef}
            pont = pontryagin_density(:g; registry=reg)
            @test pont isa TProduct
            @test length(pont.factors) == 3

            # Euler: Riem² - 4Ric² + R²
            euler = euler_density(:g; registry=reg)
            @test euler isa TSum
            @test length(euler.terms) == 3

            # Chern-Simons action: ϑ * ★(R∧R)
            register_tensor!(reg, TensorProperties(name=:ϑ, manifold=:M4, rank=(0,0)))
            cs = chern_simons_action(Tensor(:ϑ, TIndex[]), :g; registry=reg)
            @test cs isa TProduct
        end
    end

    # ── Background rules ──
    @testset "Maximally-symmetric background" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            maximally_symmetric_background!(reg, :M4; metric=:g)

            # R = 4Λ
            R = simplify(Tensor(:RicScalar, TIndex[]))
            @test R isa TProduct
            @test R.scalar == 4 // 1

            # Ric_{ab} = Λ g_{ab}
            Ric = simplify(Tensor(:Ric, [down(:a), down(:b)]))
            @test Ric isa TProduct
        end
    end

    @testset "Vacuum background" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            vacuum_background!(reg, :M4; metric=:g)

            @test simplify(Tensor(:RicScalar, TIndex[])) == TScalar(0//1)
            @test simplify(Tensor(:Ric, [down(:a), down(:b)])) == TScalar(0//1)
        end
    end

    # ── Spin projectors ──
    @testset "Barnes-Rivers spin projectors" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=η
            register_tensor!(reg, TensorProperties(name=:k, manifold=:M4, rank=(0,1)))

            # θ_{μν} = η_{μν} - k_μ k_ν / k²
            θ = theta_projector(down(:a), down(:b); metric=:η)
            @test θ isa TSum
            @test length(θ.terms) == 2

            # ω_{μν} = k_μ k_ν / k²
            ω = omega_projector(down(:a), down(:b))
            @test ω isa TProduct

            # Completeness: θ + ω = η (structurally, these are the two parts)
            # P^(2) exists and is a sum
            p2 = spin2_projector(down(:a), down(:b), down(:c), down(:d); dim=4, metric=:η)
            @test p2 isa TSum

            # P^(1) exists
            p1 = spin1_projector(down(:a), down(:b), down(:c), down(:d); metric=:η)
            @test p1 isa TProduct || p1 isa TSum

            # P^(0-s) and P^(0-w) exist
            p0s = spin0s_projector(down(:a), down(:b), down(:c), down(:d); dim=4, metric=:η)
            @test p0s isa TProduct || p0s isa TSum
            p0w = spin0w_projector(down(:a), down(:b), down(:c), down(:d))
            @test p0w isa TProduct

            # Transfer operators exist
            tsw = transfer_sw(down(:a), down(:b), down(:c), down(:d); dim=4, metric=:η)
            @test tsw isa TProduct || tsw isa TSum
        end
    end

    # ── Box operator + scalar CovD helpers ──
    @testset "Box operator" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            phi = Tensor(:phi, TIndex[])
            register_tensor!(reg, TensorProperties(name=:phi, manifold=:M4, rank=(0,0)))

            # box(φ, :g) = g^{ab} ∂_a ∂_b φ
            b = box(phi, :g)
            @test b isa TProduct
            @test length(b.factors) == 2  # g^{ab} and ∂_a(∂_b(φ))
            # The metric factor should have two up indices
            gfactor = b.factors[1]
            @test gfactor isa Tensor && gfactor.name == :g
            @test all(idx -> idx.position == Up, gfactor.indices)

            # grad_squared(φ, :g) = g^{ab} ∂_a φ ∂_b φ
            gs = grad_squared(phi, :g)
            @test gs isa TProduct
            @test length(gs.factors) == 3  # g^{ab}, ∂_a φ, ∂_b φ

            # covd_chain
            chain = covd_chain(phi, [down(:a), down(:b), down(:c)])
            @test chain isa TDeriv
            @test chain.index == down(:a)
            @test chain.arg isa TDeriv
            @test chain.arg.index == down(:b)

            # covd_product
            cp = covd_product(phi, down(:a), down(:b))
            @test cp isa TProduct

            # Scalar commutation: [∂_a, ∂_b]φ = 0
            expr = covd_chain(phi, [down(:b), down(:a)])  # ∂_b(∂_a(φ))
            sorted = commute_covds(expr, :∇g)
            # Should swap to ∂_a(∂_b(φ)) with zero commutator
            @test sorted isa TDeriv
            @test sorted.index.name == :a
        end
    end

    # ── G7: Tensor weight ──
    @testset "TensorWeight" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M, rank=(0,2),
            weight=1))
        tp = get_tensor(reg, :T)
        @test tp.weight == 1
    end
end
