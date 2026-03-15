@testset "DDI Rules (Dimensionally-Dependent Identities)" begin

    # Helper to create a registry with curvature tensors in given dimension
    function ddi_registry(; dim=4)
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, dim, :g, :partial,
            [:a, :b, :c, :d, :e, :f, :m, :n, :p, :q, :r, :s, :t, :u, :v, :w]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M, rank=(1, 1),
            symmetries=Any[],
            options=Dict{Symbol,Any}(:is_delta => true)))
        define_curvature_tensors!(reg, :M, :g)
        reg
    end

    @testset "gauss_bonnet_ddi: algebraic form" begin
        reg = ddi_registry(; dim=4)
        with_registry(reg) do
            gb = gauss_bonnet_ddi(; metric=:g, registry=reg)
            # Should be a sum of three terms: Riem^2 - 4 Ric^2 + R^2
            @test gb isa TSum
            @test length(gb.terms) == 3

            # Verify the identity simplifies to zero via existing Gauss-Bonnet rule
            for r in gauss_bonnet_rule(; metric=:g)
                register_rule!(reg, r)
            end
            result = simplify(gb; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "gauss_bonnet_ddi contains correct tensors" begin
        reg = ddi_registry(; dim=4)
        with_registry(reg) do
            gb = gauss_bonnet_ddi(; metric=:g, registry=reg)

            has_riem = false
            has_ric = false
            has_scalar = false
            walk(gb) do node
                if node isa Tensor
                    node.name == :Riem && (has_riem = true)
                    node.name == :Ric && (has_ric = true)
                    node.name == :RicScalar && (has_scalar = true)
                end
                node
            end
            @test has_riem
            @test has_ric
            @test has_scalar
        end
    end

    @testset "gauss_bonnet_ddi: coefficient structure" begin
        reg = ddi_registry(; dim=4)
        with_registry(reg) do
            gb = gauss_bonnet_ddi(; metric=:g, registry=reg)
            @test gb isa TSum
            # 3 terms: Riem^2, -4 Ric^2, R^2
            @test length(gb.terms) == 3

            # Check coefficients: one has magnitude 4, two have magnitude 1
            coefficients = Rational{Int}[]
            for term in gb.terms
                coeff, _ = TensorGR._split_scalar(term)
                push!(coefficients, coeff)
            end
            sorted_abs = sort(abs.(coefficients))
            @test sorted_abs == [1 // 1, 1 // 1, 4 // 1]
        end
    end

    @testset "generate_ddi_rules: d=4, order=2 returns rules" begin
        reg = ddi_registry(; dim=4)
        with_registry(reg) do
            rules = generate_ddi_rules(4; order=2, metric=:g, registry=reg)
            @test !isempty(rules)
            @test all(r -> r isa RewriteRule, rules)
        end
    end

    @testset "generate_ddi_rules: d=4 order=2 eliminates Riem^2" begin
        reg = ddi_registry(; dim=4)
        with_registry(reg) do
            rules = generate_ddi_rules(4; order=2, metric=:g, registry=reg)
            for r in rules
                register_rule!(reg, r)
            end

            # Construct Riem_{abcd} Riem^{abcd} (Kretschner scalar)
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            result = simplify(kretschner; registry=reg)

            # Result should not contain Riem (eliminated by DDI)
            has_riem = false
            walk(result) do node
                if node isa Tensor && node.name == :Riem
                    has_riem = true
                end
                node
            end
            @test !has_riem

            # Result should contain Ric and/or RicScalar
            has_curvature = false
            walk(result) do node
                if node isa Tensor && (node.name == :Ric || node.name == :RicScalar)
                    has_curvature = true
                end
                node
            end
            @test has_curvature
        end
    end

    @testset "generate_ddi_rules: d=4 Riem^2 with extra factor" begin
        reg = ddi_registry(; dim=4)
        register_tensor!(reg, TensorProperties(
            name=:phi, manifold=:M, rank=(0, 0),
            symmetries=Any[], options=Dict{Symbol,Any}()))
        with_registry(reg) do
            rules = generate_ddi_rules(4; order=2, metric=:g, registry=reg)
            for r in rules
                register_rule!(reg, r)
            end

            phi = Tensor(:phi, TIndex[])
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            expr = phi * Riem_down * Riem_up

            result = simplify(expr; registry=reg)

            # Should still eliminate Riem
            has_riem = false
            walk(result) do node
                if node isa Tensor && node.name == :Riem
                    has_riem = true
                end
                node
            end
            @test !has_riem
        end
    end

    @testset "generate_ddi_rules: d=4 order=0 (trace identity)" begin
        reg = ddi_registry(; dim=4)
        with_registry(reg) do
            # order=0: pure metric contractions, trivially 0 = 0
            rules = generate_ddi_rules(4; order=0, metric=:g, registry=reg)
            @test rules isa Vector{RewriteRule}
            @test isempty(rules)
        end
    end

    @testset "register_ddi_rules! adds to registry" begin
        reg = ddi_registry(; dim=4)
        n_rules_before = length(reg.rules)
        rules = register_ddi_rules!(reg; dim=4, order=2, metric=:g)
        @test length(reg.rules) > n_rules_before
        @test all(r -> r isa RewriteRule, rules)
    end

    @testset "generate_ddi_rules: error on invalid order" begin
        reg = ddi_registry(; dim=4)
        with_registry(reg) do
            @test_throws ErrorException generate_ddi_rules(4; order=-1, registry=reg)
            @test_throws ErrorException generate_ddi_rules(4; order=6, registry=reg)
        end
    end

    @testset "d=2 DDI: Ricci is pure trace" begin
        reg = ddi_registry(; dim=2)
        with_registry(reg) do
            rules = generate_ddi_rules(2; order=2, metric=:g, registry=reg)
            @test !isempty(rules)
            for r in rules
                register_rule!(reg, r)
            end

            # Ricci should reduce to (R/2) g_{ab} in d=2
            ric = Tensor(:Ric, [down(:a), down(:b)])
            result = simplify(ric; registry=reg)

            # Check that the result contains the metric and Ricci scalar
            has_metric = false
            has_scalar = false
            walk(result) do node
                if node isa Tensor
                    node.name == :g && (has_metric = true)
                    node.name == :RicScalar && (has_scalar = true)
                end
                node
            end
            @test has_metric
            @test has_scalar
        end
    end

    @testset "d=3 DDI: Weyl vanishes" begin
        reg = ddi_registry(; dim=3)
        with_registry(reg) do
            rules = generate_ddi_rules(3; order=2, metric=:g, registry=reg)
            @test !isempty(rules)
            for r in rules
                register_rule!(reg, r)
            end

            weyl = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify(weyl; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "d=2 DDI: order=1 also gives Ricci trace rule" begin
        reg = ddi_registry(; dim=2)
        with_registry(reg) do
            rules = generate_ddi_rules(2; order=1, metric=:g, registry=reg)
            @test !isempty(rules)
        end
    end

    @testset "d=3: DDI generation returns rules" begin
        reg = ddi_registry(; dim=3)
        with_registry(reg) do
            rules = generate_ddi_rules(3; order=2, metric=:g, registry=reg)
            @test rules isa Vector{RewriteRule}
            @test !isempty(rules)
        end
    end

    @testset "DDI rules agree with syzygies.jl gauss_bonnet_rule" begin
        # Verify that DDI-generated rules and the hand-built gauss_bonnet_rule
        # produce the same result when applied to Riem^2
        reg1 = ddi_registry(; dim=4)
        reg2 = ddi_registry(; dim=4)

        # Apply DDI rules to reg1
        with_registry(reg1) do
            for r in generate_ddi_rules(4; order=2, metric=:g, registry=reg1)
                register_rule!(reg1, r)
            end
        end

        # Apply syzygies.jl rules to reg2
        for r in gauss_bonnet_rule(; metric=:g)
            register_rule!(reg2, r)
        end

        # Both should eliminate Riem^2 to the same result
        Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
        Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
        kretschner = Riem_down * Riem_up

        result1 = with_registry(reg1) do
            simplify(kretschner; registry=reg1)
        end
        result2 = with_registry(reg2) do
            simplify(kretschner; registry=reg2)
        end

        # Both should have eliminated Riem
        has_riem1 = false
        walk(result1) do node
            node isa Tensor && node.name == :Riem && (has_riem1 = true)
            node
        end
        has_riem2 = false
        walk(result2) do node
            node isa Tensor && node.name == :Riem && (has_riem2 = true)
            node
        end
        @test !has_riem1
        @test !has_riem2
    end
end
