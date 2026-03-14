@testset "Curvature Syzygies" begin
    function syzygy_registry(; dim=4)
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, dim, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M, rank=(1,1),
            symmetries=Any[],
            options=Dict{Symbol,Any}(:is_delta => true)))
        define_curvature_tensors!(reg, :M, :g)
        reg
    end

    @testset "Gauss-Bonnet rule creation" begin
        rules = gauss_bonnet_rule(; metric=:g)
        @test length(rules) >= 1
        @test all(r -> r isa RewriteRule, rules)
    end

    @testset "Gauss-Bonnet: Riem^2 -> 4 Ric^2 - R^2" begin
        reg = syzygy_registry(; dim=4)
        for r in gauss_bonnet_rule(; metric=:g)
            register_rule!(reg, r)
        end
        with_registry(reg) do
            # Construct Riem_{abcd} Riem^{abcd}
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            result = simplify(kretschner)

            # Result should not contain :Riem anymore
            has_riem = false
            walk(result) do node
                if node isa Tensor && node.name == :Riem
                    has_riem = true
                end
                node
            end
            @test !has_riem

            # Result should contain Ric and RicScalar terms
            has_ric = false
            has_scalar = false
            walk(result) do node
                if node isa Tensor && node.name == :Ric
                    has_ric = true
                end
                if node isa Tensor && node.name == :RicScalar
                    has_scalar = true
                end
                node
            end
            @test has_ric
            @test has_scalar
        end
    end

    @testset "Gauss-Bonnet: Riem^2 with extra factor" begin
        reg = syzygy_registry(; dim=4)
        for r in gauss_bonnet_rule(; metric=:g)
            register_rule!(reg, r)
        end
        with_registry(reg) do
            # phi * Riem_{abcd} Riem^{abcd}
            register_tensor!(reg, TensorProperties(
                name=:phi, manifold=:M, rank=(0,0),
                symmetries=Any[],
                options=Dict{Symbol,Any}()))
            phi = Tensor(:phi, TIndex[])
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            expr = phi * Riem_down * Riem_up

            result = simplify(expr)

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

    @testset "Weyl vanishing in D=3" begin
        rules = weyl_vanishing_rule()
        @test length(rules) == 1

        reg = syzygy_registry(; dim=3)
        for r in rules
            register_rule!(reg, r)
        end
        with_registry(reg) do
            weyl = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify(weyl)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Weyl vanishing in product" begin
        reg = syzygy_registry(; dim=3)
        for r in weyl_vanishing_rule()
            register_rule!(reg, r)
        end
        with_registry(reg) do
            weyl = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M, rank=(1,0),
                symmetries=Any[],
                options=Dict{Symbol,Any}()))
            v = Tensor(:V, [up(:e)])
            expr = weyl * v
            result = simplify(expr)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Ricci trace rule in D=2" begin
        rules = ricci_trace_rule(; metric=:g, dim=2)
        @test length(rules) == 1

        reg = syzygy_registry(; dim=2)
        for r in rules
            register_rule!(reg, r)
        end
        with_registry(reg) do
            ric = Tensor(:Ric, [down(:a), down(:b)])
            result = simplify(ric)
            # Should be (1/2) g_{ab} R
            @test result isa TProduct
            # Check it contains the metric and Ricci scalar
            has_metric = false
            has_scalar = false
            walk(result) do node
                if node isa Tensor && node.name == :g
                    has_metric = true
                end
                if node isa Tensor && node.name == :RicScalar
                    has_scalar = true
                end
                node
            end
            @test has_metric
            @test has_scalar
        end
    end

    @testset "Riemann vanishing in D=1" begin
        rules = riemann_vanishing_rule()
        @test length(rules) == 1

        reg = syzygy_registry(; dim=1)
        for r in rules
            register_rule!(reg, r)
        end
        with_registry(reg) do
            riem = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify(riem)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "syzygy_rules convenience" begin
        # D=4: should get Gauss-Bonnet
        rules4 = syzygy_rules(; dim=4)
        @test length(rules4) >= 1

        # D=3: should get Weyl vanishing + Gauss-Bonnet(? no, only <= 3 for Weyl)
        rules3 = syzygy_rules(; dim=3)
        @test length(rules3) >= 1

        # D=2: Weyl vanishing + Ricci trace
        rules2 = syzygy_rules(; dim=2)
        @test length(rules2) >= 2

        # D=1: Riemann vanishing only
        rules1 = syzygy_rules(; dim=1)
        @test length(rules1) >= 1
    end

    @testset "Fully contracted indices detection" begin
        # Internal function test: _are_fully_contracted
        idxs1 = [down(:a), down(:b), down(:c), down(:d)]
        idxs2 = [up(:a), up(:b), up(:c), up(:d)]
        @test TensorGR._are_fully_contracted(idxs1, idxs2)

        # Not contracted: same positions
        idxs3 = [down(:a), down(:b), down(:c), down(:d)]
        @test !TensorGR._are_fully_contracted(idxs1, idxs3)

        # Partial contraction
        idxs4 = [up(:a), up(:b), down(:c), down(:d)]
        @test !TensorGR._are_fully_contracted(idxs1, idxs4)
    end
end
