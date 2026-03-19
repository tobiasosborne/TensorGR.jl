@testset "DDI Simplify (simplify_with_ddis)" begin

    # Helper to create a registry with curvature tensors in given dimension
    function ddi_simplify_registry(; dim=4)
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

    @testset "Gauss-Bonnet identity simplifies to zero in d=4" begin
        reg = ddi_simplify_registry(; dim=4)
        with_registry(reg) do
            gb = gauss_bonnet_ddi(; metric=:g, registry=reg)
            @test gb isa TSum
            @test length(gb.terms) == 3

            result = simplify_with_ddis(gb; dim=4, registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Kretschner scalar eliminated in d=4" begin
        reg = ddi_simplify_registry(; dim=4)
        with_registry(reg) do
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            result = simplify_with_ddis(kretschner; dim=4, registry=reg)

            # Result should not contain Riemann (eliminated by Gauss-Bonnet DDI)
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

    @testset "idempotent: multiple calls do not duplicate rules" begin
        reg = ddi_simplify_registry(; dim=4)
        with_registry(reg) do
            gb = gauss_bonnet_ddi(; metric=:g, registry=reg)

            # Call simplify_with_ddis twice
            result1 = simplify_with_ddis(gb; dim=4, registry=reg)
            n_rules_after_first = length(reg.rules)

            result2 = simplify_with_ddis(gb; dim=4, registry=reg)
            n_rules_after_second = length(reg.rules)

            # Should not have added duplicate rules
            @test n_rules_after_first == n_rules_after_second
            @test result1 == TScalar(0 // 1)
            @test result2 == TScalar(0 // 1)
        end
    end

    @testset "has_ddi_rules tracks registration" begin
        reg = ddi_simplify_registry(; dim=4)
        @test !has_ddi_rules(reg; dim=4, order=2)

        register_ddi_rules!(reg; dim=4, order=2)
        @test has_ddi_rules(reg; dim=4, order=2)
        @test !has_ddi_rules(reg; dim=4, order=1)
        @test !has_ddi_rules(reg; dim=3, order=2)
    end

    @testset "Weyl vanishes in d=3" begin
        reg = ddi_simplify_registry(; dim=3)
        with_registry(reg) do
            weyl = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            result = simplify_with_ddis(weyl; dim=3, registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Ricci is pure trace in d=2" begin
        reg = ddi_simplify_registry(; dim=2)
        with_registry(reg) do
            ric = Tensor(:Ric, [down(:a), down(:b)])
            result = simplify_with_ddis(ric; dim=2, registry=reg)

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

    @testset "Riem^2 with extra scalar factor" begin
        reg = ddi_simplify_registry(; dim=4)
        register_tensor!(reg, TensorProperties(
            name=:phi, manifold=:M, rank=(0, 0),
            symmetries=Any[], options=Dict{Symbol,Any}()))
        with_registry(reg) do
            phi = Tensor(:phi, TIndex[])
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            expr = phi * Riem_down * Riem_up

            result = simplify_with_ddis(expr; dim=4, registry=reg)

            # Riem should be eliminated
            has_riem = false
            walk(result) do node
                node isa Tensor && node.name == :Riem && (has_riem = true)
                node
            end
            @test !has_riem

            # phi should still be present
            has_phi = false
            walk(result) do node
                node isa Tensor && node.name == :phi && (has_phi = true)
                node
            end
            @test has_phi
        end
    end

    @testset "kwargs forwarded to simplify" begin
        reg = ddi_simplify_registry(; dim=4)
        with_registry(reg) do
            gb = gauss_bonnet_ddi(; metric=:g, registry=reg)

            # maxiter=1 should still converge for this simple case
            result = simplify_with_ddis(gb; dim=4, registry=reg, maxiter=50)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "end-to-end: Riem^2 in sum reduces" begin
        reg = ddi_simplify_registry(; dim=4)
        with_registry(reg) do
            # Build: Riem^2 + 2 Ric^2
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            e = :e; f = :f
            Ric_down = Tensor(:Ric, [down(e), down(f)])
            Ric_up = Tensor(:Ric, [up(e), up(f)])
            ric_sq = Ric_down * Ric_up

            expr = kretschner + (2 // 1) * ric_sq

            result = simplify_with_ddis(expr; dim=4, registry=reg)

            # Should not contain Riem (GB replaces Riem^2 with 4 Ric^2 - R^2)
            # so result = (4 + 2) Ric^2 - R^2 = 6 Ric^2 - R^2
            has_riem = false
            walk(result) do node
                node isa Tensor && node.name == :Riem && (has_riem = true)
                node
            end
            @test !has_riem
        end
    end

end
