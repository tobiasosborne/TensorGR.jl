@testset "Riemann DDI (generate_riemann_ddi)" begin

    # Helper to create a registry with curvature tensors in given dimension
    function riem_ddi_registry(; dim=4)
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

    @testset "order=2 reproduces Gauss-Bonnet" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            # riemann_ddi_expr at order=2 should give the Gauss-Bonnet expression
            expr = riemann_ddi_expr(4, 2; metric=:g, registry=reg)
            @test expr isa TSum
            @test length(expr.terms) == 3

            # Check it contains Riem, Ric, RicScalar
            has_riem = false
            has_ric = false
            has_scalar = false
            walk(expr) do node
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

            # Check coefficients: 1, -4, 1
            coefficients = Rational{Int}[]
            for term in expr.terms
                coeff, _ = TensorGR._split_scalar(term)
                push!(coefficients, coeff)
            end
            sorted_abs = sort(abs.(coefficients))
            @test sorted_abs == [1 // 1, 1 // 1, 4 // 1]
        end
    end

    @testset "order=2 simplifies to zero with GB rule" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            expr = riemann_ddi_expr(4, 2; registry=reg)
            for r in gauss_bonnet_rule(; metric=:g)
                register_rule!(reg, r)
            end
            result = simplify(expr; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "generate_riemann_ddi(4, 2) returns rules" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            rules = generate_riemann_ddi(4, 2; metric=:g, registry=reg)
            @test !isempty(rules)
            @test all(r -> r isa RewriteRule, rules)
        end
    end

    @testset "order=2 rules eliminate Riem^2" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            rules = generate_riemann_ddi(4, 2; metric=:g, registry=reg)
            for r in rules
                register_rule!(reg, r)
            end

            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschner = Riem_down * Riem_up

            result = simplify(kretschner; registry=reg)

            has_riem = false
            walk(result) do node
                node isa Tensor && node.name == :Riem && (has_riem = true)
                node
            end
            @test !has_riem
        end
    end

    @testset "order=3 cubic DDI expression structure" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            expr = riemann_ddi_expr(4, 3; metric=:g, registry=reg)
            @test expr isa TSum
            @test length(expr.terms) == 5

            # Check it contains the expected tensor types
            has_triple_riem = false
            has_riem_r = false
            has_ric3 = false
            has_ric2_r = false
            has_r3 = false

            for term in expr.terms
                riem_count = 0
                ric_count = 0
                r_count = 0
                walk(term) do node
                    if node isa Tensor
                        node.name == :Riem && (riem_count += 1)
                        node.name == :Ric && (ric_count += 1)
                        node.name == :RicScalar && (r_count += 1)
                    end
                    node
                end
                if riem_count == 3
                    has_triple_riem = true
                elseif riem_count == 2 && r_count == 1
                    has_riem_r = true
                elseif ric_count == 3
                    has_ric3 = true
                elseif ric_count == 2 && r_count == 1
                    has_ric2_r = true
                elseif r_count == 3
                    has_r3 = true
                end
            end
            @test has_triple_riem
            @test has_riem_r
            @test has_ric3
            @test has_ric2_r
            @test has_r3
        end
    end

    @testset "order=3 cubic DDI coefficients" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            expr = riemann_ddi_expr(4, 3; metric=:g, registry=reg)
            @test expr isa TSum

            # Fulling et al. (1992) Table 1, s=6:
            # I1 - (1/4) I2 + 2 I3 - I4 + (1/4) I5 = 0
            # Extract coefficients by counting Riemann/Ricci/RicScalar factors
            for term in expr.terms
                c, _ = TensorGR._split_scalar(term)
                riem_count = 0
                ric_count = 0
                r_count = 0
                walk(term) do node
                    if node isa Tensor
                        node.name == :Riem && (riem_count += 1)
                        node.name == :Ric && (ric_count += 1)
                        node.name == :RicScalar && (r_count += 1)
                    end
                    node
                end
                if riem_count == 3
                    @test c == 1 // 1  # I1 coefficient
                elseif riem_count == 2 && r_count == 1
                    @test c == -1 // 4  # I2 coefficient
                elseif ric_count == 3
                    @test c == 2 // 1  # I3 coefficient
                elseif ric_count == 2 && r_count == 1
                    @test c == -1 // 1  # I4 coefficient
                elseif r_count == 3
                    @test c == 1 // 4  # I5 coefficient
                end
            end
        end
    end

    @testset "generate_riemann_ddi(4, 3) returns rules" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            rules = generate_riemann_ddi(4, 3; metric=:g, registry=reg)
            @test !isempty(rules)
            @test all(r -> r isa RewriteRule, rules)
        end
    end

    @testset "order=3 rule matches triple Riemann" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            rules = generate_riemann_ddi(4, 3; metric=:g, registry=reg)
            for r in rules
                register_rule!(reg, r)
            end

            # Build a cyclic triple Riemann contraction: R_{abcd}R^{cd}_{ef}R^{efab}
            I1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)]) *
                 Tensor(:Riem, [up(:c), up(:d), down(:e), down(:f)]) *
                 Tensor(:Riem, [up(:e), up(:f), up(:a), up(:b)])

            result = simplify(I1; registry=reg)

            # After applying the cubic DDI rule, the result should not contain
            # three Riemann tensors (the triple contraction is eliminated)
            triple_riem = false
            if result isa TProduct
                riem_count = count(f -> f isa Tensor && f.name == :Riem, result.factors)
                triple_riem = riem_count >= 3
            end
            @test !triple_riem
        end
    end

    @testset "error on invalid order" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            @test_throws ErrorException generate_riemann_ddi(4, 1; registry=reg)
            @test_throws ErrorException generate_riemann_ddi(4, 4; registry=reg)
            @test_throws ErrorException riemann_ddi_expr(4, 1; registry=reg)
        end
    end

    @testset "error on order too large for dimension" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            # order=4 requires n_R=3, p=5, 2*3=6 > 5
            @test_throws ErrorException generate_riemann_ddi(4, 4; registry=reg)
        end
    end

    @testset "error on dim < 4 for order=3" begin
        reg = riem_ddi_registry(; dim=3)
        with_registry(reg) do
            @test_throws ErrorException riemann_ddi_expr(3, 3; registry=reg)
        end
    end

    @testset "order=2 agrees with gauss_bonnet_ddi" begin
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            riem_expr = riemann_ddi_expr(4, 2; metric=:g, registry=reg)
            gb_expr = gauss_bonnet_ddi(; metric=:g, registry=reg)

            # Both should be sums of the same 3 terms (Riem^2, Ric^2, R^2)
            @test riem_expr isa TSum
            @test gb_expr isa TSum
            @test length(riem_expr.terms) == length(gb_expr.terms)

            # The difference should simplify to zero
            for r in gauss_bonnet_rule(; metric=:g)
                register_rule!(reg, r)
            end
            diff = riem_expr - gb_expr
            result = simplify(diff; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "cubic DDI on vacuum (Ric=0, R=0) reduces to I1=0" begin
        # For vacuum (Ricci-flat) solutions, the cubic DDI becomes:
        # I1 - 0 + 0 - 0 + 0 = I1 = 0
        # meaning the cyclic Riemann contraction vanishes.
        # This is consistent with the known result for Schwarzschild.
        reg = riem_ddi_registry(; dim=4)
        with_registry(reg) do
            expr = riemann_ddi_expr(4, 3; registry=reg)

            # Set Ric and RicScalar to zero (vacuum)
            set_vanishing!(reg, :Ric)
            set_vanishing!(reg, :RicScalar)

            result = simplify(expr; registry=reg)

            # After setting Ric=0, R=0, only the I1 term survives
            # and the identity says I1 = 0
            riem_count = 0
            walk(result) do node
                node isa Tensor && node.name == :Riem && (riem_count += 1)
                node
            end
            # Either the result is zero, or it's just the I1 term
            # (which the identity asserts = 0)
            if result != TScalar(0 // 1)
                @test riem_count == 3  # Only I1 term survives
            end
        end
    end
end
