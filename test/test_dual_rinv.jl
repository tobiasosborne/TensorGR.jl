@testset "DualRInv: Dual Curvature Invariants with Levi-Civita" begin
    # ---- Helper: standard 4D registry with curvature + epsilon ----
    function drinv_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # ---- Construction from RInv ----
    @testset "DualRInv construction" begin
        kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

        # Left dual on factor 1
        d1 = left_dual(kr, 1)
        @test d1 isa DualRInv
        @test d1.base == kr
        @test d1.dual_positions == [(1, :left)]

        # Right dual on factor 2
        d2 = right_dual(kr, 2)
        @test d2.base == kr
        @test d2.dual_positions == [(2, :right)]

        # Double dual on factor 1
        d3 = double_dual(kr, 1)
        @test d3.dual_positions == [(1, :double)]

        # Multiple duals on different factors
        d4 = left_dual(d2, 1)
        @test length(d4.dual_positions) == 2
        @test (1, :left) in d4.dual_positions
        @test (2, :right) in d4.dual_positions
    end

    # ---- Validation errors ----
    @testset "DualRInv validation" begin
        kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

        # Factor out of range
        @test_throws ErrorException left_dual(kr, 0)
        @test_throws ErrorException left_dual(kr, 3)

        # Invalid side
        @test_throws ErrorException DualRInv(kr, [(1, :invalid)])

        # Duplicate factor
        @test_throws ErrorException DualRInv(kr, [(1, :left), (1, :right)])
    end

    # ---- Composing duals: left + right = double ----
    @testset "Composing duals" begin
        kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

        # left_dual then right_dual on same factor -> double
        d1 = left_dual(kr, 1)
        d2 = right_dual(d1, 1)
        @test d2.dual_positions == [(1, :double)]

        # right_dual then left_dual on same factor -> double
        d3 = right_dual(kr, 1)
        d4 = left_dual(d3, 1)
        @test d4.dual_positions == [(1, :double)]
    end

    # ---- Left dual inserts epsilon on first two indices ----
    @testset "Left dual: epsilon on first pair" begin
        reg = drinv_registry()
        with_registry(reg) do
            kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            d = left_dual(kr, 1)

            expr = to_tensor_expr(d; registry=reg, metric=:g)
            @test expr isa TProduct

            # Should contain exactly 2 Riemann tensors
            riems = filter(f -> f isa Tensor && f.name == :Riem, expr.factors)
            @test length(riems) == 2

            # Should contain exactly 1 epsilon tensor (left dual on one factor)
            epsilons = filter(f -> f isa Tensor && f.name == :εg, expr.factors)
            @test length(epsilons) == 1

            # The epsilon should have 4 indices
            @test length(epsilons[1].indices) == 4

            # Left dual: epsilon_{ext1, ext2}^{int1, int2}
            # First two indices Down (external, in the contraction slot)
            @test epsilons[1].indices[1].position == Down
            @test epsilons[1].indices[2].position == Down

            # Last two indices Up (internal, contract with Riemann's down indices)
            @test epsilons[1].indices[3].position == Up
            @test epsilons[1].indices[4].position == Up

            # No free indices (fully contracted scalar)
            @test isempty(free_indices(expr))

            # Prefactor should include 1/2
            @test expr.scalar == 1 // 2
        end
    end

    # ---- Right dual inserts epsilon on last two indices ----
    @testset "Right dual: epsilon on second pair" begin
        reg = drinv_registry()
        with_registry(reg) do
            kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            d = right_dual(kr, 2)

            expr = to_tensor_expr(d; registry=reg, metric=:g)
            @test expr isa TProduct

            riems = filter(f -> f isa Tensor && f.name == :Riem, expr.factors)
            @test length(riems) == 2

            epsilons = filter(f -> f isa Tensor && f.name == :εg, expr.factors)
            @test length(epsilons) == 1

            # Right dual: epsilon^{int1, int2}_{ext1, ext2}
            # First two indices Up (internal, contract with Riemann's down indices)
            @test epsilons[1].indices[1].position == Up
            @test epsilons[1].indices[2].position == Up
            # Last two indices Down (external, in the contraction slot)
            @test epsilons[1].indices[3].position == Down
            @test epsilons[1].indices[4].position == Down

            @test isempty(free_indices(expr))
            @test expr.scalar == 1 // 2
        end
    end

    # ---- Double dual: two epsilons ----
    @testset "Double dual: two epsilons" begin
        reg = drinv_registry()
        with_registry(reg) do
            kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            d = double_dual(kr, 1)

            expr = to_tensor_expr(d; registry=reg, metric=:g)
            @test expr isa TProduct

            epsilons = filter(f -> f isa Tensor && f.name == :εg, expr.factors)
            @test length(epsilons) == 2

            @test isempty(free_indices(expr))

            # Prefactor: (1/2)^2 = 1/4
            @test expr.scalar == 1 // 4
        end
    end

    # ---- Pontryagin density as DualRInv ----
    @testset "Pontryagin density" begin
        reg = drinv_registry()
        with_registry(reg) do
            p = pontryagin_rinv()
            @test p isa DualRInv
            @test p.base.degree == 2

            # Should have exactly one dual position
            @test length(p.dual_positions) == 1
            @test p.dual_positions[1][2] == :right

            expr = to_tensor_expr(p; registry=reg, metric=:g)
            @test expr isa TProduct

            # Should have 2 Riemann tensors
            riems = filter(f -> f isa Tensor && f.name == :Riem, expr.factors)
            @test length(riems) == 2

            # Should have 1 epsilon tensor
            epsilons = filter(f -> f isa Tensor && f.name == :εg, expr.factors)
            @test length(epsilons) == 1

            # No free indices
            @test isempty(free_indices(expr))
        end
    end

    # ---- Epsilon count ----
    @testset "Epsilon count" begin
        kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

        @test TensorGR._n_epsilons(left_dual(kr, 1)) == 1
        @test TensorGR._n_epsilons(right_dual(kr, 2)) == 1
        @test TensorGR._n_epsilons(double_dual(kr, 1)) == 2

        # Two single duals on different factors
        d = left_dual(right_dual(kr, 2), 1)
        @test TensorGR._n_epsilons(d) == 2
    end

    # ---- Equality ----
    @testset "DualRInv equality" begin
        kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

        d1 = left_dual(kr, 1)
        d2 = left_dual(kr, 1)
        @test d1 == d2

        d3 = right_dual(kr, 1)
        @test d1 != d3  # left vs right on same factor

        # Different factors
        d4 = left_dual(kr, 2)
        @test d1 != d4
    end

    # ---- to_tensor_expr round-trip consistency ----
    @testset "to_tensor_expr round-trip consistency" begin
        reg = drinv_registry()
        with_registry(reg) do
            kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

            # Generate expr from DualRInv, check it is self-consistent
            for (name, drinv) in [
                ("left_dual_f1", left_dual(kr, 1)),
                ("right_dual_f2", right_dual(kr, 2)),
                ("double_dual_f1", double_dual(kr, 1)),
                ("pontryagin", pontryagin_rinv()),
            ]
                expr = to_tensor_expr(drinv; registry=reg, metric=:g)

                # Must be a fully contracted scalar (no free indices)
                @test isempty(free_indices(expr))

                # Must be a TProduct
                @test expr isa TProduct

                # Count factors: k Riemann + n_eps epsilon + metrics
                riems = filter(f -> f isa Tensor && f.name == :Riem, expr.factors)
                epsilons = filter(f -> f isa Tensor && f.name == :εg, expr.factors)
                metrics = filter(f -> f isa Tensor && f.name == :g, expr.factors)

                @test length(riems) == drinv.base.degree
                n_eps = TensorGR._n_epsilons(drinv)
                @test length(epsilons) == n_eps

                # Each epsilon index that is Down contracts with a Riemann index;
                # each epsilon index that is Up sits in the "external" position
                # corresponding to the original contraction slot.
                # Total metric contractions = 2k (same as base RInv)
                @test length(metrics) == 2 * drinv.base.degree
            end
        end
    end

    # ---- Degree-3 dual ----
    @testset "Degree-3 with dual" begin
        # Goroff-Sagnotti type: R R R with one dual
        gs = RInv(3, [11, 12, 5, 6, 3, 4, 9, 10, 7, 8, 1, 2])
        d = left_dual(gs, 2)
        @test d.base.degree == 3
        @test length(d.dual_positions) == 1

        reg = drinv_registry()
        with_registry(reg) do
            expr = to_tensor_expr(d; registry=reg, metric=:g)
            @test isempty(free_indices(expr))

            epsilons = filter(f -> f isa Tensor && f.name == :εg, expr.factors)
            @test length(epsilons) == 1
        end
    end
end
