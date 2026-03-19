# Ground truth: McCrea, CQG 9, 553 (1992); Hehl et al, Phys. Rep. 258 (1995);
#               Helpin & Volkov, arXiv:2407.18019.

# Include the source into TensorGR module (orchestrator will add to TensorGR.jl later)
Base.include(TensorGR, joinpath(@__DIR__, "..", "src", "metric_affine", "brauer.jl"))

using TensorGR: brauer_piece_names, brauer_piece_dimensions, brauer_symmetric_split,
                brauer_decompose, BrauerDecomposition,
                _ma_ricci_trace, _ma_scalar_trace

@testset "Brauer algebra 11-piece decomposition" begin

    # ── Setup helper ──
    function _brauer_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
        end
        ac = define_affine_connection!(reg, :Gamma; manifold=:M4, metric=:g)
        fs = define_ma_curvature!(reg, ac)
        reg, ac, fs
    end

    @testset "Piece names" begin
        names = brauer_piece_names()
        @test length(names) == 11
        @test :WEYL_S in names
        @test :SCALAR_S in names
        @test :RICCI_S in names
        @test :PAIRSYM_S in names
        @test :RICANTI_S in names
        @test :PAIRSKEW_S in names
        @test :WEYL_A in names
        @test :RICCI_Z1 in names
        @test :RICCI_Z2 in names
        @test :PAIRSKEW_A in names
        @test :PAIRSYM_A in names
        # All names are unique
        @test length(Set(names)) == 11
    end

    @testset "Dimensions sum to 96 for d=4" begin
        dims = brauer_piece_dimensions(4)
        @test length(dims) == 11
        @test sum(values(dims)) == 96

        # Individual dimensions
        @test dims[:WEYL_S] == 10
        @test dims[:RICCI_S] == 9
        @test dims[:SCALAR_S] == 1
        @test dims[:RICANTI_S] == 6

        # Symmetric sector sums to 60
        sym_pieces = [:WEYL_S, :RICCI_S, :SCALAR_S, :PAIRSYM_S, :RICANTI_S, :PAIRSKEW_S]
        @test sum(dims[p] for p in sym_pieces) == 60

        # Antisymmetric sector sums to 36
        anti_pieces = [:WEYL_A, :RICCI_Z1, :RICCI_Z2, :PAIRSKEW_A, :PAIRSYM_A]
        @test sum(dims[p] for p in anti_pieces) == 36
    end

    @testset "Dimensions for other d values" begin
        # d=2: total = 8*1/2 = 4
        dims2 = brauer_piece_dimensions(2)
        @test sum(values(dims2)) == 4

        # d=3: total = 27*2/2 = 27
        dims3 = brauer_piece_dimensions(3)
        @test sum(values(dims3)) == 27

        # d=5: total = 125*4/2 = 250
        dims5 = brauer_piece_dimensions(5)
        @test sum(values(dims5)) == 250

        # d=2: Weyl vanishes
        @test dims2[:WEYL_S] == 0
    end

    @testset "Symmetric/antisymmetric split" begin
        reg, ac, fs = _brauer_reg()
        with_registry(reg) do
            # Build a general R_{abcd} with only cd antisymmetry
            R = Tensor(fs.riemann_name, [up(:a), down(:b), down(:c), down(:d)])

            # Lower the first index: R_{abcd} = g_{ae} R^e_{bcd}
            R_down = tproduct(1 // 1, TensorExpr[
                Tensor(:g, [down(:a), down(:e)]),
                Tensor(fs.riemann_name, [up(:e), down(:b), down(:c), down(:d)])
            ])

            W, Z = brauer_symmetric_split(R_down)

            # W and Z should each have 4 free indices
            @test length(free_indices(W)) == 4
            @test length(free_indices(Z)) == 4

            # Sum should reconstruct original (at the expression level)
            total = tsum(TensorExpr[W, Z])
            @test total isa TSum
            @test length(free_indices(total)) == 4
        end
    end

    @testset "BrauerDecomposition struct" begin
        reg, ac, fs = _brauer_reg()
        with_registry(reg) do
            R_down = tproduct(1 // 1, TensorExpr[
                Tensor(:g, [down(:a), down(:e)]),
                Tensor(fs.riemann_name, [up(:e), down(:b), down(:c), down(:d)])
            ])

            bd = brauer_decompose(R_down; metric=:g, dim=4, registry=reg)

            @test bd isa BrauerDecomposition
            @test bd.dim == 4
            @test bd.metric == :g
            @test length(bd.pieces) == 11
            @test length(bd.dimensions) == 11

            # All 11 pieces present
            for name in brauer_piece_names()
                @test haskey(bd.pieces, name)
                @test haskey(bd.dimensions, name)
            end

            # Total dimension = 96
            @test sum(values(bd.dimensions)) == 96
        end
    end

    @testset "Pieces have correct free indices" begin
        reg, ac, fs = _brauer_reg()
        with_registry(reg) do
            R_down = tproduct(1 // 1, TensorExpr[
                Tensor(:g, [down(:a), down(:e)]),
                Tensor(fs.riemann_name, [up(:e), down(:b), down(:c), down(:d)])
            ])

            bd = brauer_decompose(R_down; metric=:g, dim=4, registry=reg)

            # Each piece should have 4 free indices (a,b,c,d)
            for (name, expr) in bd.pieces
                fi = free_indices(expr)
                @test length(fi) == 4
            end
        end
    end

    @testset "Scalar trace extraction" begin
        reg, ac, fs = _brauer_reg()
        with_registry(reg) do
            R_down = tproduct(1 // 1, TensorExpr[
                Tensor(:g, [down(:a), down(:e)]),
                Tensor(fs.riemann_name, [up(:e), down(:b), down(:c), down(:d)])
            ])

            # The scalar trace g^{ac}g^{bd}R_{abcd} should produce a scalar
            scalar = TensorGR._ma_scalar_trace(R_down, :a, :b, :c, :d; metric=:g)
            @test length(free_indices(scalar)) == 0
        end
    end

    @testset "Ricci trace extraction" begin
        reg, ac, fs = _brauer_reg()
        with_registry(reg) do
            R_down = tproduct(1 // 1, TensorExpr[
                Tensor(:g, [down(:a), down(:e)]),
                Tensor(fs.riemann_name, [up(:e), down(:b), down(:c), down(:d)])
            ])

            # Ricci trace g^{ac}R_{abcd} should have 2 free indices
            ric = TensorGR._ma_ricci_trace(R_down, :a, :c; metric=:g)
            @test length(free_indices(ric)) == 2
        end
    end

    @testset "Display" begin
        dims = brauer_piece_dimensions(4)
        bd = BrauerDecomposition(
            Dict{Symbol, TensorExpr}(),
            dims, 4, :g)
        s = sprint(show, bd)
        @test occursin("BrauerDecomposition", s)
        @test occursin("11", s)
        @test occursin("96", s)
    end

    @testset "Completeness: sum of pieces = original" begin
        reg, ac, fs = _brauer_reg()
        with_registry(reg) do
            R_down = tproduct(1 // 1, TensorExpr[
                Tensor(:g, [down(:a), down(:e)]),
                Tensor(fs.riemann_name, [up(:e), down(:b), down(:c), down(:d)])
            ])

            bd = brauer_decompose(R_down; metric=:g, dim=4, registry=reg)

            # Sum all 11 pieces
            all_pieces = TensorExpr[bd.pieces[name] for name in brauer_piece_names()]
            total = tsum(all_pieces)

            # The total should have the same free indices as the original
            @test length(free_indices(total)) == 4

            # The total should reproduce the original R_{abcd}.
            # We verify this structurally: total - R_down should simplify to zero.
            diff = tsum(TensorExpr[total, tproduct(-1 // 1, TensorExpr[R_down])])

            # Simplify the difference
            simplified = simplify(diff; registry=reg)

            # The simplified result should be zero (TScalar(0) or equivalent)
            is_zero = (simplified isa TScalar && simplified.val == 0) ||
                      (simplified isa TProduct && simplified.scalar == 0 // 1) ||
                      (simplified == TScalar(0))
            @test is_zero
        end
    end
end
