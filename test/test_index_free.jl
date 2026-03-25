@testset "Index-Free Notation" begin
    using TensorGR: IndexFree, to_index_free, from_index_free,
                    index_free_structure, same_tensor_structure,
                    GammaMatrix,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    Symmetric, AntiSymmetric, RiemannSymmetry,
                    Tensor, TProduct, TSum, TScalar,
                    up, down, with_registry, register_tensor!,
                    current_registry, has_tensor,
                    free_indices, simplify

    function _make_index_free_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(1, 0), symmetries=SymmetrySpec[]))
            register_tensor!(reg, TensorProperties(
                name=:W, manifold=:M4, rank=(0, 1), symmetries=SymmetrySpec[]))
            register_tensor!(reg, TensorProperties(
                name=:T, manifold=:M4, rank=(1, 1), symmetries=SymmetrySpec[]))
            register_tensor!(reg, TensorProperties(
                name=:F, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[AntiSymmetric(1, 2)]))
        end
        return reg
    end

    # ── to_index_free: basic conversion ──────────────────────────────────

    @testset "to_index_free: single tensor" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            V = Tensor(:V, [up(:a)])
            ifree = to_index_free(V)
            @test ifree isa IndexFree
            @test ifree.factors == [:V]
            @test ifree.slot_count == 1
            @test isempty(ifree.contractions)
            @test ifree.free_slots == [1]
            @test ifree.free_positions == [Up]
            @test ifree.scalar == 1 // 1
        end
    end

    @testset "to_index_free: product with contraction" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            # V^a W_a (contracted)
            expr = TProduct(1 // 1, [
                Tensor(:V, [up(:a)]),
                Tensor(:W, [down(:a)])
            ])
            ifree = to_index_free(expr)
            @test ifree.factors == [:V, :W]
            @test ifree.slot_count == 2
            @test length(ifree.contractions) == 1
            @test ifree.contractions[1] == (1, 2)
            @test isempty(ifree.free_slots)
        end
    end

    @testset "to_index_free: product with free indices" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            # T^a_b (no contraction, two free indices)
            T = Tensor(:T, [up(:a), down(:b)])
            ifree = to_index_free(T)
            @test ifree.factors == [:T]
            @test ifree.slot_count == 2
            @test isempty(ifree.contractions)
            @test ifree.free_slots == [1, 2]
            @test ifree.free_positions == [Up, Down]
        end
    end

    @testset "to_index_free: Ric_{ab} Ric^{ab}" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            expr = TProduct(1 // 1, [
                Tensor(:Ric, [down(:a), down(:b)]),
                Tensor(:Ric, [up(:a), up(:b)])
            ])
            ifree = to_index_free(expr)
            @test ifree.factors == [:Ric, :Ric]
            @test ifree.slot_count == 4
            @test length(ifree.contractions) == 2
            @test isempty(ifree.free_slots)
        end
    end

    @testset "to_index_free: scalar" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            s = TScalar(3 // 2)
            ifree = to_index_free(s)
            @test ifree.scalar == 3 // 2
            @test isempty(ifree.factors)
            @test ifree.slot_count == 0
        end
    end

    @testset "to_index_free: scalar coefficient preserved" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            expr = TProduct(3 // 7, [
                Tensor(:V, [up(:a)]),
                Tensor(:W, [down(:a)])
            ])
            ifree = to_index_free(expr)
            @test ifree.scalar == 3 // 7
        end
    end

    @testset "to_index_free: mixed free and contracted" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            # T^a_b V^b → has contraction on b, free index a
            expr = TProduct(1 // 1, [
                Tensor(:T, [up(:a), down(:b)]),
                Tensor(:V, [up(:b)])
            ])
            ifree = to_index_free(expr)
            @test ifree.factors == [:T, :V]
            @test ifree.slot_count == 3
            @test length(ifree.contractions) == 1
            @test ifree.contractions[1] == (2, 3)  # T's 2nd slot ↔ V's slot
            @test ifree.free_slots == [1]
            @test ifree.free_positions == [Up]
        end
    end

    @testset "to_index_free: Riemann" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            # R_{abcd} — all free
            R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            ifree = to_index_free(R)
            @test ifree.factors == [:Riem]
            @test ifree.slot_count == 4
            @test isempty(ifree.contractions)
            @test length(ifree.free_slots) == 4
        end
    end

    # ── from_index_free: reconstruction ──────────────────────────────────

    @testset "from_index_free: single tensor" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            ifree = IndexFree(
                1 // 1, [:V], 1,
                Tuple{Int,Int}[], [1], [Up], [:Tangent]
            )
            result = from_index_free(ifree; free_names=[:a])
            @test result isa Tensor
            @test result.name === :V
            @test length(result.indices) == 1
            @test result.indices[1].name === :a
            @test result.indices[1].position === Up
        end
    end

    @testset "from_index_free: contracted product" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            ifree = IndexFree(
                1 // 1, [:V, :W], 2,
                [(1, 2)], Int[], IndexPosition[], [:Tangent, :Tangent]
            )
            result = from_index_free(ifree)
            @test result isa TProduct
            @test length(result.factors) == 2
            # Both factors should share a dummy index
            idx1 = result.factors[1].indices[1]
            idx2 = result.factors[2].indices[1]
            @test idx1.name == idx2.name
            @test idx1.position != idx2.position
        end
    end

    @testset "from_index_free: with free names" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            ifree = IndexFree(
                1 // 1, [:T], 2,
                Tuple{Int,Int}[], [1, 2], [Up, Down], [:Tangent, :Tangent]
            )
            result = from_index_free(ifree; free_names=[:a, :b])
            @test result isa Tensor
            @test result.indices[1].name === :a
            @test result.indices[2].name === :b
        end
    end

    @testset "from_index_free: scalar" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            ifree = IndexFree(
                5 // 3, Symbol[], 0,
                Tuple{Int,Int}[], Int[], IndexPosition[], Symbol[]
            )
            result = from_index_free(ifree)
            @test result isa TScalar
            @test result.val == 5 // 3
        end
    end

    @testset "from_index_free: coefficient preserved" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            ifree = IndexFree(
                3 // 7, [:V, :W], 2,
                [(1, 2)], Int[], IndexPosition[], [:Tangent, :Tangent]
            )
            result = from_index_free(ifree)
            @test result isa TProduct
            @test result.scalar == 3 // 7
        end
    end

    # ── Round-trip tests ─────────────────────────────────────────────────

    @testset "round-trip: V^a" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            V = Tensor(:V, [up(:a)])
            ifree = to_index_free(V)
            result = from_index_free(ifree; free_names=[:a])
            @test result isa Tensor
            @test result.name === :V
            @test result.indices[1] == up(:a)
        end
    end

    @testset "round-trip: V^a W_a (scalar)" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            expr = TProduct(1 // 1, [
                Tensor(:V, [up(:a)]),
                Tensor(:W, [down(:a)])
            ])
            ifree = to_index_free(expr)
            result = from_index_free(ifree)

            # Should produce a product with contracted indices
            @test result isa TProduct
            fi = free_indices(result)
            @test isempty(fi)  # fully contracted
        end
    end

    @testset "round-trip: T^a_b preserves structure" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            T = Tensor(:T, [up(:a), down(:b)])
            ifree = to_index_free(T)
            result = from_index_free(ifree; free_names=[:a, :b])
            @test result isa Tensor
            @test result.name === :T
            @test result.indices[1].position === Up
            @test result.indices[2].position === Down
        end
    end

    @testset "round-trip: Ric Ric contraction" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            expr = TProduct(1 // 1, [
                Tensor(:Ric, [down(:a), down(:b)]),
                Tensor(:Ric, [up(:a), up(:b)])
            ])
            ifree = to_index_free(expr)
            result = from_index_free(ifree)

            @test result isa TProduct
            fi = free_indices(result)
            @test isempty(fi)

            # Both are Ric⊗Ric contractions — same tensor structure
            @test length(result.factors) == 2
            @test all(f -> f isa Tensor && f.name === :Ric, result.factors)
        end
    end

    # ── index_free_structure ─────────────────────────────────────────────

    @testset "index_free_structure: extracts tensor names" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            expr = TProduct(1 // 1, [
                Tensor(:Ric, [down(:a), down(:b)]),
                Tensor(:Riem, [up(:a), down(:c), up(:b), down(:d)])
            ])
            names = index_free_structure(expr)
            @test names == [:Ric, :Riem]
        end
    end

    # ── same_tensor_structure ────────────────────────────────────────────

    @testset "same_tensor_structure: same contractions" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            expr1 = TProduct(2 // 1, [
                Tensor(:V, [up(:a)]),
                Tensor(:W, [down(:a)])
            ])
            expr2 = TProduct(5 // 1, [
                Tensor(:V, [up(:b)]),
                Tensor(:W, [down(:b)])
            ])
            if1 = to_index_free(expr1)
            if2 = to_index_free(expr2)
            @test same_tensor_structure(if1, if2)
        end
    end

    @testset "same_tensor_structure: different contractions" begin
        reg = _make_index_free_registry()
        with_registry(reg) do
            # R_{abcd} R^{acbd} vs R_{abcd} R^{abcd}
            expr1 = TProduct(1 // 1, [
                Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)]),
                Tensor(:Riem, [up(:a), up(:c), up(:b), up(:d)])
            ])
            expr2 = TProduct(1 // 1, [
                Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)]),
                Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            ])
            if1 = to_index_free(expr1)
            if2 = to_index_free(expr2)
            # Different contraction patterns
            @test !same_tensor_structure(if1, if2)
        end
    end

    # ── Display ──────────────────────────────────────────────────────────

    @testset "IndexFree display" begin
        ifree = IndexFree(
            1 // 1, [:Ric, :Ric], 4,
            [(1, 3), (2, 4)], Int[], IndexPosition[],
            [:Tangent, :Tangent, :Tangent, :Tangent]
        )
        s = sprint(show, ifree)
        @test occursin("Ric", s)
        @test occursin("↔", s)
    end
end
