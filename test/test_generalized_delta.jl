@testset "Generalized Kronecker Delta" begin

    @testset "is_zero_by_dimension" begin
        # p <= dim: not zero
        @test !is_zero_by_dimension(1, 4)
        @test !is_zero_by_dimension(2, 4)
        @test !is_zero_by_dimension(3, 4)
        @test !is_zero_by_dimension(4, 4)

        # p > dim: zero (DDI)
        @test is_zero_by_dimension(5, 4)
        @test is_zero_by_dimension(6, 4)
        @test is_zero_by_dimension(10, 3)

        # Edge cases
        @test !is_zero_by_dimension(0, 4)
        @test is_zero_by_dimension(1, 0)
        @test !is_zero_by_dimension(0, 0)
    end

    @testset "p=0: scalar identity" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            result = generalized_delta(Symbol[], Symbol[])
            @test result == TScalar(1 // 1)

            result2 = generalized_delta(0, 4)
            @test result2 == TScalar(1 // 1)
        end
    end

    @testset "p=1: ordinary Kronecker delta" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            result = generalized_delta([:a], [:b])
            @test result isa Tensor
            @test result.name == :δ
            @test result.indices == [up(:a), down(:b)]
        end
    end

    @testset "p=2: antisymmetric product of two deltas" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            # delta^{ab}_{cd} = delta^a_c delta^b_d - delta^a_d delta^b_c
            result = generalized_delta([:a, :b], [:c, :d])
            @test result isa TSum
            @test length(result.terms) == 2

            # Check the two terms: one positive, one negative
            positive_found = false
            negative_found = false
            for term in result.terms
                @test term isa TProduct
                @test length(term.factors) == 2
                # Both factors should be delta tensors
                for f in term.factors
                    @test f isa Tensor
                    @test f.name == :δ
                end
                if term.scalar == 1 // 1
                    positive_found = true
                    # delta^a_c * delta^b_d (identity permutation)
                    @test term.factors[1].indices == [up(:a), down(:c)]
                    @test term.factors[2].indices == [up(:b), down(:d)]
                elseif term.scalar == -1 // 1
                    negative_found = true
                    # delta^a_d * delta^b_c (swap permutation)
                    @test term.factors[1].indices == [up(:a), down(:d)]
                    @test term.factors[2].indices == [up(:b), down(:c)]
                end
            end
            @test positive_found
            @test negative_found
        end
    end

    @testset "p=3: six terms" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            result = generalized_delta([:a, :b, :c], [:d, :e, :f])
            @test result isa TSum
            @test length(result.terms) == 6  # 3! = 6

            # Each term is a product of 3 deltas with coefficient +/-1
            for term in result.terms
                @test term isa TProduct
                @test abs(term.scalar) == 1 // 1
                @test length(term.factors) == 3
                for f in term.factors
                    @test f isa Tensor
                    @test f.name == :δ
                end
            end

            # Check sign balance: 3 even perms (+1) and 3 odd perms (-1)
            pos_count = count(t -> t.scalar == 1 // 1, result.terms)
            neg_count = count(t -> t.scalar == -1 // 1, result.terms)
            @test pos_count == 3
            @test neg_count == 3
        end
    end

    @testset "DDI: p=5 vanishes in d=4" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d,
            [:a, :b, :c, :d, :e, :f, :m, :n, :p, :q]))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            # Explicit index version
            result = generalized_delta([:a, :b, :c, :d, :e], [:f, :m, :n, :p, :q])
            @test result == TScalar(0 // 1)

            # Fresh index version
            result2 = generalized_delta(5, 4)
            @test result2 == TScalar(0 // 1)
        end
    end

    @testset "DDI: p=d does not vanish" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            # p=4 in d=4: does NOT vanish (Levi-Civita-like)
            result = generalized_delta([:a, :b, :c, :d], [:e, :f, :m, :n])
            @test result isa TSum
            @test length(result.terms) == 24  # 4! = 24
        end
    end

    @testset "DDI: various dimensions" begin
        # d=2: p=3 vanishes, p=2 does not
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 2, :g, :d, [:a, :b, :c, :d]))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            @test generalized_delta(3, 2) == TScalar(0 // 1)
            result2 = generalized_delta(2, 2)
            @test result2 isa TSum
            @test length(result2.terms) == 2
        end
    end

    @testset "fresh index version" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            result = generalized_delta(2, 4)
            @test result isa TSum
            @test length(result.terms) == 2

            # All delta factors should have unique index names
            all_idx_names = Symbol[]
            for term in result.terms
                for f in term.factors
                    for idx in f.indices
                        push!(all_idx_names, idx.name)
                    end
                end
            end
            # 2 terms x 2 factors x 2 indices = 8, but indices are shared across terms
            # The up indices and down indices should each be unique sets
            up_names = Set{Symbol}()
            down_names = Set{Symbol}()
            # Check just the identity-permutation term
            id_term = nothing
            for term in result.terms
                if term.scalar == 1 // 1
                    id_term = term
                    break
                end
            end
            @test id_term !== nothing
            for f in id_term.factors
                for idx in f.indices
                    if idx.position == Up
                        push!(up_names, idx.name)
                    else
                        push!(down_names, idx.name)
                    end
                end
            end
            @test length(up_names) == 2
            @test length(down_names) == 2
            @test isempty(intersect(up_names, down_names))
        end
    end

    @testset "trace of generalized delta: delta^{ab}_{ab} = d(d-1)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            # delta^{ab}_{cd} contracted with same indices:
            # delta^{ab}_{ab} = delta^a_a * delta^b_b - delta^a_b * delta^b_a
            #                 = d*d - d = d(d-1) = 4*3 = 12
            gd = generalized_delta([:a, :b], [:a, :b])
            result = simplify(gd)
            @test result == TScalar(12 // 1)
        end
    end

    @testset "partial trace: delta^{ab}_{ac} = (d-1) delta^b_c" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            # delta^{ab}_{ac} = delta^a_a * delta^b_c - delta^a_c * delta^b_a
            #                 = d * delta^b_c - delta^b_c = (d-1) delta^b_c
            gd = generalized_delta([:a, :b], [:a, :c])
            result = simplify(gd)
            expected = tproduct(3 // 1, TensorExpr[Tensor(:δ, [up(:b), down(:c)])])
            @test result == expected
        end
    end

    @testset "error: mismatched index counts" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d]))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            @test_throws ErrorException generalized_delta([:a, :b], [:c])
        end
    end

    @testset "error: negative p" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d]))
        register_tensor!(reg, TensorProperties(
            name=:δ, manifold=:M, rank=(1, 1),
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))

        with_registry(reg) do
            @test_throws ErrorException generalized_delta(-1, 4)
        end
    end
end
