@testset "Symmetrize / Antisymmetrize" begin

    @testset "_permutations_with_sign" begin
        using TensorGR: _permutations_with_sign

        # n=2: two permutations
        p2 = _permutations_with_sign(2)
        @test length(p2) == 2
        @test ([1, 2], 1) in p2
        @test ([2, 1], -1) in p2

        # n=3: six permutations
        p3 = _permutations_with_sign(3)
        @test length(p3) == 6
        # Check all permutations present
        perms_only = Set([p[1] for p in p3])
        @test length(perms_only) == 6
        # Check signs: even permutations have sign +1, odd have -1
        for (perm, sgn) in p3
            # Compute sign by counting inversions
            inv_count = 0
            for i in 1:length(perm), j in (i+1):length(perm)
                perm[i] > perm[j] && (inv_count += 1)
            end
            expected_sign = iseven(inv_count) ? 1 : -1
            @test sgn == expected_sign
        end
    end

    @testset "symmetrize 2 indices" begin
        T = Tensor(:T, [down(:a), down(:b)])

        sym = symmetrize(T, [:a, :b])
        # Should be (1/2)(T_ab + T_ba)
        @test sym isa TSum
        @test length(sym.terms) == 2

        # Collect the terms: one should have indices [a,b], the other [b,a]
        idx_sets = Set()
        for term in sym.terms
            # Each term is (1//2) * T_{...}
            if term isa TProduct
                @test term.scalar == 1 // 2
                t = term.factors[1]::Tensor
                @test t.name == :T
                push!(idx_sets, [i.name for i in t.indices])
            else
                # If the identity perm collapses (1//2)*T to just a product
                t = term::Tensor
                push!(idx_sets, [i.name for i in t.indices])
            end
        end
        @test [:a, :b] in idx_sets
        @test [:b, :a] in idx_sets
    end

    @testset "antisymmetrize 2 indices" begin
        T = Tensor(:T, [down(:a), down(:b)])

        asym = antisymmetrize(T, [:a, :b])
        # Should be (1/2)(T_ab - T_ba)
        @test asym isa TSum
        @test length(asym.terms) == 2

        # Check coefficients
        for term in asym.terms
            @test term isa TProduct
            t = term.factors[1]::Tensor
            if [i.name for i in t.indices] == [:a, :b]
                @test term.scalar == 1 // 2
            else
                @test [i.name for i in t.indices] == [:b, :a]
                @test term.scalar == -1 // 2
            end
        end
    end

    @testset "symmetric tensor: T_{(ab)} = T_{ab} after simplify" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:S, manifold=:M, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)]))

        with_registry(reg) do
            S = Tensor(:S, [down(:a), down(:b)])
            sym_S = symmetrize(S, [:a, :b])
            result = simplify(sym_S)
            @test result == S
        end
    end

    @testset "symmetric tensor: T_{[ab]} = 0 after simplify" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:S, manifold=:M, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)]))

        with_registry(reg) do
            S = Tensor(:S, [down(:a), down(:b)])
            asym_S = antisymmetrize(S, [:a, :b])
            result = simplify(asym_S)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "T_{(ab)} + T_{(ba)} = 2 T_{(ab)}" begin
        T = Tensor(:T, [down(:a), down(:b)])

        sym1 = symmetrize(T, [:a, :b])
        # T_{(ba)}: symmetrize over b,a (same expression, just different input order)
        T_ba = Tensor(:T, [down(:b), down(:a)])
        sym2 = symmetrize(T_ba, [:b, :a])

        # Both should expand the same way; their sum should have 4 terms
        total = sym1 + sym2
        # After collect_terms, each unique term should have coefficient 2*(1/2) = 1
        # So we get T_ab + T_ba = 2 * T_{(ab)}
        # Verify structurally: the sum should have 4 terms before simplification
        @test total isa TSum
        @test length(total.terms) == 4

        # After collecting terms, should reduce to 2 terms with coefficient 1
        collected = collect_terms(total)
        if collected isa TSum
            for term in collected.terms
                sc, _ = TensorGR._split_scalar(term)
                # Each unique tensor configuration gets 2 * (1//2) = 1//1
                @test sc == 1 // 1
            end
        end
    end

    @testset "T_{[ab]} + T_{[ba]} = 0" begin
        T = Tensor(:T, [down(:a), down(:b)])
        asym_ab = antisymmetrize(T, [:a, :b])
        T_ba = Tensor(:T, [down(:b), down(:a)])
        asym_ba = antisymmetrize(T_ba, [:b, :a])

        total = asym_ab + asym_ba
        collected = collect_terms(total)
        @test collected == TScalar(0 // 1)
    end

    @testset "3-index symmetrization" begin
        T = Tensor(:T, [down(:a), down(:b), down(:c)])
        sym = symmetrize(T, [:a, :b, :c])

        # Should have 3! = 6 terms, each with coefficient 1/6
        @test sym isa TSum
        @test length(sym.terms) == 6

        for term in sym.terms
            @test term isa TProduct
            @test term.scalar == 1 // 6
            t = term.factors[1]::Tensor
            @test t.name == :T
            @test length(t.indices) == 3
            # All indices should be some permutation of {a, b, c}
            names = Set(i.name for i in t.indices)
            @test names == Set([:a, :b, :c])
        end
    end

    @testset "3-index antisymmetrization" begin
        T = Tensor(:T, [down(:a), down(:b), down(:c)])
        asym = antisymmetrize(T, [:a, :b, :c])

        @test asym isa TSum
        @test length(asym.terms) == 6

        for term in asym.terms
            @test term isa TProduct
            @test abs(term.scalar) == 1 // 6
        end

        # The identity permutation [a,b,c] should have +1/6
        # The swap [b,a,c] should have -1/6
        for term in asym.terms
            t = term.factors[1]::Tensor
            names = [i.name for i in t.indices]
            if names == [:a, :b, :c]
                @test term.scalar == 1 // 6
            elseif names == [:b, :a, :c]
                @test term.scalar == -1 // 6
            end
        end
    end

    @testset "single index is identity" begin
        T = Tensor(:T, [down(:a), down(:b)])
        @test symmetrize(T, [:a]) === T
        @test antisymmetrize(T, [:a]) === T
    end

    @testset "antisymmetric tensor: T_{[ab]} = T_{ab}" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:A, manifold=:M, rank=(0, 2),
            symmetries=Any[AntiSymmetric(1, 2)]))

        with_registry(reg) do
            A = Tensor(:A, [down(:a), down(:b)])
            asym_A = antisymmetrize(A, [:a, :b])
            result = simplify(asym_A)
            @test result == A
        end
    end

    @testset "antisymmetric tensor: T_{(ab)} = 0" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M, 4, :g, :d, [:a, :b, :c, :d, :e, :f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:A, manifold=:M, rank=(0, 2),
            symmetries=Any[AntiSymmetric(1, 2)]))

        with_registry(reg) do
            A = Tensor(:A, [down(:a), down(:b)])
            sym_A = symmetrize(A, [:a, :b])
            result = simplify(sym_A)
            @test result == TScalar(0 // 1)
        end
    end
end
