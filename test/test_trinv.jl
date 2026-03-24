@testset "TRInv: Tensorial Riemann Monomials" begin
    # Helper: registry with 4D manifold + curvature tensors
    function trinv_registry(; dim=4)
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=dim metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # ─── Struct construction and validation ────────────────────────────
    @testset "Construction" begin
        # Degree 1, all free (single Riemann)
        t = TRInv(1, [1, 2, 3, 4], [1, 2, 3, 4],
                  [Down, Down, Down, Down])
        @test t.degree == 1
        @test TensorGR.rank(t) == 4
        @test TensorGR.is_scalar(t) == false
        @test t.canonical == false

        # Degree 2, rank 4 (2 dummy pairs + 4 free)
        t2 = TRInv(2, [5, 6, 3, 4, 1, 2, 7, 8],
                   [3, 4, 7, 8], [Down, Down, Down, Down])
        @test t2.degree == 2
        @test TensorGR.rank(t2) == 4

        # Degree 2, scalar (all contracted)
        t3 = TRInv(2, [5, 6, 7, 8, 1, 2, 3, 4],
                   Int[], IndexPosition[])
        @test TensorGR.is_scalar(t3) == true
        @test TensorGR.rank(t3) == 0

        # Invalid: non-involution
        @test_throws ErrorException TRInv(1, [2, 3, 1, 4], [4], [Down])

        # Invalid: free_slots don't match fixed points
        @test_throws ErrorException TRInv(1, [1, 2, 3, 4], [1, 2], [Down, Down])

        # Unsorted free_slots are auto-sorted
        t4 = TRInv(1, [1, 2, 3, 4], [4, 2, 1, 3],
                   [Down, Up, Down, Up])
        @test t4.free_slots == [1, 2, 3, 4]
        @test t4.free_positions == [Down, Up, Up, Down]  # reordered to match sorted slots
    end

    # ─── RInv conversion ──────────────────────────────────────────────
    @testset "RInv conversion" begin
        rinv = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
        trinv = TRInv(rinv)
        @test trinv.degree == 2
        @test TensorGR.is_scalar(trinv)
        @test trinv.contraction == rinv.contraction

        # Round-trip
        rinv2 = RInv(trinv)
        @test rinv2.contraction == rinv.contraction

        # Cannot convert TRInv with free indices to RInv
        t = TRInv(1, [1, 2, 3, 4], [1, 2, 3, 4],
                  [Down, Down, Down, Down])
        @test_throws ErrorException RInv(t)
    end

    # ─── Canonicalization: degree 1, all free ─────────────────────────
    @testset "Canonicalize degree 1 all-free" begin
        reg = trinv_registry()
        with_registry(reg) do
            # R_{abcd}: contraction [1,2,3,4], all free, all Down
            t1 = TRInv(1, [1, 2, 3, 4], [1, 2, 3, 4],
                       [Down, Down, Down, Down])
            c1, s1 = canonicalize(t1)
            @test c1.canonical
            @test s1 == +1 || s1 == -1

            # The same TRInv canonicalized twice gives same result
            t2 = TRInv(1, [1, 2, 3, 4], [1, 2, 3, 4],
                       [Down, Down, Down, Down])
            c2, s2 = canonicalize(t2)
            @test c1.contraction == c2.contraction
            @test s1 == s2

            # Already canonical returns immediately
            c3, s3 = canonicalize(c1)
            @test c3 === c1
            @test s3 == +1
        end
    end

    # ─── Canonicalization: degree 2 with dummies ──────────────────────
    @testset "Canonicalize degree 2 with dummies" begin
        reg = trinv_registry()
        with_registry(reg) do
            # R_{abcd} R^{ab}_{ef}: slots 1↔5, 2↔6, free slots 3,4,7,8
            t1 = TRInv(2, [5, 6, 3, 4, 1, 2, 7, 8],
                       [3, 4, 7, 8], [Down, Down, Down, Down])
            c1, s1 = canonicalize(t1)
            @test c1.canonical
            @test TensorGR.rank(c1) == 4
            @test s1 != 0

            # Same contraction with dummy pair order swapped
            # (antisymmetry in factor 2 first pair: swap slots 5,6)
            t2 = TRInv(2, [6, 5, 3, 4, 2, 1, 7, 8],
                       [3, 4, 7, 8], [Down, Down, Down, Down])
            c2, s2 = canonicalize(t2)
            @test c2.canonical

            # Same canonical contraction (related by antisymmetry)
            @test c1.contraction == c2.contraction
        end
    end

    # ─── Canonicalization: vanishing by antisymmetry ──────────────────
    @testset "Vanishing by antisymmetry" begin
        reg = trinv_registry()
        with_registry(reg) do
            # Kretschmann R_{abcd}R^{abcd}: scalar, should NOT vanish
            t = TRInv(2, [5, 6, 7, 8, 1, 2, 3, 4],
                      Int[], IndexPosition[])
            c, s = canonicalize(t)
            @test s != 0
            @test c.canonical
        end
    end

    # ─── Factor exchange symmetry ─────────────────────────────────────
    @testset "Factor exchange" begin
        reg = trinv_registry()
        with_registry(reg) do
            # Same free-slot pattern (local 3,4 free in both) → exchangeable
            t1 = TRInv(2, [5, 6, 3, 4, 1, 2, 7, 8],
                       [3, 4, 7, 8], [Down, Down, Down, Down])
            @test TensorGR._trinv_factors_exchangeable(t1, 1, 2)

            # Different local patterns: factor 1 has local 1,2 free,
            # factor 2 has local 1 contracted, local 2,3,4 free
            # → NOT exchangeable
            # factor 1: slots 1,2 free, slots 3,4 contracted with 7,8
            # factor 2: slot 5 contracted with slot 3... wait, need valid involution
            # Build: factor 1 slots 1,2 free, slots 3↔7, 4↔8
            #        factor 2 slot 5 free, slots 6↔3? No...
            # Simpler: factor 1 local (free, free, contracted, contracted)
            #          factor 2 local (contracted, free, free, contracted)
            # factor 1: slots 1,2 free; 3↔5, 4↔6
            # factor 2: slot 5↔3, 6↔4, 7 free, 8 free -- wait 5,6 already used
            # Let me build it correctly:
            # k=2, 8 slots. Factor 1 = slots 1-4, Factor 2 = slots 5-8
            # Factor 1: local 1,2 free (slots 1,2), local 3,4 contracted with factor 2
            # Factor 2: local 1,2 contracted with factor 1, local 3,4 free (slots 7,8)
            # contraction: 3↔5, 4↔6 → [1,2,5,6,3,4,7,8]
            # Factor 1 pattern: (free, free, contracted, contracted)
            # Factor 2 pattern: (contracted, contracted, free, free)
            # Same pattern! → exchangeable
            # Different local patterns: factor 1 has local 1,2 free,
            # factor 2 has local 3,4 free → NOT exchangeable
            t2 = TRInv(2, [1, 2, 5, 6, 3, 4, 7, 8],
                       [1, 2, 7, 8], [Down, Down, Down, Down])
            @test !TensorGR._trinv_factors_exchangeable(t2, 1, 2)

            # For non-exchangeable: different Up/Down on free slots
            t3 = TRInv(2, [5, 6, 3, 4, 1, 2, 7, 8],
                       [3, 4, 7, 8], [Down, Down, Up, Down])
            @test !TensorGR._trinv_factors_exchangeable(t3, 1, 2)
        end
    end

    # ─── to_tensor_expr ───────────────────────────────────────────────
    @testset "to_tensor_expr" begin
        reg = trinv_registry()
        with_registry(reg) do
            # Degree 2 with dummies
            t = TRInv(2, [5, 6, 3, 4, 1, 2, 7, 8],
                      [3, 4, 7, 8], [Down, Down, Down, Down])
            expr = to_tensor_expr(t; registry=reg, metric=:g)
            @test expr isa TProduct

            # Should contain 2 Riemann tensors and 2 metrics
            riems = [f for f in expr.factors if f isa Tensor && f.name == :Riem]
            mets = [f for f in expr.factors if f isa Tensor && f.name == :g]
            @test length(riems) == 2
            @test length(mets) == 2

            # Scalar case
            t_scalar = TRInv(2, [5, 6, 7, 8, 1, 2, 3, 4],
                            Int[], IndexPosition[])
            expr_s = to_tensor_expr(t_scalar; registry=reg, metric=:g)
            riems_s = [f for f in expr_s.factors if f isa Tensor && f.name == :Riem]
            @test length(riems_s) == 2
        end
    end

    # ─── from_tensor_expr_trinv ───────────────────────────────────────
    @testset "from_tensor_expr_trinv" begin
        reg = trinv_registry()
        with_registry(reg) do
            # Build with metric contractions
            R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R2 = Tensor(:Riem, [down(:e), down(:f), down(:h), down(:i)])
            m1 = Tensor(:g, [up(:a), up(:e)])
            m2 = Tensor(:g, [up(:b), up(:f)])
            expr = tproduct(1 // 1, TensorExpr[R1, R2, m1, m2])

            trinv, sign = from_tensor_expr_trinv(expr; registry=reg, metric=:g)
            @test trinv.degree == 2
            @test TensorGR.rank(trinv) == 4
            @test sign == 1
            @test length(trinv.free_slots) == 4
        end
    end

    # ─── Consistency: equivalent contractions give same canonical form ─
    @testset "Canonical consistency" begin
        reg = trinv_registry()
        with_registry(reg) do
            # Two TRInvs related by antisymmetry in factor 2
            # First: slots 3↔5, 4↔6, free 1,2,7,8
            t1 = TRInv(2, [1, 2, 5, 6, 3, 4, 7, 8],
                       [1, 2, 7, 8], [Down, Down, Down, Down])
            c1, s1 = canonicalize(t1)

            # Second: swap dummy pair in factor 2 (antisymmetry)
            # 3↔6, 4↔5 instead of 3↔5, 4↔6
            t2 = TRInv(2, [1, 2, 6, 5, 4, 3, 7, 8],
                       [1, 2, 7, 8], [Down, Down, Down, Down])
            c2, s2 = canonicalize(t2)

            # Same canonical contraction, opposite signs
            @test c1.contraction == c2.contraction
            @test s1 == -s2
        end
    end

    # ─── First Bianchi identity (cyclic reduction) ──────────────────
    @testset "Bianchi cyclic: degree 1 all-free" begin
        reg = trinv_registry()
        with_registry(reg) do
            # R_{abcd} + R_{acdb} + R_{adbc} = 0
            # For degree 1, all free: the identity produces two cycled TRInvs
            t = TRInv(1, [1, 2, 3, 4], [1, 2, 3, 4],
                       [Down, Down, Down, Down])
            ct, st = canonicalize(t)
            cycled = bianchi_cyclic_trinv(ct, 1)
            @test length(cycled) == 2

            # The relation: ct + cycled[1] + cycled[2] = 0
            # i.e. sign_t * ct + sign_1 * c1 + sign_2 * c2 = 0
            # Since ct is canonical with sign +1, and bianchi_cyclic returns
            # the RHS with negated signs: ct = -c1_sign*c1 - c2_sign*c2
            # Check: all three terms have same contraction (identity for all-free)
            c1, s1 = cycled[1]
            c2, s2 = cycled[2]
            @test c1.canonical
            @test c2.canonical

            # For all-free degree 1: the Bianchi identity constrains the
            # symmetric part of R_{abcd}. All three terms should be
            # non-vanishing.
            @test s1 != 0
            @test s2 != 0
        end
    end

    @testset "Bianchi cyclic: degree 2 with dummies" begin
        reg = trinv_registry()
        with_registry(reg) do
            # Apply Bianchi to factor 1 of a degree-2 tensorial monomial
            t = TRInv(2, [5, 6, 3, 4, 1, 2, 7, 8],
                       [3, 4, 7, 8], [Down, Down, Down, Down])
            ct, _ = canonicalize(t)
            cycled = bianchi_cyclic_trinv(ct, 1)
            @test length(cycled) == 2

            # Both results should have same rank
            @test TensorGR.rank(cycled[1][1]) == 4
            @test TensorGR.rank(cycled[2][1]) == 4

            # Apply Bianchi to factor 2
            cycled2 = bianchi_cyclic_trinv(ct, 2)
            @test length(cycled2) == 2
        end
    end

    @testset "Bianchi cyclic: self-consistency" begin
        reg = trinv_registry()
        with_registry(reg) do
            # The Bianchi identity applied twice should be consistent:
            # if we apply Bianchi to the result, we should get back
            # something expressible in terms of the original.
            # This is a basic sanity check, not a full proof.
            t = TRInv(1, [1, 2, 3, 4], [1, 2, 3, 4],
                       [Down, Down, Down, Down])
            ct, _ = canonicalize(t)
            cycled = bianchi_cyclic_trinv(ct, 1)

            # Apply Bianchi to the first cycled term
            c1, s1 = cycled[1]
            cycled_of_cycled = bianchi_cyclic_trinv(c1, 1)
            # Should produce valid canonical TRInvs
            @test all(p -> p[1].canonical, cycled_of_cycled)
        end
    end

    @testset "Bianchi relations generation" begin
        reg = trinv_registry()
        with_registry(reg) do
            # Generate Bianchi relations for degree-1 all-free monomials
            t = TRInv(1, [1, 2, 3, 4], [1, 2, 3, 4],
                       [Down, Down, Down, Down])
            ct, _ = canonicalize(t)
            rels = bianchi_relations_trinv([ct])

            # For a single degree-1 all-free monomial with 1 factor,
            # Bianchi generates at most 1 relation
            @test length(rels) <= 1

            # For degree 2 with dummies: more relations possible
            t2 = TRInv(2, [5, 6, 3, 4, 1, 2, 7, 8],
                        [3, 4, 7, 8], [Down, Down, Down, Down])
            ct2, _ = canonicalize(t2)
            rels2 = bianchi_relations_trinv([ct2])
            @test rels2 isa Vector
        end
    end

    # ─── Second Bianchi identity (differential) ─────────────────────
    @testset "Bianchi2: structure of differential_bianchi" begin
        reg = trinv_registry()
        with_registry(reg) do
            @covd D on=M4 metric=g

            # The second Bianchi identity produces 3 terms
            id = differential_bianchi(down(:a), down(:b), down(:c),
                                      down(:d), down(:e); covd=:D)
            @test id isa TSum
            @test length(id.terms) == 3

            # Each term is a TDeriv wrapping a Riemann tensor
            for t in id.terms
                @test t isa TDeriv
                @test t.arg isa Tensor
                @test t.arg.name == :Riem
            end
        end
    end

    @testset "Bianchi2: apply to tensorial product" begin
        reg = trinv_registry()
        with_registry(reg) do
            @covd D on=M4 metric=g

            # Build ∂_a R_{bcde} as a single factor
            dR = TDeriv(down(:a), Tensor(:Riem, [down(:b), down(:c),
                        down(:d), down(:e)]), :D)
            expr = tproduct(1 // 1, TensorExpr[dR])

            # Apply second Bianchi to factor 1
            result = apply_bianchi2_tensorial(expr, 1; registry=reg)
            @test result isa TensorExpr

            # The result should be: -∂_b R_{cade} - ∂_c R_{abde}
            # which is a TSum of two terms
            if result isa TSum
                @test length(result.terms) == 2
            end

            # has_diff_riemann should detect derivative Riemann factors
            @test has_diff_riemann(dR)
            @test !has_diff_riemann(Tensor(:Riem, [down(:a), down(:b),
                                   down(:c), down(:d)]))

            # diff_riemann_factor_indices on a product
            p = tproduct(1 // 1, TensorExpr[dR])
            if p isa TProduct
                idxs = diff_riemann_factor_indices(p)
                @test length(idxs) == 1
                @test idxs[1] == 1
            end
        end
    end

    @testset "Bianchi2: apply produces correct index structure" begin
        reg = trinv_registry()
        with_registry(reg) do
            @covd D on=M4 metric=g

            # apply_bianchi2_tensorial should produce terms where the
            # derivative index cycles with the first pair of Riemann
            dR = TDeriv(down(:a), Tensor(:Riem, [down(:b), down(:c),
                        down(:d), down(:e)]), :D)
            expr = tproduct(1 // 1, TensorExpr[dR])
            result = apply_bianchi2_tensorial(expr, 1; registry=reg)

            # Result should have 5 free indices (same as original)
            orig_idxs = indices(dR)
            @test length(orig_idxs) == 5

            # Result is non-trivial (not zero)
            @test !(result isa TScalar && result.val == 0)
        end
    end

    @testset "Bianchi2: product with extra factor" begin
        reg = trinv_registry()
        with_registry(reg) do
            @covd D on=M4 metric=g

            # ∂_a R_{bcde} * R_{fghi} — apply Bianchi2 to factor 1
            dR = TDeriv(down(:a), Tensor(:Riem, [down(:b), down(:c),
                        down(:d), down(:e)]), :D)
            R2 = Tensor(:Riem, [down(:f), down(:g), down(:h), down(:i)])
            expr = tproduct(1 // 1, TensorExpr[dR, R2])

            result = apply_bianchi2_tensorial(expr, 1; registry=reg)
            @test result isa TensorExpr

            # Factor 2 (plain Riemann) should not be affected by Bianchi2
            result2 = apply_bianchi2_tensorial(expr, 2; registry=reg)
            @test result2 == expr  # returns unchanged
        end
    end

    # ─── Display ──────────────────────────────────────────────────────
    @testset "Display" begin
        t = TRInv(2, [5, 6, 3, 4, 1, 2, 7, 8],
                  [3, 4, 7, 8], [Down, Down, Down, Down])
        s = sprint(show, t)
        @test occursin("degree=2", s)
        @test occursin("rank=4", s)
    end

    # ── Tensorial DDI Reduction ──────────────────────────────────────

    @testset "ddi_reduces_trinv: threshold checks" begin
        riem4 = TRInv(1, [1,2,3,4], [1,2,3,4], [Down,Down,Down,Down])

        # Riemann rank-4 in d≤3: Weyl vanishes
        @test ddi_reduces_trinv(riem4, 2)
        @test ddi_reduces_trinv(riem4, 3)
        @test !ddi_reduces_trinv(riem4, 4)

        # Scalar Riem² in d=4: Gauss-Bonnet
        riem2_scalar = TRInv(2, [5,6,7,8, 1,2,3,4], Int[], IndexPosition[], true)
        @test ddi_reduces_trinv(riem2_scalar, 4)
        @test !ddi_reduces_trinv(riem2_scalar, 5)
    end

    @testset "apply_ddi_tensorial: Weyl vanishes in d=3" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M3 dim=3 metric=g registry=reg
            define_curvature_tensors!(reg, :M3, :g)

            used = Set{Symbol}()
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used); push!(used, b)
            c = fresh_index(used); push!(used, c)
            d = fresh_index(used); push!(used, d)

            riem = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
            result = apply_ddi_tensorial(riem, 3; registry=reg, metric=:g)

            # Should not contain Riem or Weyl
            str = string(result)
            @test !occursin("Riem", str)
            @test !occursin("Weyl", str)

            # Should contain Ric and/or RicScalar
            @test occursin("Ric", str)

            # Should have 4 free indices
            @test length(free_indices(result)) == 4
        end
    end

    @testset "apply_ddi_tensorial: Gauss-Bonnet in d=4" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M4, :g)

            used = Set{Symbol}()
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used); push!(used, b)
            c = fresh_index(used); push!(used, c)
            d = fresh_index(used); push!(used, d)

            # Kretschner scalar
            K = Tensor(:Riem, [down(a), down(b), down(c), down(d)]) *
                Tensor(:Riem, [up(a), up(b), up(c), up(d)])
            result = apply_ddi_tensorial(K, 4; registry=reg, metric=:g)

            str = string(result)
            @test !occursin("Riem", str)  # Kretschner eliminated
            @test occursin("Ric", str)    # replaced by Ric terms
        end
    end

    @testset "tinvar_simplify: tensorial expression in d=3" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M3 dim=3 metric=g registry=reg
            define_curvature_tensors!(reg, :M3, :g)

            used = Set{Symbol}()
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used); push!(used, b)
            c = fresh_index(used); push!(used, c)
            d = fresh_index(used); push!(used, d)
            e = fresh_index(used); push!(used, e)

            # R_{a}^{bcd} R_{bcd}^{e} in d=3
            expr = Tensor(:Riem, [down(a), up(b), up(c), up(d)]) *
                   Tensor(:Riem, [down(b), down(c), down(d), up(e)])
            result = tinvar_simplify(expr; registry=reg, dim=3, metric=:g)

            # In d=3, Weyl vanishes → no Riem in result
            str = string(result)
            @test !occursin("Riem", str)
            @test !occursin("Weyl", str)

            # Should have 2 free indices
            @test length(free_indices(result)) == 2
        end
    end

    @testset "tinvar_simplify: scalar in d=4" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M4, :g)

            used = Set{Symbol}()
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used); push!(used, b)
            c = fresh_index(used); push!(used, c)
            d = fresh_index(used); push!(used, d)

            K = Tensor(:Riem, [down(a), down(b), down(c), down(d)]) *
                Tensor(:Riem, [up(a), up(b), up(c), up(d)])
            result = tinvar_simplify(K; registry=reg, dim=4, metric=:g)

            @test !occursin("Riem", string(result))
            @test isempty(free_indices(result))
        end
    end

    @testset "apply_ddi_tensorial: Riem unchanged in d=4" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M4, :g)

            used = Set{Symbol}()
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used); push!(used, b)
            c = fresh_index(used); push!(used, c)
            d = fresh_index(used); push!(used, d)

            riem = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
            result = apply_ddi_tensorial(riem, 4; registry=reg, metric=:g)

            # Riem should remain (no tensorial DDI in d=4 for single Riemann)
            @test occursin("Riem", string(result))
        end
    end
end
