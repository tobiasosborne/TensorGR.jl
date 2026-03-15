@testset "RInv: Contraction Permutation Representation" begin
    # ---- Helper: standard 4D registry with curvature tensors ----
    function rinv_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # ---- Construction and validation ----
    @testset "Construction and validation" begin
        # Kretschmann: R_{abcd} R^{abcd}, k=2, sigma=(1,5)(2,6)(3,7)(4,8)
        kretschmann = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
        @test kretschmann.degree == 2
        @test kretschmann.contraction == [5, 6, 7, 8, 1, 2, 3, 4]
        @test kretschmann.canonical == false

        # Length mismatch
        @test_throws ErrorException RInv(2, [5, 6, 7, 8, 1, 2, 3])

        # Fixed point
        @test_throws ErrorException RInv(1, [1, 3, 2, 4])

        # Not an involution
        @test_throws ErrorException RInv(1, [2, 3, 4, 1])

        # Out of range
        @test_throws ErrorException RInv(1, [2, 9, 4, 3])
    end

    # ---- Degree-1: Ricci scalar R = R^a{}_{ab}{}^b ----
    @testset "Degree 1: Ricci scalar contraction" begin
        # k=1: slots 1-4, contraction (1,3)(2,4) -> R^{ab}{}_{ab}
        r_scalar = RInv(1, [3, 4, 1, 2])
        @test r_scalar.degree == 1
    end

    # ---- Canonical form ----
    @testset "Canonicalize" begin
        reg = rinv_registry()
        with_registry(reg) do
            # Two equivalent representations of Kretschmann:
            #   sigma1 = (1,5)(2,6)(3,7)(4,8)  -- slots 1-4 pair with 5-8
            #   sigma2 = (1,6)(2,5)(3,8)(4,7)  -- swapped within pairs
            kr1 = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            kr2 = RInv(2, [6, 5, 8, 7, 2, 1, 4, 3])

            ckr1 = canonicalize(kr1)
            ckr2 = canonicalize(kr2)

            @test ckr1.canonical == true
            @test ckr2.canonical == true
            @test ckr1.contraction == ckr2.contraction
        end
    end

    # ---- Equality via canonical comparison ----
    @testset "Equality" begin
        reg = rinv_registry()
        with_registry(reg) do
            kr1 = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            kr2 = RInv(2, [6, 5, 8, 7, 2, 1, 4, 3])
            @test kr1 == kr2

            # Different invariant: Ricci squared R_{ab}R^{ab} at degree 2
            # as Riemann with self-contractions: R^c_{acb} R^e_{def}
            # contraction: slot 1->3, slot 5->7 (self-contract each factor)
            # plus cross: slot 2->6, slot 4->8
            ric_sq = RInv(2, [3, 6, 1, 8, 7, 2, 5, 4])
            @test kr1 != ric_sq  # different invariant classes

            # Degree mismatch -> not equal
            r1 = RInv(1, [3, 4, 1, 2])
            @test r1 != kr1
        end
    end

    # ---- Idempotent canonicalization ----
    @testset "Canonicalize idempotent" begin
        reg = rinv_registry()
        with_registry(reg) do
            kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            c1 = canonicalize(kr)
            c2 = canonicalize(c1)
            @test c1.contraction == c2.contraction
            @test c2.canonical == true
        end
    end

    # ---- to_tensor_expr ----
    @testset "to_tensor_expr" begin
        reg = rinv_registry()
        with_registry(reg) do
            # Kretschmann scalar
            kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            expr = to_tensor_expr(kr; registry=reg, metric=:g)

            # Should be a TProduct
            @test expr isa TProduct

            # No free indices (fully contracted scalar)
            @test isempty(free_indices(expr))

            # Should contain exactly 2 Riemann tensors
            riems = filter(f -> f isa Tensor && f.name == :Riem, expr.factors)
            @test length(riems) == 2

            # Should contain 4 metric tensors (4 pairs for k=2)
            mets = filter(f -> f isa Tensor && f.name == :g, expr.factors)
            @test length(mets) == 4
        end
    end

    # ---- from_tensor_expr ----
    @testset "from_tensor_expr" begin
        reg = rinv_registry()
        with_registry(reg) do
            # Build Kretschmann by hand: Riem_{abcd} Riem_{efgh} g^{ae} g^{bf} g^{cg} g^{dh}
            a, b, c, d = :a, :b, :c, :d
            e, f, gg, h = :e, :f, :gg, :h
            R1 = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
            R2 = Tensor(:Riem, [down(e), down(f), down(gg), down(h)])
            expr = tproduct(1 // 1, TensorExpr[
                R1, R2,
                Tensor(:g, [up(a), up(e)]),
                Tensor(:g, [up(b), up(f)]),
                Tensor(:g, [up(c), up(gg)]),
                Tensor(:g, [up(d), up(h)]),
            ])

            rinv = from_tensor_expr(expr; registry=reg, metric=:g)
            @test rinv.degree == 2
            # The contraction should pair slot 1 with 5, 2 with 6, etc.
            @test rinv.contraction == [5, 6, 7, 8, 1, 2, 3, 4]
        end
    end

    # ---- Round-trip: from_tensor_expr . to_tensor_expr ----
    @testset "Round-trip from_tensor_expr(to_tensor_expr(...))" begin
        reg = rinv_registry()
        with_registry(reg) do
            kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])
            expr = to_tensor_expr(kr; registry=reg, metric=:g)
            kr2 = from_tensor_expr(expr; registry=reg, metric=:g)
            @test kr == kr2
        end
    end

    # ---- Degree-2 invariant count (ZM97) ----
    @testset "Degree-2 invariant count" begin
        reg = rinv_registry()
        with_registry(reg) do
            # At degree 2, there are exactly 3 independent Riemann monomials:
            #  1. R^2 (double self-contraction): (1,2)(3,4)(5,6)(7,8) -- self-contracts each factor
            #  2. Ric^2: cross-contract some slots
            #  3. Kretschmann: full cross-contraction

            # R^2 as Riemann: R^{ab}{}_{ab} * R^{cd}{}_{cd}
            # self-contracts: (1,2)(3,4) for factor 1, (5,6)(7,8) for factor 2
            r_sq = RInv(2, [2, 1, 4, 3, 6, 5, 8, 7])

            # Ric^2 as Riemann: R^c_{acb} R^f_{dfe} g^{ad} g^{be}
            # Factor 1: slots 1,2,3,4 = c,a,c,b -> self-contract 1<->3
            # Factor 2: slots 5,6,7,8 = f,d,f,e -> self-contract 5<->7
            # Cross: 2<->6, 4<->8
            ric_sq = RInv(2, [3, 6, 1, 8, 7, 2, 5, 4])

            # Kretschmann: full cross-contraction
            kretschmann = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])

            # All three should be distinct
            @test r_sq != ric_sq
            @test r_sq != kretschmann
            @test ric_sq != kretschmann
        end
    end

    # ---- Error on non-Riemann factors ----
    @testset "from_tensor_expr errors" begin
        reg = rinv_registry()
        with_registry(reg) do
            # Non-Riemann tensor should error
            bad = Tensor(:Ric, [down(:a), down(:b)])
            @test_throws ErrorException from_tensor_expr(bad; registry=reg, metric=:g)

            # TScalar should error
            @test_throws ErrorException from_tensor_expr(TScalar(1); registry=reg, metric=:g)
        end
    end

    # ---- Degree-1 round-trip ----
    @testset "Degree-1 round-trip" begin
        reg = rinv_registry()
        with_registry(reg) do
            # R = R^{ab}{}_{ab} -> contraction (1,3)(2,4)
            r1 = RInv(1, [3, 4, 1, 2])
            expr = to_tensor_expr(r1; registry=reg, metric=:g)
            @test expr isa TProduct
            @test isempty(free_indices(expr))

            r2 = from_tensor_expr(expr; registry=reg, metric=:g)
            @test r1 == r2
        end
    end

    # ---- Degree-3 construction ----
    @testset "Degree-3 construction" begin
        # Goroff-Sagnotti: R_{ab}^{cd} R_{cd}^{ef} R_{ef}^{ab}
        # Contraction: (1,11)(2,12)(3,5)(4,6)(7,9)(8,10)
        gs = RInv(3, [11, 12, 5, 6, 3, 4, 9, 10, 7, 8, 1, 2])
        @test gs.degree == 3
        c = canonicalize(gs)
        @test c.canonical == true
    end
end
