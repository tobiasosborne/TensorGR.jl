@testset "SymH unified symmetry handler" begin

    # ── Construction from FullySymmetric ──────────────────────────────
    @testset "SymH from FullySymmetric" begin
        spec = FullySymmetric(1, 2, 3)
        symh = SymH(spec, 3)
        @test symh.nslots == 3
        # FullySymmetric(1,2,3) gives 2 adjacent transposition generators
        @test length(symh.monoterm) == 2
        @test is_monoterm_only(symh)
        # Generators: swap (1,2) with sign +1, swap (2,3) with sign +1
        @test symh.monoterm[1].sign == +1
        @test symh.monoterm[2].sign == +1
    end

    # ── Construction from FullyAntiSymmetric ──────────────────────────
    @testset "SymH from FullyAntiSymmetric" begin
        spec = FullyAntiSymmetric(1, 2, 3)
        symh = SymH(spec, 3)
        @test symh.nslots == 3
        @test length(symh.monoterm) == 2
        @test is_monoterm_only(symh)
        # Generators should have sign -1
        @test symh.monoterm[1].sign == -1
        @test symh.monoterm[2].sign == -1
    end

    # ── Construction from raw generators ──────────────────────────────
    @testset "SymH from raw generators" begin
        gens = [([2, 1], +1)]  # symmetric pair
        symh = SymH(gens, 2)
        @test symh.nslots == 2
        @test length(symh.monoterm) == 1
        @test symh.monoterm[1].perm == [2, 1]
        @test symh.monoterm[1].sign == +1
        @test isempty(symh.multiterm)
    end

    # ── Riemann SymH ──────────────────────────────────────────────────
    @testset "riemann_symh" begin
        rs = riemann_symh()
        @test rs.nslots == 4

        # 3 monoterm generators: anti(1,2), anti(3,4), pair(1,2,3,4)
        @test length(rs.monoterm) == 3

        # Check anti(1,2): perm = [2,1,3,4], sign = -1
        @test rs.monoterm[1].perm == [2, 1, 3, 4]
        @test rs.monoterm[1].sign == -1

        # Check anti(3,4): perm = [1,2,4,3], sign = -1
        @test rs.monoterm[2].perm == [1, 2, 4, 3]
        @test rs.monoterm[2].sign == -1

        # Check pair(1,2,3,4): perm = [3,4,1,2], sign = +1
        @test rs.monoterm[3].perm == [3, 4, 1, 2]
        @test rs.monoterm[3].sign == +1

        # 1 multi-term relation: first Bianchi
        @test length(rs.multiterm) == 1
        @test !is_monoterm_only(rs)

        bianchi = rs.multiterm[1]
        @test bianchi.nslots == 4
        @test length(bianchi.terms) == 3
        # Check the three permutations of Bianchi
        @test bianchi.terms[1] == (1 // 1, [1, 2, 3, 4])
        @test bianchi.terms[2] == (1 // 1, [1, 3, 4, 2])
        @test bianchi.terms[3] == (1 // 1, [1, 4, 2, 3])
    end

    # ── Component counting: symmetric rank-2 ──────────────────────────
    @testset "n_independent_components: symmetric rank-2 in d=4" begin
        spec = FullySymmetric(1, 2)
        symh = SymH(spec, 2)
        # d(d+1)/2 = 4*5/2 = 10
        @test n_independent_components(symh, 4) == 10
    end

    @testset "n_independent_components: symmetric rank-2 in d=3" begin
        spec = FullySymmetric(1, 2)
        symh = SymH(spec, 2)
        # d(d+1)/2 = 3*4/2 = 6
        @test n_independent_components(symh, 3) == 6
    end

    # ── Component counting: antisymmetric rank-2 ──────────────────────
    @testset "n_independent_components: antisymmetric rank-2 in d=4" begin
        spec = FullyAntiSymmetric(1, 2)
        symh = SymH(spec, 2)
        # d(d-1)/2 = 4*3/2 = 6
        @test n_independent_components(symh, 4) == 6
    end

    # ── Component counting: Riemann tensor ────────────────────────────
    @testset "n_independent_components: Riemann in d=4 gives 20" begin
        rs = riemann_symh()
        # d^2(d^2-1)/12 = 16*15/12 = 20
        @test n_independent_components(rs, 4) == 20
    end

    @testset "n_independent_components: Riemann in d=3 gives 6" begin
        rs = riemann_symh()
        # d^2(d^2-1)/12 = 9*8/12 = 6
        @test n_independent_components(rs, 3) == 6
    end

    @testset "n_independent_components: Riemann in d=2 gives 1" begin
        rs = riemann_symh()
        # d^2(d^2-1)/12 = 4*3/12 = 1
        @test n_independent_components(rs, 2) == 1
    end

    # ── Monoterm-only Riemann (no Bianchi) gives 21 in d=4 ────────────
    @testset "monoterm-only Riemann (no Bianchi) in d=4" begin
        # Riemann without Bianchi: 21 components in d=4
        gens = [
            MonotermSym([2, 1, 3, 4], -1),
            MonotermSym([1, 2, 4, 3], -1),
            MonotermSym([3, 4, 1, 2], +1),
        ]
        symh_no_bianchi = SymH(4, gens, MultitermSym[])
        @test n_independent_components(symh_no_bianchi, 4) == 21
    end

    # ── Round-trip: SymmetrySpec -> SymH -> SymmetrySpec ───────────────
    @testset "round-trip SymmetrySpec -> SymH -> SymmetrySpec" begin
        # Symmetric pair
        spec = Symmetric(1, 2)
        symh = SymH(spec, 3)
        specs_back = to_symmetry_spec(symh)
        @test length(specs_back) == 1
        @test specs_back[1] isa Symmetric
        @test specs_back[1].i == 1
        @test specs_back[1].j == 2

        # AntiSymmetric pair
        spec_a = AntiSymmetric(2, 3)
        symh_a = SymH(spec_a, 3)
        specs_a = to_symmetry_spec(symh_a)
        @test length(specs_a) == 1
        @test specs_a[1] isa AntiSymmetric
        @test specs_a[1].i == 2
        @test specs_a[1].j == 3

        # PairSymmetric
        spec_p = PairSymmetric(1, 2, 3, 4)
        symh_p = SymH(spec_p, 4)
        specs_p = to_symmetry_spec(symh_p)
        @test length(specs_p) == 1
        @test specs_p[1] isa PairSymmetric
    end

    # ── Bianchi as MultitermSym ───────────────────────────────────────
    @testset "Bianchi identity as MultitermSym" begin
        bianchi = TensorGR._bianchi_multiterm()
        @test bianchi.nslots == 4
        @test length(bianchi.terms) == 3
        # All coefficients are +1
        for (c, _) in bianchi.terms
            @test c == 1 // 1
        end
        # The three permutations encode R_{abcd} + R_{acdb} + R_{adbc} = 0
        perms = [p for (_, p) in bianchi.terms]
        @test [1, 2, 3, 4] in perms
        @test [1, 3, 4, 2] in perms
        @test [1, 4, 2, 3] in perms
    end

    # ── SymH from RiemannSymmetry spec ────────────────────────────────
    @testset "SymH from RiemannSymmetry" begin
        spec = RiemannSymmetry()
        symh = SymH(spec, 4)
        # Should have 3 monoterm generators + 1 multiterm
        @test length(symh.monoterm) == 3
        @test length(symh.multiterm) == 1
        # Component count should give 20 in d=4
        @test n_independent_components(symh, 4) == 20
    end

    # ── xperm generator round-trip ────────────────────────────────────
    @testset "to_xperm_generators round-trip" begin
        spec = RiemannSymmetry()
        symh = SymH(spec, 4)
        xperm_gens = to_xperm_generators(symh)
        # Compare with direct symmetry_generators
        direct_gens = symmetry_generators([spec], 4)
        @test length(xperm_gens) == length(direct_gens)
        for (xg, dg) in zip(xperm_gens, direct_gens)
            @test xg.data == dg.data
        end
    end

    # ── Fully symmetric rank-3 in various dimensions ──────────────────
    @testset "n_independent_components: fully symmetric rank-3" begin
        symh = SymH(FullySymmetric(1, 2, 3), 3)
        # Binomial(d+2, 3) = d(d+1)(d+2)/6
        @test n_independent_components(symh, 2) == 4    # 2*3*4/6
        @test n_independent_components(symh, 3) == 10   # 3*4*5/6
        @test n_independent_components(symh, 4) == 20   # 4*5*6/6
    end

    # ── Fully antisymmetric rank-3 in various dimensions ──────────────
    @testset "n_independent_components: fully antisymmetric rank-3" begin
        symh = SymH(FullyAntiSymmetric(1, 2, 3), 3)
        # Binomial(d, 3) = d(d-1)(d-2)/6
        @test n_independent_components(symh, 3) == 1    # 3*2*1/6
        @test n_independent_components(symh, 4) == 4    # 4*3*2/6
        @test n_independent_components(symh, 5) == 10   # 5*4*3/6
    end

    # ── SymH canonicalization ────────────────────────────────────────
    @testset "canonicalize_symh: monoterm" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)

            # Riemann tensor: canonicalize via SymH should match
            # standard canonicalize
            R = Tensor(:Riem, [down(:c), down(:a), down(:b), down(:d)])
            symh = riemann_symh()

            csymh = canonicalize_symh(R, symh; registry=reg)
            cstd = canonicalize(R)

            @test csymh == cstd
        end
    end

    @testset "canonicalize_symh: symmetric tensor" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            register_tensor!(reg, TensorProperties(
                name=:T, manifold=:M4, rank=(0,2),
                symmetries=[Symmetric(1, 2)]))

            T_ba = Tensor(:T, [down(:b), down(:a)])
            symh = SymH(Symmetric(1, 2), 2)

            result = canonicalize_symh(T_ba, symh; registry=reg)
            expected = canonicalize(T_ba)
            @test result == expected
        end
    end

    # ── symmetrize_symh ──────────────────────────────────────────────
    @testset "symmetrize_symh: symmetric projection" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(0,2),
                symmetries=SymmetrySpec[]))

            V = Tensor(:V, [down(:a), down(:b)])
            symh = SymH(Symmetric(1, 2), 2)

            # Symmetrizing V_{ab} should give (V_{ab} + V_{ba})/2
            result = symmetrize_symh(V, symh; registry=reg)
            @test result isa TensorExpr

            # The group has order 2: {id, (12)}, so the projector is
            # P(V_{ab}) = (1/2)(V_{ab} + V_{ba})
            # After simplification this should be a sum of two terms
            # with coefficient 1/2 each
        end
    end

    @testset "symmetrize_symh: antisymmetric projection" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(0,2),
                symmetries=SymmetrySpec[]))

            V = Tensor(:V, [down(:a), down(:b)])
            symh = SymH(AntiSymmetric(1, 2), 2)

            result = symmetrize_symh(V, symh; registry=reg)
            @test result isa TensorExpr
        end
    end

    # ── verify_symh ──────────────────────────────────────────────────
    @testset "verify_symh: Riemann" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)

            R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            symh = riemann_symh()

            # Riemann tensor should satisfy its own symmetry
            @test verify_symh(R, symh; registry=reg)
        end
    end

    @testset "verify_symh: wrong symmetry" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)

            R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            # Fully symmetric SymH — Riemann is NOT fully symmetric
            wrong_symh = SymH(FullySymmetric(1, 2, 3, 4), 4)

            @test !verify_symh(R, wrong_symh; registry=reg)
        end
    end

    # ── SymH Arithmetic ──────────────────────────────────────────────

    @testset "symh_product: Riem⊗Riem" begin
        riem = riemann_symh()
        prod = symh_product(riem, riem)

        @test prod.nslots == 8
        @test length(prod.monoterm) == 6   # 3 from each factor
        @test length(prod.multiterm) == 2  # Bianchi from each factor

        # First factor generators act on slots 1-4, identity on 5-8
        for m in prod.monoterm[1:3]
            @test all(m.perm[5:8] .== 5:8)
        end
        # Second factor generators: identity on 1-4, act on 5-8
        for m in prod.monoterm[4:6]
            @test all(m.perm[1:4] .== 1:4)
        end

        # Independent components: 20 * 20 = 400 (no exchange)
        @test n_independent_components(prod, 4) == 400
    end

    @testset "symh_product: Sym(2) ⊗ AntiSym(2)" begin
        sym2 = SymH(FullySymmetric(1, 2), 2)
        anti2 = SymH(FullyAntiSymmetric(1, 2), 2)
        prod = symh_product(sym2, anti2)

        @test prod.nslots == 4
        @test length(prod.monoterm) == 2  # 1 from each

        # d=3: sym2 has 6 components, anti2 has 3 → product = 18
        @test n_independent_components(prod, 3) == 18
    end

    @testset "symh_exchange: Riem⊗Riem" begin
        riem = riemann_symh()
        prod = symh_product(riem, riem)
        prod_ex = symh_exchange(prod, 4; sign=1)

        @test prod_ex.nslots == 8
        @test length(prod_ex.monoterm) == 7  # 6 + 1 exchange

        # Symmetric product: 20*21/2 = 210
        @test n_independent_components(prod_ex, 4) == 210

        # Wrong nslots throws
        @test_throws ArgumentError symh_exchange(prod, 3)
    end

    @testset "symh_trace: Riemann slots (1,3)" begin
        riem = riemann_symh()
        traced = symh_trace(riem, 1, 3)

        @test traced.nslots == 2
        # anti(1,2) maps 1→2,2→1: slot 1 goes to 2 (not in {1,3}) → filtered out
        # anti(3,4) maps 3→4,4→3: slot 3 goes to 4 (not in {1,3}) → filtered out
        # pair(1,2,3,4) maps 1→3,3→1: {1,3}→{3,1}={1,3} → kept!
        @test length(traced.monoterm) == 1
        # The kept generator swaps the two remaining slots → symmetric!
        @test traced.monoterm[1].sign == 1  # pair symmetry has sign +1
    end

    @testset "symh_trace: edge cases" begin
        # Trace of symmetric rank-2: scalar (0 slots)
        sym2 = SymH(FullySymmetric(1, 2), 2)
        traced = symh_trace(sym2, 1, 2)
        @test traced.nslots == 0

        # Invalid indices throw
        @test_throws ArgumentError symh_trace(riemann_symh(), 3, 1)
        @test_throws ArgumentError symh_trace(riemann_symh(), 0, 2)
    end
end
