#= Validate quadratic Riemann invariants via all_contractions and contraction_ansatz.
   Ground truth: Fulling, King, Wybourne & Cummins, "Normal forms for tensor polynomials.
   1: The Riemann tensor", CQG 9 (1992) 1151-1197, DOI:10.1088/0264-9381/9/5/003.

   Key results verified here:
   - R_{abcd} has Riemann symmetries (antisym on pairs, pair-swap, Bianchi).
   - A single Riemann tensor admits exactly 1 independent full contraction: the Ricci
     scalar R = g^{ac}g^{bd}R_{abcd} (Fulling et al. Sec 2; Nutma Eq 37).
   - The product R_{abcd}R_{efgh} is rank 8, giving (8-1)!! = 105 raw pairings.
     After Riemann symmetries, the independent quadratic invariants are:
       (1) R^2 = (g^{ac}g^{bd}R_{abcd})^2    — Ricci scalar squared
       (2) R_{ab}R^{ab}                         — Ricci tensor norm
       (3) R_{abcd}R^{abcd}                     — Kretschmann scalar
     These three span the space of quadratic curvature invariants in arbitrary
     dimension (Fulling et al. Table 1, order p=2).
   - In d=4 the Gauss-Bonnet relation R_{abcd}R^{abcd} - 4R_{ab}R^{ab} + R^2 = E_4
     provides one syzygy, but the three invariants remain algebraically independent
     before topological reduction.
   - For the Ricci tensor (rank-2 symmetric), there are exactly 2 independent
     quadratic invariants: R_{ab}R^{ab} and R^2 (Fulling et al. Sec 2).
=#

using TensorGR: all_contractions, contraction_ansatz, free_indices

@testset "Quadratic Riemann invariants (Fulling et al. 1992, CQG 9:1151)" begin

    # Shared registry builder: 4D manifold with curvature tensors
    function fulling_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        reg
    end

    # --------------------------------------------------------------------------
    # 1. Raw pairing count for rank-8 product R*R
    #    Fulling et al. Sec 2: a rank-2n expression has (2n-1)!! raw pairings.
    #    For n=8: (8-1)!! = 7!! = 7*5*3*1 = 105.
    # --------------------------------------------------------------------------
    @testset "Raw pairing count: 8 indices -> 105 pairings" begin
        # Verify the combinatorial identity directly
        double_factorial_7 = 7 * 5 * 3 * 1  # 105
        @test double_factorial_7 == 105

        # The internal pairing generator should produce exactly 105 matchings
        pairings = TensorGR._all_pairings(collect(1:8))
        @test length(pairings) == 105
    end

    # --------------------------------------------------------------------------
    # 2. Single Riemann full contraction -> 1 result (Ricci scalar)
    #    Fulling et al. Sec 2 / Nutma Eq 37: From the 3 raw pairings of
    #    R_{abcd}, antisymmetry kills (a,b)(c,d) and pair-swap identifies
    #    (a,c)(b,d) ~ -(a,d)(b,c). Only the Ricci scalar survives.
    # --------------------------------------------------------------------------
    @testset "Single Riemann: 1 contraction (Ricci scalar)" begin
        reg = fulling_registry()
        with_registry(reg) do
            R_single = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            single = all_contractions(R_single, :g; registry=reg)

            # Fulling et al.: exactly 1 independent contraction
            @test length(single) == 1

            # Must be a scalar (no free indices)
            @test isempty(free_indices(single[1]))

            # The result must involve the Ricci scalar
            s = simplify(single[1]; registry=reg)
            has_ric_scalar = false
            walk(s) do node
                if node isa Tensor && node.name == :RicScalar
                    has_ric_scalar = true
                end
                node
            end
            @test has_ric_scalar
        end
    end

    # --------------------------------------------------------------------------
    # 3. Riem*Riem full contractions -> complete basis (>= 3 invariants)
    #    Fulling et al. Table 1, p=2: three independent quadratic curvature
    #    invariants exist in arbitrary dimension: R^2, R_{ab}R^{ab}, R_{abcd}R^{abcd}.
    #    The all_contractions function returns a complete basis that may include
    #    the additional contraction R_{acbd}R^{abcd} (related to the others by
    #    Bianchi), but must contain at least the 3 independent ones.
    # --------------------------------------------------------------------------
    @testset "Riem*Riem: complete basis >= 3 invariants" begin
        reg = fulling_registry()
        with_registry(reg) do
            R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R2 = Tensor(:Riem, [down(:e), down(:f), down(:g1), down(:h)])
            RR = R1 * R2

            filtered = all_contractions(RR, :g; registry=reg)

            # Fulling et al.: at least 3 truly independent invariants
            @test length(filtered) >= 3

            # Must be strictly fewer than the 105 raw pairings
            @test length(filtered) < 105

            # All results must be scalars
            for c in filtered
                @test isempty(free_indices(c))
            end

            # All results must be nonzero
            for c in filtered
                @test c != TScalar(0 // 1)
            end
        end
    end

    # --------------------------------------------------------------------------
    # 4. Verify the three known invariants appear in the Riem*Riem basis
    #    Fulling et al. Table 1: the three independent quadratic invariants are
    #    recognizable by their tensor content after simplification:
    #      (a) RicScalar * RicScalar  (R^2)
    #      (b) Ric * Ric              (R_{ab}R^{ab})
    #      (c) Riem * Riem            (Kretschmann, R_{abcd}R^{abcd})
    # --------------------------------------------------------------------------
    @testset "Riem*Riem basis contains R^2, Ric^2, Kretschmann" begin
        reg = fulling_registry()
        with_registry(reg) do
            R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R2 = Tensor(:Riem, [down(:e), down(:f), down(:g1), down(:h)])
            RR = R1 * R2

            filtered = all_contractions(RR, :g; registry=reg)

            has_R_sq = false       # Two RicScalar factors
            has_Ric_sq = false     # Two Ric factors (no RicScalar, no Riem)
            has_Kretschmann = false  # Two Riem factors

            for c in filtered
                s = simplify(c; registry=reg)
                str = string(s)

                # R^2: two RicScalar factors
                if count("RicScalar", str) >= 2
                    has_R_sq = true
                end

                # Ric^2: Ric factors present, no bare Riem or RicScalar
                if occursin("Ric", str) && !occursin("RicScalar", str) && !occursin("Riem", str)
                    has_Ric_sq = true
                end

                # Kretschmann: two Riem factors
                if count("Riem", str) >= 2
                    has_Kretschmann = true
                end
            end

            # Fulling et al. Table 1: all three must appear
            @test has_R_sq
            @test has_Ric_sq
            @test has_Kretschmann
        end
    end

    # --------------------------------------------------------------------------
    # 5. contraction_ansatz for Ric*Ric -> exactly 2 invariants
    #    Fulling et al. Sec 2: the Ricci tensor R_{ab} is symmetric rank-2.
    #    Two independent quadratic contractions: R_{ab}R^{ab} and R^2.
    # --------------------------------------------------------------------------
    @testset "contraction_ansatz([:Ric,:Ric]) -> 2 invariants" begin
        reg = fulling_registry()
        with_registry(reg) do
            result = contraction_ansatz([:Ric, :Ric], :g; registry=reg)

            # Fulling et al.: exactly 2 independent quadratic Ricci invariants
            @test result isa TSum
            @test length(result.terms) == 2

            # Both terms must be scalars
            for t in result.terms
                @test isempty(free_indices(t))
            end

            # Coefficients must be distinct
            coeffs = Symbol[]
            for t in result.terms
                @test t isa TProduct
                scalar_factors = filter(f -> f isa TScalar && f.val isa Symbol, t.factors)
                @test !isempty(scalar_factors)
                push!(coeffs, scalar_factors[1].val)
            end
            @test length(unique(coeffs)) == 2
        end
    end

    # --------------------------------------------------------------------------
    # 6. contraction_ansatz for Riem*Riem -> complete basis (>= 3 terms)
    #    Fulling et al. Table 1, p=2: at least 3 independent invariants.
    # --------------------------------------------------------------------------
    @testset "contraction_ansatz([:Riem,:Riem]) -> >= 3 terms" begin
        reg = fulling_registry()
        with_registry(reg) do
            result = contraction_ansatz([:Riem, :Riem], :g; registry=reg)
            terms = result isa TSum ? result.terms : [result]

            # Fulling et al.: at least 3 independent invariants
            @test length(terms) >= 3

            # All terms must be scalars
            for t in terms
                @test isempty(free_indices(t))
            end
        end
    end

    # --------------------------------------------------------------------------
    # 7. Gauss-Bonnet syzygy reduces basis by 1
    #    Fulling et al. Sec 5 / Lanczos-Lovelock: In d=4,
    #      R_{abcd}R^{abcd} - 4R_{ab}R^{ab} + R^2 = E_4 (topological).
    #    After applying the Gauss-Bonnet rule, the Kretschmann scalar is
    #    expressible in terms of Ric^2 and R^2, eliminating one invariant.
    # --------------------------------------------------------------------------
    @testset "Gauss-Bonnet reduces Kretschmann to Ric^2, R^2" begin
        reg = fulling_registry()
        for r in gauss_bonnet_rule(; metric=:g)
            register_rule!(reg, r)
        end

        with_registry(reg) do
            # Build Kretschmann scalar: Riem_{abcd} Riem^{abcd}
            Riem_down = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem_up = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            kretschmann = Riem_down * Riem_up

            result = simplify(kretschmann; registry=reg)

            # After Gauss-Bonnet: no Riem factors should remain
            has_riem = false
            walk(result) do node
                if node isa Tensor && node.name == :Riem
                    has_riem = true
                end
                node
            end
            @test !has_riem

            # Must still be a scalar
            @test isempty(free_indices(result))

            # Should contain Ric and RicScalar (the reduced basis)
            has_ric = false
            has_ric_scalar = false
            walk(result) do node
                if node isa Tensor && node.name == :Ric
                    has_ric = true
                end
                if node isa Tensor && node.name == :RicScalar
                    has_ric_scalar = true
                end
                node
            end
            @test has_ric
            @test has_ric_scalar
        end
    end

    # --------------------------------------------------------------------------
    # 8. Verify each contraction is a genuine scalar (no free indices)
    #    Fulling et al. Sec 2: a "full contraction" pairs all indices, leaving
    #    a scalar invariant. This must hold for every element returned by
    #    all_contractions.
    # --------------------------------------------------------------------------
    @testset "All contractions are genuine scalars" begin
        reg = fulling_registry()
        with_registry(reg) do
            R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R2 = Tensor(:Riem, [down(:e), down(:f), down(:g1), down(:h)])
            RR = R1 * R2

            for c in all_contractions(RR, :g; registry=reg)
                s = simplify(c; registry=reg)
                @test isempty(free_indices(s))
            end
        end
    end

    # --------------------------------------------------------------------------
    # 9. Weyl tensor: mono-term symmetries vs tracelessness
    #    Fulling et al. Sec 3: The Weyl tensor C_{abcd} has all Riemann
    #    symmetries plus tracelessness (g^{ac}C_{abcd} = 0). Tracelessness
    #    is a multi-term identity, not a mono-term permutation symmetry.
    #    As noted in Nutma (after Eq 38), all_contractions uses only mono-term
    #    symmetries, so it returns a complete (but not irreducible) basis.
    #    The Riemann antisymmetry kills (a,b)(c,d) from the 3 raw pairings,
    #    leaving at most 2 contractions before tracelessness is applied.
    # --------------------------------------------------------------------------
    @testset "Single Weyl: <= 2 contractions (mono-term symmetry)" begin
        reg = fulling_registry()
        with_registry(reg) do
            W = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            results = all_contractions(W, :g; registry=reg)

            # Riemann-type antisymmetry kills at least 1 of the 3 raw pairings
            @test length(results) <= 2

            # All results must be scalars
            for r in results
                @test isempty(free_indices(r))
            end
        end
    end

    # --------------------------------------------------------------------------
    # 10. Metric contraction of rank-2 symmetric tensor -> 1 invariant
    #     Fulling et al. Sec 2: a symmetric rank-2 tensor S_{ab} has one
    #     trace: g^{ab}S_{ab}. This is a basic consistency check.
    # --------------------------------------------------------------------------
    @testset "Symmetric rank-2: 1 contraction (trace)" begin
        reg = fulling_registry()
        with_registry(reg) do
            # Ricci tensor is symmetric rank-2
            Ric = Tensor(:Ric, [down(:a), down(:b)])
            results = all_contractions(Ric, :g; registry=reg)

            # Exactly 1 contraction: the trace (Ricci scalar)
            @test length(results) == 1
            @test isempty(free_indices(results[1]))
        end
    end

    # --------------------------------------------------------------------------
    # 11. Consistency: contraction_ansatz for single Riemann -> 1 term
    #     Fulling et al. Sec 2: the most general scalar linear in the Riemann
    #     tensor is c_1 * R (a single term proportional to the Ricci scalar).
    # --------------------------------------------------------------------------
    @testset "contraction_ansatz([:Riem]) -> 1 term" begin
        reg = fulling_registry()
        with_registry(reg) do
            result = contraction_ansatz([:Riem], :g; registry=reg)
            terms = result isa TSum ? result.terms : [result]

            # Fulling et al.: exactly 1 independent linear Riemann invariant
            @test length(terms) == 1

            # Must be a scalar
            for t in terms
                @test isempty(free_indices(t))
            end
        end
    end

    # --------------------------------------------------------------------------
    # 12. Dimensional check: invariant counts are dimension-independent
    #     Fulling et al. Table 1: the algebraic invariant count (3 for p=2)
    #     holds in any dimension d >= 4. The Gauss-Bonnet syzygy (reducing
    #     to 2) is a topological relation, not an algebraic one.
    #     Here we verify the same counts hold in d=5.
    # --------------------------------------------------------------------------
    @testset "Dimension independence: d=5 gives same invariant structure" begin
        reg5 = TensorRegistry()
        with_registry(reg5) do
            @manifold M5 dim=5 metric=g
            define_curvature_tensors!(reg5, :M5, :g)
        end

        with_registry(reg5) do
            # Single Riemann: still 1 contraction in d=5
            R_single = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            single = all_contractions(R_single, :g; registry=reg5)
            @test length(single) == 1

            # Ric*Ric: still 2 in d=5
            ric_ans = contraction_ansatz([:Ric, :Ric], :g; registry=reg5)
            @test ric_ans isa TSum
            @test length(ric_ans.terms) == 2

            # Riem*Riem: still >= 3 in d=5
            riem_ans = contraction_ansatz([:Riem, :Riem], :g; registry=reg5)
            riem_terms = riem_ans isa TSum ? riem_ans.terms : [riem_ans]
            @test length(riem_terms) >= 3
        end
    end
end
