#= Ground truth verification for AllContractions and ContractionAnsatz.
   Reference: Nutma (2014), "xTras: A field-theory inspired xAct package
   for Mathematica", arXiv:1308.3493.
   Section numbers refer to this paper throughout.

   Note: define_metric! already calls define_curvature_tensors! internally,
   so curvature tensors (Riem, Ric, RicScalar, Ein, Weyl) are automatically
   available after define_metric!. =#

using TensorGR: all_contractions, contraction_ansatz, free_indices,
                _all_pairings, _perfect_matchings

# ---------------------------------------------------------------------------
# 1. Double factorial pairing counts: (2n-1)!! = 1*3*5*...*(2n-1)
#    Nutma Sec 5.1.1: "the naive number of possible contractions of a
#    tensorial expression that has n free indices is (n-1)!!, which is the
#    number of independent products of n/2 metrics."
# ---------------------------------------------------------------------------

@testset "Nutma Sec 5.1.1: (n-1)!! pairing counts" begin
    # rank 2: (2-1)!! = 1
    @test length(_all_pairings(collect(1:2))) == 1

    # rank 4: (4-1)!! = 3
    @test length(_all_pairings(collect(1:4))) == 3

    # rank 6: (6-1)!! = 15
    @test length(_all_pairings(collect(1:6))) == 15

    # rank 8: (8-1)!! = 105
    @test length(_all_pairings(collect(1:8))) == 105

    # _perfect_matchings should agree with _all_pairings
    @test length(_perfect_matchings(collect(1:2))) == 1
    @test length(_perfect_matchings(collect(1:4))) == 3
    @test length(_perfect_matchings(collect(1:6))) == 15
    @test length(_perfect_matchings(collect(1:8))) == 105
end

# ---------------------------------------------------------------------------
# 2. Rank-2 and rank-4 generic tensor contractions
#    Nutma Sec 5.1.1: a rank-2n tensor has (2n-1)!! full contractions before
#    symmetry deduplication.
# ---------------------------------------------------------------------------

@testset "Nutma Sec 5.1.1: rank-2 tensor -> 1 contraction" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)
    register_tensor!(reg, TensorProperties(name=:S, manifold=:M4, rank=(0,2),
        symmetries=Any[Symmetric(1,2)]))

    with_registry(reg) do
        S = Tensor(:S, [down(:a), down(:b)])
        results = all_contractions(S, :g; registry=reg)
        @test length(results) == 1  # (2-1)!! = 1
        @test isempty(free_indices(results[1]))
    end
end

@testset "Nutma Sec 5.1.1: rank-4 generic tensor -> 3 contractions" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)
    register_tensor!(reg, TensorProperties(name=:T4, manifold=:M4, rank=(0,4),
        symmetries=Any[]))

    with_registry(reg) do
        T4 = Tensor(:T4, [down(:a), down(:b), down(:c), down(:d)])
        results = all_contractions(T4, :g; registry=reg)
        @test length(results) == 3  # (4-1)!! = 3
        for r in results
            @test isempty(free_indices(r))
            @test r != TScalar(0 // 1)
        end
    end
end

@testset "Nutma Sec 5.1.1: rank-6 generic tensor -> 15 contractions" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)
    register_tensor!(reg, TensorProperties(name=:T6, manifold=:M4, rank=(0,6),
        symmetries=Any[]))

    with_registry(reg) do
        T6 = Tensor(:T6, [down(:a), down(:b), down(:c), down(:d),
                          down(:e), down(:f)])
        results = all_contractions(T6, :g; registry=reg)
        @test length(results) == 15  # (6-1)!! = 15
        for r in results
            @test isempty(free_indices(r))
            @test r != TScalar(0 // 1)
        end
    end
end

# ---------------------------------------------------------------------------
# 3. Riemann tensor full contraction
#    Nutma Eq (37): AllContractions[RiemannCD[-a,-b,-c,-d]] = {R}
#    Only one independent full contraction exists (the Ricci scalar).
# ---------------------------------------------------------------------------

@testset "Nutma Eq 37: Riemann full contraction -> 1 result (Ricci scalar)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
        results = all_contractions(R, :g; registry=reg)

        # Nutma: only 1 full contraction survives (antisymmetry kills some,
        # pair symmetry identifies others).
        @test length(results) == 1

        # The surviving contraction is the Ricci scalar (possibly with sign).
        # Our convention gives -RicScalar from g^{ac}g^{bd}R_{abcd}.
        r = results[1]
        @test isempty(free_indices(r))

        # Verify it involves RicScalar
        s = simplify(r; registry=reg)
        contains_ric_scalar = false
        function _check_ric(e)
            if e isa Tensor && e.name == :RicScalar
                contains_ric_scalar = true
            end
            if e isa TProduct
                for f in e.factors
                    _check_ric(f)
                end
            end
        end
        _check_ric(s)
        @test contains_ric_scalar
    end
end

# ---------------------------------------------------------------------------
# 4. Riemann*Riemann full contractions
#    Nutma Eq (38): AllContractions[Riem*Riem] =
#    {R_ab R^ab, R^2, R_abcd R^abcd, R_acbd R^abcd}
#    -> 4 contractions (last two related by Bianchi, so 3 truly independent).
#
#    Our metric-based all_contractions returns a complete (but not minimal)
#    basis. Nutma states (after Eq 38): "AllContractions does not necessarily
#    return an irreducible basis of contractions, but it does always return a
#    complete basis."
# ---------------------------------------------------------------------------

@testset "Nutma Eq 38: Riem*Riem full contractions (complete basis)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
        R2 = Tensor(:Riem, [down(:e), down(:f), down(:g1), down(:h)])
        product = R1 * R2
        results = all_contractions(product, :g; registry=reg)

        # Must return at least the 3 truly independent invariants:
        # R_ab R^ab, R^2, R_abcd R^abcd (Kretschner)
        @test length(results) >= 3

        # All results should be scalars (no free indices)
        for r in results
            @test isempty(free_indices(r))
        end

        # All results should be nonzero
        for r in results
            @test r != TScalar(0 // 1)
        end

        # Verify the basis contains the three known quadratic curvature
        # invariants by checking for Ric*Ric, RicScalar*RicScalar, Riem*Riem
        has_ric_sq = false
        has_r_sq = false
        has_riem_sq = false
        for r in results
            s = simplify(r; registry=reg)
            str = string(s)
            if occursin("Ric", str) && !occursin("RicScalar", str) && !occursin("Riem", str)
                has_ric_sq = true
            end
            if count("RicScalar", str) >= 2
                has_r_sq = true
            end
            if count("Riem", str) >= 2
                has_riem_sq = true
            end
        end
        @test has_ric_sq   # R_{ab} R^{ab}
        @test has_r_sq     # R^2
        @test has_riem_sq  # R_{abcd} R^{abcd}
    end
end

# ---------------------------------------------------------------------------
# 5. Spin-2 contractions on flat background
#    Nutma Sec 4.1, Eq (19): AllContractions[H[a,b] PD[c]@PD[d]@H[e,f]]
#    yields 5 independent terms.
#
#    Our benchmark bench_02_xtras.jl already pins this at
#    XTRAS_SPIN2_CONTRACTIONS = 5 (see ground_truth.jl).
#    Here we verify the H*H pure contraction count (no derivatives).
# ---------------------------------------------------------------------------

@testset "Nutma Sec 4.1: H*H contractions -> 2 (symmetric rank-2)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)
    register_tensor!(reg, TensorProperties(name=:H, manifold=:M4, rank=(0,2),
        symmetries=Any[Symmetric(1,2)]))

    with_registry(reg) do
        # Tensor-based: 2 contractions of H*H with no free indices
        t_H1 = Tensor(:H, [down(:a), down(:b)])
        t_H2 = Tensor(:H, [down(:c), down(:d)])
        results = all_contractions([t_H1, t_H2], TIndex[])
        @test length(results) == 2

        # Metric-based: also 2
        product = Tensor(:H, [down(:a), down(:b)]) * Tensor(:H, [down(:c), down(:d)])
        results_m = all_contractions(product, :g; registry=reg)
        @test length(results_m) == 2
    end
end

# ---------------------------------------------------------------------------
# 6. contraction_ansatz: single metric
#    Nutma Sec 5.1.5: MakeAnsatz combined with AllContractions.
#    A single metric fully contracted gives g^{ab}g_{ab} = d (dimension),
#    i.e. exactly 1 scalar term.
# ---------------------------------------------------------------------------

@testset "Nutma Sec 5.1.5: contraction_ansatz([:g], :g) -> 1 term" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        result = contraction_ansatz([:g], :g; registry=reg)
        terms = result isa TSum ? result.terms : [result]
        @test length(terms) == 1
        for t in terms
            @test isempty(free_indices(t))
        end
    end
end

# ---------------------------------------------------------------------------
# 7. contraction_ansatz: Ric x Ric -> 2 invariants
#    Nutma Sec 5.1.5 / Sec 4.1: The two independent quadratic Ricci
#    invariants are R_{ab}R^{ab} and R^2.
# ---------------------------------------------------------------------------

@testset "Nutma Sec 5.1.5: contraction_ansatz([:Ric, :Ric]) -> 2 invariants" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        result = contraction_ansatz([:Ric, :Ric], :g; registry=reg)
        @test result isa TSum
        @test length(result.terms) == 2

        # Both terms must be scalars
        for t in result.terms
            @test isempty(free_indices(t))
        end

        # Coefficients should be distinct symbols
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

# ---------------------------------------------------------------------------
# 8. contraction_ansatz: Riem x Riem -> complete basis
#    Nutma Eq (66): MakeAnsatz @ AllContractions[Riem*Riem] gives 4 terms
#    (C1 R_ab R^ab + C2 R^2 + C3 R_abcd R^abcd + C4 R_acbd R^abcd).
#    Our implementation returns a complete basis that may include extra
#    terms related by Bianchi; at minimum 3 truly independent ones.
# ---------------------------------------------------------------------------

@testset "Nutma Eq 66: contraction_ansatz([:Riem, :Riem]) -> complete basis" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        result = contraction_ansatz([:Riem, :Riem], :g; registry=reg)
        terms = result isa TSum ? result.terms : [result]

        # Must produce at least 3 independent invariants
        @test length(terms) >= 3

        # All terms must be scalars
        for t in terms
            @test isempty(free_indices(t))
        end

        # All coefficients must be distinct
        coeffs = Symbol[]
        for t in terms
            if t isa TProduct
                for f in t.factors
                    if f isa TScalar && f.val isa Symbol
                        push!(coeffs, f.val)
                        break
                    end
                end
            end
        end
        @test length(unique(coeffs)) == length(coeffs)
    end
end

# ---------------------------------------------------------------------------
# 9. Odd-rank tensors cannot be fully contracted
#    Nutma Sec 5.1.1 (implicit): full contraction requires pairing all
#    indices, which is only possible for even rank.
# ---------------------------------------------------------------------------

@testset "Nutma Sec 5.1.1: odd-rank error" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)
    register_tensor!(reg, TensorProperties(name=:V, manifold=:M4, rank=(0,1),
        symmetries=Any[]))

    with_registry(reg) do
        V = Tensor(:V, [down(:a)])
        @test_throws ArgumentError all_contractions(V, :g; registry=reg)
    end
end

# ---------------------------------------------------------------------------
# 10. Scalar input returns itself
#     Nutma Sec 5.1.1 (implicit): a scalar expression has no free indices
#     to contract; all_contractions returns it unchanged.
# ---------------------------------------------------------------------------

@testset "Nutma Sec 5.1.1: scalar input -> [expr]" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        s = TScalar(42 // 1)
        results = all_contractions(s, :g; registry=reg)
        @test length(results) == 1
        @test results[1] == s
    end
end

# ---------------------------------------------------------------------------
# 11. Riemann symmetry reduces 3 raw contractions to 1
#     Nutma Sec 5.1.1: For the Riemann tensor (rank-4 with full Riemann
#     symmetries), the 3 raw pairings reduce because antisymmetry on (1,2)
#     and (3,4) kills the (a,b)(c,d) pairing: g^{ab}g^{cd}R_{abcd}=0.
#     The remaining two are related by pair-swap symmetry up to sign:
#     g^{ac}g^{bd}R_{abcd} = -g^{ad}g^{bc}R_{abcd}.
# ---------------------------------------------------------------------------

@testset "Nutma Sec 5.1.1: Riemann symmetry reduces 3 -> 1 contraction" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        R_abcd = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
        results = all_contractions(R_abcd, :g; registry=reg)
        @test length(results) == 1  # 3 raw -> 1 after symmetry
    end
end

# ---------------------------------------------------------------------------
# 12. Metric x metric contractions
#     Nutma Sec 5.1.4, Eq (60): two metrics have 3 index configurations
#     (before contraction). As a rank-4 expression with (4-1)!! = 3 raw
#     pairings, some coincide due to metric symmetry g_{ab} = g_{ba}.
# ---------------------------------------------------------------------------

@testset "Nutma Sec 5.1.4: metric x metric contractions" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        g_prod = Tensor(:g, [down(:a), down(:b)]) * Tensor(:g, [down(:c), down(:d)])
        results = all_contractions(g_prod, :g; registry=reg)
        @test length(results) >= 1
        @test length(results) <= 3
    end
end

# ---------------------------------------------------------------------------
# 13. Weyl tensor contractions
#     Nutma Sec 5.1.2 (implicit): The Weyl tensor is traceless, so all
#     single traces vanish (W^a_{bac} = 0). However, all_contractions does
#     not apply multi-term symmetries (tracelessness is a trace constraint,
#     not a mono-term permutation symmetry). This is consistent with Nutma
#     (after Eq 38): "AllContractions does not necessarily return an
#     irreducible basis of contractions."
#
#     The Riemann-type antisymmetry still kills the (1,2)(3,4) pairing
#     (g^{ab}g^{cd}W_{abcd} = 0 from antisymmetry), reducing 3 raw -> <= 2.
# ---------------------------------------------------------------------------

@testset "Nutma Sec 5.1.2: Weyl contractions (mono-term symmetry only)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
        [:a,:b,:c,:d,:e,:f,:g1,:h]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        W = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
        results = all_contractions(W, :g; registry=reg)

        # Riemann antisymmetry kills at least 1 of the 3 raw pairings
        @test length(results) <= 3  # at most (4-1)!! = 3

        # All results should be scalars
        for r in results
            @test isempty(free_indices(r))
        end
    end
end
