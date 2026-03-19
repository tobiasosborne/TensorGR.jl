#= Brauer algebra 11-piece irreducible decomposition of the metric-affine
#  Riemann tensor R_{abcd}(Gamma) under GL(d,R).
#
#  In metric-affine gravity the connection is independent of the metric.
#  The curvature R^a_{bcd} has only antisymmetry in (c,d), NOT pair
#  symmetry or the first Bianchi identity.
#
#  After lowering: R_{abcd} = g_{ae} R^e_{bcd}, with R_{abcd} = -R_{abdc}.
#
#  The 96 independent components (d=4) decompose into 11 irreducible
#  representations of GL(4,R) via the Brauer algebra.
#
#  Step 1: Split (a,b) into symmetric W_{(ab)cd} and antisymmetric Z_{[ab]cd}.
#    - W: d(d+1)/2 * d(d-1)/2 = 60 (d=4)
#    - Z: d(d-1)/2 * d(d-1)/2 = 36 (d=4)
#
#  Step 2: Further decompose each sector by traces and pair-exchange symmetry.
#
#  The 11 pieces are (following McCrea 1992, Hehl et al 1995):
#
#  Symmetric sector W_{(ab)cd} (6 pieces, 60 components in d=4):
#    W^+ (Riemann-like, 20): WEYL_S(10) + RICCI_S(9) + SCALAR_S(1)
#    W^- (pair-antisymmetric, 40): PAIRSYM_S(9) + RICANTI_S(6) + PAIRSKEW_S(25)
#
#  Antisymmetric sector Z_{[ab]cd} (5 pieces, 36 components in d=4):
#    Z^+ (pair-symmetric, 21): WEYL_A(6) + RICCI_Z1(9) + RICCI_Z2(6)
#    Z^- (pair-antisymmetric, 15): PAIRSYM_A(6) + PAIRSKEW_A(9)
#
#  Ground truth: Helpin & Volkov, arXiv:2407.18019;
#                McCrea, Class. Quant. Grav. 9, 553 (1992);
#                Hehl, McCrea, Mielke & Ne'eman, Phys. Rep. 258 (1995) Table C.3.1.
=#

"""
    BrauerDecomposition

The 11-piece irreducible decomposition of the metric-affine Riemann tensor
R_{abcd}(Gamma) under GL(d,R).

# Fields
- `pieces::Dict{Symbol, TensorExpr}` -- maps piece names to projected tensor expressions
- `dimensions::Dict{Symbol, Int}` -- expected dimension of each irreducible piece
- `dim::Int` -- spacetime dimension d
- `metric::Symbol` -- metric used for traces and index lowering
"""
struct BrauerDecomposition
    pieces::Dict{Symbol, TensorExpr}
    dimensions::Dict{Symbol, Int}
    dim::Int
    metric::Symbol
end

function Base.show(io::IO, bd::BrauerDecomposition)
    total = sum(values(bd.dimensions))
    print(io, "BrauerDecomposition(d=$(bd.dim), $(length(bd.dimensions)) pieces, $(total) components)")
end

# ──────────────────────────────────────────────────────────────────────
#  Piece names and dimensions
# ──────────────────────────────────────────────────────────────────────

"""
    brauer_piece_names() -> Vector{Symbol}

Return the canonical names of the 11 irreducible pieces of the metric-affine
Riemann tensor decomposition.

The first 6 are from the symmetric sector W_{(ab)cd}, the last 5 from the
antisymmetric sector Z_{[ab]cd}.

Naming follows McCrea (1992) / Hehl et al (1995).
"""
function brauer_piece_names()
    Symbol[
        # Symmetric sector W_{(ab)cd} — 6 pieces
        :WEYL_S,       # Weyl-like (totally tracefree, pair-symmetric, window Young diagram)
        :PAIRSYM_S,    # Pair-symmetric tracefree hook
        :RICCI_S,      # Symmetric tracefree Ricci (from pair-symmetric trace)
        :SCALAR_S,     # Scalar curvature
        :PAIRSKEW_S,   # Pair-antisymmetric tracefree part
        :RICANTI_S,    # Antisymmetric Ricci (from pair-antisymmetric trace)
        # Antisymmetric sector Z_{[ab]cd} — 5 pieces
        :WEYL_A,       # Antisymmetric Weyl-like (tracefree, cd<->ab symmetric)
        :PAIRSYM_A,    # Pair-symmetric tracefree part from Z
        :RICCI_Z1,     # First antisymmetric Ricci piece (symmetric)
        :RICCI_Z2,     # Second antisymmetric Ricci piece (antisymmetric)
        :PAIRSKEW_A,   # Pair-antisymmetric from antisymmetric sector
    ]
end

"""
    brauer_piece_dimensions(d::Int=4) -> Dict{Symbol, Int}

Return the dimension (number of independent components) of each of the 11
irreducible pieces in spacetime dimension `d`.

The dimensions are derived from GL(d,R) representation theory. The total
must equal d^2 * d(d-1)/2 = d^3(d-1)/2 (= 96 for d=4).

Ground truth: McCrea (1992) Table 1; Hehl et al (1995) Table C.3.1.
"""
function brauer_piece_dimensions(d::Int=4)
    d >= 2 || error("Dimension must be >= 2, got $d")

    # Total components: R_{abcd} with antisymmetry in cd
    # = d*d * d(d-1)/2 = d^3(d-1)/2
    total = d^3 * (d - 1) ÷ 2

    # Symmetric sector W_{(ab)cd}: d(d+1)/2 * d(d-1)/2 components
    sym_total = (d * (d + 1) ÷ 2) * (d * (d - 1) ÷ 2)

    # Antisymmetric sector Z_{[ab]cd}: d(d-1)/2 * d(d-1)/2 components
    anti_total = (d * (d - 1) ÷ 2)^2

    @assert sym_total + anti_total == total

    n2 = d * (d - 1) ÷ 2  # dim of Lambda^2(V)

    # ═══════════════════════════════════════════════════════════════════
    # W^+ (pair-symmetric part of symmetric sector = Riemann symmetries)
    # ═══════════════════════════════════════════════════════════════════
    # dim = d^2(d^2-1)/12 (standard Riemann component count)
    dim_Wplus = d^2 * (d^2 - 1) ÷ 12

    # Weyl: (d+2)(d+1)d(d-3)/12 for d >= 3, vanishes for d <= 2
    dim_WEYL_S = d >= 3 ? (d + 2) * (d + 1) * d * (d - 3) ÷ 12 : 0

    # Scalar: always 1 (assuming d >= 2 with non-degenerate metric)
    dim_SCALAR_S = 1

    # Tracefree symmetric Ricci: by subtraction from Riemann count
    # For d >= 3: d(d+1)/2 - 1. For d=2: 0.
    dim_RICCI_S = dim_Wplus - dim_WEYL_S - dim_SCALAR_S

    # ═══════════════════════════════════════════════════════════════════
    # W^- (pair-antisymmetric part of symmetric sector)
    # ═══════════════════════════════════════════════════════════════════
    dim_Wminus = sym_total - dim_Wplus

    # The trace g^{ac}W^-_{(ab)cd} gives a rank-2 tensor in (b,d).
    # Double trace vanishes by pair-antisymmetry.
    # Trace components: symmetric tracefree (PAIRSYM_S) + antisymmetric (RICANTI_S).
    # The trace map has rank min(d^2 - 1, dim_Wminus).
    trace_Wm = min(d^2 - 1, dim_Wminus)

    # Split trace into symmetric tracefree and antisymmetric parts.
    # Antisymmetric part: min(d(d-1)/2, trace_Wm)
    dim_RICANTI_S = min(d * (d - 1) ÷ 2, trace_Wm)

    # Symmetric tracefree part: remainder of trace
    dim_PAIRSYM_S = trace_Wm - dim_RICANTI_S

    # Tracefree part: whatever is left
    dim_PAIRSKEW_S = dim_Wminus - trace_Wm

    # ═══════════════════════════════════════════════════════════════════
    # Z^+ (pair-symmetric part of antisymmetric sector)
    # ═══════════════════════════════════════════════════════════════════
    # Symmetric matrix on Lambda^2: n2(n2+1)/2
    dim_Zplus = n2 * (n2 + 1) ÷ 2

    # Trace gives rank-2 with vanishing double trace: d^2 - 1 components max
    trace_Zp = min(d^2 - 1, dim_Zplus)

    # Split: antisymmetric (RICCI_Z2) + symmetric tracefree (RICCI_Z1)
    dim_RICCI_Z2 = min(d * (d - 1) ÷ 2, trace_Zp)
    dim_RICCI_Z1 = trace_Zp - dim_RICCI_Z2

    # Tracefree remainder (WEYL_A)
    dim_WEYL_A = dim_Zplus - trace_Zp

    # ═══════════════════════════════════════════════════════════════════
    # Z^- (pair-antisymmetric part of antisymmetric sector)
    # ═══════════════════════════════════════════════════════════════════
    # Antisymmetric matrix on Lambda^2: n2(n2-1)/2
    dim_Zminus = n2 * (n2 - 1) ÷ 2

    # Trace gives antisymmetric rank-2: d(d-1)/2 components max
    trace_Zm = min(d * (d - 1) ÷ 2, dim_Zminus)

    dim_PAIRSYM_A = trace_Zm
    dim_PAIRSKEW_A = dim_Zminus - trace_Zm

    # ═══════════════════════════════════════════════════════════════════
    # Verify total
    # ═══════════════════════════════════════════════════════════════════
    total_check = dim_WEYL_S + dim_RICCI_S + dim_SCALAR_S +
                  dim_PAIRSYM_S + dim_RICANTI_S + dim_PAIRSKEW_S +
                  dim_WEYL_A + dim_RICCI_Z1 + dim_RICCI_Z2 +
                  dim_PAIRSKEW_A + dim_PAIRSYM_A

    @assert total_check == total "Dimension sum $total_check != $total for d=$d"

    Dict{Symbol, Int}(
        :WEYL_S     => dim_WEYL_S,
        :RICCI_S    => dim_RICCI_S,
        :SCALAR_S   => dim_SCALAR_S,
        :PAIRSYM_S  => dim_PAIRSYM_S,
        :RICANTI_S  => dim_RICANTI_S,
        :PAIRSKEW_S => dim_PAIRSKEW_S,
        :WEYL_A     => dim_WEYL_A,
        :RICCI_Z1   => dim_RICCI_Z1,
        :RICCI_Z2   => dim_RICCI_Z2,
        :PAIRSKEW_A => dim_PAIRSKEW_A,
        :PAIRSYM_A  => dim_PAIRSYM_A,
    )
end

# ──────────────────────────────────────────────────────────────────────
#  Symmetric / Antisymmetric split
# ──────────────────────────────────────────────────────────────────────

"""
    brauer_symmetric_split(R_expr::TensorExpr) -> (W, Z)

Split R_{abcd} into symmetric and antisymmetric parts in (a,b):

    W_{abcd} = R_{(ab)cd} = (1/2)(R_{abcd} + R_{bacd})
    Z_{abcd} = R_{[ab]cd} = (1/2)(R_{abcd} - R_{bacd})

Returns a tuple (W, Z) of TensorExpr.

The expression must have exactly 4 free indices, with the first two
corresponding to the (a,b) pair.
"""
function brauer_symmetric_split(R_expr::TensorExpr)
    fi = free_indices(R_expr)
    length(fi) == 4 || error("Expected 4 free indices, got $(length(fi))")

    a_name = fi[1].name
    b_name = fi[2].name

    W = symmetrize(R_expr, [a_name, b_name])
    Z = antisymmetrize(R_expr, [a_name, b_name])

    (W, Z)
end

# ──────────────────────────────────────────────────────────────────────
#  Pair exchange operator
# ──────────────────────────────────────────────────────────────────────

"""
    _pair_exchange(expr::TensorExpr, a::Symbol, b::Symbol, c::Symbol, d::Symbol)

Apply the pair exchange (ab) <-> (cd) to a 4-index tensor expression.
Returns the expression with a<->c and b<->d simultaneously.

This is used to decompose into pair-symmetric and pair-antisymmetric parts.
"""
function _pair_exchange(expr::TensorExpr, a::Symbol, b::Symbol, c::Symbol, d::Symbol)
    # Swap a<->c and b<->d using fresh temporaries
    fi = free_indices(expr)
    used = Set(idx.name for idx in fi)
    t1 = fresh_index(used); push!(used, t1)
    t2 = fresh_index(used)

    # a -> t1, c -> a, t1 -> c (swap a,c)
    e1 = rename_dummy(expr, a, t1)
    e2 = rename_dummy(e1, c, a)
    e3 = rename_dummy(e2, t1, c)

    # b -> t2, d -> b, t2 -> d (swap b,d)
    e4 = rename_dummy(e3, b, t2)
    e5 = rename_dummy(e4, d, b)
    e6 = rename_dummy(e5, t2, d)

    e6
end

"""
    _pair_symmetric_part(expr, a, b, c, d) -> TensorExpr

Compute the pair-symmetric part: (1/2)(T_{abcd} + T_{cdab}).
"""
function _pair_symmetric_part(expr::TensorExpr, a::Symbol, b::Symbol,
                               c::Symbol, d::Symbol)
    exchanged = _pair_exchange(expr, a, b, c, d)
    tsum(TensorExpr[
        tproduct(1 // 2, TensorExpr[expr]),
        tproduct(1 // 2, TensorExpr[exchanged])
    ])
end

"""
    _pair_antisymmetric_part(expr, a, b, c, d) -> TensorExpr

Compute the pair-antisymmetric part: (1/2)(T_{abcd} - T_{cdab}).
"""
function _pair_antisymmetric_part(expr::TensorExpr, a::Symbol, b::Symbol,
                                   c::Symbol, d::Symbol)
    exchanged = _pair_exchange(expr, a, b, c, d)
    tsum(TensorExpr[
        tproduct(1 // 2, TensorExpr[expr]),
        tproduct(-1 // 2, TensorExpr[exchanged])
    ])
end

# ──────────────────────────────────────────────────────────────────────
#  Trace extraction
# ──────────────────────────────────────────────────────────────────────

"""
    _ma_ricci_trace(expr::TensorExpr, a::Symbol, c::Symbol;
                    metric::Symbol=:g) -> TensorExpr

Compute the Ricci-type trace g^{ac} T_{abcd}, contracting indices a and c.
Result is a rank-2 tensor in (b,d).
"""
function _ma_ricci_trace(expr::TensorExpr, a::Symbol, c::Symbol;
                          metric::Symbol=:g)
    fi = free_indices(expr)
    used = Set(idx.name for idx in fi)

    # Determine positions of a and c
    a_pos = nothing
    c_pos = nothing
    for idx in fi
        if idx.name == a
            a_pos = idx.position
        elseif idx.name == c
            c_pos = idx.position
        end
    end

    a_pos === nothing && error("Index $a not found")
    c_pos === nothing && error("Index $c not found")

    # Insert metric with appropriate index positions to contract a and c
    if a_pos == Down && c_pos == Down
        g = Tensor(metric, [up(a), up(c)])
        return tproduct(1 // 1, TensorExpr[g, expr])
    elseif a_pos == Up && c_pos == Up
        g = Tensor(metric, [down(a), down(c)])
        return tproduct(1 // 1, TensorExpr[g, expr])
    else
        # Mixed: just rename to contract
        return rename_dummy(expr, c, a)
    end
end

"""
    _ma_scalar_trace(expr::TensorExpr, a::Symbol, b::Symbol,
                     c::Symbol, d::Symbol; metric::Symbol=:g) -> TensorExpr

Compute the double trace g^{ac}g^{bd} T_{abcd}. Result is a scalar.
"""
function _ma_scalar_trace(expr::TensorExpr, a::Symbol, b::Symbol,
                           c::Symbol, d::Symbol; metric::Symbol=:g)
    # First trace on (a,c)
    r2 = _ma_ricci_trace(expr, a, c; metric=metric)
    # Now r2 is a tensor in (b,d); trace on (b,d)
    fi = free_indices(r2)
    b_pos = nothing
    d_pos = nothing
    for idx in fi
        if idx.name == b
            b_pos = idx.position
        elseif idx.name == d
            d_pos = idx.position
        end
    end

    b_pos === nothing && error("Index $b not found after first trace")
    d_pos === nothing && error("Index $d not found after first trace")

    if b_pos == Down && d_pos == Down
        g = Tensor(metric, [up(b), up(d)])
        return tproduct(1 // 1, TensorExpr[g, r2])
    elseif b_pos == Up && d_pos == Up
        g = Tensor(metric, [down(b), down(d)])
        return tproduct(1 // 1, TensorExpr[g, r2])
    else
        return rename_dummy(r2, d, b)
    end
end

# ──────────────────────────────────────────────────────────────────────
#  Full Brauer decomposition
# ──────────────────────────────────────────────────────────────────────

"""
    brauer_decompose(R_expr::TensorExpr;
                     metric::Symbol=:g, dim::Int=4,
                     registry::TensorRegistry=current_registry())
        -> BrauerDecomposition

Decompose a metric-affine curvature tensor R_{abcd} (all indices down,
antisymmetric in cd only) into the 11 irreducible GL(d,R) pieces.

The decomposition proceeds as:

1. Split into symmetric W_{(ab)cd} and antisymmetric Z_{[ab]cd} in (a,b).

2. Split each by pair exchange symmetry (ab<->cd).

3. Extract traces and tracefree parts from each sub-sector:
   - W^+ (pair-symmetric part of W): Weyl + Ricci + scalar (3 pieces)
   - W^- (pair-antisymmetric part of W): tracefree + sym Ricci + antisym Ricci (3 pieces)
   - Z^+ (pair-symmetric part of Z): tracefree + sym Ricci + antisym Ricci (3 pieces)
   - Z^- (pair-antisymmetric part of Z): tracefree + antisym Ricci (2 pieces)

Returns a `BrauerDecomposition` containing the 11 pieces as tensor expressions.

Ground truth: McCrea (1992); Hehl et al (1995) App. C; Helpin & Volkov (2407.18019).
"""
function brauer_decompose(R_expr::TensorExpr;
                           metric::Symbol=:g, dim::Int=4,
                           registry::TensorRegistry=current_registry())
    fi = free_indices(R_expr)
    length(fi) == 4 || error("Expected 4 free indices for R_{abcd}, got $(length(fi))")

    a = fi[1].name
    b = fi[2].name
    c = fi[3].name
    d = fi[4].name

    dims = brauer_piece_dimensions(dim)
    pieces = Dict{Symbol, TensorExpr}()

    # ── Step 1: (a,b) symmetry split ──
    W, Z = brauer_symmetric_split(R_expr)

    # ── Step 2: Pair exchange split ──
    Wplus = _pair_symmetric_part(W, a, b, c, d)
    Wminus = _pair_antisymmetric_part(W, a, b, c, d)
    Zplus = _pair_symmetric_part(Z, a, b, c, d)
    Zminus = _pair_antisymmetric_part(Z, a, b, c, d)

    # ── Step 3a: W^+ decomposition (Riemann-like: Weyl + Ricci + scalar) ──
    # W^+ has full Riemann symmetries. Standard Weyl decomposition:
    # W^+_{abcd} = WEYL_S + RICCI_S + SCALAR_S
    scalar_val = _ma_scalar_trace(Wplus, a, b, c, d; metric=metric)

    g_ac = Tensor(metric, [down(a), down(c)])
    g_bd = Tensor(metric, [down(b), down(d)])
    g_ad = Tensor(metric, [down(a), down(d)])
    g_bc = Tensor(metric, [down(b), down(c)])

    scalar_coeff = 1 // ((dim - 1) * (dim - 2))

    # (g_{ac}g_{bd} - g_{ad}g_{bc})
    gg_plus = tproduct(1 // 1, TensorExpr[g_ac, g_bd])
    gg_minus = tproduct(1 // 1, TensorExpr[g_ad, g_bc])
    gg_diff = tsum(TensorExpr[gg_plus, tproduct(-1 // 1, TensorExpr[gg_minus])])

    pieces[:SCALAR_S] = tproduct(scalar_coeff, TensorExpr[scalar_val, gg_diff])

    # Ricci trace of W^+: Ric^{(+)}_{bd} = g^{ac} W^+_{abcd}
    ric_plus = _ma_ricci_trace(Wplus, a, c; metric=metric)

    # Tracefree Ricci: Ric^TF_{bd} = Ric_{bd} - (1/d) g_{bd} R
    ric_plus_tf = tsum(TensorExpr[
        ric_plus,
        tproduct(-1 // dim, TensorExpr[scalar_val, g_bd])
    ])

    # Build RICCI_S from tracefree Ricci using the standard Weyl decomposition:
    # RICCI_S = (1/(d-2)) * (g_{ac}Ric^TF_{bd} + g_{bd}Ric^TF_{ac}
    #           - g_{ad}Ric^TF_{bc} - g_{bc}Ric^TF_{ad})
    fi_used = Set(idx.name for idx in fi)
    e = fresh_index(fi_used); push!(fi_used, e)
    f1 = fresh_index(fi_used); push!(fi_used, f1)
    f2 = fresh_index(fi_used); push!(fi_used, f2)

    ricci_coeff = 1 // (dim - 2)

    # Rename tracefree Ricci to various index pairs via temporaries
    ric_tf_ac = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_plus_tf, b, f1), d, f2), f1, a), f2, c)
    ric_tf_bc = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_plus_tf, b, f1), d, f2), f1, b), f2, c)
    ric_tf_ad = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_plus_tf, b, f1), d, f2), f1, a), f2, d)
    ric_tf_bd = ric_plus_tf

    pieces[:RICCI_S] = tsum(TensorExpr[
        tproduct(ricci_coeff, TensorExpr[g_ac, ric_tf_bd]),
        tproduct(ricci_coeff, TensorExpr[g_bd, ric_tf_ac]),
        tproduct(-ricci_coeff, TensorExpr[g_ad, ric_tf_bc]),
        tproduct(-ricci_coeff, TensorExpr[g_bc, ric_tf_ad]),
    ])

    # WEYL_S = W^+ - RICCI_S - SCALAR_S
    pieces[:WEYL_S] = tsum(TensorExpr[
        Wplus,
        tproduct(-1 // 1, TensorExpr[pieces[:RICCI_S]]),
        tproduct(-1 // 1, TensorExpr[pieces[:SCALAR_S]]),
    ])

    # ── Step 3b: W^- decomposition ──
    #
    # W^- is the pair-antisymmetric part of W_{(ab)cd}.
    # Its trace g^{ac}W^-_{abcd} gives a rank-2 tensor whose:
    #   - symmetric tracefree part → PAIRSYM_S
    #   - antisymmetric part → RICANTI_S
    #
    # The tracefree remainder → PAIRSKEW_S

    ric_minus = _ma_ricci_trace(Wminus, a, c; metric=metric)
    # ric_minus has free indices (b, d)

    # Symmetric part of ric_minus:
    ric_minus_sym = symmetrize(ric_minus, [b, d])
    # Since double trace of W^- vanishes (shown above), ric_minus_sym is already tracefree.
    # But let's not assume — make it explicitly tracefree:
    scalar_minus = _ma_scalar_trace(Wminus, a, b, c, d; metric=metric)
    ric_minus_sym_tf = tsum(TensorExpr[
        ric_minus_sym,
        tproduct(-1 // dim, TensorExpr[scalar_minus, g_bd])
    ])

    # Antisymmetric part of ric_minus:
    ric_minus_anti = antisymmetrize(ric_minus, [b, d])

    # Build the PAIRSYM_S piece from the symmetric tracefree Ricci of W^-:
    # Analogous to the Ricci decomposition but for the pair-antisymmetric sector.
    # The reconstruction formula for a tensor T_{abcd} symmetric in (ab),
    # antisymmetric in (cd), pair-antisymmetric, from its trace Ric_{bd}:
    # Uses the same Ricci-to-curvature formula adapted for the symmetry.
    #
    # For the symmetric tracefree Ricci part:
    ric_ms_tf_ac = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_minus_sym_tf, b, f1), d, f2), f1, a), f2, c)
    ric_ms_tf_bc = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_minus_sym_tf, b, f1), d, f2), f1, b), f2, c)
    ric_ms_tf_ad = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_minus_sym_tf, b, f1), d, f2), f1, a), f2, d)
    ric_ms_tf_bd = ric_minus_sym_tf

    pieces[:PAIRSYM_S] = tsum(TensorExpr[
        tproduct(ricci_coeff, TensorExpr[g_ac, ric_ms_tf_bd]),
        tproduct(ricci_coeff, TensorExpr[g_bd, ric_ms_tf_ac]),
        tproduct(-ricci_coeff, TensorExpr[g_ad, ric_ms_tf_bc]),
        tproduct(-ricci_coeff, TensorExpr[g_bc, ric_ms_tf_ad]),
    ])

    # For the antisymmetric Ricci part (RICANTI_S):
    ric_ma_ac = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_minus_anti, b, f1), d, f2), f1, a), f2, c)
    ric_ma_bc = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_minus_anti, b, f1), d, f2), f1, b), f2, c)
    ric_ma_ad = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_minus_anti, b, f1), d, f2), f1, a), f2, d)
    ric_ma_bd = ric_minus_anti

    pieces[:RICANTI_S] = tsum(TensorExpr[
        tproduct(ricci_coeff, TensorExpr[g_ac, ric_ma_bd]),
        tproduct(ricci_coeff, TensorExpr[g_bd, ric_ma_ac]),
        tproduct(-ricci_coeff, TensorExpr[g_ad, ric_ma_bc]),
        tproduct(-ricci_coeff, TensorExpr[g_bc, ric_ma_ad]),
    ])

    # PAIRSKEW_S = W^- - PAIRSYM_S - RICANTI_S
    pieces[:PAIRSKEW_S] = tsum(TensorExpr[
        Wminus,
        tproduct(-1 // 1, TensorExpr[pieces[:PAIRSYM_S]]),
        tproduct(-1 // 1, TensorExpr[pieces[:RICANTI_S]]),
    ])

    # ── Step 3c: Z^+ decomposition ──
    #
    # Z^+ is pair-symmetric, antisymmetric in both (ab) and (cd).
    # Trace g^{ac}Z^+_{abcd}: general rank-2 in (b,d), double trace = 0.
    # Split into:
    #   RICCI_Z1: symmetric tracefree part (dim = d(d+1)/2 - 1)
    #   RICCI_Z2: antisymmetric part (dim = d(d-1)/2)
    #   WEYL_A: tracefree remainder

    ric_zplus = _ma_ricci_trace(Zplus, a, c; metric=metric)

    ric_zplus_sym = symmetrize(ric_zplus, [b, d])
    ric_zplus_anti = antisymmetrize(ric_zplus, [b, d])

    # Make symmetric part explicitly tracefree (double trace should vanish)
    scalar_zplus = _ma_scalar_trace(Zplus, a, b, c, d; metric=metric)
    ric_zplus_sym_tf = tsum(TensorExpr[
        ric_zplus_sym,
        tproduct(-1 // dim, TensorExpr[scalar_zplus, g_bd])
    ])

    # Build RICCI_Z1 (symmetric tracefree Ricci from Z^+):
    rz1_ac = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_zplus_sym_tf, b, f1), d, f2), f1, a), f2, c)
    rz1_bc = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_zplus_sym_tf, b, f1), d, f2), f1, b), f2, c)
    rz1_ad = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_zplus_sym_tf, b, f1), d, f2), f1, a), f2, d)
    rz1_bd = ric_zplus_sym_tf

    pieces[:RICCI_Z1] = tsum(TensorExpr[
        tproduct(ricci_coeff, TensorExpr[g_ac, rz1_bd]),
        tproduct(ricci_coeff, TensorExpr[g_bd, rz1_ac]),
        tproduct(-ricci_coeff, TensorExpr[g_ad, rz1_bc]),
        tproduct(-ricci_coeff, TensorExpr[g_bc, rz1_ad]),
    ])

    # Build RICCI_Z2 (antisymmetric Ricci from Z^+):
    rz2_ac = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_zplus_anti, b, f1), d, f2), f1, a), f2, c)
    rz2_bc = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_zplus_anti, b, f1), d, f2), f1, b), f2, c)
    rz2_ad = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_zplus_anti, b, f1), d, f2), f1, a), f2, d)
    rz2_bd = ric_zplus_anti

    pieces[:RICCI_Z2] = tsum(TensorExpr[
        tproduct(ricci_coeff, TensorExpr[g_ac, rz2_bd]),
        tproduct(ricci_coeff, TensorExpr[g_bd, rz2_ac]),
        tproduct(-ricci_coeff, TensorExpr[g_ad, rz2_bc]),
        tproduct(-ricci_coeff, TensorExpr[g_bc, rz2_ad]),
    ])

    # WEYL_A = Z^+ - RICCI_Z1 - RICCI_Z2
    pieces[:WEYL_A] = tsum(TensorExpr[
        Zplus,
        tproduct(-1 // 1, TensorExpr[pieces[:RICCI_Z1]]),
        tproduct(-1 // 1, TensorExpr[pieces[:RICCI_Z2]]),
    ])

    # ── Step 3d: Z^- decomposition ──
    #
    # Z^- is pair-antisymmetric, antisymmetric in both (ab) and (cd).
    # Trace: g^{ac}Z^-_{abcd} gives a rank-2 tensor, decomposed into
    #   antisymmetric trace → PAIRSYM_A
    #   The tracefree remainder → PAIRSKEW_A

    ric_zminus = _ma_ricci_trace(Zminus, a, c; metric=metric)

    # For Z^-, the trace is purely antisymmetric in (b,d) due to the
    # combination of antisymmetries and pair-antisymmetry.
    # (The symmetric part of the trace vanishes for Z^-.)
    # We extract both parts for safety:
    ric_zminus_anti = antisymmetrize(ric_zminus, [b, d])

    # Build PAIRSYM_A (antisymmetric Ricci from Z^-):
    rza_ac = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_zminus_anti, b, f1), d, f2), f1, a), f2, c)
    rza_bc = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_zminus_anti, b, f1), d, f2), f1, b), f2, c)
    rza_ad = rename_dummy(rename_dummy(rename_dummy(rename_dummy(
        ric_zminus_anti, b, f1), d, f2), f1, a), f2, d)
    rza_bd = ric_zminus_anti

    pieces[:PAIRSYM_A] = tsum(TensorExpr[
        tproduct(ricci_coeff, TensorExpr[g_ac, rza_bd]),
        tproduct(ricci_coeff, TensorExpr[g_bd, rza_ac]),
        tproduct(-ricci_coeff, TensorExpr[g_ad, rza_bc]),
        tproduct(-ricci_coeff, TensorExpr[g_bc, rza_ad]),
    ])

    # PAIRSKEW_A = Z^- - PAIRSYM_A
    pieces[:PAIRSKEW_A] = tsum(TensorExpr[
        Zminus,
        tproduct(-1 // 1, TensorExpr[pieces[:PAIRSYM_A]]),
    ])

    BrauerDecomposition(pieces, dims, dim, metric)
end
