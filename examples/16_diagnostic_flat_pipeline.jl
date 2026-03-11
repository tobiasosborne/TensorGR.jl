#= Diagnostic: flat-space perturbation engine vs FP kernel
#
# Root cause: to_fourier replaces ∂→k uniformly, but in a bilinear action
# the two fields carry momenta k and -k. For derivatives on the LEFT h,
# ∂ → ik. For derivatives on the RIGHT h, ∂ → -ik = i(-k).
#
# For N total derivatives with n_L on left and n_R on right:
#   true sign = i^N × (-1)^{n_R}
#   our convention sign = 1  (since we just replace ∂→k)
#   correction = i^N × (-1)^{n_R}
#
# For N=2 (EH action):
#   1+1: correction = (-1)×(-1) = +1  (correct as-is)
#   2+0: correction = (-1)×(+1) = -1  (need to flip sign)
#   0+2: correction = (-1)×(+1) = -1  (need to flip sign)
#
# Fix: symmetrize to 1+1 via IBP before Fourier. Each symmetrizing IBP
# picks up a -1 sign, which exactly compensates the convention error.
=#

using TensorGR

println("=" ^ 70)
println("  Diagnostic: Flat Pipeline — Fourier sign correction")
println("=" ^ 70)

# ═══════════════════════════════════════════════════════════════════════
# Derivative depth counting and symmetrization
# ═══════════════════════════════════════════════════════════════════════

"""Count derivative depth (TDeriv layers) on expr."""
function _deriv_depth(expr::TDeriv)
    1 + _deriv_depth(expr.arg)
end
_deriv_depth(::TensorExpr) = 0

"""Get the innermost expression (strip all TDeriv layers)."""
function _innermost(expr::TDeriv)
    _innermost(expr.arg)
end
_innermost(expr::TensorExpr) = expr

"""Check if expr is or contains (at the bottom of TDeriv chain) a tensor named `name`."""
function _is_field(expr::TensorExpr, name::Symbol)
    inner = _innermost(expr)
    inner isa Tensor && inner.name == name
end

"""Symmetrize derivative distribution in a bilinear-in-field expression.

For each product term with two h factors, if one h has more derivatives
than the other, peel the outermost derivative from the more-derivatived h
and wrap it around the less-derivatived h. This changes the derivative
distribution (e.g., 2+0 → 1+1) and introduces a sign flip from IBP.

Only does ONE IBP step per term (sufficient for N=2 total derivatives).
For N=4, call repeatedly until symmetric.
"""
function symmetrize_bilinear_derivs(expr::TSum, field::Symbol)
    tsum(TensorExpr[symmetrize_bilinear_derivs(t, field) for t in expr.terms])
end

function symmetrize_bilinear_derivs(expr::TProduct, field::Symbol)
    factors = collect(expr.factors)
    # Find the two field factors (possibly wrapped in TDeriv)
    h_pos = Int[]
    for (i, f) in enumerate(factors)
        if _is_field(f, field)
            push!(h_pos, i)
        end
    end
    length(h_pos) != 2 && return expr

    i1, i2 = h_pos
    f1, f2 = factors[i1], factors[i2]
    n1 = _deriv_depth(f1)
    n2 = _deriv_depth(f2)

    # Already symmetric
    n1 == n2 && return expr

    # Identify which has more derivatives
    if n1 > n2
        # Peel outermost derivative from f1, wrap it around f2
        # f1 = TDeriv(∂_a, inner), replace with inner, and f2 → TDeriv(∂_a, f2)
        @assert f1 isa TDeriv
        new_factors = copy(factors)
        new_factors[i1] = f1.arg          # strip outer derivative
        new_factors[i2] = TDeriv(f1.index, f2, f1.covd)  # add it to other h
        return tproduct(-expr.scalar, new_factors)  # IBP sign
    else
        # Peel outermost derivative from f2, wrap it around f1
        @assert f2 isa TDeriv
        new_factors = copy(factors)
        new_factors[i2] = f2.arg
        new_factors[i1] = TDeriv(f2.index, f1, f2.covd)
        return tproduct(-expr.scalar, new_factors)
    end
end

symmetrize_bilinear_derivs(expr::TensorExpr, ::Symbol) = expr

# ═══════════════════════════════════════════════════════════════════════
# Main diagnostic
# ═══════════════════════════════════════════════════════════════════════

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)

    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Riem)
    set_vanishing!(reg, :Ric)
    set_vanishing!(reg, :RicScalar)

    k2 = 1.7

    # ─── Part A: δ²R with NAIVE Fourier (∂→k, no sign fix) ───
    println("\n── Part A: δ²R with naive Fourier ──")
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)
    n2 = δ2R isa TSum ? length(δ2R.terms) : 1
    println("  δ²R: $n2 terms")

    # Show derivative distribution
    if δ2R isa TSum
        for (i, t) in enumerate(δ2R.terms)
            if t isa TProduct
                h_depths = Int[]
                for f in t.factors
                    if _is_field(f, :h)
                        push!(h_depths, _deriv_depth(f))
                    end
                end
                if length(h_depths) == 2
                    println("    term $i: deriv distribution $(h_depths[1])+$(h_depths[2])")
                end
            end
        end
    end

    δ2R_f = to_fourier(δ2R)
    δ2R_f = simplify(δ2R_f; registry=reg)
    δ2R_f = fix_dummy_positions(δ2R_f)
    K_A = extract_kernel(δ2R_f, :h; registry=reg)

    s2_A  = spin_project(K_A, :spin2;  registry=reg)
    s0s_A = spin_project(K_A, :spin0s; registry=reg)
    s1_A  = spin_project(K_A, :spin1;  registry=reg)
    s0w_A = spin_project(K_A, :spin0w; registry=reg)

    v2_A  = _eval_spin_scalar(s2_A,  k2)
    v1_A  = _eval_spin_scalar(s1_A,  k2)
    v0s_A = _eval_spin_scalar(s0s_A, k2)
    v0w_A = _eval_spin_scalar(s0w_A, k2)

    println("  Naive Fourier: spin-2=$(round(v2_A;digits=4)), spin-1=$(round(v1_A;digits=4)), spin-0s=$(round(v0s_A;digits=4)), spin-0w=$(round(v0w_A;digits=4))")
    println("  Expected (FP): spin-2=$(round(2.5k2;digits=4)), spin-1=0, spin-0s=$(round(-k2;digits=4)), spin-0w=0")

    # ─── Part B: Symmetrized derivatives + Fourier ───
    println("\n── Part B: Symmetrize derivatives (1 round), then Fourier ──")
    δ2R_sym = symmetrize_bilinear_derivs(δ2R, :h)
    δ2R_sym = simplify(δ2R_sym; registry=reg)
    n_sym = δ2R_sym isa TSum ? length(δ2R_sym.terms) : 1
    println("  δ²R symmetrized: $n_sym terms")

    # Show derivative distribution after symmetrization
    if δ2R_sym isa TSum
        for (i, t) in enumerate(δ2R_sym.terms)
            if t isa TProduct
                h_depths = Int[]
                for f in t.factors
                    if _is_field(f, :h)
                        push!(h_depths, _deriv_depth(f))
                    end
                end
                if length(h_depths) == 2
                    println("    term $i: deriv distribution $(h_depths[1])+$(h_depths[2])")
                end
            end
        end
    end

    δ2R_sym_f = to_fourier(δ2R_sym)
    δ2R_sym_f = simplify(δ2R_sym_f; registry=reg)
    δ2R_sym_f = fix_dummy_positions(δ2R_sym_f)
    K_B = extract_kernel(δ2R_sym_f, :h; registry=reg)

    s2_B  = spin_project(K_B, :spin2;  registry=reg)
    s0s_B = spin_project(K_B, :spin0s; registry=reg)
    s1_B  = spin_project(K_B, :spin1;  registry=reg)
    s0w_B = spin_project(K_B, :spin0w; registry=reg)

    v2_B  = _eval_spin_scalar(s2_B,  k2)
    v1_B  = _eval_spin_scalar(s1_B,  k2)
    v0s_B = _eval_spin_scalar(s0s_B, k2)
    v0w_B = _eval_spin_scalar(s0w_B, k2)

    println("  Symmetrized:   spin-2=$(round(v2_B;digits=4)), spin-1=$(round(v1_B;digits=4)), spin-0s=$(round(v0s_B;digits=4)), spin-0w=$(round(v0w_B;digits=4))")

    # ─── Part C: Symmetrized + √g correction ───
    println("\n── Part C: Symmetrized δ²R + ½h·δR (full L_FP) ──")
    δR = simplify(δricci_scalar(mp, 1); registry=reg)
    h_trace = Tensor(:g, [up(:z1), up(:z2)]) * Tensor(:h, [down(:z1), down(:z2)])
    half_h_δR = tproduct(1 // 2, TensorExpr[h_trace]) * δR
    # ½h·δR has derivative distribution 0+0 for h_trace and 2+0 for the h inside δR
    # But ½h·δR has 3 h factors (trace h + 2 h's in δR)... wait, δR is scalar with 1 h
    # ½h·δR = ½(g^{ab}h_{ab})(∂∂h - □h) has TWO h's. Need to symmetrize this too.
    half_h_δR_sym = symmetrize_bilinear_derivs(simplify(half_h_δR; registry=reg), :h)
    half_h_δR_sym = simplify(half_h_δR_sym; registry=reg)

    Q = δ2R_sym + half_h_δR_sym
    Q = simplify(Q; registry=reg)
    nQ = Q isa TSum ? length(Q.terms) : 1
    println("  Q = δ²R_sym + ½h·δR_sym: $nQ terms")

    Q_f = to_fourier(Q)
    Q_f = simplify(Q_f; registry=reg)
    Q_f = fix_dummy_positions(Q_f)
    K_C = extract_kernel(Q_f, :h; registry=reg)

    s2_C  = spin_project(K_C, :spin2;  registry=reg)
    s0s_C = spin_project(K_C, :spin0s; registry=reg)
    s1_C  = spin_project(K_C, :spin1;  registry=reg)
    s0w_C = spin_project(K_C, :spin0w; registry=reg)

    v2_C  = _eval_spin_scalar(s2_C,  k2)
    v1_C  = _eval_spin_scalar(s1_C,  k2)
    v0s_C = _eval_spin_scalar(s0s_C, k2)
    v0w_C = _eval_spin_scalar(s0w_C, k2)

    println("  Full L_FP:     spin-2=$(round(v2_C;digits=4)), spin-1=$(round(v1_C;digits=4)), spin-0s=$(round(v0s_C;digits=4)), spin-0w=$(round(v0w_C;digits=4))")

    # ─── Part D: Reference FP ───
    println("\n── Part D: Reference FP kernel ──")
    K_FP = build_FP_momentum_kernel(reg)
    s2_D  = spin_project(K_FP, :spin2;  registry=reg)
    s0s_D = spin_project(K_FP, :spin0s; registry=reg)
    v2_D  = _eval_spin_scalar(s2_D,  k2)
    v0s_D = _eval_spin_scalar(spin_project(K_FP, :spin0s; registry=reg), k2)
    println("  FP ref:        spin-2=$(round(v2_D;digits=4)), spin-0s=$(round(v0s_D;digits=4))")

    # ─── Summary ───
    println("\n" * "=" ^ 70)
    println("  SUMMARY (k²=$k2)")
    println("=" ^ 70)
    println("  Method              spin-2   spin-1   spin-0s  spin-0w")
    println("  FP reference        $(lpad(round(v2_D;digits=3),8))   $(lpad("0.0",8))   $(lpad(round(v0s_D;digits=3),8))   $(lpad("0.0",8))")
    println("  A: Naive Fourier    $(lpad(round(v2_A;digits=3),8))   $(lpad(round(v1_A;digits=3),8))   $(lpad(round(v0s_A;digits=3),8))   $(lpad(round(v0w_A;digits=3),8))")
    println("  B: Sym-IBP δ²R      $(lpad(round(v2_B;digits=3),8))   $(lpad(round(v1_B;digits=3),8))   $(lpad(round(v0s_B;digits=3),8))   $(lpad(round(v0w_B;digits=3),8))")
    println("  C: Sym-IBP full L₂  $(lpad(round(v2_C;digits=3),8))   $(lpad(round(v1_C;digits=3),8))   $(lpad(round(v0s_C;digits=3),8))   $(lpad(round(v0w_C;digits=3),8))")

    pass = abs(v2_C - v2_D) < 0.01 && abs(v0s_C - v0s_D) < 0.01
    pass2 = abs(v2_B - v2_D) < 0.01 && abs(v0s_B - v0s_D) < 0.01
    println("\n  δ²R alone (sym-IBP) matches FP? $(pass2 ? "YES ✓" : "NO — need √g correction")")
    println("  Full L₂ (sym-IBP) matches FP?   $(pass ? "YES ✓" : "NO")")
    println("=" ^ 70)
end
