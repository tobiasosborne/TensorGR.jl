# ============================================================================
# TensorGR.jl — Curvature Decomposition
#
# Decompose the Riemann tensor into Weyl + Ricci parts,
# build the Gauss-Bonnet invariant, and construct the Kretschmann scalar.
# ============================================================================

using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)

    a, b, c, d = down(:a), down(:b), down(:c), down(:d)

    # --- (1) Riemann -> Weyl decomposition ---
    # R_{abcd} = C_{abcd} + (Ricci terms) + (scalar terms)
    decomp = riemann_to_weyl(a, b, c, d, :g; dim=4)
    println("R_{abcd} = ", to_unicode(decomp))

    # --- (2) Inverse: Weyl -> Riemann ---
    inv_decomp = weyl_to_riemann(a, b, c, d, :g; dim=4)
    println("\nC_{abcd} = ", to_unicode(inv_decomp))

    # --- (3) Schouten tensor ---
    schouten = schouten_to_ricci(a, b, :g; dim=4)
    println("\nP_{ab} = ", to_unicode(schouten))

    # --- (4) Trace-free Ricci ---
    tfric = tfricci_expr(a, b, :g; dim=4)
    println("S_{ab} = ", to_unicode(tfric))

    # --- (5) Einstein tensor from Ricci ---
    einstein = einstein_to_ricci(a, b, :g)
    println("G_{ab} = ", to_unicode(einstein))

    # --- (6) Ricci from Einstein (inverse) ---
    ricci_back = ricci_to_einstein(a, b, :g)
    println("R_{ab} = ", to_unicode(ricci_back))

    # --- (7) Gauss-Bonnet invariant ---
    # G = R^2 - 4 R_{ab}R^{ab} + R_{abcd}R^{abcd}
    # Build each term abstractly:
    R_scalar = Tensor(:RicScalar, TIndex[])
    R_squared = R_scalar * R_scalar

    Ric_ab = Tensor(:Ric, [down(:a), down(:b)])
    Ric_up = Tensor(:Ric, [up(:a), up(:b)])
    Ric_sq = Ric_ab * Ric_up

    Kretschmann = kretschmann_expr(:g; dim=4)

    gauss_bonnet = R_squared - 4 * Ric_sq + Kretschmann
    println("\nGauss-Bonnet (abstract): ", to_unicode(gauss_bonnet))

    # --- (8) Convert everything to Ricci form ---
    weyl_expr = Tensor(:Weyl, [a, b, c, d])
    in_ricci = to_ricci(weyl_expr; metric=:g, dim=4)
    println("\nWeyl -> Ricci form: Weyl stays (it's traceless)")

    # --- (9) Convert to Riemann form (replaces Einstein, Schouten, etc.) ---
    ein_expr = Tensor(:Ein, [a, b])
    in_riemann = to_riemann(ein_expr; metric=:g, dim=4)
    println("G_{ab} -> Riemann form: ", to_unicode(in_riemann))

    # --- (10) Kretschmann expression ---
    println("\nKretschmann scalar (abstract): ", to_unicode(Kretschmann))

    println("\nAll curvature decompositions computed successfully!")
end
