# ============================================================================
# TensorGR.jl — Gauge Theory with VBundles
#
# Define an SU(2) gauge bundle, create field strength tensors with mixed
# spacetime and gauge indices, and verify that contraction respects
# bundle boundaries.
# ============================================================================

using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g

    # --- (1) Define a gauge vector bundle ---
    define_vbundle!(reg, :SU2; manifold=:M4, dim=3,
                    indices=[:A, :B, :C, :D, :E])
    println("Registered SU(2) bundle: dim=3, indices A,B,C,D,E")
    println("  Tangent bundle: dim=", get_vbundle(reg, :Tangent).dim)
    println("  SU(2) bundle:   dim=", get_vbundle(reg, :SU2).dim)

    # --- (2) Field strength with mixed indices ---
    # F^A_{mu nu}: gauge index A (SU2), spacetime indices mu,nu (Tangent)
    F = Tensor(:F, [up(:A, :SU2), down(:mu), down(:nu)])
    println("\nField strength: ", to_unicode(F))
    println("  Index 1: name=", F.indices[1].name, " vbundle=", F.indices[1].vbundle)
    println("  Index 2: name=", F.indices[2].name, " vbundle=", F.indices[2].vbundle)
    println("  Index 3: name=", F.indices[3].name, " vbundle=", F.indices[3].vbundle)

    # --- (3) Cross-bundle contraction is refused ---
    # If two indices share the same name but different bundles,
    # they are NOT recognized as a dummy pair.
    T_tangent = Tensor(:T, [up(:a, :Tangent)])
    S_gauge   = Tensor(:S, [down(:a, :SU2)])
    product = T_tangent * S_gauge

    dp = dummy_pairs(product)
    a_pairs = filter(p -> p[1].name == :a, dp)
    println("\nCross-bundle 'a' indices contracted? ", !isempty(a_pairs))
    @assert isempty(a_pairs) "Cross-bundle contraction should be refused"
    println("  Correctly refused: Tangent 'a' and SU2 'a' are independent")

    fi = free_indices(product)
    a_free = filter(i -> i.name == :a, fi)
    println("  Free 'a' indices: ", length(a_free), " (both are free)")
    @assert length(a_free) == 2

    # --- (4) Same-bundle contraction works ---
    T1 = Tensor(:T, [up(:A, :SU2), down(:mu)])
    T2 = Tensor(:S, [down(:A, :SU2), up(:nu)])
    same_prod = T1 * T2

    dp2 = dummy_pairs(same_prod)
    A_pairs = filter(p -> p[1].name == :A, dp2)
    println("\nSame-bundle SU2 'A' contracted? ", !isempty(A_pairs))
    @assert length(A_pairs) == 1

    # --- (5) Gauge-covariant objects ---
    # Build F^A_{mu nu} F^{mu nu}_A (Yang-Mills Lagrangian density term)
    F_up = Tensor(:F, [up(:A, :SU2), down(:mu), down(:nu)])
    F_contracted = Tensor(:F, [down(:A, :SU2), up(:mu), up(:nu)])
    lagrangian = F_up * F_contracted
    println("\nYang-Mills term F^A_{mn} F_A^{mn}:")
    println("  ", to_unicode(lagrangian))

    dp_ym = dummy_pairs(lagrangian)
    println("  Dummy pairs: ", length(dp_ym))
    # Should have 3 dummy pairs: A(SU2), mu(Tangent), nu(Tangent)
    @assert length(dp_ym) == 3

    # --- (6) Dagger preserves vbundle ---
    F_dag = dagger(F)
    println("\nDagger preserves bundle:")
    println("  F^dag index 1: vbundle=", F_dag.indices[1].vbundle)
    @assert F_dag.indices[1].vbundle == :SU2
    @assert F_dag.indices[2].vbundle == :Tangent

    # --- (7) LaTeX output with gauge indices ---
    println("\nLaTeX: ", to_latex(F_up))
    println("LaTeX: ", to_latex(lagrangian))

    println("\nAll gauge theory checks passed!")
end
