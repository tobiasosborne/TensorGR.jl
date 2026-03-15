# Spinor VBundle infrastructure for 2-component spinors (Penrose & Rindler Vol 1, Ch 2).
#
# Registers :SL2C (undotted/unprimed) and :SL2C_dot (dotted/primed) bundles as
# conjugate pairs on a spacetime manifold. The dotted/undotted distinction is
# encoded via the vbundle field of TIndex, not via IndexPosition.

"""
    define_spinor_bundles!(reg; manifold=:M4)

Register the spinor VBundles `:SL2C` (undotted, dim=2) and `:SL2C_dot` (dotted, dim=2)
on `manifold` as complex-conjugate pairs.

Standard index alphabets follow Penrose-Rindler conventions:
- SL2C (undotted):  A, B, C, D, E, F
- SL2C_dot (dotted): Ap, Bp, Cp, Dp, Ep, Fp

The conjugation relationship is stored on each `VBundleProperties` via the
`conjugate_bundle` field, enabling downstream engines (contraction, canonicalization)
to recognise that these two bundles are related by complex conjugation.

# Reference
Penrose & Rindler, *Spinors and Space-Time* Vol 1 (1984), Section 2.5.
"""
function define_spinor_bundles!(reg::TensorRegistry; manifold::Symbol=:M4)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    undotted_indices = [:A, :B, :C, :D, :E, :F]
    dotted_indices   = [:Ap, :Bp, :Cp, :Dp, :Ep, :Fp]

    # Register SL2C (undotted) with conjugate pointing to SL2C_dot
    define_vbundle!(reg, :SL2C;
                    manifold=manifold, dim=2,
                    indices=undotted_indices,
                    conjugate_bundle=:SL2C_dot)

    # Register SL2C_dot (dotted) with conjugate pointing to SL2C
    define_vbundle!(reg, :SL2C_dot;
                    manifold=manifold, dim=2,
                    indices=dotted_indices,
                    conjugate_bundle=:SL2C)

    nothing
end

# ── Convenience constructors ──────────────────────────────────────────────

"""
    spin_up(s::Symbol) -> TIndex

Contravariant (upper) undotted spinor index on the SL2C bundle.
"""
spin_up(s::Symbol) = TIndex(s, Up, :SL2C)

"""
    spin_down(s::Symbol) -> TIndex

Covariant (lower) undotted spinor index on the SL2C bundle.
"""
spin_down(s::Symbol) = TIndex(s, Down, :SL2C)

"""
    spin_dot_up(s::Symbol) -> TIndex

Contravariant (upper) dotted spinor index on the SL2C_dot bundle.
"""
spin_dot_up(s::Symbol) = TIndex(s, Up, :SL2C_dot)

"""
    spin_dot_down(s::Symbol) -> TIndex

Covariant (lower) dotted spinor index on the SL2C_dot bundle.
"""
spin_dot_down(s::Symbol) = TIndex(s, Down, :SL2C_dot)

# ── Query functions ───────────────────────────────────────────────────────

"""
    is_dotted(idx::TIndex) -> Bool

Return `true` if `idx` lives in the dotted (primed) spinor bundle `:SL2C_dot`.
"""
is_dotted(idx::TIndex) = idx.vbundle === :SL2C_dot

"""
    is_spinor_index(idx::TIndex) -> Bool

Return `true` if `idx` lives in either spinor bundle (`:SL2C` or `:SL2C_dot`).
"""
is_spinor_index(idx::TIndex) = idx.vbundle === :SL2C || idx.vbundle === :SL2C_dot

"""
    conjugate_index(idx::TIndex) -> TIndex

Swap the vbundle of a spinor index between `:SL2C` and `:SL2C_dot`,
preserving name and position. Errors on non-spinor indices.

This implements complex conjugation of spinor indices: an undotted index
A becomes a dotted index A (in the conjugate bundle), and vice versa.
"""
function conjugate_index(idx::TIndex)
    if idx.vbundle === :SL2C
        return TIndex(idx.name, idx.position, :SL2C_dot)
    elseif idx.vbundle === :SL2C_dot
        return TIndex(idx.name, idx.position, :SL2C)
    else
        error("conjugate_index: index $(idx.name) is not a spinor index (vbundle=$(idx.vbundle))")
    end
end
