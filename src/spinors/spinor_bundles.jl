#= Spinor VBundle registration.

Registers SL(2,C) and its conjugate SL(2,C)_dot as vector bundles on a
spacetime manifold. The dotted/undotted distinction is carried by the
vbundle field of TIndex, not by index position or a special index type.

Reference: Penrose & Rindler, "Spinors and Space-Time" Vol 1 (1984), Sec 2.5.
=#

"""
    define_spinor_bundles!(reg; manifold=:M4)

Register `:SL2C` (undotted, dim=2) and `:SL2C_dot` (dotted, dim=2) as
complex-conjugate spinor bundles on `manifold`.

Index alphabets: A,B,C,D,E,F (undotted) and Ap,Bp,Cp,Dp,Ep,Fp (dotted).
ASCII names only (Julia cannot parse prime characters in Symbols).

# Example
```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g
define_spinor_bundles!(reg; manifold=:M4)
# Now: spin_up(:A) creates TIndex(:A, Up, :SL2C)
```
"""
function define_spinor_bundles!(reg::TensorRegistry; manifold::Symbol=:M4)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    undotted_indices = [:A, :B, :C, :D, :E, :F]
    dotted_indices   = [:Ap, :Bp, :Cp, :Dp, :Ep, :Fp]

    define_vbundle!(reg, :SL2C;
                    manifold=manifold, dim=2,
                    indices=undotted_indices,
                    conjugate_bundle=:SL2C_dot)

    define_vbundle!(reg, :SL2C_dot;
                    manifold=manifold, dim=2,
                    indices=dotted_indices,
                    conjugate_bundle=:SL2C)

    nothing
end

# ── Convenience constructors ─────────────────────────────────────────────

"""Create an undotted spinor index in the Up position."""
spin_up(s::Symbol) = TIndex(s, Up, :SL2C)

"""Create an undotted spinor index in the Down position."""
spin_down(s::Symbol) = TIndex(s, Down, :SL2C)

"""Create a dotted spinor index in the Up position."""
spin_dot_up(s::Symbol) = TIndex(s, Up, :SL2C_dot)

"""Create a dotted spinor index in the Down position."""
spin_dot_down(s::Symbol) = TIndex(s, Down, :SL2C_dot)

# ── Predicates ───────────────────────────────────────────────────────────

"""Test whether an index belongs to a spinor bundle (SL2C or SL2C_dot)."""
is_spinor_index(idx::TIndex) = idx.vbundle === :SL2C || idx.vbundle === :SL2C_dot

"""Test whether an index is dotted (belongs to SL2C_dot)."""
is_dotted(idx::TIndex) = idx.vbundle === :SL2C_dot

# ── Index conjugation ────────────────────────────────────────────────────

"""
    conjugate_index(idx::TIndex) -> TIndex

Return the complex-conjugate spinor index: swap SL2C <-> SL2C_dot,
preserving name and position. Errors for non-spinor indices.
"""
function conjugate_index(idx::TIndex)
    idx.vbundle === :SL2C     && return TIndex(idx.name, idx.position, :SL2C_dot)
    idx.vbundle === :SL2C_dot && return TIndex(idx.name, idx.position, :SL2C)
    error("conjugate_index: $(idx.name) is not a spinor index (vbundle=$(idx.vbundle))")
end

"""
    conjugate_index(idx::TIndex, reg::TensorRegistry) -> TIndex

Registry-aware conjugation: looks up the conjugate bundle from VBundleProperties.
Works for any conjugate pair, not just SL2C.
"""
function conjugate_index(idx::TIndex, reg::TensorRegistry)
    conj = conjugate_vbundle(reg, idx.vbundle)
    conj === nothing && error("No conjugate bundle for $(idx.vbundle)")
    TIndex(idx.name, idx.position, conj)
end
