# Frame/tetrad index VBundle for Lorentz (frame) indices.
#
# Registers a :Lorentz VBundle for orthonormal or null-frame indices,
# separate from :Tangent (coordinate) indices. The tetrad e^a_I is a
# mixed tensor: one :Tangent Up index, one :Lorentz Down index.
#
# The orthonormal vs null-frame distinction lives in the frame metric
# eta_{IJ} (diag(-1,1,1,1) vs NP null metric), not in the index type.
#
# References:
#   Chandrasekhar, *The Mathematical Theory of Black Holes* (1983), Ch 1.
#   Nakahara, *Geometry, Topology and Physics* (2003), Sec 7.8.

"""
    define_frame_bundle!(reg; manifold=:M4, dim=4)

Register a `:Lorentz` VBundle on `manifold` with fiber dimension `dim`
for tetrad/frame indices.

Also registers the frame metric `eta` (symmetric, `is_metric=true`) and
Kronecker delta `delta_frame` (`is_delta=true`) on the `:Lorentz` bundle.

Frame indices use the capital Latin alphabet: I, J, K, L, M, N.

# Convention
Following xCoba: `frame_up(:I)` for frame indices vs `up(:a)` for
coordinate (Tangent) indices. The tetrad `e^a_I` has one :Tangent and
one :Lorentz index.

# References
- Chandrasekhar, *The Mathematical Theory of Black Holes* (1983), Ch 1.
- Nakahara, *Geometry, Topology and Physics* (2003), Sec 7.8.
"""
function define_frame_bundle!(reg::TensorRegistry;
                              manifold::Symbol=:M4, dim::Int=4)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    if !has_vbundle(reg, :Lorentz)
        define_vbundle!(reg, :Lorentz;
                        manifold=manifold, dim=dim,
                        indices=[:I, :J, :K, :L, :M, :N])
    end

    # Save manifold-level caches so register_tensor! doesn't overwrite
    saved_metric = get(reg.metric_cache, manifold, nothing)
    saved_delta  = get(reg.delta_cache, manifold, nothing)

    # ── Frame metric eta_{IJ} ─────────────────────────────────────────
    if !has_tensor(reg, :eta)
        register_tensor!(reg, TensorProperties(
            name=:eta, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true,
            options=Dict{Symbol,Any}(
                :is_metric => true,
                :vbundle => :Lorentz,
                :vbundle_dim => dim)))
    end

    # ── Frame delta: delta^I_J ────────────────────────────────────────
    if !has_tensor(reg, :delta_frame)
        register_tensor!(reg, TensorProperties(
            name=:delta_frame, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            is_delta=true,
            options=Dict{Symbol,Any}(
                :is_delta => true,
                :vbundle => :Lorentz,
                :vbundle_dim => dim)))
    end

    # ── Restore manifold-level caches ─────────────────────────────────
    if saved_metric !== nothing
        reg.metric_cache[manifold] = saved_metric
    end
    if saved_delta !== nothing
        reg.delta_cache[manifold] = saved_delta
    end

    # ── Populate caches keyed by vbundle ──────────────────────────────
    reg.metric_cache[:Lorentz] = :eta
    reg.delta_cache[:Lorentz]  = :delta_frame

    nothing
end

# ── Convenience constructors ──────────────────────────────────────────────

"""Contravariant (upper) frame index on the :Lorentz bundle."""
frame_up(s::Symbol) = TIndex(s, Up, :Lorentz)

"""Covariant (lower) frame index on the :Lorentz bundle."""
frame_down(s::Symbol) = TIndex(s, Down, :Lorentz)

"""Return `true` if `idx` lives in the `:Lorentz` (frame) VBundle."""
is_frame_index(idx::TIndex) = idx.vbundle === :Lorentz
