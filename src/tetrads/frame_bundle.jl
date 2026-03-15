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

# ── Standard index alphabet for frame indices ─────────────────────────────
const _FRAME_INDICES = [:I, :J, :K, :L, :M, :N]
const _FRAME_FRESH   = [:I, :J, :K, :L, :M, :N]

"""
    define_frame_bundle!(reg; manifold=:M4, dim=4)

Register a `:Lorentz` VBundle on `manifold` with fiber dimension `dim`
for tetrad/frame indices.

Also registers the frame metric `eta` (symmetric, `is_metric=true`) and
Kronecker delta `delta_frame` (`is_delta=true`) on the `:Lorentz` bundle.

Frame indices use the capital Latin alphabet: I, J, K, L, M, N.

# Convention
Following xCoba: `up(:I, :Lorentz)` for frame indices vs `up(:a)` for
coordinate (Tangent) indices. The tetrad `e^a_I` has one :Tangent and
one :Lorentz index.

# References
- Chandrasekhar, *The Mathematical Theory of Black Holes* (1983), Ch 1.
- Nakahara, *Geometry, Topology and Physics* (2003), Sec 7.8.
"""
function define_frame_bundle!(reg::TensorRegistry;
                              manifold::Symbol=:M4, dim::Int=4)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    define_vbundle!(reg, :Lorentz;
                    manifold=manifold, dim=dim,
                    indices=copy(_FRAME_INDICES))

    # Save manifold-level caches so register_tensor! doesn't overwrite
    # the spacetime metric/delta with the frame ones.
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

"""
    frame_up(s::Symbol) -> TIndex

Contravariant (upper) frame index on the :Lorentz bundle.
"""
frame_up(s::Symbol) = TIndex(s, Up, :Lorentz)

"""
    frame_down(s::Symbol) -> TIndex

Covariant (lower) frame index on the :Lorentz bundle.
"""
frame_down(s::Symbol) = TIndex(s, Down, :Lorentz)

# ── Query function ────────────────────────────────────────────────────────

"""
    is_frame_index(idx::TIndex) -> Bool

Return `true` if `idx` lives in the `:Lorentz` (frame) VBundle.
"""
is_frame_index(idx::TIndex) = idx.vbundle === :Lorentz

# ── Fresh index generation for :Lorentz vbundle ──────────────────────────

"""
    fresh_frame_index(used::Set{Symbol}) -> Symbol

Generate a fresh frame index name not in `used`.
Tries I, J, K, L, M, N, then I1, J1, ...
"""
function fresh_frame_index(used::Set{Symbol})
    for s in _FRAME_FRESH
        s in used || return s
    end
    for n in 1:100
        for s in _FRAME_FRESH
            ext = Symbol(s, n)
            ext in used || return ext
        end
    end
    error("Could not generate fresh frame index (exhausted 600+ names)")
end

# ── Register fresh_index dispatch hook ────────────────────────────────────
# Chain onto the existing _FRESH_SPINOR_HOOK (which handles :SL2C/:SL2C_dot).
# We wrap the previous hook and add :Lorentz dispatch.

let _prev_hook = _FRESH_SPINOR_HOOK[]
    _FRESH_SPINOR_HOOK[] = function(used::Set{Symbol}, vbundle::Symbol)
        if vbundle === :Lorentz
            return fresh_frame_index(used)
        end
        # Delegate to previous hook (spinor dispatch or nothing)
        _prev_hook === nothing ? nothing : _prev_hook(used, vbundle)
    end
end
