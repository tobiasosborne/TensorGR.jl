# Spinor metric epsilon_{AB} for SL(2,C) index raising/lowering.
#
# The spin metric is the fundamental SL(2,C) invariant tensor:
#   epsilon_{AB} = -epsilon_{BA}   (antisymmetric)
#   epsilon_{01} = 1, epsilon_{10} = -1  (Penrose-Rindler convention)
#   epsilon^{AC} epsilon_{CB} = delta^A_C
#
# Unlike the spacetime metric g_{ab}, the spin metric is antisymmetric,
# so raising/lowering spinor indices picks up signs (the "see-saw" rule):
#   psi_A chi^A = -psi^A chi_A
#
# Reference: Penrose & Rindler, Spinors and Space-Time Vol 1 (1984), Sec 2.5.

"""
    define_spin_metric!(reg::TensorRegistry; manifold::Symbol=:M4)

Register the SL(2,C) spin metrics and their associated deltas:

- `:eps_spin` on `:SL2C` -- antisymmetric rank-(0,2) metric, `is_metric=true`
- `:eps_spin_dot` on `:SL2C_dot` -- conjugate antisymmetric metric
- `:delta_spin` on `:SL2C` -- Kronecker delta, `is_delta=true`
- `:delta_spin_dot` on `:SL2C_dot` -- conjugate delta

Stores entries in `metric_cache` and `delta_cache` keyed by vbundle symbol
(`:SL2C` and `:SL2C_dot`), extending the existing manifold-keyed caches.

Requires that spinor bundles are already registered via [`define_spinor_bundles!`](@ref).

# Reference
Penrose & Rindler, *Spinors and Space-Time* Vol 1 (1984), Eqs 2.5.2--2.5.8.
"""
function define_spin_metric!(reg::TensorRegistry; manifold::Symbol=:M4)
    has_vbundle(reg, :SL2C) || error("SL2C bundle not registered; call define_spinor_bundles! first")
    has_vbundle(reg, :SL2C_dot) || error("SL2C_dot bundle not registered; call define_spinor_bundles! first")

    sl2c = get_vbundle(reg, :SL2C)
    sl2c_dot = get_vbundle(reg, :SL2C_dot)

    # Save existing manifold-level cache entries so register_tensor! does not
    # overwrite the spacetime metric/delta with the spinor ones (both live on
    # the same manifold).
    saved_metric = get(reg.metric_cache, manifold, nothing)
    saved_delta = get(reg.delta_cache, manifold, nothing)

    # ── Undotted spin metric eps_{AB} ──────────────────────────────────────
    if !has_tensor(reg, :eps_spin)
        register_tensor!(reg, TensorProperties(
            name=:eps_spin, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[AntiSymmetric(1, 2)],
            is_metric=true,
            options=Dict{Symbol,Any}(
                :is_metric => true,
                :vbundle => :SL2C,
                :vbundle_dim => sl2c.dim)))
    end

    # ── Undotted delta: delta^A_B ──────────────────────────────────────────
    if !has_tensor(reg, :delta_spin)
        register_tensor!(reg, TensorProperties(
            name=:delta_spin, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            is_delta=true,
            options=Dict{Symbol,Any}(
                :is_delta => true,
                :vbundle => :SL2C,
                :vbundle_dim => sl2c.dim)))
    end

    # ── Dotted spin metric eps_{A'B'} ─────────────────────────────────────
    if !has_tensor(reg, :eps_spin_dot)
        register_tensor!(reg, TensorProperties(
            name=:eps_spin_dot, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[AntiSymmetric(1, 2)],
            is_metric=true,
            options=Dict{Symbol,Any}(
                :is_metric => true,
                :vbundle => :SL2C_dot,
                :vbundle_dim => sl2c_dot.dim)))
    end

    # ── Dotted delta: delta^{A'}_{B'} ─────────────────────────────────────
    if !has_tensor(reg, :delta_spin_dot)
        register_tensor!(reg, TensorProperties(
            name=:delta_spin_dot, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            is_delta=true,
            options=Dict{Symbol,Any}(
                :is_delta => true,
                :vbundle => :SL2C_dot,
                :vbundle_dim => sl2c_dot.dim)))
    end

    # ── Restore manifold-level caches ─────────────────────────────────────
    # register_tensor! overwrites metric_cache[manifold] for every is_metric
    # tensor; restore the spacetime entries.
    if saved_metric !== nothing
        reg.metric_cache[manifold] = saved_metric
    end
    if saved_delta !== nothing
        reg.delta_cache[manifold] = saved_delta
    end

    # ── Populate caches keyed by vbundle ───────────────────────────────────
    reg.metric_cache[:SL2C] = :eps_spin
    reg.metric_cache[:SL2C_dot] = :eps_spin_dot
    reg.delta_cache[:SL2C] = :delta_spin
    reg.delta_cache[:SL2C_dot] = :delta_spin_dot

    nothing
end

"""
    spin_metric(vbundle::Symbol; registry::TensorRegistry=current_registry()) -> Tensor

Return the spin metric tensor for the given spinor vbundle (`:SL2C` or `:SL2C_dot`)
with two abstract down indices.

# Examples
```julia
spin_metric(:SL2C)       # eps_{AB}   with fresh A, B indices
spin_metric(:SL2C_dot)   # eps_{A'B'} with fresh Ap, Bp indices
```
"""
function spin_metric(vbundle::Symbol; registry::TensorRegistry=current_registry())
    metric_name = get(registry.metric_cache, vbundle, nothing)
    metric_name === nothing && error("No spin metric registered for vbundle $vbundle")
    has_tensor(registry, metric_name) || error("Spin metric $metric_name not in registry")

    # Generate two fresh down indices on the appropriate vbundle
    used = Set{Symbol}()
    i1 = fresh_index(used; vbundle=vbundle)
    push!(used, i1)
    i2 = fresh_index(used; vbundle=vbundle)

    Tensor(metric_name, [TIndex(i1, Down, vbundle), TIndex(i2, Down, vbundle)])
end
