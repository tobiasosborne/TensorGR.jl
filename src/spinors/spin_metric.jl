#= Spin metric (epsilon tensor).

The SL(2,C) spin metric ε_{AB} is the antisymmetric 2-form that raises
and lowers spinor indices. It is the spinor analogue of the spacetime
metric g_{ab}, but ANTISYMMETRIC: ε_{AB} = -ε_{BA}.

Reference: Penrose & Rindler, "Spinors and Space-Time" Vol 1 (1984),
Eqs 2.5.2-2.5.8.

Key identities:
  ε^{AC} ε_{CB} = δ^A_B
  ε^{AB} ε_{AB} = 2  (dim of SL2C)
  φ^A = ε^{AB} φ_B   (raising: slot 2 is the natural contraction slot)
=#

"""
    define_spin_metric!(reg; manifold=:M4)

Register the SL(2,C) spin metric epsilon and its conjugate, plus the
corresponding Kronecker deltas. Requires `define_spinor_bundles!` first.

Populates metric_cache and delta_cache with vbundle-keyed entries:
  metric_cache[:SL2C] = :eps_spin
  delta_cache[:SL2C] = :delta_spin
  (and conjugate versions for :SL2C_dot)
"""
function define_spin_metric!(reg::TensorRegistry; manifold::Symbol=:M4)
    has_vbundle(reg, :SL2C) || error("SL2C vbundle not registered; call define_spinor_bundles! first")
    has_vbundle(reg, :SL2C_dot) || error("SL2C_dot vbundle not registered; call define_spinor_bundles! first")

    # Save existing metric/delta cache for the manifold (register_tensor! overwrites)
    saved_metric = get(reg.metric_cache, manifold, nothing)
    saved_delta = get(reg.delta_cache, manifold, nothing)

    # ── Undotted spin metric: ε_{AB} ──
    register_tensor!(reg, TensorProperties(
        name=:eps_spin, manifold=manifold, rank=(0, 2),
        symmetries=SymmetrySpec[AntiSymmetric(1, 2)],
        options=Dict{Symbol,Any}(:is_metric => true, :vbundle => :SL2C, :vbundle_dim => 2)
    ))

    # ── Dotted spin metric: ε_{A'B'} ──
    register_tensor!(reg, TensorProperties(
        name=:eps_spin_dot, manifold=manifold, rank=(0, 2),
        symmetries=SymmetrySpec[AntiSymmetric(1, 2)],
        options=Dict{Symbol,Any}(:is_metric => true, :vbundle => :SL2C_dot, :vbundle_dim => 2)
    ))

    # ── Undotted spinor delta: δ^A_B ──
    register_tensor!(reg, TensorProperties(
        name=:delta_spin, manifold=manifold, rank=(1, 1),
        options=Dict{Symbol,Any}(:is_delta => true, :vbundle => :SL2C, :vbundle_dim => 2)
    ))

    # ── Dotted spinor delta: δ^{A'}_{B'} ──
    register_tensor!(reg, TensorProperties(
        name=:delta_spin_dot, manifold=manifold, rank=(1, 1),
        options=Dict{Symbol,Any}(:is_delta => true, :vbundle => :SL2C_dot, :vbundle_dim => 2)
    ))

    # Restore spacetime metric/delta cache entries (register_tensor! clobbered them)
    if saved_metric !== nothing
        reg.metric_cache[manifold] = saved_metric
    end
    if saved_delta !== nothing
        reg.delta_cache[manifold] = saved_delta
    end

    # Populate vbundle-keyed cache entries
    reg.metric_cache[:SL2C] = :eps_spin
    reg.metric_cache[:SL2C_dot] = :eps_spin_dot
    reg.delta_cache[:SL2C] = :delta_spin
    reg.delta_cache[:SL2C_dot] = :delta_spin_dot

    nothing
end
