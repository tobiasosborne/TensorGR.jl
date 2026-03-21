#= SU(2) spatial spinor infrastructure for hypersurfaces.
#
# SU(2) spinors on spacelike hypersurfaces use a separate VBundle :SU2
# (dim 2, indices P,Q,R,S,T,U). The key objects are:
#
# 1. Spatial spin metric eps_space_{PQ} — antisymmetric, raises/lowers SU(2) indices
# 2. Soldering form tau^i_{PQ} — SYMMETRIC in PQ (unlike 4D sigma^a_{AA'})
#
# Reference: Sommers (1980), Sen (1981), Ashtekar (1991) Ch 2,
#            Penrose & Rindler (1984) Vol 1.
=#

"""
    define_space_spinors!(reg::TensorRegistry;
                          manifold::Symbol=:Sigma,
                          spatial_dim::Int=3,
                          indices::Vector{Symbol}=[:P,:Q,:R,:S,:T,:U],
                          metric::Symbol=:gamma)

Register SU(2) spatial spinor infrastructure on `manifold`:

1. SU(2) VBundle (dim 2) with the given index alphabet
2. Spatial spin metric `eps_space_{PQ}` (antisymmetric) + delta `delta_SU2`
3. Soldering form `tau^i_{PQ}` (symmetric in PQ)

Requires that `manifold` is already registered with a spatial metric.

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold Sigma dim=3 metric=gamma
    define_space_spinors!(reg; manifold=:Sigma, metric=:gamma)
end
```
"""
function define_space_spinors!(reg::TensorRegistry;
                               manifold::Symbol=:Sigma,
                               spatial_dim::Int=3,
                               indices::Vector{Symbol}=[:P,:Q,:R,:S,:T,:U],
                               metric::Symbol=:gamma)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    # 1. Register SU(2) VBundle (dim 2)
    if !has_vbundle(reg, :SU2)
        define_vbundle!(reg, :SU2; manifold=manifold, dim=2, indices=indices)
    end

    # Save existing metric/delta cache for the manifold
    saved_metric = get(reg.metric_cache, manifold, nothing)
    saved_delta = get(reg.delta_cache, manifold, nothing)

    # 2. Register spatial spin metric eps_space_{PQ} (antisymmetric)
    if !has_tensor(reg, :eps_space)
        register_tensor!(reg, TensorProperties(
            name=:eps_space, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[AntiSymmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true, :vbundle => :SU2, :vbundle_dim => 2)
        ))
    end

    # 3. Register SU(2) Kronecker delta
    if !has_tensor(reg, :delta_SU2)
        register_tensor!(reg, TensorProperties(
            name=:delta_SU2, manifold=manifold, rank=(1, 1),
            options=Dict{Symbol,Any}(:is_delta => true, :vbundle => :SU2, :vbundle_dim => 2)
        ))
    end

    # Restore spacetime metric/delta cache entries (register_tensor! clobbers them)
    if saved_metric !== nothing
        reg.metric_cache[manifold] = saved_metric
    end
    if saved_delta !== nothing
        reg.delta_cache[manifold] = saved_delta
    end

    # Populate vbundle-keyed cache entries for SU(2)
    reg.metric_cache[:SU2] = :eps_space
    reg.delta_cache[:SU2] = :delta_SU2

    # 4. Register soldering form tau^i_{PQ} (symmetric in spinor indices 2,3)
    if !has_tensor(reg, :tau)
        register_tensor!(reg, TensorProperties(
            name=:tau, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[Symmetric(2, 3)],
            options=Dict{Symbol,Any}(
                :is_soldering => true,
                :is_space_soldering => true,
                :index_vbundles => [:Tangent, :SU2, :SU2])
        ))
    end

    nothing
end

# ── Convenience constructors ─────────────────────────────────────────────

"""Create an SU(2) spatial spinor index in the Up position."""
space_spin_up(s::Symbol) = TIndex(s, Up, :SU2)

"""Create an SU(2) spatial spinor index in the Down position."""
space_spin_down(s::Symbol) = TIndex(s, Down, :SU2)

# ── Predicates ───────────────────────────────────────────────────────────

"""Test whether an index belongs to the SU(2) spatial spinor bundle."""
is_space_spinor_index(idx::TIndex) = idx.vbundle === :SU2

# ── Expression builders ─────────────────────────────────────────────────

"""
    space_spin_metric_expr(; registry=current_registry()) -> Tensor

Return the spatial spin metric `eps_space_{PQ}` with fresh SU(2) indices.
"""
function space_spin_metric_expr(; registry::TensorRegistry=current_registry())
    has_tensor(registry, :eps_space) || error("eps_space not registered; call define_space_spinors! first")
    used = Set{Symbol}()
    i1 = fresh_index(used; vbundle=:SU2)
    push!(used, i1)
    i2 = fresh_index(used; vbundle=:SU2)
    Tensor(:eps_space, [TIndex(i1, Down, :SU2), TIndex(i2, Down, :SU2)])
end

"""
    soldering_form_expr(; registry=current_registry()) -> Tensor

Return the spatial soldering form `tau^i_{PQ}` with fresh indices:
one Tangent (Up) and two SU(2) (Down, symmetric).
"""
function soldering_form_expr(; registry::TensorRegistry=current_registry())
    has_tensor(registry, :tau) || error("tau not registered; call define_space_spinors! first")
    used = Set{Symbol}()
    i_tang = fresh_index(used; vbundle=:Tangent)
    push!(used, i_tang)
    i1 = fresh_index(used; vbundle=:SU2)
    push!(used, i1)
    i2 = fresh_index(used; vbundle=:SU2)
    Tensor(:tau, [TIndex(i_tang, Up, :Tangent),
                  TIndex(i1, Down, :SU2),
                  TIndex(i2, Down, :SU2)])
end

"""
    space_spinor_completeness(; registry=current_registry()) -> TProduct

Return the completeness relation for the spatial soldering form:

    tau^i_{PQ} tau_i^{RS}

which equals `eps^{(R}_{(P} eps^{S)}_{Q)}` (symmetrized product of deltas).
The returned expression is the LHS as a product of two tau tensors with
the spatial index contracted.
"""
function space_spinor_completeness(; registry::TensorRegistry=current_registry())
    has_tensor(registry, :tau) || error("tau not registered; call define_space_spinors! first")
    used = Set{Symbol}()

    # Contracted spatial index
    i_sp = fresh_index(used; vbundle=:Tangent)
    push!(used, i_sp)

    # First tau: tau^i_{PQ}
    p1 = fresh_index(used; vbundle=:SU2)
    push!(used, p1)
    q1 = fresh_index(used; vbundle=:SU2)
    push!(used, q1)

    # Second tau: tau_i^{RS}
    r1 = fresh_index(used; vbundle=:SU2)
    push!(used, r1)
    s1 = fresh_index(used; vbundle=:SU2)

    tau1 = Tensor(:tau, [TIndex(i_sp, Up, :Tangent),
                         TIndex(p1, Down, :SU2),
                         TIndex(q1, Down, :SU2)])
    tau2 = Tensor(:tau, [TIndex(i_sp, Down, :Tangent),
                         TIndex(r1, Up, :SU2),
                         TIndex(s1, Up, :SU2)])

    tproduct(1 // 1, TensorExpr[tau1, tau2])
end
