#= Differential invariant database: scalar invariants with up to 2 covariant
#  derivatives of curvature tensors.
#
#  Differential invariants are scalars built from covariant derivatives of
#  curvature, fully contracted. Unlike purely algebraic invariants (RInv),
#  these involve TDeriv nodes and cannot be represented as contraction
#  permutations on Riemann slots alone.
#
#  Order 4 (case {2}): 2 covariant derivatives + 1 curvature factor
#  ---------------------------------------------------------------
#  Independent invariants:
#    box_R       :  Box R = nabla^a nabla_a R                (total derivative)
#    grad_R_sq   :  |nabla R|^2 = nabla_a R nabla^a R
#    grad_Ric_sq :  |nabla Ric|^2 = nabla_a R_{bc} nabla^a R^{bc}
#    grad_Riem_sq:  |nabla Riem|^2 = nabla_a R_{bcde} nabla^a R^{bcde}
#
#  Key relation (contracted second Bianchi identity):
#    nabla_a nabla_b R^{ab} = (1/2) Box R
#
#  Ground truth: Garcia-Parrado & Martin-Garcia (2007), Sec 6;
#                Fulling, King, Wybourne & Cummins (1992).
=#

# ---- DiffInvariantEntry struct -----------------------------------------------

"""
    DiffInvariantEntry

Catalog entry for a scalar differential curvature invariant (involving
covariant derivatives of curvature).

# Fields
- `name::Symbol` -- canonical key (e.g. `:box_R`, `:grad_R_sq`)
- `n_derivs::Int` -- number of covariant derivatives
- `n_riemann::Int` -- number of Riemann/Ricci/scalar curvature factors
- `order::Int` -- total derivative order (= 2*n_riemann + n_derivs)
- `description::String` -- human-readable description
- `expression_fn::Function` -- `(reg, manifold, metric, covd) -> TensorExpr`
- `is_total_derivative::Bool` -- true if the expression is a total divergence
"""
struct DiffInvariantEntry
    name::Symbol
    n_derivs::Int
    n_riemann::Int
    order::Int
    description::String
    expression_fn::Function
    is_total_derivative::Bool
end

# ---- Expression builders (private) ------------------------------------------

# Box R = nabla^a nabla_a R = g^{ab} D_a D_b R
function _build_box_R(reg::TensorRegistry, manifold::Symbol,
                      metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)
    R = Tensor(:RicScalar, TIndex[])
    Tensor(metric, [up(a), up(b)]) * TDeriv(down(a), TDeriv(down(b), R, covd), covd)
end

# |nabla R|^2 = nabla_a R nabla^a R = g^{ab} (D_a R)(D_b R)
function _build_grad_R_sq(reg::TensorRegistry, manifold::Symbol,
                          metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)
    R1 = Tensor(:RicScalar, TIndex[])
    R2 = Tensor(:RicScalar, TIndex[])
    Tensor(metric, [up(a), up(b)]) * TDeriv(down(a), R1, covd) * TDeriv(down(b), R2, covd)
end

# |nabla Ric|^2 = nabla_a R_{bc} nabla^a R^{bc}
# = g^{ae} g^{bf} g^{cg} (D_a Ric_{bc})(D_e Ric_{fg})
function _build_grad_Ric_sq(reg::TensorRegistry, manifold::Symbol,
                            metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g_idx = fresh_index(used)

    Ric1 = Tensor(:Ric, [down(b), down(c)])
    Ric2 = Tensor(:Ric, [down(f), down(g_idx)])
    dRic1 = TDeriv(down(a), Ric1, covd)
    dRic2 = TDeriv(down(e), Ric2, covd)

    Tensor(metric, [up(a), up(e)]) *
        Tensor(metric, [up(b), up(f)]) *
        Tensor(metric, [up(c), up(g_idx)]) *
        dRic1 * dRic2
end

# |nabla Riem|^2 = nabla_a R_{bcde} nabla^a R^{bcde}
# = g^{af} g^{bg} g^{ch} g^{di} g^{ej} (D_a Riem_{bcde})(D_f Riem_{ghij})
function _build_grad_Riem_sq(reg::TensorRegistry, manifold::Symbol,
                             metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g_idx = fresh_index(used); push!(used, g_idx)
    h = fresh_index(used); push!(used, h)
    i_idx = fresh_index(used); push!(used, i_idx)
    j = fresh_index(used)

    Riem1 = Tensor(:Riem, [down(b), down(c), down(d), down(e)])
    Riem2 = Tensor(:Riem, [down(g_idx), down(h), down(i_idx), down(j)])
    dRiem1 = TDeriv(down(a), Riem1, covd)
    dRiem2 = TDeriv(down(f), Riem2, covd)

    Tensor(metric, [up(a), up(f)]) *
        Tensor(metric, [up(b), up(g_idx)]) *
        Tensor(metric, [up(c), up(h)]) *
        Tensor(metric, [up(d), up(i_idx)]) *
        Tensor(metric, [up(e), up(j)]) *
        dRiem1 * dRiem2
end

# nabla_a nabla_b R^{ab} (reduces to (1/2) Box R via contracted Bianchi)
# = g^{ac} g^{bd} D_a D_b Ric_{cd}
function _build_div_grad_Ric(reg::TensorRegistry, manifold::Symbol,
                             metric::Symbol, covd::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    Ric = Tensor(:Ric, [down(c), down(d)])
    Tensor(metric, [up(a), up(c)]) *
        Tensor(metric, [up(b), up(d)]) *
        TDeriv(down(a), TDeriv(down(b), Ric, covd), covd)
end

# ---- Catalog ----------------------------------------------------------------

"""
    _DIFF_INVAR_CATALOG :: Dict{Symbol, DiffInvariantEntry}

Module-level catalog of differential curvature invariants (involving
covariant derivatives of curvature).

Order 4 (2 derivatives, 1 curvature): `:box_R`, `:grad_R_sq`,
`:grad_Ric_sq`, `:grad_Riem_sq`
"""
const _DIFF_INVAR_CATALOG = Dict{Symbol, DiffInvariantEntry}(
    :box_R => DiffInvariantEntry(
        :box_R, 2, 1, 4,
        "Box R = nabla^a nabla_a R (d'Alembertian of Ricci scalar)",
        _build_box_R,
        true,  # is total derivative
    ),
    :grad_R_sq => DiffInvariantEntry(
        :grad_R_sq, 2, 1, 4,
        "|nabla R|^2 = nabla_a R nabla^a R (gradient of scalar curvature squared)",
        _build_grad_R_sq,
        false,
    ),
    :grad_Ric_sq => DiffInvariantEntry(
        :grad_Ric_sq, 2, 1, 4,
        "|nabla Ric|^2 = nabla_a R_{bc} nabla^a R^{bc} (Ricci gradient norm)",
        _build_grad_Ric_sq,
        false,
    ),
    :grad_Riem_sq => DiffInvariantEntry(
        :grad_Riem_sq, 2, 1, 4,
        "|nabla Riem|^2 = nabla_a R_{bcde} nabla^a R^{bcde} (Riemann gradient norm)",
        _build_grad_Riem_sq,
        false,
    ),
)

# ---- Public API --------------------------------------------------------------

"""
    diff_invariant(name::Symbol;
                   registry=current_registry(),
                   manifold=:M4,
                   metric=:g,
                   covd=:D) -> TensorExpr

Look up `name` in the differential invariant catalog and return the
corresponding abstract tensor expression.

The covariant derivative `covd` must already be defined in the registry.

# Examples
```julia
box_R = diff_invariant(:box_R; covd=:D)
grad_R2 = diff_invariant(:grad_R_sq; covd=:D)
```
"""
function diff_invariant(name::Symbol;
                        registry::TensorRegistry=current_registry(),
                        manifold::Symbol=:M4,
                        metric::Symbol=:g,
                        covd::Symbol=:D)
    haskey(_DIFF_INVAR_CATALOG, name) ||
        error("Unknown differential invariant: $name. " *
              "Available: $(sort(collect(keys(_DIFF_INVAR_CATALOG))))")

    entry = _DIFF_INVAR_CATALOG[name]
    entry.expression_fn(registry, manifold, metric, covd)
end

"""
    list_diff_invariants(; order=nothing, n_derivs=nothing) -> Vector{NamedTuple}

Return metadata for all cataloged differential invariants, optionally
filtered by total derivative order and/or number of covariant derivatives.

Each entry is a NamedTuple with fields:
`(name, n_derivs, n_riemann, order, description, is_total_derivative)`.

# Examples
```julia
list_diff_invariants()            # all differential invariants
list_diff_invariants(order=4)     # order-4 differential invariants
list_diff_invariants(n_derivs=2)  # invariants with exactly 2 derivatives
```
"""
function list_diff_invariants(; order::Union{Int,Nothing}=nothing,
                                n_derivs::Union{Int,Nothing}=nothing)
    result = NamedTuple{(:name, :n_derivs, :n_riemann, :order, :description,
                          :is_total_derivative),
                         Tuple{Symbol, Int, Int, Int, String, Bool}}[]
    for (_, entry) in sort(collect(_DIFF_INVAR_CATALOG);
                           by=e -> (e.second.order, e.second.name))
        (order !== nothing && entry.order != order) && continue
        (n_derivs !== nothing && entry.n_derivs != n_derivs) && continue
        push!(result, (name=entry.name, n_derivs=entry.n_derivs,
                       n_riemann=entry.n_riemann, order=entry.order,
                       description=entry.description,
                       is_total_derivative=entry.is_total_derivative))
    end
    result
end

"""
    diff_invariant_count(order::Int; dim::Union{Int,Nothing}=nothing) -> Int

Return the number of independent differential invariants at the given
total derivative order.

Currently implemented:
- Order 4 (2 derivatives, 1 curvature): 4 independent invariants
  (3 non-total-derivative + 1 total derivative)
- Order 6 (4 derivatives + 1 curvature, or 2 derivatives + 2 curvature):
  6 independent invariants (5 non-total-derivative + 1 total derivative)

The count is dimension-independent for dim >= 4. In lower dimensions,
some invariants become dependent through dimension-specific identities
(not yet implemented).
"""
function diff_invariant_count(order::Int; dim::Union{Int,Nothing}=nothing)
    if order == 4
        return 4
    elseif order == 6
        return 6
    end
    error("diff_invariant_count: order $order not yet implemented (available: 4, 6)")
end

"""
    _build_div_grad_Ric_expr(; registry=current_registry(),
                               manifold=:M4, metric=:g, covd=:D) -> TensorExpr

Build the expression nabla_a nabla_b R^{ab}. This is NOT independent:
by the contracted second Bianchi identity, it equals (1/2) Box R.

This is provided as a helper for testing the Bianchi relation, not as
a catalog entry.
"""
function _build_div_grad_Ric_expr(; registry::TensorRegistry=current_registry(),
                                    manifold::Symbol=:M4,
                                    metric::Symbol=:g,
                                    covd::Symbol=:D)
    _build_div_grad_Ric(registry, manifold, metric, covd)
end
