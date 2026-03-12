#= Curvature invariant catalog.

A structured catalog of scalar curvature invariants at order 1 (linear in
curvature) and order 2 (quadratic in curvature).  Each entry stores a
constructor function `(registry, manifold, metric) -> TensorExpr` together
with metadata (order, minimum dimension, description).

Public API:
  curvature_invariant(name; registry, manifold, metric) -> TensorExpr
  list_invariants(; order) -> Vector{NamedTuple}
=#

# ─── InvariantEntry ────────────────────────────────────────────────

"""
    InvariantEntry

Catalog entry for a scalar curvature invariant.

# Fields
- `name::Symbol` — canonical key (e.g. `:R`, `:Kretschmann`)
- `order::Int` — curvature order (1 = linear, 2 = quadratic, ...)
- `expression_fn::Function` — `(reg, manifold, metric) -> TensorExpr`
- `description::String` — human-readable one-liner
- `min_dim::Int` — minimum spacetime dimension for non-trivial result
"""
struct InvariantEntry
    name::Symbol
    order::Int
    expression_fn::Function
    description::String
    min_dim::Int
end

# ─── Expression builders (private) ─────────────────────────────────

# Order 1: Ricci scalar R
function _build_R(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    Tensor(:RicScalar, TIndex[])
end

# Order 2: R^2
function _build_R_sq(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    R1 = Tensor(:RicScalar, TIndex[])
    R2 = Tensor(:RicScalar, TIndex[])
    R1 * R2
end

# Order 2: R_{ab} R^{ab}
function _build_Ric_sq(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    Ric_down = Tensor(:Ric, [down(a), down(b)])
    Ric_up   = Tensor(:Ric, [up(a), up(b)])
    Ric_down * Ric_up
end

# Order 2: R_{abcd} R^{abcd}  (Kretschmann scalar)
function _build_Kretschmann(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    Riem_down = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
    Riem_up   = Tensor(:Riem, [up(a), up(b), up(c), up(d)])
    Riem_down * Riem_up
end

# Order 2: C_{abcd} C^{abcd}  (Weyl squared)
function _build_Weyl_sq(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    Weyl_down = Tensor(:Weyl, [down(a), down(b), down(c), down(d)])
    Weyl_up   = Tensor(:Weyl, [up(a), up(b), up(c), up(d)])
    Weyl_down * Weyl_up
end

# ─── Catalog ───────────────────────────────────────────────────────

"""
    INVARIANT_CATALOG :: Dict{Symbol, InvariantEntry}

Module-level catalog of curvature invariants.  Keys are canonical names
(`:R`, `:R_sq`, `:Ric_sq`, `:Kretschmann`, `:Weyl_sq`).
"""
const INVARIANT_CATALOG = Dict{Symbol, InvariantEntry}(
    # ── Order 1 ──
    :R => InvariantEntry(
        :R, 1, _build_R,
        "Ricci scalar R",
        2,
    ),
    # ── Order 2 ──
    :R_sq => InvariantEntry(
        :R_sq, 2, _build_R_sq,
        "Ricci scalar squared R^2",
        2,
    ),
    :Ric_sq => InvariantEntry(
        :Ric_sq, 2, _build_Ric_sq,
        "Ricci tensor squared R_{ab}R^{ab}",
        2,
    ),
    :Kretschmann => InvariantEntry(
        :Kretschmann, 2, _build_Kretschmann,
        "Kretschmann scalar R_{abcd}R^{abcd}",
        2,
    ),
    :Weyl_sq => InvariantEntry(
        :Weyl_sq, 2, _build_Weyl_sq,
        "Weyl tensor squared C_{abcd}C^{abcd}",
        4,
    ),
)

# ─── Public API ────────────────────────────────────────────────────

"""
    curvature_invariant(name::Symbol;
                        registry=current_registry(),
                        manifold=:M4,
                        metric=:g) -> TensorExpr

Look up `name` in the curvature invariant catalog and return the
corresponding abstract tensor expression.

Checks that the manifold dimension is at least `entry.min_dim`.

# Examples
```julia
R  = curvature_invariant(:R)
K  = curvature_invariant(:Kretschmann)
W2 = curvature_invariant(:Weyl_sq; manifold=:M4, metric=:g)
```
"""
function curvature_invariant(name::Symbol;
                              registry::TensorRegistry=current_registry(),
                              manifold::Symbol=:M4,
                              metric::Symbol=:g)
    haskey(INVARIANT_CATALOG, name) ||
        error("Unknown curvature invariant: $name. " *
              "Available: $(sort(collect(keys(INVARIANT_CATALOG))))")

    entry = INVARIANT_CATALOG[name]

    # Dimension check
    if has_manifold(registry, manifold)
        mp = get_manifold(registry, manifold)
        dim = mp.dim
        if dim < entry.min_dim
            error("Invariant $(entry.name) requires dim >= $(entry.min_dim), " *
                  "but manifold $manifold has dim=$dim")
        end
    end

    entry.expression_fn(registry, manifold, metric)
end

"""
    list_invariants(; order=nothing) -> Vector{NamedTuple}

Return metadata for all cataloged invariants, optionally filtered by
curvature order.

Each entry is a `NamedTuple{(:name, :order, :description), Tuple{Symbol, Int, String}}`.

# Examples
```julia
list_invariants()            # all invariants
list_invariants(order=2)     # only quadratic invariants
```
"""
function list_invariants(; order::Union{Int,Nothing}=nothing)
    result = NamedTuple{(:name, :order, :description), Tuple{Symbol, Int, String}}[]
    for (_, entry) in sort(collect(INVARIANT_CATALOG); by=e -> (e.second.order, e.second.name))
        if order === nothing || entry.order == order
            push!(result, (name=entry.name, order=entry.order, description=entry.description))
        end
    end
    result
end
