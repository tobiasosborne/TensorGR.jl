#= Curvature invariant catalog.

A structured catalog of scalar curvature invariants at orders 1 (linear),
2 (quadratic), and 3 (cubic in curvature).  Each entry stores a
constructor function `(registry, manifold, metric) -> TensorExpr` together
with metadata (order, minimum dimension, description).

The 6 cubic (order-3) invariants are the independent monomials cubic in
curvature:
  I1 = R^3,  I2 = R R_{ab}R^{ab},  I3 = R_a^b R_b^c R_c^a,
  I4 = R R_{abcd}R^{abcd},  I5 = R_{ab}R^{acde}R^b_{cde},
  I6 = R_{ab}^{cd}R_{cd}^{ef}R_{ef}^{ab}  (Goroff-Sagnotti)

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

# Order 3: R^3
function _build_R_cubed(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    Tensor(:RicScalar, TIndex[]) * Tensor(:RicScalar, TIndex[]) * Tensor(:RicScalar, TIndex[])
end

# Order 3: R * R_{ab} R^{ab}
function _build_R_Ric_sq(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    R = Tensor(:RicScalar, TIndex[])
    Ric1 = Tensor(:Ric, [down(a), down(b)])
    Ric2 = Tensor(:Ric, [down(c), down(d)])
    R * Ric1 * Ric2 * Tensor(metric, [up(a), up(c)]) * Tensor(metric, [up(b), up(d)])
end

# Order 3: R_{a}^{b} R_{b}^{c} R_{c}^{a}  (Ricci cube trace)
function _build_Ric_cubed(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g = fresh_index(used)

    Ric1 = Tensor(:Ric, [down(a), down(e)])
    Ric2 = Tensor(:Ric, [down(b), down(f)])
    Ric3 = Tensor(:Ric, [down(c), down(g)])
    Ric1 * Ric2 * Ric3 *
        Tensor(metric, [up(e), up(b)]) * Tensor(metric, [up(f), up(c)]) * Tensor(metric, [up(g), up(a)])
end

# Order 3: R * R_{abcd} R^{abcd}  (scalar x Kretschmann)
function _build_R_Kretschmann(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g = fresh_index(used); push!(used, g)
    h = fresh_index(used)

    R = Tensor(:RicScalar, TIndex[])
    Riem1 = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
    Riem2 = Tensor(:Riem, [down(e), down(f), down(g), down(h)])
    R * Riem1 * Riem2 *
        Tensor(metric, [up(a), up(e)]) * Tensor(metric, [up(b), up(f)]) *
        Tensor(metric, [up(c), up(g)]) * Tensor(metric, [up(d), up(h)])
end

# Order 3: R_{ab} R^{acde} R^{b}_{cde}  (Ricci-Riemann contraction)
function _build_Ric_Riem_sq(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g = fresh_index(used); push!(used, g)
    h = fresh_index(used); push!(used, h)
    p = fresh_index(used); push!(used, p)
    q = fresh_index(used)

    Ric = Tensor(:Ric, [down(p), down(q)])
    Riem1 = Tensor(:Riem, [down(a), down(c), down(d), down(e)])
    Riem2 = Tensor(:Riem, [down(b), down(f), down(g), down(h)])
    Ric * Riem1 * Riem2 *
        Tensor(metric, [up(p), up(a)]) * Tensor(metric, [up(q), up(b)]) *
        Tensor(metric, [up(c), up(f)]) * Tensor(metric, [up(d), up(g)]) * Tensor(metric, [up(e), up(h)])
end

# Order 3: R_{ab}^{cd} R_{cd}^{ef} R_{ef}^{ab}  (Riemann cycle / Goroff-Sagnotti)
function _build_Riem_cubed(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    i = fresh_index(used); push!(used, i)
    j = fresh_index(used); push!(used, j)
    k = fresh_index(used); push!(used, k)
    l = fresh_index(used); push!(used, l)
    m = fresh_index(used); push!(used, m)
    n = fresh_index(used)

    Riem1 = Tensor(:Riem, [down(a), down(b), down(i), down(j)])
    Riem2 = Tensor(:Riem, [down(c), down(d), down(k), down(l)])
    Riem3 = Tensor(:Riem, [down(e), down(f), down(m), down(n)])
    Riem1 * Riem2 * Riem3 *
        Tensor(metric, [up(i), up(c)]) * Tensor(metric, [up(j), up(d)]) *
        Tensor(metric, [up(k), up(e)]) * Tensor(metric, [up(l), up(f)]) *
        Tensor(metric, [up(m), up(a)]) * Tensor(metric, [up(n), up(b)])
end

# ─── Catalog ───────────────────────────────────────────────────────

"""
    INVARIANT_CATALOG :: Dict{Symbol, InvariantEntry}

Module-level catalog of curvature invariants.  Keys are canonical names.

Order 1: `:R`
Order 2: `:R_sq`, `:Ric_sq`, `:Kretschmann`, `:Weyl_sq`
Order 3: `:R_cubed`, `:R_Ric_sq`, `:Ric_cubed`, `:R_Kretschmann`,
         `:Ric_Riem_sq`, `:Riem_cubed`
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
    # ── Order 3 (cubic curvature invariants) ──
    :R_cubed => InvariantEntry(
        :R_cubed, 3, _build_R_cubed,
        "Ricci scalar cubed R^3",
        2,
    ),
    :R_Ric_sq => InvariantEntry(
        :R_Ric_sq, 3, _build_R_Ric_sq,
        "Scalar times Ricci squared R R_{ab}R^{ab}",
        2,
    ),
    :Ric_cubed => InvariantEntry(
        :Ric_cubed, 3, _build_Ric_cubed,
        "Ricci cube trace R_{a}^{b}R_{b}^{c}R_{c}^{a}",
        2,
    ),
    :R_Kretschmann => InvariantEntry(
        :R_Kretschmann, 3, _build_R_Kretschmann,
        "Scalar times Kretschmann R R_{abcd}R^{abcd}",
        4,
    ),
    :Ric_Riem_sq => InvariantEntry(
        :Ric_Riem_sq, 3, _build_Ric_Riem_sq,
        "Ricci-Riemann contraction R_{ab}R^{acde}R^{b}_{cde}",
        4,
    ),
    :Riem_cubed => InvariantEntry(
        :Riem_cubed, 3, _build_Riem_cubed,
        "Riemann cycle R_{ab}^{cd}R_{cd}^{ef}R_{ef}^{ab} (Goroff-Sagnotti)",
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
