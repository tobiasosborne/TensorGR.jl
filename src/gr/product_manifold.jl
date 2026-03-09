#= Product manifolds: M = M₁ × M₂ × ⋯ with block-diagonal metric.

For a direct product of Riemannian manifolds (M₁,g₁) × (M₂,g₂):
  • Metric:    g = g₁ ⊕ g₂  (block diagonal, off-diagonal blocks vanish)
  • Riemann:   R = R₁ ⊕ R₂  (mixed-factor components vanish)
  • Ricci:     Ric = Ric₁ ⊕ Ric₂
  • Scalar:    R = R₁ + R₂
  • Einstein:  G_{ij} = G₁_{ij} - ½ Σ_{k≠i} Rₖ · g₁_{ij}  (cross-scalar terms)

Design: No new AST types. The product is a metadata container linking factor
manifolds. Decomposition functions construct expressions using factor-manifold
tensors with metric-suffixed names (Riem_g₁, Ric_g₁, …).
=#

"""
    ProductManifoldProperties

Metadata for a direct product manifold M = M₁ × M₂ × ⋯

# Fields
- `name`: product manifold identifier
- `factors`: ordered factor manifold names
- `factor_metrics`: metric on each factor (same ordering)
- `factor_dims`: dimension of each factor
"""
struct ProductManifoldProperties
    name::Symbol
    factors::Vector{Symbol}
    factor_metrics::Vector{Symbol}
    factor_dims::Vector{Int}
end

has_product_manifold(reg::TensorRegistry, name::Symbol) =
    haskey(reg.foliations, Symbol(:product_, name))

function get_product_manifold(reg::TensorRegistry, name::Symbol)
    reg.foliations[Symbol(:product_, name)]::ProductManifoldProperties
end

"""Metric-suffixed tensor name for factor curvature."""
_factor_curvature_name(base::Symbol, metric::Symbol) = Symbol(base, :_, metric)

"""
    define_product_manifold!(reg, name; factors) -> ProductManifoldProperties

Define a direct product manifold `name = factors[1] × factors[2] × ⋯`.

Each factor must be a registered manifold with a metric. Registers metric-suffixed
curvature tensors (Riem_g₁, Ric_g₁, RicScalar_g₁, Ein_g₁, Weyl_g₁) on each
factor manifold for use in decomposition expressions.

# Example
```julia
@manifold M1 dim=2 metric=g1 indices=[a,b,c,d]
@manifold S2 dim=2 metric=g2 indices=[α,β,γ,δ]
define_product_manifold!(reg, :M; factors=[:M1, :S2])

R = product_scalar_curvature(:M)    # RicScalar_g1 + RicScalar_g2
G = product_einstein(:M, :M1)       # Ein_g1_{ab} - ½ RicScalar_g2 · g1_{ab}
```
"""
function define_product_manifold!(reg::TensorRegistry, name::Symbol;
                                   factors::Vector{Symbol})
    length(factors) < 2 && error("Product manifold requires at least 2 factors")
    has_product_manifold(reg, name) && error("Product manifold $name already defined")

    factor_props = ManifoldProperties[]
    factor_metrics = Symbol[]
    factor_dims = Int[]

    for f in factors
        has_manifold(reg, f) || error("Factor manifold $f not registered")
        fp = get_manifold(reg, f)
        fp.metric === nothing && error("Factor $f has no metric; all factors require metrics")
        push!(factor_props, fp)
        push!(factor_metrics, fp.metric)
        push!(factor_dims, fp.dim)
    end

    # Register factor curvature tensors with metric-suffixed names
    for (fp, fm) in zip(factor_props, factor_metrics)
        _register_factor_curvature!(reg, fp.name, fm)
    end

    pp = ProductManifoldProperties(name, collect(factors), factor_metrics, factor_dims)
    reg.foliations[Symbol(:product_, name)] = pp
    pp
end

"""Register Riemann, Ricci, RicScalar, Einstein, Weyl for a factor (metric-suffixed)."""
function _register_factor_curvature!(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    for (base, rank, syms) in (
        (:Riem,      (0, 4), SymmetrySpec[RiemannSymmetry()]),
        (:Ric,       (0, 2), SymmetrySpec[Symmetric(1, 2)]),
        (:RicScalar, (0, 0), SymmetrySpec[]),
        (:Ein,       (0, 2), SymmetrySpec[Symmetric(1, 2)]),
        (:Weyl,      (0, 4), SymmetrySpec[RiemannSymmetry()]),
    )
        tname = _factor_curvature_name(base, metric)
        has_tensor(reg, tname) && continue
        register_tensor!(reg, TensorProperties(
            name=tname, manifold=manifold, rank=rank, symmetries=syms))
    end
end

# ── Helpers ──────────────────────────────────────────────────────────

function _factor_index(pp::ProductManifoldProperties, factor::Symbol)
    i = findfirst(==(factor), pp.factors)
    i === nothing && error("$factor is not a factor of product manifold $(pp.name)")
    i
end

function _factor_indices(reg::TensorRegistry, factor::Symbol, n::Int)
    fp = get_manifold(reg, factor)
    length(fp.indices) < n &&
        error("Factor $factor has $(length(fp.indices)) indices, need $n")
    fp.indices[1:n]
end

# ── Curvature Decomposition ─────────────────────────────────────────

"""
    product_metric(name; registry) -> TensorExpr

Block-diagonal metric: g = g₁_{ab} + g₂_{αβ} + ⋯

Each term uses the factor manifold's own index alphabet.
"""
function product_metric(name::Symbol;
                         registry::TensorRegistry=current_registry())
    pp = get_product_manifold(registry, name)
    terms = map(pp.factors, pp.factor_metrics) do f, fm
        ab = _factor_indices(registry, f, 2)
        Tensor(fm, [down(ab[1]), down(ab[2])])
    end
    tsum(collect(TensorExpr, terms))
end

"""
    product_scalar_curvature(name; registry) -> TensorExpr

Scalar curvature of a direct product: R = R₁ + R₂ + ⋯
"""
function product_scalar_curvature(name::Symbol;
                                   registry::TensorRegistry=current_registry())
    pp = get_product_manifold(registry, name)
    terms = TensorExpr[Tensor(_factor_curvature_name(:RicScalar, m), TIndex[])
                       for m in pp.factor_metrics]
    tsum(terms)
end

"""
    product_ricci(name, factor; registry) -> TensorExpr

Ricci tensor in the `factor` sector of product manifold `name`.

For a direct product, Ric_{ij} = Ric₁_{ij} (pure factor, no cross terms).
Mixed-factor Ricci components vanish identically.
"""
function product_ricci(name::Symbol, factor::Symbol;
                       registry::TensorRegistry=current_registry())
    pp = get_product_manifold(registry, name)
    fi = _factor_index(pp, factor)
    ab = _factor_indices(registry, factor, 2)
    Tensor(_factor_curvature_name(:Ric, pp.factor_metrics[fi]),
           [down(ab[1]), down(ab[2])])
end

"""
    product_riemann(name, factor; registry) -> TensorExpr

Riemann tensor in the `factor` sector of product manifold `name`.

For a direct product, R_{ijkl} = R₁_{ijkl} (pure factor).
All mixed-factor Riemann components vanish.
"""
function product_riemann(name::Symbol, factor::Symbol;
                          registry::TensorRegistry=current_registry())
    pp = get_product_manifold(registry, name)
    fi = _factor_index(pp, factor)
    abcd = _factor_indices(registry, factor, 4)
    Tensor(_factor_curvature_name(:Riem, pp.factor_metrics[fi]),
           [down(abcd[1]), down(abcd[2]), down(abcd[3]), down(abcd[4])])
end

"""
    product_einstein(name, factor; registry) -> TensorExpr

Einstein tensor in the `factor` sector of a direct product manifold.

For M = M₁ × M₂ × ⋯, the Einstein tensor acquires cross-scalar contributions:

    G_{ij} = G₁_{ij} - ½ (R₂ + R₃ + ⋯) g₁_{ij}

This arises because G = Ric - ½Rg and the total scalar curvature R = R₁ + R₂ + ⋯
is additive, but Ric and g are block-diagonal.
"""
function product_einstein(name::Symbol, factor::Symbol;
                           registry::TensorRegistry=current_registry())
    pp = get_product_manifold(registry, name)
    fi = _factor_index(pp, factor)
    fm = pp.factor_metrics[fi]
    ab = _factor_indices(registry, factor, 2)
    a, b = down(ab[1]), down(ab[2])

    ein = Tensor(_factor_curvature_name(:Ein, fm), [a, b])

    # Cross-scalar: -½ Σ_{j≠i} Rⱼ · gᵢ
    cross = TensorExpr[Tensor(_factor_curvature_name(:RicScalar, m), TIndex[])
                       for (j, m) in enumerate(pp.factor_metrics) if j != fi]

    isempty(cross) && return ein

    R_other = length(cross) == 1 ? cross[1] : tsum(cross)
    ein - (1 // 2) * R_other * Tensor(fm, [a, b])
end

"""
    product_einstein_equations(name; registry) -> Dict{Symbol, TensorExpr}

Einstein tensor decomposition for all sectors of a product manifold.
Returns a Dict mapping each factor name to its Einstein sector expression.

# Example
```julia
eqs = product_einstein_equations(:M)
eqs[:M1]  # Ein_g1_{ab} - ½ RicScalar_g2 · g1_{ab}
eqs[:S2]  # Ein_g2_{αβ} - ½ RicScalar_g1 · g2_{αβ}
```
"""
function product_einstein_equations(name::Symbol;
                                     registry::TensorRegistry=current_registry())
    pp = get_product_manifold(registry, name)
    Dict(f => product_einstein(name, f; registry) for f in pp.factors)
end
