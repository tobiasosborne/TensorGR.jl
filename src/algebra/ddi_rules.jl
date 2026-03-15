#= Dimensionally-dependent identity (DDI) generation for rank-2 tensors.

In d dimensions, the generalized Kronecker delta δ^{a₁...a_{d+1}}_{b₁...b_{d+1}} = 0.
Contracting pairs of indices with rank-2 tensors (metric g_{ab}, Ricci R_{ab}) yields
algebraic identities between curvature invariants.

The key DDIs at quadratic order (order=2) are:

  d=4: Riem² - 4 Ric² + R² = 0  (Gauss-Bonnet / Euler density)
  d=3: Weyl_{abcd} = 0  (Weyl vanishes)
  d=2: R_{ab} = (R/2) g_{ab}  (Ricci is pure trace)

The algorithm avoids brute-force expansion of the generalized delta (which has
(d+1)! terms) by using known analytical results for the contracted identities.

Ground truth:
  - Lovelock (1971), J. Math. Phys. 12, 498
  - Fulling et al. (1992), Class. Quantum Grav. 9, 1151, Sec 6

References:
  - Harvey (1995)
  - Nutma (2014), arXiv:1308.3493, Sec 3
=#

"""
    generate_ddi_rules(dim::Int; order::Int=2, metric::Symbol=:g,
                       registry::TensorRegistry=current_registry()) -> Vector{RewriteRule}

Generate dimensionally-dependent identity (DDI) rewrite rules by contracting the
vanishing generalized Kronecker delta δ^{a₁...a_{d+1}}_{b₁...b_{d+1}} = 0 with
rank-2 curvature tensors.

The `order` parameter controls how many Ricci tensors are used in the contraction:
- `order=0`: pure metric contractions (trace identities, trivial in d+1 > d)
- `order=1`: one Ricci tensor (linear curvature identities)
- `order=2`: two Ricci tensors (quadratic curvature identities, e.g., Gauss-Bonnet in d=4)

Returns a vector of `RewriteRule`s encoding the resulting identities.

# Examples
```julia
# d=4, order=2: Gauss-Bonnet identity Riem² → 4 Ric² - R²
rules = generate_ddi_rules(4; order=2)
```
"""
function generate_ddi_rules(dim::Int; order::Int=2, metric::Symbol=:g,
                             registry::TensorRegistry=current_registry())
    order < 0 && error("generate_ddi_rules: order must be non-negative (got $order)")
    order > dim + 1 && error("generate_ddi_rules: order cannot exceed dim+1 (got order=$order, dim=$dim)")

    rules = RewriteRule[]

    if order == 2
        # Quadratic DDI: the contracted delta^{d+1} with 2 Ricci and (d-1) metrics
        # yields the Lovelock identity:
        #   Riem² - 4 Ric² + R² = 0  (in d=4)
        # More generally, for any d >= 4, this gives the Gauss-Bonnet identity
        # which allows eliminating Riem² in favour of Ric² and R².
        if dim >= 4
            append!(rules, _gauss_bonnet_rewrite_rules(metric))
        end
        if dim <= 3
            # In d<=3, the Weyl tensor vanishes (stronger statement)
            append!(rules, weyl_vanishing_rule())
        end
        if dim == 2
            # In d=2, Ricci is pure trace
            append!(rules, ricci_trace_rule(; metric=metric, dim=2))
        end
    elseif order == 1
        # Linear DDI: contracted delta^{d+1} with 1 Ricci and d metrics
        if dim == 2
            append!(rules, ricci_trace_rule(; metric=metric, dim=2))
        end
        # In d>=3, the order=1 DDI is trivially satisfied (no new information)
    elseif order == 0
        # Pure metric contraction: trivially 0 = 0, no rules
    end

    rules
end

"""
    gauss_bonnet_ddi(; metric::Symbol=:g,
                       registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the Gauss-Bonnet identity in d=4 as a DDI:
  R_{abcd}R^{abcd} - 4 R_{ab}R^{ab} + R² = 0

Returns the LHS expression (which equals zero by the identity).
This is derived by contracting δ^{abcde}_{fghij} = 0 with two Ricci tensors
and three metrics.
"""
function gauss_bonnet_ddi(; metric::Symbol=:g,
                            registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)

    # Kretschner: R_{abcd}R^{abcd}
    Riem_down = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
    Riem_up   = Tensor(:Riem, [up(a), up(b), up(c), up(d)])
    kretschner = Riem_down * Riem_up

    # Ricci squared: R_{ab}R^{ab}
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    Ric_down = Tensor(:Ric, [down(e), down(f)])
    Ric_up   = Tensor(:Ric, [up(e), up(f)])
    ricci_sq = Ric_down * Ric_up

    # Scalar squared: R²
    R = Tensor(:RicScalar, TIndex[])
    scalar_sq = R * R

    # GB identity: Riem² - 4 Ric² + R² = 0
    kretschner - (4 // 1) * ricci_sq + scalar_sq
end

"""
    register_ddi_rules!(reg::TensorRegistry; dim::Int=4, order::Int=2,
                         metric::Symbol=:g) -> Vector{RewriteRule}

Generate DDI rewrite rules for the given dimension and order, and register
them in the registry. Returns the generated rules.

# Arguments
- `reg`: the TensorRegistry to modify
- `dim`: manifold dimension (default 4)
- `order`: number of Ricci tensors in the contraction (default 2)
- `metric`: metric tensor name (default `:g`)

# Examples
```julia
reg = TensorRegistry()
# ... set up manifold, metric, curvature tensors ...
register_ddi_rules!(reg; dim=4, order=2)
```
"""
function register_ddi_rules!(reg::TensorRegistry; dim::Int=4, order::Int=2,
                              metric::Symbol=:g)
    rules = with_registry(reg) do
        generate_ddi_rules(dim; order=order, metric=metric, registry=reg)
    end
    for r in rules
        register_rule!(reg, r)
    end
    rules
end

# ── Internal helpers ──────────────────────────────────────────────────

"""
Build the Gauss-Bonnet rewrite rules: Riem² → 4 Ric² - R².

This is the d=4 DDI at order 2. The rule matches any `TProduct` containing
`Riem_{abcd} * Riem^{abcd}` (fully contracted Riemann squared) and replaces
the Riemann pair with `4 Ric_{ab} Ric^{ab} - R²`.

This is the same identity as in `syzygies.jl`, generated here systematically
from the DDI framework. Uses structural matching to avoid infinite recursion.
"""
function _gauss_bonnet_rewrite_rules(metric::Symbol)
    rules = RewriteRule[]

    push!(rules, RewriteRule(
        function(expr)
            # Match a product containing Riem_{...} * Riem^{...} fully contracted
            expr isa TProduct || return false
            _find_riem_squared_indices(expr) !== nothing
        end,
        function(expr)
            _replace_riem_squared_ddi(expr, metric)
        end
    ))

    rules
end

"""Replace Riem² in a product with 4 Ric² - R² (Gauss-Bonnet DDI)."""
function _replace_riem_squared_ddi(p::TProduct, metric::Symbol)
    pair = _find_riem_squared_indices(p)
    pair === nothing && return p
    i, j = pair

    # Collect remaining factors
    other_factors = TensorExpr[p.factors[k] for k in eachindex(p.factors) if k != i && k != j]

    # Build replacement: 4 Ric_{ab} Ric^{ab} - R²
    used = Set{Symbol}()
    for f in other_factors
        for idx in indices(f)
            push!(used, idx.name)
        end
    end
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)

    Ric_down = Tensor(:Ric, [down(e), down(f)])
    Ric_up = Tensor(:Ric, [up(e), up(f)])
    ricci_sq = Ric_down * Ric_up

    R = Tensor(:RicScalar, TIndex[])
    scalar_sq = R * R

    replacement = (4 // 1) * ricci_sq - scalar_sq

    if isempty(other_factors)
        tproduct(p.scalar, TensorExpr[replacement])
    else
        tproduct(p.scalar, vcat(other_factors, TensorExpr[replacement]))
    end
end
