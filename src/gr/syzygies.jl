#= Order-2 curvature syzygies as rewrite rules.

Curvature syzygies are algebraic identities between products of curvature
tensors that hold in specific dimensions. They complement Bianchi identities
(which are linear in curvature) with quadratic relations.

1. Gauss-Bonnet / Lanczos-Lovelock identity (D=4):
   Riem² - 4 Ric² + R² = E₄  (Euler density, topological)
   → In the action (modulo boundary): Riem² = 4 Ric² - R² + E₄
   → With E₄ set to zero: Riem² → 4 Ric² - R²

2. Dimensionally-dependent identities (D≤3):
   - D≤3: Weyl_{abcd} = 0 (Weyl tensor vanishes identically)
   - D≤2: Ric_{ab} = (R/D) g_{ab} (Ricci is pure trace)
   - D=1: Riem_{abcd} = 0 (all curvature vanishes)
=#

"""
    gauss_bonnet_rule(; metric=:g, drop_euler=true) -> Vector{RewriteRule}

Create rewrite rules implementing the Gauss-Bonnet identity in 4D:
  Riem² = 4 Ric² - R² + E₄

When `drop_euler=true` (default), the Euler density is dropped (valid for
equations of motion / in the action up to a total derivative), giving:
  Riem² → 4 Ric² - R²

This eliminates Kretschner scalar (Riem²) in favour of simpler invariants.

The rule matches any `TProduct` containing `Riem_{abcd} * Riem^{abcd}`
(fully contracted Riemann squared) and replaces it.
"""
function gauss_bonnet_rule(; metric::Symbol=:g, drop_euler::Bool=true)
    rules = RewriteRule[]

    push!(rules, RewriteRule(
        function(expr)
            # Match a product containing Riem_{...} * Riem^{...} fully contracted
            expr isa TProduct || return false
            riem_indices = _find_riem_squared_indices(expr)
            riem_indices !== nothing
        end,
        function(expr)
            result = _replace_riem_squared(expr, metric, drop_euler)
            result
        end
    ))

    rules
end

"""Find two Riemann factors in a product that are fully contracted with each other."""
function _find_riem_squared_indices(p::TProduct)
    riem_positions = Int[]
    for (i, f) in enumerate(p.factors)
        f isa Tensor && f.name == :Riem && length(f.indices) == 4 && push!(riem_positions, i)
    end
    length(riem_positions) < 2 && return nothing

    # Check all pairs for full contraction
    for ii in 1:length(riem_positions)
        for jj in (ii+1):length(riem_positions)
            i, j = riem_positions[ii], riem_positions[jj]
            r1 = p.factors[i]::Tensor
            r2 = p.factors[j]::Tensor
            if _are_fully_contracted(r1.indices, r2.indices)
                return (i, j)
            end
        end
    end
    nothing
end

"""Check if two sets of 4 indices are fully contracted (each index in set 1
has a partner in set 2 with same name, opposite position)."""
function _are_fully_contracted(idxs1::Vector{TIndex}, idxs2::Vector{TIndex})
    length(idxs1) == 4 && length(idxs2) == 4 || return false
    matched = falses(4)
    for i1 in idxs1
        found = false
        for (k, i2) in enumerate(idxs2)
            if !matched[k] && i1.name == i2.name && i1.position != i2.position
                matched[k] = true
                found = true
                break
            end
        end
        found || return false
    end
    true
end

"""Replace Riem² in a product with 4 Ric² - R² (+ E₄ if not dropped)."""
function _replace_riem_squared(p::TProduct, metric::Symbol, drop_euler::Bool)
    pair = _find_riem_squared_indices(p)
    pair === nothing && return p
    i, j = pair

    # Collect remaining factors (everything except the two Riemann tensors)
    other_factors = TensorExpr[p.factors[k] for k in eachindex(p.factors) if k != i && k != j]

    # Build the replacement: 4 Ric_{ab} Ric^{ab} - R²
    used = Set{Symbol}()
    # Gather index names from other factors to avoid clashes
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

    # Riem² = 4 Ric² - R² (dropping Euler density)
    replacement = (4 // 1) * ricci_sq - scalar_sq

    if isempty(other_factors)
        tproduct(p.scalar, TensorExpr[replacement])
    else
        tproduct(p.scalar, vcat(other_factors, TensorExpr[replacement]))
    end
end

# ─── Dimensionally-dependent identities ──────────────────────────────

"""
    weyl_vanishing_rule() -> Vector{RewriteRule}

In D≤3, the Weyl tensor vanishes identically. This rule sets any
`Weyl_{abcd}` (bare or in a product) to zero.

Register when working on a manifold with dim ≤ 3.
"""
function weyl_vanishing_rule()
    rules = RewriteRule[]
    push!(rules, RewriteRule(
        function(expr)
            expr isa Tensor && expr.name == :Weyl && length(expr.indices) == 4
        end,
        _ -> ZERO
    ))
    rules
end

"""
    ricci_trace_rule(; metric=:g, dim=2) -> Vector{RewriteRule}

In D=2, the Ricci tensor is pure trace: R_{ab} = (R/2) g_{ab}.
In D=1, all curvature vanishes (handled separately by `riemann_vanishing_rule`).

This creates a pattern-based rule using pattern indices.
"""
function ricci_trace_rule(; metric::Symbol=:g, dim::Int=2)
    dim == 2 || error("ricci_trace_rule is for dim=2; got dim=$dim")
    rules = RewriteRule[]
    # R_{a_,b_} → (R/2) g_{a_,b_}
    pat = Tensor(:Ric, [down(:a_), down(:b_)])
    rep = (1 // dim) * Tensor(metric, [down(:a_), down(:b_)]) * Tensor(:RicScalar, TIndex[])
    push!(rules, RewriteRule(pat, rep))
    rules
end

"""
    riemann_vanishing_rule() -> Vector{RewriteRule}

In D=1, the Riemann tensor vanishes identically.
"""
function riemann_vanishing_rule()
    rules = RewriteRule[]
    push!(rules, RewriteRule(
        function(expr)
            expr isa Tensor && expr.name == :Riem && length(expr.indices) == 4
        end,
        _ -> ZERO
    ))
    rules
end

"""
    syzygy_rules(; manifold=:M4, metric=:g, dim=4, drop_euler=true,
                   registry=nothing) -> Vector{RewriteRule}

Convenience function: return all applicable curvature syzygies for the
given dimension.

- D=4: Gauss-Bonnet (Riem² → 4 Ric² - R²)
- D≤3: Weyl = 0, plus Gauss-Bonnet
- D=2: additionally Ric_{ab} = (R/2) g_{ab}
- D=1: Riem = 0 (all curvature vanishes)
"""
function syzygy_rules(; manifold::Symbol=:M4, metric::Symbol=:g,
                        dim::Int=4, drop_euler::Bool=true,
                        registry::Union{TensorRegistry, Nothing}=nothing)
    rules = RewriteRule[]

    if dim == 1
        append!(rules, riemann_vanishing_rule())
        return rules
    end

    if dim == 2
        append!(rules, ricci_trace_rule(; metric=metric, dim=2))
    end

    if dim <= 3
        append!(rules, weyl_vanishing_rule())
    end

    if dim >= 4
        append!(rules, gauss_bonnet_rule(; metric=metric, drop_euler=drop_euler))
    end

    rules
end
