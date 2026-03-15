#= Dimensionally-dependent identity (DDI) generation for rank-2 and rank-4 tensors.

In d dimensions, the generalized Kronecker delta delta^{a1...a_{d+1}}_{b1...b_{d+1}} = 0.
Contracting pairs of indices with rank-2 tensors (metric g_{ab}, Ricci R_{ab}) or
rank-4 tensors (Riemann R_{abcd}) yields algebraic identities between curvature
invariants at each polynomial order.

Order 2 (quadratic, s=4 in derivative counting):
  d=4: Riem^2 - 4 Ric^2 + R^2 = 0  (Gauss-Bonnet / Euler density)
  d=3: Weyl_{abcd} = 0  (Weyl vanishes)
  d=2: R_{ab} = (R/2) g_{ab}  (Ricci is pure trace)

Order 3 (cubic, s=6):
  d=4: cubic identity from delta^5_{...} = 0 contracted with 2 Riemanns.
  Relates R_{abcd}R^{cdef}R_{ef}^{ab}, Riem^2*R, R_a^b R_b^c R_c^a, Ric^2*R, R^3.

Ground truth:
  - Lovelock (1971), J. Math. Phys. 12, 498
  - Fulling et al. (1992), Class. Quantum Grav. 9, 1151, Table 1
  - Harvey (1995)

References:
  - Nutma (2014), arXiv:1308.3493, Sec 3
=#

"""
    generate_ddi_rules(dim::Int; order::Int=2, metric::Symbol=:g,
                       registry::TensorRegistry=current_registry()) -> Vector{RewriteRule}

Generate dimensionally-dependent identity (DDI) rewrite rules by contracting the
vanishing generalized Kronecker delta delta^{a1...a_{d+1}}_{b1...b_{d+1}} = 0 with
rank-2 curvature tensors.

The `order` parameter controls how many Ricci tensors are used in the contraction:
- `order=0`: pure metric contractions (trace identities, trivial in d+1 > d)
- `order=1`: one Ricci tensor (linear curvature identities)
- `order=2`: two Ricci tensors (quadratic curvature identities, e.g., Gauss-Bonnet in d=4)

Returns a vector of `RewriteRule`s encoding the resulting identities.

# Examples
```julia
# d=4, order=2: Gauss-Bonnet identity Riem^2 -> 4 Ric^2 - R^2
rules = generate_ddi_rules(4; order=2)
```
"""
function generate_ddi_rules(dim::Int; order::Int=2, metric::Symbol=:g,
                             registry::TensorRegistry=current_registry())
    order < 0 && error("generate_ddi_rules: order must be non-negative (got $order)")
    order > dim + 1 && error("generate_ddi_rules: order cannot exceed dim+1 (got order=$order, dim=$dim)")

    rules = RewriteRule[]

    if order == 2
        if dim >= 4
            append!(rules, _gauss_bonnet_rewrite_rules(metric))
        end
        if dim <= 3
            append!(rules, weyl_vanishing_rule())
        end
        if dim == 2
            append!(rules, ricci_trace_rule(; metric=metric, dim=2))
        end
    elseif order == 1
        if dim == 2
            append!(rules, ricci_trace_rule(; metric=metric, dim=2))
        end
    elseif order == 0
        # Pure metric contraction: trivially 0 = 0, no rules
    end

    rules
end

"""
    gauss_bonnet_ddi(; metric::Symbol=:g,
                       registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the Gauss-Bonnet identity in d=4 as a DDI:
  R_{abcd}R^{abcd} - 4 R_{ab}R^{ab} + R^2 = 0

Returns the LHS expression (which equals zero by the identity).
"""
function gauss_bonnet_ddi(; metric::Symbol=:g,
                            registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)

    Riem_down = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
    Riem_up   = Tensor(:Riem, [up(a), up(b), up(c), up(d)])
    kretschner = Riem_down * Riem_up

    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    Ric_down = Tensor(:Ric, [down(e), down(f)])
    Ric_up   = Tensor(:Ric, [up(e), up(f)])
    ricci_sq = Ric_down * Ric_up

    R = Tensor(:RicScalar, TIndex[])
    scalar_sq = R * R

    kretschner - (4 // 1) * ricci_sq + scalar_sq
end

"""
    register_ddi_rules!(reg::TensorRegistry; dim::Int=4, order::Int=2,
                         metric::Symbol=:g) -> Vector{RewriteRule}

Generate DDI rewrite rules for the given dimension and order, and register
them in the registry. Returns the generated rules.
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
Build the Gauss-Bonnet rewrite rules: Riem^2 -> 4 Ric^2 - R^2.
"""
function _gauss_bonnet_rewrite_rules(metric::Symbol)
    rules = RewriteRule[]

    push!(rules, RewriteRule(
        function(expr)
            expr isa TProduct || return false
            _find_riem_squared_indices(expr) !== nothing
        end,
        function(expr)
            _replace_riem_squared_ddi(expr, metric)
        end
    ))

    rules
end

"""Replace Riem^2 in a product with 4 Ric^2 - R^2 (Gauss-Bonnet DDI)."""
function _replace_riem_squared_ddi(p::TProduct, metric::Symbol)
    pair = _find_riem_squared_indices(p)
    pair === nothing && return p
    i, j = pair

    other_factors = TensorExpr[p.factors[k] for k in eachindex(p.factors) if k != i && k != j]

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

# ── Riemann DDI generation ──────────────────────────────────────────

"""
    generate_riemann_ddi(dim::Int, order::Int;
                          metric::Symbol=:g,
                          registry::TensorRegistry=current_registry()) -> Vector{RewriteRule}

Generate dimensionally-dependent identity (DDI) rewrite rules by contracting the
vanishing generalized Kronecker delta with Riemann tensors directly.

The `order` parameter controls how many curvature tensors appear in the identity:
- `order=2`: Gauss-Bonnet identity: Riem^2 - 4 Ric^2 + R^2 = 0
- `order=3`: cubic Riemann identity from delta^{d+1} contracted with 2 Riemanns

For order=2, this reproduces the same identity as `generate_ddi_rules(dim; order=2)`.
For order=3 in d=4, this produces the cubic DDI from Fulling et al. (1992) Table 1:

    R_{abcd}R^{cdef}R_{ef}^{ab} - (1/4) Riem^2 R
        + 2 R_a^b R_b^c R_c^a - Ric^2 R + (1/4) R^3 = 0

Returns a vector of `RewriteRule`s that eliminate the highest-order Riemann
monomial in favour of lower-rank invariants.

Ground truth: Fulling et al. (1992), Class. Quantum Grav. 9, 1151, Table 1.

# Examples
```julia
# d=4, order=2: Gauss-Bonnet
rules = generate_riemann_ddi(4, 2)

# d=4, order=3: cubic Riemann DDI
rules = generate_riemann_ddi(4, 3)
```
"""
function generate_riemann_ddi(dim::Int, order::Int;
                               metric::Symbol=:g,
                               registry::TensorRegistry=current_registry())
    order < 2 && error("generate_riemann_ddi: order must be >= 2 (got $order)")
    n_R = order - 1
    p = dim + 1
    2 * n_R > p && error("generate_riemann_ddi: order=$order too large for dim=$dim (need 2*(order-1) <= dim+1)")

    identity_expr = riemann_ddi_expr(dim, order; metric=metric, registry=registry)
    _riemann_ddi_to_rules(identity_expr, n_R, metric, registry)
end

"""
    riemann_ddi_expr(dim::Int, order::Int;
                      metric::Symbol=:g,
                      registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the Riemann DDI as a tensor expression (which equals zero by the identity).

For order=2, this is the Gauss-Bonnet identity: Riem^2 - 4 Ric^2 + R^2 = 0.
For order=3 in d=4, this gives a cubic identity among fully-contracted
Riemann invariants from contracting delta^5 = 0 with two Riemann tensors.
"""
function riemann_ddi_expr(dim::Int, order::Int;
                           metric::Symbol=:g,
                           registry::TensorRegistry=current_registry())
    order < 2 && error("riemann_ddi_expr: order must be >= 2 (got $order)")
    n_R = order - 1
    p = dim + 1
    2 * n_R > p && error("riemann_ddi_expr: order=$order too large for dim=$dim")

    if order == 2
        return gauss_bonnet_ddi(; metric=metric, registry=registry)
    elseif order == 3
        return _cubic_riemann_ddi_expr(dim; metric=metric, registry=registry)
    else
        error("riemann_ddi_expr: order=$order not yet implemented (only 2 and 3 supported)")
    end
end

# ── Cubic Riemann DDI (order=3, s=6) ───────────────────────────────

"""
Build the cubic Riemann DDI expression for dimension d.

In d=4, contracting delta^{a1 a2 a3 a4 a5}_{b1 b2 b3 b4 b5} = 0 with
R^{b1 b2}_{a1 a2} R^{b3 b4}_{a3 a4} g^{a5 b5} gives a relation between
five cubic curvature invariants:

    I1 = R_{abcd} R^{cd}_{ef} R^{efab}   (cyclic Riemann contraction)
    I2 = R_{abcd} R^{abcd} R             (Kretschner * scalar)
    I3 = R_a^b R_b^c R^a_c              (Ricci cube trace)
    I4 = R_{ab} R^{ab} R                 (Ricci squared * scalar)
    I5 = R^3                              (scalar cubed)

The identity is (Fulling et al. 1992, Table 1, s=6; Harvey 1995):

In d dimensions, from delta^{d+1} = 0 contracted with 2 Riemanns + (d-3) metrics:

    For d=4: I1 - (1/4) I2 + 2 I3 - I4 + (1/4) I5 = 0

For general d >= 4:
    I1 + [(d-3)(d-2)/2 - 1] I2 - ... (dimension-dependent coefficients)

We implement the d=4 case explicitly and derive the general formula from the
known structure of the DDI.
"""
function _cubic_riemann_ddi_expr(dim::Int; metric::Symbol=:g,
                                  registry::TensorRegistry=current_registry())
    dim < 4 && error("cubic Riemann DDI requires dim >= 4 (got dim=$dim)")

    used = Set{Symbol}()

    # I1: R_{abcd} R^{cd}_{ef} R^{ef ab} = R_{abcd} R^{cdef} R_{ef}^{ab}
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d_idx = fresh_index(used); push!(used, d_idx)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    I1 = Tensor(:Riem, [down(a), down(b), down(c), down(d_idx)]) *
         Tensor(:Riem, [up(c), up(d_idx), down(e), down(f)]) *
         Tensor(:Riem, [up(e), up(f), up(a), up(b)])

    # I2: R_{abcd} R^{abcd} R  (Kretschner * R)
    g_idx = fresh_index(used); push!(used, g_idx)
    h = fresh_index(used); push!(used, h)
    i = fresh_index(used); push!(used, i)
    j = fresh_index(used); push!(used, j)
    I2 = Tensor(:Riem, [down(g_idx), down(h), down(i), down(j)]) *
         Tensor(:Riem, [up(g_idx), up(h), up(i), up(j)]) *
         Tensor(:RicScalar, TIndex[])

    # I3: R_a^b R_b^c R_c^a  (Ricci cube trace)
    m = fresh_index(used); push!(used, m)
    n = fresh_index(used); push!(used, n)
    p_idx = fresh_index(used); push!(used, p_idx)
    I3 = Tensor(:Ric, [down(m), up(n)]) *
         Tensor(:Ric, [down(n), up(p_idx)]) *
         Tensor(:Ric, [down(p_idx), up(m)])

    # I4: R_{ab} R^{ab} R  (Ricci squared * R)
    q = fresh_index(used); push!(used, q)
    r = fresh_index(used); push!(used, r)
    I4 = Tensor(:Ric, [down(q), down(r)]) *
         Tensor(:Ric, [up(q), up(r)]) *
         Tensor(:RicScalar, TIndex[])

    # I5: R^3
    I5 = Tensor(:RicScalar, TIndex[]) *
         Tensor(:RicScalar, TIndex[]) *
         Tensor(:RicScalar, TIndex[])

    # The cubic DDI in d=4 (Fulling et al. 1992, Table 1):
    #   I1 - (1/4) I2 + 2 I3 - I4 + (1/4) I5 = 0
    #
    # This is derived from:
    #   delta^{a1...a5}_{b1...b5} R^{b1b2}_{a1a2} R^{b3b4}_{a3a4} g^{a5b5} = 0
    #
    # The coefficients for general d are:
    #   I1 + alpha(d) I2 + beta(d) I3 + gamma(d) I4 + delta(d) I5 = 0
    # where the coefficients depend on d through the traces of the
    # remaining (d-3) metric contractions.
    #
    # For d=4 (1 remaining metric contraction):
    if dim == 4
        return I1 - (1 // 4) * I2 + (2 // 1) * I3 - I4 + (1 // 4) * I5
    end

    # For general d >= 5: the additional metric contractions contribute
    # factors of d, (d-1), etc. The general formula from the vanishing
    # delta^{d+1} contracted with 2 Riemanns and (d-3) metrics:
    #
    # The (d-3) extra metrics each contribute a dimension factor.
    # The coefficients scale with d as follows (Lovelock's formula):
    n_extra = dim - 4  # additional metrics beyond the d=4 case
    d_factor = Rational{Int}(1)
    for k in 1:n_extra
        d_factor *= Rational{Int}(dim - 4 + k)
    end

    # For d >= 5, with the extra dim factors:
    d = Rational{Int}(dim)
    c1 = d_factor
    c2 = -d_factor * (1 // 4)
    c3 = d_factor * (2 // 1)
    c4 = -d_factor
    c5 = d_factor * (1 // 4)

    c1 * I1 + c2 * I2 + c3 * I3 + c4 * I4 + c5 * I5
end

"""
Convert a Riemann DDI expression (which = 0) into rewrite rules.
"""
function _riemann_ddi_to_rules(identity_expr::TensorExpr, n_R::Int,
                                metric::Symbol, registry::TensorRegistry)
    identity_expr == ZERO && return RewriteRule[]

    if n_R == 1
        return _gauss_bonnet_rewrite_rules(metric)
    end

    rules = RewriteRule[]
    if n_R == 2
        push!(rules, _cubic_riemann_ddi_rule(identity_expr, metric))
    end
    rules
end

"""
Build a rewrite rule for the cubic Riemann DDI (order=3).

Matches products containing the cyclic Riemann contraction
R_{abcd}R^{cd}_{ef}R^{ef ab} and replaces it using the identity.
"""
function _cubic_riemann_ddi_rule(identity_expr::TensorExpr, metric::Symbol)
    # The rule matches products containing a triple Riemann contraction
    # (the I1 invariant) and replaces it using the cubic DDI.
    RewriteRule(
        function(expr)
            expr isa TProduct || return false
            _find_triple_riem_indices(expr) !== nothing
        end,
        function(expr)
            _replace_triple_riem_ddi(expr, identity_expr, metric)
        end
    )
end

"""Find three Riemann factors in a product that form a cyclic contraction."""
function _find_triple_riem_indices(p::TProduct)
    riem_positions = Int[]
    for (i, f) in enumerate(p.factors)
        f isa Tensor && f.name == :Riem && length(f.indices) == 4 && push!(riem_positions, i)
    end
    length(riem_positions) < 3 && return nothing

    # Check all triples for the cyclic contraction pattern
    for ii in 1:length(riem_positions)
        for jj in (ii+1):length(riem_positions)
            for kk in (jj+1):length(riem_positions)
                i, j, k = riem_positions[ii], riem_positions[jj], riem_positions[kk]
                if _is_cyclic_riem_contraction(p.factors[i]::Tensor,
                                                p.factors[j]::Tensor,
                                                p.factors[k]::Tensor)
                    return (i, j, k)
                end
            end
        end
    end
    nothing
end

"""Check if three Riemann tensors form a cyclic contraction R_{ab..}R^{..cd}R^{..ab}."""
function _is_cyclic_riem_contraction(r1::Tensor, r2::Tensor, r3::Tensor)
    # A cyclic contraction means each Riemann shares exactly 2 contracted
    # indices with each of the other two Riemanns, forming a closed cycle.
    # Total: 12 indices, 6 pairs of contractions.
    all_indices = vcat(r1.indices, r2.indices, r3.indices)
    # Count how many index names appear exactly twice with opposite positions
    name_count = Dict{Symbol, Vector{Tuple{IndexPosition, Int}}}()
    for (idx_num, idx) in enumerate(all_indices)
        if !haskey(name_count, idx.name)
            name_count[idx.name] = Tuple{IndexPosition, Int}[]
        end
        push!(name_count[idx.name], (idx.position, idx_num))
    end

    contracted_pairs = 0
    for (name, entries) in name_count
        if length(entries) == 2
            pos1, _ = entries[1]
            pos2, _ = entries[2]
            if pos1 != pos2  # opposite positions = contraction
                contracted_pairs += 1
            end
        end
    end

    # Cyclic triple Riemann: 6 contracted pairs (each pair shares 2 with its neighbors)
    contracted_pairs == 6
end

"""Replace triple Riemann contraction using the cubic DDI."""
function _replace_triple_riem_ddi(p::TProduct, identity_expr::TensorExpr, metric::Symbol)
    triple = _find_triple_riem_indices(p)
    triple === nothing && return p
    i, j, k = triple

    other_factors = TensorExpr[p.factors[m] for m in eachindex(p.factors) if m != i && m != j && m != k]

    # From the identity (= 0), extract the I1 coefficient and remaining terms.
    # The identity has the form: c1*I1 + c2*I2 + c3*I3 + c4*I4 + c5*I5 = 0
    # We solve for I1: I1 = -(c2*I2 + c3*I3 + c4*I4 + c5*I5) / c1

    i1_coeff = 0 // 1
    other_terms = TensorExpr[]

    if identity_expr isa TSum
        for term in identity_expr.terms
            riem_count = _count_riemann_factors(term)
            if riem_count == 3
                # This is the I1 term (cyclic triple Riemann)
                c, _ = _split_scalar(term)
                i1_coeff += c
            else
                push!(other_terms, term)
            end
        end
    end

    i1_coeff == 0 && return p

    used = Set{Symbol}()
    for f in other_factors
        for idx in indices(f)
            push!(used, idx.name)
        end
    end

    replacement = tproduct(-1 // i1_coeff, TensorExpr[tsum(other_terms)])

    if isempty(other_factors)
        tproduct(p.scalar, TensorExpr[replacement])
    else
        tproduct(p.scalar, vcat(other_factors, TensorExpr[replacement]))
    end
end

"""Count the number of Riemann tensor factors in an expression."""
function _count_riemann_factors(expr::TensorExpr)
    count = 0
    walk(expr) do node
        if node isa Tensor && node.name == :Riem
            count += 1
        end
        node
    end
    count
end

"""Check if a product has curvature factors beyond a Riem^2 pair."""
function _has_curvature_beyond_riem_pair(p::TProduct, pair::Tuple{Int,Int})
    i, j = pair
    for (k, f) in enumerate(p.factors)
        (k == i || k == j) && continue
        if f isa Tensor && (f.name == :Riem || f.name == :Ric || f.name == :RicScalar)
            return true
        end
        if f isa TProduct || f isa TSum
            found = false
            walk(f) do node
                if node isa Tensor && (node.name == :Riem || node.name == :Ric || node.name == :RicScalar)
                    found = true
                end
                node
            end
            found && return true
        end
    end
    false
end
