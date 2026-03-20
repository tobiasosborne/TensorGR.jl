#= Invar six-level simplification algorithm -- Level 1.
#
# Level 1: Permutation symmetries for Riemann monomials.
# Ensures canonical ordering of indices in products of curvature tensors
# using the existing Butler-Portugal (xperm) canonicalization.
#
# The six levels are (Garcia-Parrado & Martin-Garcia 2007, Sec 4):
#   1. Permutation symmetries (Riemann pair-swap, antisymmetry)
#   2. Cyclic symmetry (first Bianchi identity)
#   3. Dimensionally-dependent identities (DDIs)
#   4. Product symmetries (commutativity of tensor products)
#   5. Renaming symmetries (dummy index relabeling)
#   6. Schouten/Lovelock identities
#
# Level 1 is handled by the existing canonicalize() via xperm.c,
# which already knows about Riemann symmetries. This module provides
# a thin wrapper and targeted verification tests.
#
# Ground truth: Garcia-Parrado & Martin-Garcia, Comp. Phys. Comm.
#              176 (2007) 246, Section 4, Level 1.
=#

"""
    simplify_level1(expr::TensorExpr; registry::TensorRegistry=current_registry())
        -> TensorExpr

Apply Level 1 (permutation symmetries) of the Invar simplification algorithm.

This delegates to the existing `canonicalize()` function, which uses
Butler-Portugal (xperm.c) to find the canonical form under:
- Riemann pair symmetry: R_{abcd} = R_{cdab}
- Riemann antisymmetry: R_{abcd} = -R_{bacd} = -R_{abdc}
- Ricci symmetry: R_{ab} = R_{ba}
- Metric symmetry: g_{ab} = g_{ba}
- Products: reorder factors by canonical index pattern

For Riemann monomials (products of n Riemann tensors with contracted
indices), this is the first and most basic simplification step.

Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4, Level 1.
"""
function simplify_level1(expr::TensorExpr;
                          registry::TensorRegistry=current_registry())
    with_registry(registry) do
        canonicalize(expr)
    end
end

"""
    is_riemann_monomial(expr::TensorExpr) -> Bool

Check if an expression is a scalar Riemann monomial: a product of
curvature tensors (Riemann, Ricci, RicScalar, Weyl) with all indices
contracted (no free indices).

This is the class of expressions that the Invar algorithm operates on.
"""
function is_riemann_monomial(expr::TensorExpr)
    # Must have no free indices (scalar)
    !isempty(free_indices(expr)) && return false

    # Must be a product or single curvature tensor
    _is_curvature_product(expr)
end

function _is_curvature_product(expr::Tensor)
    expr.name in (:Riem, :Ric, :RicScalar, :Weyl, :Ein, :Schouten) ||
        expr.name in (:g, :delta)  # metrics/deltas allowed in contractions
end

_is_curvature_product(::TScalar) = true  # scalar coefficients ok

function _is_curvature_product(p::TProduct)
    all(_is_curvature_product, p.factors)
end

function _is_curvature_product(s::TSum)
    all(_is_curvature_product, s.terms)
end

_is_curvature_product(::TDeriv) = false  # derivatives not in Invar scope
_is_curvature_product(::TensorExpr) = false

"""
    count_riemann_degree(expr::TensorExpr) -> Int

Count the number of Riemann/Weyl tensor factors in a monomial.
Ricci counts as 1 (trace of Riemann). RicScalar counts as 1.

This is the "degree" of the invariant in the Invar classification.
"""
function count_riemann_degree(expr::Tensor)
    expr.name in (:Riem, :Ric, :RicScalar, :Weyl, :Ein, :Schouten) ? 1 : 0
end

count_riemann_degree(::TScalar) = 0

function count_riemann_degree(p::TProduct)
    sum(count_riemann_degree(f) for f in p.factors; init=0)
end

function count_riemann_degree(s::TSum)
    isempty(s.terms) ? 0 : maximum(count_riemann_degree(t) for t in s.terms)
end

count_riemann_degree(d::TDeriv) = count_riemann_degree(d.arg)

# ────────────────────────────────────────────────────────────────────
# Level 2: Cyclic symmetry (first Bianchi identity)
# ────────────────────────────────────────────────────────────────────

#= The first Bianchi identity: R_{a[bcd]} = 0, equivalently:
#    R_{abcd} + R_{acdb} + R_{adbc} = 0
#
# For a product of n Riemann tensors, applying this identity to any
# factor generates a linear relation among monomials. Collecting all
# such relations and reducing yields the independent basis.
#
# At degree 2 in d=4: 6 naive contractions reduce to 3 independent
# invariants (R², R_{ab}R^{ab}, R_{abcd}R^{abcd}).
#
# Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4.1.
=#

"""
    apply_bianchi_cyclic(expr::TensorExpr, factor_idx::Int;
                          registry::TensorRegistry=current_registry())
        -> TensorExpr

Apply the first Bianchi identity R_{abcd} + R_{acdb} + R_{adbc} = 0
to the `factor_idx`-th Riemann factor in a product.

Returns the expression with the cycled terms subtracted:
    expr |_{R_{abcd}} -> -expr|_{R_{acdb}} - expr|_{R_{adbc}}

The result is then Level-1 canonicalized.
"""
function apply_bianchi_cyclic(expr::TensorExpr, factor_idx::Int;
                               registry::TensorRegistry=current_registry())
    expr isa TProduct || return expr

    factors = collect(expr.factors)
    (1 <= factor_idx <= length(factors)) ||
        error("factor_idx $factor_idx out of range [1, $(length(factors))]")

    riem = factors[factor_idx]
    riem isa Tensor && riem.name == :Riem || return expr

    idxs = riem.indices
    length(idxs) == 4 || return expr

    a, b, c, d = idxs

    # Original: R_{abcd}
    # Bianchi: R_{abcd} = -R_{acdb} - R_{adbc}
    riem_cycled1 = Tensor(:Riem, [a, c, d, b])
    riem_cycled2 = Tensor(:Riem, [a, d, b, c])

    other_factors = TensorExpr[factors[i] for i in 1:length(factors) if i != factor_idx]

    term1 = tproduct(-expr.scalar, TensorExpr[vcat(other_factors, [riem_cycled1])...])
    term2 = tproduct(-expr.scalar, TensorExpr[vcat(other_factors, [riem_cycled2])...])

    result = tsum(TensorExpr[term1, term2])

    # Level-1 canonicalize
    with_registry(registry) do
        canonicalize(result)
    end
end

"""
    simplify_level2(expr::TensorExpr;
                     registry::TensorRegistry=current_registry()) -> TensorExpr

Apply Level 2 (cyclic symmetry / first Bianchi identity) of the Invar
simplification algorithm.

For each Riemann factor in each term, applies the Bianchi identity
and collects terms. This may reduce the number of independent invariants.

At degree 2 in d=4: reduces from 6 naive to 3 independent invariants.

Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4.1, Level 2.
"""
function simplify_level2(expr::TensorExpr;
                          registry::TensorRegistry=current_registry())
    # First apply Level 1
    expr1 = simplify_level1(expr; registry=registry)

    # Then simplify via the full pipeline which includes Bianchi rules
    with_registry(registry) do
        simplify(expr1; registry=registry)
    end
end

"""
    bianchi_relation(a::TIndex, b::TIndex, c::TIndex, d::TIndex) -> TensorExpr

Construct the first Bianchi identity as an expression:

    R_{abcd} + R_{acdb} + R_{adbc} = 0

Returns the LHS (which should simplify to zero).
"""
function bianchi_relation(a::TIndex, b::TIndex, c::TIndex, d::TIndex)
    R1 = Tensor(:Riem, [a, b, c, d])
    R2 = Tensor(:Riem, [a, c, d, b])
    R3 = Tensor(:Riem, [a, d, b, c])
    tsum(TensorExpr[R1, R2, R3])
end

# ────────────────────────────────────────────────────────────────────
# Level 3: Second (differential) Bianchi identity
# ────────────────────────────────────────────────────────────────────

#= The second Bianchi identity: nabla_{[a} R_{bc]de} = 0
#
#   nabla_a R_{bcde} + nabla_b R_{cade} + nabla_c R_{abde} = 0
#
# Contracted form:
#   nabla_a R^{abcd} = nabla^b R^{cd} - nabla^c R^{bd}
#
# Or equivalently (contracted once more):
#   nabla^a R_{ab} = (1/2) nabla_b R
#   nabla^a G_{ab} = 0
#
# Level 3 applies this to differential invariants (those containing
# covariant derivatives of curvature tensors).
#
# Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4.2.
=#

"""
    differential_bianchi(a::TIndex, b::TIndex, c::TIndex, d::TIndex, e::TIndex;
                         covd::Symbol=:D) -> TensorExpr

Construct the second Bianchi identity as an expression:

    nabla_a R_{bcde} + nabla_b R_{cade} + nabla_c R_{abde} = 0

Returns the LHS (which should simplify to zero).
"""
function differential_bianchi(a::TIndex, b::TIndex, c::TIndex, d::TIndex, e::TIndex;
                              covd::Symbol=:D)
    R1 = TDeriv(a, Tensor(:Riem, [b, c, d, e]), covd)
    R2 = TDeriv(b, Tensor(:Riem, [c, a, d, e]), covd)
    R3 = TDeriv(c, Tensor(:Riem, [a, b, d, e]), covd)
    tsum(TensorExpr[R1, R2, R3])
end

"""
    contracted_bianchi(; covd::Symbol=:D) -> TensorExpr

Construct the contracted second Bianchi identity:

    nabla_a R^{abcd} - nabla^b R^{cd} + nabla^c R^{bd} = 0

Returns the LHS (which should simplify to zero).
Uses fresh indices to avoid collisions.
"""
function contracted_bianchi(; covd::Symbol=:D)
    # nabla_a R^{abcd} - nabla^b R^{cd} + nabla^c R^{bd} = 0
    term1 = TDeriv(down(:a), Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)]), covd)
    term2 = tproduct(-1 // 1, TensorExpr[
        TDeriv(up(:b), Tensor(:Ric, [up(:c), up(:d)]), covd)
    ])
    term3 = TDeriv(up(:c), Tensor(:Ric, [up(:b), up(:d)]), covd)

    tsum(TensorExpr[term1, term2, term3])
end

"""
    simplify_level3(expr::TensorExpr;
                     covd::Symbol=:D,
                     registry::TensorRegistry=current_registry()) -> TensorExpr

Apply Level 3 (second Bianchi identity) of the Invar simplification algorithm.

For differential invariants (expressions containing nabla_a R_{bcde} terms),
applies the contracted second Bianchi identity to replace divergences of
the Riemann tensor with derivatives of the Ricci tensor.

Specifically, the rule nabla^a R_{abcd} = nabla_b R_{cd} - nabla_c R_{bd}
is applied via `commute_covds` and the existing Bianchi rules in the registry.

This includes Level 1 and Level 2 as prerequisites.

Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4.2, Level 3.
"""
function simplify_level3(expr::TensorExpr;
                          covd::Symbol=:D,
                          registry::TensorRegistry=current_registry())
    # First apply Level 2
    expr2 = simplify_level2(expr; registry=registry)

    # Then apply the full simplify pipeline with covd commutation
    # which integrates the second Bianchi identity via existing rules
    with_registry(registry) do
        simplify(expr2; registry=registry, commute_covds_name=covd)
    end
end

# ────────────────────────────────────────────────────────────────────
# Level 4: Derivative commutation for differential invariants
# ────────────────────────────────────────────────────────────────────

#= Covariant derivative commutation: [nabla_a, nabla_b] T_{c...} = Riemann terms.
#
# For a (0,r) tensor T_{c1...cr}:
#   (nabla_a nabla_b - nabla_b nabla_a) T_{c1...cr}
#     = - sum_i R^d_{c_i a b} T_{c1...d...cr}
#
# Level 4 uses this to canonically order covariant derivatives acting
# on curvature tensors, generating additional Riemann terms from the
# commutation. This is the derivative analogue of Level 1 (permutation
# symmetries for undifferentiated monomials).
#
# The existing commute_covds() infrastructure (src/gr/sort_covds.jl)
# already handles the commutation. Level 4 wraps it specifically for
# differential invariants.
#
# Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4.3, Level 4.
=#

"""
    simplify_level4(expr::TensorExpr;
                     covd::Symbol=:D,
                     registry::TensorRegistry=current_registry()) -> TensorExpr

Apply Level 4 (derivative commutation) of the Invar simplification algorithm.

For differential invariants containing `nabla_a nabla_b R_{cdef}`, commute
the covariant derivatives to canonical order, generating Riemann commutator
terms via `[nabla_a, nabla_b] T = -R^d_{c ab} T_{...d...}`.

This includes Levels 1-3 as prerequisites, then applies `commute_covds`
to sort derivative indices and `simplify` to collect terms.

Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4.3, Level 4.
"""
function simplify_level4(expr::TensorExpr;
                          covd::Symbol=:D,
                          registry::TensorRegistry=current_registry())
    # First apply Level 3 (includes Levels 1-2)
    expr3 = simplify_level3(expr; covd=covd, registry=registry)

    # Apply derivative commutation to sort covds, generating Riemann terms
    with_registry(registry) do
        commuted = commute_covds(expr3, covd; registry=registry)
        # Final simplify pass to collect terms from commutation
        simplify(commuted; registry=registry, commute_covds_name=covd)
    end
end

# ────────────────────────────────────────────────────────────────────
# Level 5: Dimensionally-dependent identities (DDIs)
# ────────────────────────────────────────────────────────────────────

#= Dimensionally-dependent identities arise from the vanishing of the
# generalized Kronecker delta when its rank exceeds the manifold dimension d:
#
#   delta^{a1...a_{d+1}}_{b1...b_{d+1}} = 0
#
# Contracting pairs of indices with curvature tensors yields algebraic
# identities between curvature invariants at each polynomial order:
#
# Order 2 (quadratic):
#   d=4: Riem^2 - 4 Ric^2 + R^2 = 0  (Gauss-Bonnet / Euler density)
#   d=3: C_{abcd} = 0  (Weyl vanishes identically)
#   d=2: R_{ab} = (R/2) g_{ab}  (Ricci is pure trace)
#
# Order 3 (cubic):
#   d=4: I1 - (1/4) I2 + 2 I3 - I4 + (1/4) I5 = 0
#   (Fulling et al. 1992, Table 1)
#
# Level 5 integrates these DDIs into the Invar simplification pipeline.
# It delegates to the existing DDI infrastructure in src/algebra/ddi_rules.jl:
#   - generate_ddi_rules / generate_riemann_ddi: construct DDI rewrite rules
#   - register_ddi_rules!: register rules in the registry (idempotent)
#   - simplify_with_ddis: convenience wrapper for simplify + DDIs
#
# Ground truth:
#   - Garcia-Parrado & Martin-Garcia (2007) Sec 4, Level 3 (their numbering)
#   - Lovelock (1971), J. Math. Phys. 12, 498
#   - Fulling et al. (1992), Class. Quantum Grav. 9, 1151, Table 1
=#

"""
    simplify_level5(expr::TensorExpr;
                     covd::Symbol=:D,
                     dim::Int=4,
                     max_order::Int=0,
                     registry::TensorRegistry=current_registry()) -> TensorExpr

Apply Level 5 (dimensionally-dependent identities) of the Invar simplification
algorithm.

Dimensionally-dependent identities (DDIs) are algebraic relations between
curvature invariants that arise from the vanishing of the generalized Kronecker
delta `delta^{a1...a_{d+1}}_{b1...b_{d+1}} = 0` in `d` dimensions. Contracting
this identity with curvature tensors yields relations such as:

- **d=4, order 2 (Gauss-Bonnet)**: `Riem^2 = 4 Ric^2 - R^2`
- **d=3**: Weyl tensor vanishes identically
- **d=2**: `R_{ab} = (R/2) g_{ab}` (Ricci is pure trace)
- **d=4, order 3 (cubic DDI)**: relates `R_{abcd}R^{cdef}R_{ef}^{ab}` to
  lower-rank invariants (Fulling et al. 1992, Table 1)

This level first applies Level 4 (which includes Levels 1-3: permutation
symmetries, cyclic Bianchi, differential Bianchi, and derivative commutation),
then registers and applies DDI rewrite rules for the given manifold dimension.

# Arguments
- `expr`: tensor expression to simplify
- `covd::Symbol=:D`: name of the covariant derivative (for Levels 3-4)
- `dim::Int=4`: manifold dimension (determines which DDIs apply)
- `max_order::Int=0`: highest polynomial order of DDI rules to apply.
  If `0` (default), automatically determined from the Riemann degree of `expr`:
  order 2 for quadratic or lower, order 3 for cubic, etc.
- `registry`: the TensorRegistry to use

# Ground truth
Garcia-Parrado & Martin-Garcia (2007) Sec 4, Level 3 (their numbering);
Lovelock (1971); Fulling et al. (1992) Table 1.

# Examples
```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g registry=reg
define_curvature_tensors!(reg, :M4, :g)
@covd D on=M4 metric=g registry=reg

# Gauss-Bonnet identity simplifies to zero
gb = gauss_bonnet_ddi(; metric=:g, registry=reg)
simplify_level5(gb; dim=4, registry=reg)  # => 0

# Kretschner scalar eliminated in favour of Ricci invariants
K = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)]) *
    Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
simplify_level5(K; dim=4, registry=reg)  # => 4 Ric^2 - R^2
```

See also: [`simplify_with_ddis`](@ref), [`register_ddi_rules!`](@ref),
[`generate_ddi_rules`](@ref), [`simplify_level4`](@ref)
"""
function simplify_level5(expr::TensorExpr;
                          covd::Symbol=:D,
                          dim::Int=4,
                          max_order::Int=0,
                          registry::TensorRegistry=current_registry())
    # First apply Level 4 (includes Levels 1-3)
    expr4 = simplify_level4(expr; covd=covd, registry=registry)

    # Determine the DDI order from the expression's Riemann degree if not specified
    ddi_order = max_order
    if ddi_order <= 0
        degree = count_riemann_degree(expr4)
        # DDI order = degree (quadratic invariants use order 2, cubic use order 3)
        # Minimum order 2 since DDIs start at order 2
        ddi_order = max(degree, 2)
    end

    # Cap the DDI order at what the dimension supports:
    # generate_riemann_ddi requires 2*(order-1) <= dim+1
    # i.e., order <= (dim+1)/2 + 1 = (dim+3)/2
    max_supported = (dim + 3) ÷ 2
    ddi_order = min(ddi_order, max_supported)

    # Apply DDIs via the existing infrastructure
    with_registry(registry) do
        simplify_with_ddis(expr4; dim=dim, order=ddi_order, registry=registry,
                           commute_covds_name=covd)
    end
end

# ────────────────────────────────────────────────────────────────────
# Level 6: Dual invariant product relations
# ────────────────────────────────────────────────────────────────────

#= Dual invariant product relations arise from the Hodge dual of the
# Riemann tensor in d=4.  The Hodge duals are:
#
#   *R_{abcd}  = (1/2) ε_{ab}^{ef} R_{efcd}          (left dual)
#   R*_{abcd}  = (1/2) R_{abef} ε^{ef}_{cd}           (right dual)
#   *R*_{abcd} = (1/4) ε_{ab}^{ef} R_{efgh} ε^{gh}_{cd} (double dual)
#
# Key identity in d=4 (pair symmetry + epsilon contraction):
#
#   *R*_{abcd} = R_{abcd}
#
# That is, the double Hodge dual of the Riemann tensor equals the
# original Riemann tensor in 4 dimensions.  This is a consequence of
# the identity ε_{abef} ε^{ghef} = -2 (δ^g_a δ^h_b - δ^h_a δ^g_b)
# combined with the pair and antisymmetry of Riemann.
#
# Consequences for scalar invariants at degree 2:
#
#   (*R*)_{abcd} (*R*)^{abcd} = R_{abcd} R^{abcd}
#   (double-dual Kretschner = Kretschner)
#
# Level 6 wraps the existing DualRInv infrastructure and applies
# these reduction rules after Level 5 (DDIs).
#
# Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4, Level 6;
#               Zakhary & McIntosh (1997) GRG 29, 539.
=#

"""
    double_dual_identity(; metric::Symbol=:g,
                           registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the double-dual Kretschner identity in d=4:

    (*R*)_{abcd} (*R*)^{abcd} - R_{abcd} R^{abcd} = 0

Returns the LHS as a `TensorExpr`.  In 4 dimensions, this expression
equals zero because the double Hodge dual of the Riemann tensor equals
the original: `*R*_{abcd} = R_{abcd}`.

The double-dual term is constructed using the `DualRInv` infrastructure
with explicit Levi-Civita tensors.

# Arguments
- `metric::Symbol=:g`: the metric name (determines the epsilon tensor name `ε\$metric`)
- `registry::TensorRegistry`: the registry to use

# Returns
A `TensorExpr` representing `(*R*)^2 - R^2`, which should be zero in d=4.
"""
function double_dual_identity(; metric::Symbol=:g,
                                registry::TensorRegistry=current_registry())
    with_registry(registry) do
        # (*R*)_{abcd} (*R*)^{abcd} via DualRInv infrastructure
        kr = RInv(2, [5, 6, 7, 8, 1, 2, 3, 4])  # Kretschner contraction pattern
        # Double dual on BOTH factors: use DualRInv constructor directly
        dd_kretschner = DualRInv(kr, [(1, :double), (2, :double)])
        dd_expr = to_tensor_expr(dd_kretschner; registry=registry, metric=metric)

        # Ordinary Kretschner: R_{abcd} R^{abcd}
        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        Riem_down = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
        Riem_up = Tensor(:Riem, [up(a), up(b), up(c), up(d)])
        kretschner = Riem_down * Riem_up

        # Identity: (*R*)^2 - R^2 = 0
        dd_expr - kretschner
    end
end

"""
    register_dual_rules!(reg::TensorRegistry;
                          dim::Int=4,
                          metric::Symbol=:g) -> Vector{RewriteRule}

Register rewrite rules for dual invariant product identities in `dim`
dimensions.

In d=4, the key rules are:
1. The double-dual Kretschner equals the ordinary Kretschner:
   `(*R*)_{abcd} (*R*)^{abcd} → R_{abcd} R^{abcd}`

These rules reduce products involving dual Riemann tensors to
expressions involving only the undualised Riemann tensor.

The rules are registered in the given registry and also returned.

# Arguments
- `reg::TensorRegistry`: the registry in which to register rules
- `dim::Int=4`: manifold dimension (dual rules currently only for d=4)
- `metric::Symbol=:g`: the metric name

# Returns
`Vector{RewriteRule}` of the registered rules.
"""
function register_dual_rules!(reg::TensorRegistry;
                               dim::Int=4,
                               metric::Symbol=:g)
    rules = RewriteRule[]

    # Dual rules are specific to d=4
    dim == 4 || return rules

    eps_name = Symbol(:ε, metric)

    # Rule: products of epsilon pairs contracting with Riemann tensors
    # can be reduced using ε_{abef} ε^{ghef} = -2(δ^g_a δ^h_b - δ^g_b δ^h_a)
    #
    # Rather than building pattern-matching rules for the complex epsilon
    # contraction structure, we register an identity rule via the algebra:
    # *R*_{abcd} = R_{abcd} in d=4.
    #
    # Construct: (1/4) ε_{ab}^{ef} R_{efgh} ε^{gh}_{cd} -> R_{abcd}
    #
    # This is encoded as a rewrite rule on the explicit epsilon-Riemann product.
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g_idx = fresh_index(used); push!(used, g_idx)
    h = fresh_index(used); push!(used, h)

    # LHS: (1/4) ε_{ab}^{ef} R_{efgh} ε^{gh}_{cd}
    eps_left = Tensor(eps_name, [down(a), down(b), up(e), up(f)])
    riem = Tensor(:Riem, [down(e), down(f), down(g_idx), down(h)])
    eps_right = Tensor(eps_name, [up(g_idx), up(h), down(c), down(d)])
    lhs = tproduct(1 // 4, TensorExpr[eps_left, riem, eps_right])

    # RHS: R_{abcd}
    rhs = Tensor(:Riem, [down(a), down(b), down(c), down(d)])

    rule = RewriteRule(lhs, rhs)
    push!(rules, rule)
    register_rule!(reg, rule)

    rules
end

# Module-level tracking of which registries have dual rules registered.
const _DUAL_RULES_REGISTERED = Dict{UInt, Set{Tuple{Int,Symbol}}}()

"""
    has_dual_rules(reg::TensorRegistry; dim::Int=4, metric::Symbol=:g) -> Bool

Check whether dual invariant rules for the given dimension and metric have
already been registered in the registry via `register_dual_rules!`.
"""
function has_dual_rules(reg::TensorRegistry; dim::Int=4, metric::Symbol=:g)
    key = objectid(reg)
    haskey(_DUAL_RULES_REGISTERED, key) && (dim, metric) in _DUAL_RULES_REGISTERED[key]
end

"""
    simplify_level6(expr::TensorExpr;
                     covd::Symbol=:D,
                     dim::Int=4,
                     max_order::Int=0,
                     registry::TensorRegistry=current_registry()) -> TensorExpr

Apply Level 6 (dual invariant product relations) of the Invar simplification
algorithm.

Level 6 handles algebraic identities between curvature invariants involving
the Hodge dual of the Riemann tensor.  In d=4, the key identity is:

    *R*_{abcd} = R_{abcd}  (double dual = original)

which implies that any scalar invariant built from the double-dual Riemann
equals the corresponding invariant built from the undualised Riemann.

This level first applies Level 5 (which includes Levels 1-4: permutation
symmetries, cyclic/differential Bianchi, derivative commutation, and DDIs),
then registers and applies dual invariant reduction rules.

# Arguments
- `expr`: tensor expression to simplify
- `covd::Symbol=:D`: name of the covariant derivative (for Levels 3-4)
- `dim::Int=4`: manifold dimension (dual rules currently for d=4 only)
- `max_order::Int=0`: forwarded to Level 5 (DDI polynomial order)
- `registry`: the TensorRegistry to use

# Ground truth
Garcia-Parrado & Martin-Garcia (2007) Sec 4, Level 6 (their numbering: Schouten/Lovelock);
Zakhary & McIntosh (1997) GRG 29, 539.

# Examples
```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g registry=reg
define_curvature_tensors!(reg, :M4, :g)
@covd D on=M4 metric=g registry=reg

# Pontryagin density is a well-formed scalar
pont = pontryagin_density(:g; registry=reg)
result = simplify_level6(pont; dim=4, registry=reg)
@test isempty(free_indices(result))
```

See also: [`simplify_level5`](@ref), [`register_dual_rules!`](@ref),
[`double_dual_identity`](@ref), [`DualRInv`](@ref)
"""
function simplify_level6(expr::TensorExpr;
                          covd::Symbol=:D,
                          dim::Int=4,
                          max_order::Int=0,
                          registry::TensorRegistry=current_registry())
    # First apply Level 5 (includes Levels 1-4)
    expr5 = simplify_level5(expr; covd=covd, dim=dim, max_order=max_order,
                            registry=registry)

    # Register dual rules if not already present (idempotent)
    metric_name = _dual_metric_name(registry)
    if dim == 4 && !has_dual_rules(registry; dim=dim, metric=metric_name)
        with_registry(registry) do
            register_dual_rules!(registry; dim=dim, metric=metric_name)
        end
        # Track registration
        key = objectid(registry)
        if !haskey(_DUAL_RULES_REGISTERED, key)
            _DUAL_RULES_REGISTERED[key] = Set{Tuple{Int,Symbol}}()
        end
        push!(_DUAL_RULES_REGISTERED[key], (dim, metric_name))
    end

    # Apply the full simplify pipeline with dual rules in the registry
    with_registry(registry) do
        simplify(expr5; registry=registry)
    end
end

"""Look up the metric name from the registry for dual rule registration."""
function _dual_metric_name(reg::TensorRegistry)
    isempty(reg.metric_cache) ? :g : first(values(reg.metric_cache))
end
