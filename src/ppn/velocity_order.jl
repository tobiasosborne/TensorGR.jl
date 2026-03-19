#= PPN velocity-order (v/c) expansion.
#
# In the PPN formalism, each quantity has a definite order in v/c:
#   U ~ O(2), Phi_W ~ O(4), Phi_1..4 ~ O(4), A_ppn ~ O(4)
#   V_ppn, W_ppn ~ O(3), U_ppn ~ O(2), v^i ~ O(1)
#   partial_0 ~ O(1) relative to partial_i (for bound systems)
#   g_{00} ~ -1 + O(2) + O(4), g_{0i} ~ O(3), g_{ij} ~ delta + O(2)
#
# Ground truth: Will (2018) Ch 4, Sec 4.1 (order counting).
=#

# ────────────────────────────────────────────────────────────────────
# PPN order assignments
# ────────────────────────────────────────────────────────────────────

"""
    PPN_ORDER_TABLE

Default v/c order assignments for standard PPN tensors.
Order means the expression is O((v/c)^order) relative to the background.

Ground truth: Will (2018) Table 4.1 and surrounding discussion.
"""
const PPN_ORDER_TABLE = Dict{Symbol, Int}(
    :U       => 2,    # Newtonian potential
    :U_ppn   => 2,    # Superpotential U_{ij}
    :Phi_W   => 4,    # Whitehead potential
    :Phi_1   => 4,    # Post-Newtonian potential
    :Phi_2   => 4,    # Post-Newtonian potential
    :Phi_3   => 4,    # Post-Newtonian potential
    :Phi_4   => 4,    # Post-Newtonian potential
    :A_ppn   => 4,    # Vector potential scalar piece
    :V_ppn   => 3,    # Gravito-magnetic V_i
    :W_ppn   => 3,    # Gravito-magnetic W_i
)

"""
    ppn_order(expr::TensorExpr; order_table=PPN_ORDER_TABLE) -> Int

Determine the PPN velocity order (v/c power) of an expression.

Each PPN potential has a fixed order from `order_table`. Products add
orders. Sums take the minimum order (lowest-order term dominates).
Derivatives add +1 for temporal derivatives (∂₀ counts as O(1) for
bound systems), +0 for spatial derivatives.

Returns the minimum v/c order present in the expression.
"""
function ppn_order(expr::Tensor; order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    get(order_table, expr.name, 0)
end

ppn_order(::TScalar; order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE) = 0

function ppn_order(p::TProduct; order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    sum(ppn_order(f; order_table=order_table) for f in p.factors; init=0)
end

function ppn_order(s::TSum; order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    isempty(s.terms) && return 0
    minimum(ppn_order(t; order_table=order_table) for t in s.terms)
end

function ppn_order(d::TDeriv; order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    # Temporal derivatives add O(1) for bound systems (∂₀ ~ v/r ~ O(1))
    # Spatial derivatives are O(0) (∂_i ~ 1/r ~ O(0))
    # We cannot determine temporal vs spatial from abstract index alone,
    # so return the order of the argument (conservative: spatial derivative)
    ppn_order(d.arg; order_table=order_table)
end

"""
    ppn_max_order(expr::TensorExpr; order_table=PPN_ORDER_TABLE) -> Int

Return the maximum PPN velocity order present in the expression.
For sums, this is the highest-order term. For products, the sum of factors.
"""
function ppn_max_order(expr::Tensor; order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    get(order_table, expr.name, 0)
end

ppn_max_order(::TScalar; order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE) = 0

function ppn_max_order(p::TProduct; order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    sum(ppn_max_order(f; order_table=order_table) for f in p.factors; init=0)
end

function ppn_max_order(s::TSum; order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    isempty(s.terms) && return 0
    maximum(ppn_max_order(t; order_table=order_table) for t in s.terms)
end

function ppn_max_order(d::TDeriv; order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    ppn_max_order(d.arg; order_table=order_table)
end

# ────────────────────────────────────────────────────────────────────
# PPN truncation
# ────────────────────────────────────────────────────────────────────

"""
    truncate_ppn(expr::TensorExpr, max_order::Int;
                 order_table=PPN_ORDER_TABLE) -> TensorExpr

Truncate an expression at a given PPN velocity order, discarding
terms with v/c order > max_order.

For a TSum, each term is checked individually. For other expressions,
returns zero if the expression exceeds max_order.

The PPN metric requires:
- g_{00}: through O(4) for 2PN accuracy
- g_{0i}: through O(3)
- g_{ij}: through O(2)

Ground truth: Will (2018) Ch 4 order counting.
"""
function truncate_ppn(expr::TSum, max_order::Int;
                      order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    kept = TensorExpr[t for t in expr.terms
                       if ppn_order(t; order_table=order_table) <= max_order]
    tsum(kept)
end

function truncate_ppn(expr::TProduct, max_order::Int;
                      order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    ppn_order(expr; order_table=order_table) <= max_order ? expr : ZERO
end

function truncate_ppn(expr::TensorExpr, max_order::Int;
                      order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    ppn_order(expr; order_table=order_table) <= max_order ? expr : ZERO
end

# ────────────────────────────────────────────────────────────────────
# PPN metric order requirements
# ────────────────────────────────────────────────────────────────────

"""
    PPN_METRIC_ORDERS

Required v/c orders for each metric component at different PN accuracy.

At 1PN (order=1): need g_{00} through O(2), g_{0i} = 0, g_{ij} through O(2)
At 2PN (order=2): need g_{00} through O(4), g_{0i} through O(3), g_{ij} through O(2)

Ground truth: Will (2018) Sec 4.1.
"""
const PPN_METRIC_ORDERS = Dict{Symbol, Dict{Int, Int}}(
    :g00 => Dict(1 => 2, 2 => 4),
    :g0i => Dict(1 => 0, 2 => 3),
    :gij => Dict(1 => 2, 2 => 2),
)

"""
    ppn_truncate_metric(mc::PPNMetricComponents, pn_order::Int;
                        order_table=PPN_ORDER_TABLE) -> PPNMetricComponents

Truncate PPN metric components to the required v/c orders for a given PN accuracy.

At 1PN: g_{00} through O(2), g_{0i} = 0, g_{ij} through O(2).
At 2PN: g_{00} through O(4), g_{0i} through O(3), g_{ij} through O(2).
"""
function ppn_truncate_metric(mc::PPNMetricComponents, pn_order_val::Int;
                             order_table::Dict{Symbol,Int}=PPN_ORDER_TABLE)
    pn_order_val in (1, 2) ||
        error("PPN order must be 1 or 2, got $pn_order_val")

    g00_max = PPN_METRIC_ORDERS[:g00][pn_order_val]
    g0i_max = PPN_METRIC_ORDERS[:g0i][pn_order_val]
    gij_max = PPN_METRIC_ORDERS[:gij][pn_order_val]

    PPNMetricComponents(
        truncate_ppn(mc.g00, g00_max; order_table=order_table),
        g0i_max == 0 ? TScalar(0 // 1) : truncate_ppn(mc.g0i, g0i_max; order_table=order_table),
        truncate_ppn(mc.gij, gij_max; order_table=order_table)
    )
end
