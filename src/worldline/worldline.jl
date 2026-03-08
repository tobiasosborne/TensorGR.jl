#= Worldline formalism for point-particle EFT.

A worldline represents a parametric curve x^μ(s) in spacetime,
with velocity v^μ = dx^μ/ds and acceleration a^μ = Dv^μ/ds.

Used in the EFTofPNG (Effective Field Theory of Post-Newtonian Gravity)
for describing compact binary dynamics.
=#

"""
    Worldline(label, parameter, manifold)

A parametric worldline in spacetime, representing a point particle's
trajectory x^μ(s).
"""
struct Worldline
    label::Symbol        # particle label (:A, :B, :1, :2, ...)
    parameter::Symbol    # curve parameter (:s, :τ, ...)
    manifold::Symbol     # ambient manifold
    velocity::Symbol     # velocity tensor name
    position::Symbol     # position tensor name
end

function Worldline(label::Symbol; parameter::Symbol=:τ, manifold::Symbol=:M4)
    Worldline(label, parameter, manifold,
              Symbol(:v, label), Symbol(:x, label))
end

"""
    define_worldline!(reg, wl::Worldline; metric=:g)

Register the worldline tensors (velocity, position) in the registry.
Adds the normalization rule g_{ab} v^a v^b = -1 (timelike).
"""
function define_worldline!(reg::TensorRegistry, wl::Worldline;
                            metric::Symbol=:g, signature::Int=-1)
    # Velocity v^μ (contravariant vector)
    if !has_tensor(reg, wl.velocity)
        register_tensor!(reg, TensorProperties(
            name=wl.velocity, manifold=wl.manifold, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_velocity => true,
                                     :worldline => wl.label)))
    end

    # Normalization rule: g_{ab} v^a v^b = signature (-1 for timelike)
    register_rule!(reg, RewriteRule(
        function(expr)
            expr isa TProduct || return false
            v_count = count(f -> f isa Tensor && f.name == wl.velocity, expr.factors)
            v_count < 2 && return false
            # Check for metric contraction between two velocities
            m_count = count(f -> f isa Tensor && f.name == metric, expr.factors)
            m_count < 1 && return false
            true
        end,
        function(expr)
            # This is a simplistic rule — proper contraction is done
            # by the metric contraction engine. This rule catches
            # the specific pattern g_{ab} v^a v^b after contraction.
            tproduct(expr.scalar * signature, TensorExpr[
                f for f in expr.factors
                if !(f isa Tensor && (f.name == wl.velocity || f.name == metric))
            ])
        end
    ))

    wl
end

"""
    pn_order(expr::TensorExpr, velocity::Symbol) -> Int

Determine the post-Newtonian order of an expression by counting
powers of the velocity tensor. Each v counts as O(ε), where ε ~ v/c.
PN order n means O(v^{2n}) = O(ε^{2n}).
"""
function pn_order(expr::Tensor, velocity::Symbol)
    expr.name == velocity ? 1 : 0
end

pn_order(::TScalar, ::Symbol) = 0

function pn_order(d::TDeriv, velocity::Symbol)
    pn_order(d.arg, velocity)
end

function pn_order(p::TProduct, velocity::Symbol)
    sum(pn_order(f, velocity) for f in p.factors; init=0)
end

function pn_order(s::TSum, velocity::Symbol)
    isempty(s.terms) ? 0 : maximum(pn_order(t, velocity) for t in s.terms)
end

"""
    truncate_pn(expr::TensorExpr, max_order::Int, velocity::Symbol) -> TensorExpr

Truncate an expression at a given post-Newtonian order,
discarding terms with velocity power > 2*max_order.
"""
function truncate_pn(expr::TSum, max_order::Int, velocity::Symbol)
    kept = TensorExpr[t for t in expr.terms
                       if pn_order(t, velocity) <= 2 * max_order]
    tsum(kept)
end

function truncate_pn(expr::TProduct, max_order::Int, velocity::Symbol)
    pn_order(expr, velocity) <= 2 * max_order ? expr : ZERO
end

truncate_pn(expr::TensorExpr, ::Int, ::Symbol) = expr
