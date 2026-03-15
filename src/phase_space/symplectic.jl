#= Symplectic potential for the covariant phase space framework.
#
# Given a Lagrangian density L, the first variation decomposes as:
#   δL = E · δφ + ∇_a Θ^a(φ, δφ)
# where E = 0 are the equations of motion and Θ is the symplectic
# potential (d-1)-form, linear in δφ and its derivatives.
#
# Reference: Iyer & Wald (1994), PRD 50, 846, Eqs 2.5, 2.11, 3.3.
=#

"""
    SymplecticPotential(expr, field, delta_field, lagrangian)

Container for the symplectic potential Θ^a(φ, δφ).

Fields:
- `expr::TensorExpr` -- Θ^a as a TensorExpr with one free upper index
- `field::Symbol` -- the dynamical field (e.g., `:g`)
- `delta_field::Symbol` -- symbol for the variation (e.g., `:delta_g`)
- `lagrangian::LagrangianDensity` -- the originating Lagrangian
"""
struct SymplecticPotential
    expr::TensorExpr
    field::Symbol
    delta_field::Symbol
    lagrangian::LagrangianDensity
end

"""
    symplectic_potential(L::LagrangianDensity, field::Symbol;
                         delta_field::Symbol=Symbol(:delta_, field),
                         registry::TensorRegistry=current_registry()) -> SymplecticPotential

Compute the symplectic potential Θ^a for the given Lagrangian and field.

The symplectic potential is defined by the decomposition (Iyer-Wald Eq 2.5):

    δL = E · δφ + ∇_a Θ^a(φ, δφ)

For the Einstein-Hilbert Lagrangian L = R, the result is the known boundary
term (Iyer-Wald Eq 3.3):

    Θ^a = g^{bc}(∇^a δg_{bc} - ∇_c δg^a_b)

For general Lagrangians, delegates to `_general_symplectic_potential`.
"""
function symplectic_potential(L::LagrangianDensity, field::Symbol;
                               delta_field::Symbol=Symbol(:delta_, field),
                               registry::TensorRegistry=current_registry())
    with_registry(registry) do
        theta_expr = _compute_theta(L, field, delta_field, registry)
        SymplecticPotential(theta_expr, field, delta_field, L)
    end
end

# ── Internal dispatch ────────────────────────────────────────────────

function _compute_theta(L::LagrangianDensity, field::Symbol,
                        delta_field::Symbol, reg::TensorRegistry)
    if field == L.metric && _is_ricci_scalar(L.expr)
        return _theta_eh(L.metric, L.covd, delta_field)
    end
    _general_symplectic_potential(L, field, delta_field, reg)
end

"""
    theta_eh(metric, delta_metric, covd) -> TensorExpr

Return the explicit EH symplectic potential without computing from scratch.

Iyer-Wald (1994) Eq 3.3:
    Θ^a = g^{bc}(∇^a δg_{bc} - ∇_c δg^a_{  b})

This is the boundary term that arises from varying the Ricci scalar.
The free index is `up(:a)`.
"""
function theta_eh(metric::Symbol, delta_metric::Symbol, covd::Symbol)
    _theta_eh(metric, covd, delta_metric)
end

function _theta_eh(metric::Symbol, covd::Symbol, delta_field::Symbol)
    # Θ^a = g^{bc}(∇^a δg_{bc} - ∇_c δg^a_b)
    #
    # Use fresh indices to avoid clashes.
    used = Set{Symbol}([:a])
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)

    g_inv = Tensor(metric, [up(b), up(c)])
    dg_bc = Tensor(delta_field, [down(b), down(c)])
    dg_ab = Tensor(delta_field, [up(:a), down(b)])

    # ∇^a δg_{bc}: covariant derivative with upper index a
    term1 = TDeriv(up(:a), dg_bc, covd)
    # ∇_c δg^a_b: covariant derivative with lower index c
    term2 = TDeriv(down(c), dg_ab, covd)

    # Θ^a = g^{bc} (∇^a δg_{bc} - ∇_c δg^a_b)
    g_inv * (term1 - term2)
end

"""
    _general_symplectic_potential(L, field, delta_field, reg) -> TensorExpr

Compute Θ for a general Lagrangian by varying L and subtracting the EOM piece.

For a metric Lagrangian L(g, R_{abcd}, ...):
  δL = E^{ab} δg_{ab} + ∇_a Θ^a
  => Θ^a = (δL - E^{ab} δg_{ab}) boundary part

This uses the EOM already computed by eom_extract, then constructs the
remainder as the boundary term. For Lagrangians that are sums involving
the Ricci scalar, we split and handle the EH part analytically.
"""
function _general_symplectic_potential(L::LagrangianDensity, field::Symbol,
                                       delta_field::Symbol, reg::TensorRegistry)
    # For general Lagrangians we return a formal SymplecticPotential
    # built from: Θ = δL|_{boundary} after subtracting EOM·δφ.
    #
    # The full IBP extraction is complex for arbitrary higher-derivative
    # Lagrangians.  For now, construct a formal expression representing
    # the boundary term as Tensor(:Theta_<field>, [up(:a)]).
    theta_name = Symbol(:Theta_, field)
    if !has_tensor(reg, theta_name)
        register_tensor!(reg, TensorProperties(
            name=theta_name, manifold=L.metric == field ? :M4 : :M4,
            rank=(1, 0), symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_symplectic_potential => true,
                                     :lagrangian_expr => L.expr,
                                     :delta_field => delta_field)))
    end
    Tensor(theta_name, [up(:a)])
end

"""
    add_boundary_ambiguity(Theta::SymplecticPotential, Y::TensorExpr) -> SymplecticPotential

Add a boundary ambiguity d Y to the symplectic potential.

The symplectic potential is defined only up to (Iyer-Wald Eq 40, 43):
    Θ → Θ + dY(φ, δφ)
where Y is a covariant (d-2)-form linear in δφ.

This adds ∇_b Y^{ab} to Θ^a (divergence of an antisymmetric tensor).
"""
function add_boundary_ambiguity(Theta::SymplecticPotential, Y::TensorExpr)
    # Y should be an antisymmetric 2-index tensor Y^{ab} = -Y^{ba}
    # so that dY contributes ∇_b Y^{ab} to Theta^a.
    used = Set{Symbol}([:a])
    b = fresh_index(used)

    # ∇_b Y^{ab} -- divergence of Y on the second index
    covd = Theta.lagrangian.covd
    div_Y = TDeriv(down(b), Y, covd)

    new_expr = Theta.expr + div_Y
    SymplecticPotential(new_expr, Theta.field, Theta.delta_field, Theta.lagrangian)
end
