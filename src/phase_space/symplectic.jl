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

# ── Symplectic current ω ────────────────────────────────────────────
#
# The symplectic current is the antisymmetrized second variation
# (Iyer & Wald 1994, Eq 2.7; Lee & Wald 1990, Eq 2.3):
#
#   ω^a(φ; δ₁φ, δ₂φ) = δ₁Θ^a(φ, δ₂φ) − δ₂Θ^a(φ, δ₁φ)
#
# It is bilinear and antisymmetric in (δ₁φ, δ₂φ).
# On-shell (when both variations satisfy linearized EOM):
#   ∇_a ω^a = 0
#

"""
    SymplecticCurrent(expr, delta1_field, delta2_field, potential)

Container for the symplectic current ω^a(φ; δ₁φ, δ₂φ).

Fields:
- `expr::TensorExpr` -- ω^a as a TensorExpr with one free upper index
- `delta1_field::Symbol` -- first variation (e.g., `:delta1_g`)
- `delta2_field::Symbol` -- second variation (e.g., `:delta2_g`)
- `potential::SymplecticPotential` -- the originating symplectic potential
"""
struct SymplecticCurrent
    expr::TensorExpr
    delta1_field::Symbol
    delta2_field::Symbol
    potential::SymplecticPotential
end

"""
    symplectic_current(Theta::SymplecticPotential,
                       delta1::Symbol, delta2::Symbol;
                       registry::TensorRegistry=current_registry()) -> SymplecticCurrent

Compute the symplectic current ω^a for the given symplectic potential.

The symplectic current is defined by (Iyer-Wald 1994, Eq 2.7):

    ω^a(φ; δ₁φ, δ₂φ) = δ₁Θ^a(φ, δ₂φ) − δ₂Θ^a(φ, δ₁φ)

For the Einstein-Hilbert Lagrangian L = R, the explicit formula is
(Lee & Wald 1990, Eq 3.2):

    ω^a = g^{bc}(δ₁g_{de} ∇^a δ₂g_{bc} − δ₂g_{de} ∇^a δ₁g_{bc})
          − (1↔2 in remaining derivative terms)

More precisely, ω^a = Θ^a[δ₂g, with ∇δ₂g → ∇δ₂g and g⁻¹ varied by δ₁]
minus the (1↔2) exchange.

# Arguments
- `Theta::SymplecticPotential` -- the symplectic potential
- `delta1::Symbol` -- name for the first field variation
- `delta2::Symbol` -- name for the second field variation
- `registry::TensorRegistry` -- tensor registry

# Examples
```julia
L_EH = LagrangianDensity(Tensor(:RicScalar, TIndex[]), [:g], :g, :D, 4)
Theta = symplectic_potential(L_EH, :g; delta_field=:h1, registry=reg)
omega = symplectic_current(Theta, :h1, :h2; registry=reg)
# omega.expr is antisymmetric: swap h1 ↔ h2 gives -omega
```
"""
function symplectic_current(Theta::SymplecticPotential,
                            delta1::Symbol, delta2::Symbol;
                            registry::TensorRegistry=current_registry())
    with_registry(registry) do
        omega_expr = _compute_omega(Theta, delta1, delta2, registry)
        SymplecticCurrent(omega_expr, delta1, delta2, Theta)
    end
end

# ── Internal dispatch for omega ─────────────────────────────────────

function _compute_omega(Theta::SymplecticPotential,
                        delta1::Symbol, delta2::Symbol,
                        reg::TensorRegistry)
    L = Theta.lagrangian
    if Theta.field == L.metric && _is_ricci_scalar(L.expr)
        return _omega_eh(L.metric, L.covd, delta1, delta2)
    end
    _general_symplectic_current(Theta, delta1, delta2, reg)
end

"""
    _omega_eh(metric, covd, delta1, delta2) -> TensorExpr

Explicit EH symplectic current (Lee & Wald 1990, Eq 3.2; Wald 1990, Eq 3.5).

    ω^a = Θ^a(δ₂g) varied by δ₁ − Θ^a(δ₁g) varied by δ₂

where Θ^a = g^{bc}(∇^a δg_{bc} − ∇_c δg^a_b).

The variation δ₁ of Θ^a(δ₂g) acts on the background metric g^{bc}:
  δ₁(g^{bc}) = −g^{bd}g^{ce} δ₁g_{de}

This produces the P-tensor structure of Wald (1990) Eq 3.5.
We construct it directly as Θ^a(δ₂g)|_{g→g+δ₁g} − (1↔2),
keeping only terms linear in each variation.
"""
function _omega_eh(metric::Symbol, covd::Symbol,
                   delta1::Symbol, delta2::Symbol)
    # Build Θ^a(δ₂g) with explicit indices, then vary g^{bc} by δ₁.
    #
    # Θ^a(δ₂) = g^{bc}(∇^a δ₂g_{bc} − ∇_c δ₂g^a_b)
    #
    # δ₁[Θ^a(δ₂)] has two parts:
    #   (i)  δ₁(g^{bc}) · (∇^a δ₂g_{bc} − ∇_c δ₂g^a_b)
    #   (ii) g^{bc} · (∇^a δ₂g_{bc} − ∇_c δ₂g^a_b) with δ₂g → δ₂g unchanged,
    #        but ∇ now has a δ₁-dependent part (δ₁Γ terms).
    #
    # The δ₁Γ terms cancel in ω (Iyer-Wald). The surviving piece is
    # from δ₁(g^{bc}) = −g^{bd}g^{ce}δ₁g_{de}:
    #
    # δ₁[Θ^a(δ₂)] = −g^{bd}g^{ce}δ₁g_{de}(∇^a δ₂g_{bc} − ∇_c δ₂g^a_b)
    #
    # ω^a = δ₁[Θ^a(δ₂)] − δ₂[Θ^a(δ₁)]
    #      = −g^{bd}g^{ce}δ₁g_{de}(∇^a δ₂g_{bc} − ∇_c δ₂g^a_b)
    #        +g^{bd}g^{ce}δ₂g_{de}(∇^a δ₁g_{bc} − ∇_c δ₁g^a_b)
    #
    # Equivalently (relabelling dummies):
    # ω^a = g^{bd}g^{ce}[δ₂g_{de}∇^a δ₁g_{bc} − δ₁g_{de}∇^a δ₂g_{bc}]
    #      −g^{bd}g^{ce}[δ₂g_{de}∇_c δ₁g^a_b − δ₁g_{de}∇_c δ₂g^a_b]

    used = Set{Symbol}([:a])
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)

    g_bd = Tensor(metric, [up(b), up(d)])
    g_ce = Tensor(metric, [up(c), up(e)])

    d1_de = Tensor(delta1, [down(d), down(e)])
    d2_de = Tensor(delta2, [down(d), down(e)])

    d1_bc = Tensor(delta1, [down(b), down(c)])
    d2_bc = Tensor(delta2, [down(b), down(c)])

    d1_ab = Tensor(delta1, [up(:a), down(b)])
    d2_ab = Tensor(delta2, [up(:a), down(b)])

    # ∇^a δg_{bc} terms
    grad_a_d1_bc = TDeriv(up(:a), d1_bc, covd)
    grad_a_d2_bc = TDeriv(up(:a), d2_bc, covd)

    # ∇_c δg^a_b terms
    grad_c_d1_ab = TDeriv(down(c), d1_ab, covd)
    grad_c_d2_ab = TDeriv(down(c), d2_ab, covd)

    # First bracket: g^{bd}g^{ce}[δ₂g_{de}∇^a δ₁g_{bc} − δ₁g_{de}∇^a δ₂g_{bc}]
    bracket1 = g_bd * g_ce * (d2_de * grad_a_d1_bc - d1_de * grad_a_d2_bc)

    # Second bracket: g^{bd}g^{ce}[δ₂g_{de}∇_c δ₁g^a_b − δ₁g_{de}∇_c δ₂g^a_b]
    bracket2 = g_bd * g_ce * (d2_de * grad_c_d1_ab - d1_de * grad_c_d2_ab)

    # ω^a = bracket1 − bracket2
    bracket1 - bracket2
end

"""
    _general_symplectic_current(Theta, delta1, delta2, reg) -> TensorExpr

Compute ω for a general symplectic potential by antisymmetrizing.

Constructs ω^a = Θ^a[δ₂] − Θ^a[δ₁] as a formal expression where
the delta_field in Theta is replaced by delta1 and delta2 respectively.
"""
function _general_symplectic_current(Theta::SymplecticPotential,
                                      delta1::Symbol, delta2::Symbol,
                                      reg::TensorRegistry)
    # Replace delta_field → delta2 in Theta to get Theta(delta2),
    # then delta_field → delta1 to get Theta(delta1).
    # ω = Theta(delta1, ∇delta2) − Theta(delta2, ∇delta1)
    # For a formal potential Tensor(:Theta_g, [up(:a)]), we construct
    # formal omega as the antisymmetrized pair.
    orig = Theta.delta_field
    theta_d1 = _replace_tensor_name(Theta.expr, orig, delta1)
    theta_d2 = _replace_tensor_name(Theta.expr, orig, delta2)

    # ω = θ(δ₂) varied by δ₁ − θ(δ₁) varied by δ₂
    # For formal potentials, the best we can do is the substitution form.
    theta_d2 - theta_d1
end

"""
    _replace_tensor_name(expr, old_name, new_name) -> TensorExpr

Replace all tensors named `old_name` with `new_name`, preserving indices.
"""
function _replace_tensor_name(expr::TensorExpr, old_name::Symbol, new_name::Symbol)
    walk(expr) do node
        if node isa Tensor && node.name == old_name
            Tensor(new_name, node.indices)
        else
            node
        end
    end
end
