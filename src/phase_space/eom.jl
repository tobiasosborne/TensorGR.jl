#= Equations of motion extraction for the covariant phase space framework.
#
# Given a Lagrangian density L, compute:
#   delta L = E * delta_phi + d Theta
# where E = 0 are the equations of motion and Theta is the symplectic
# potential (d-1)-form.
#
# Reference: Iyer & Wald (1994), PRD 50, 846, Eqs 2.1-2.5, 3.20.
=#

"""
    LagrangianDensity(expr, fields, metric, covd, dim)

Container for a Lagrangian density and its context.

Fields:
- `expr::TensorExpr` -- the Lagrangian scalar density (e.g., RicScalar for EH gravity)
- `fields::Vector{Symbol}` -- dynamical fields (e.g., `[:g]` for pure gravity)
- `metric::Symbol` -- background metric name
- `covd::Symbol` -- covariant derivative name
- `dim::Int` -- spacetime dimension
"""
struct LagrangianDensity
    expr::TensorExpr
    fields::Vector{Symbol}
    metric::Symbol
    covd::Symbol
    dim::Int
end

"""
    EOMResult(eom, theta, lagrangian, field)

Container for the result of equations-of-motion extraction.

Fields:
- `eom::TensorExpr` -- equations of motion E (set to zero: E = 0)
- `theta::Union{TensorExpr, Nothing}` -- symplectic potential Theta (nothing if not yet computed)
- `lagrangian::LagrangianDensity` -- the original Lagrangian
- `field::Symbol` -- the field with respect to which variation was taken
"""
struct EOMResult
    eom::TensorExpr
    theta::Union{TensorExpr, Nothing}
    lagrangian::LagrangianDensity
    field::Symbol
end

"""
    eom_extract(L::LagrangianDensity, field::Symbol;
                registry::TensorRegistry=current_registry()) -> EOMResult

Extract the equations of motion from a Lagrangian density by varying with
respect to the given field.

For a scalar field Lagrangian L(phi, partial phi), this computes the
Euler-Lagrange equation via `variational_derivative`.

For a metric Lagrangian L(g, Riem, ...), this computes the metric
equations of motion via `metric_variation`.

The symplectic potential Theta is not yet extracted by this function
(placeholder for TGR-s50.3). The returned `EOMResult` has `theta = nothing`.

# Examples

```julia
# Einstein-Hilbert: L = R
reg = TensorRegistry()
@manifold M4 dim=4 metric=g
define_curvature_tensors!(reg, :M4, :g)
define_covd!(reg, :D; manifold=:M4, metric=:g)
L_EH = LagrangianDensity(Tensor(:RicScalar, TIndex[]), [:g], :g, :D, 4)
result = eom_extract(L_EH, :g; registry=reg)
# result.eom is the Einstein tensor G_{ab}
```
"""
function eom_extract(L::LagrangianDensity, field::Symbol;
                     registry::TensorRegistry=current_registry())
    with_registry(registry) do
        eom = _compute_eom(L, field, registry)
        EOMResult(eom, nothing, L, field)
    end
end

"""
    eom_extract(L::LagrangianDensity;
                registry::TensorRegistry=current_registry()) -> Vector{EOMResult}

Extract equations of motion for ALL dynamical fields in the Lagrangian.
"""
function eom_extract(L::LagrangianDensity;
                     registry::TensorRegistry=current_registry())
    [eom_extract(L, f; registry=registry) for f in L.fields]
end

# ── Internal: compute EOM ────────────────────────────────────────────

"""
Compute the equations of motion for the given field.

For the metric field, uses `metric_variation` (variation with respect to g^{ab}).
For other fields, uses `variational_derivative` (Euler-Lagrange).
"""
function _compute_eom(L::LagrangianDensity, field::Symbol,
                      reg::TensorRegistry)
    if field == L.metric
        _metric_eom(L, reg)
    else
        variational_derivative(L.expr, field)
    end
end

"""
Metric equations of motion via variation with respect to g^{ab}.

For L = R (Einstein-Hilbert), this returns G_{ab} = R_{ab} - (1/2)g_{ab}R.

The general formula is:
  (1/sqrt(-g)) * delta(sqrt(-g) * L) / delta(g^{ab})
= (1/2) g_{ab} L + partial L / partial g^{ab}

For curvature Lagrangians, we use known results:
- L = R -> E_{ab} = G_{ab}
- L = Ric_{cd}Ric^{cd} -> E_{ab} from variation of Ric^2
- General: fall back to metric_variation
"""
function _metric_eom(L::LagrangianDensity, reg::TensorRegistry)
    expr = L.expr
    metric = L.metric
    dim = L.dim

    # Check for known Lagrangians
    if _is_ricci_scalar(expr)
        return _einstein_tensor(metric)
    end

    # General case: use metric_variation
    # The full metric EOM is:
    #   E_{ab} = (1/2) g_{ab} L + (partial L / partial g^{ab})
    # The second term comes from varying the explicit metric dependence.
    # For the inverse metric variation delta(g^{cd})/delta(g^{ab}), we use
    # metric_variation which handles the Leibniz rule.
    idx_a = down(:a)
    idx_b = down(:b)

    # Variation of L with respect to g^{ab}
    var_L = metric_variation(expr, metric, idx_a, idx_b)

    # The (1/2) g_{ab} L term arises from varying sqrt(-g):
    #   delta(sqrt(-g)) = (1/2) sqrt(-g) g^{ab} delta(g_{ab})
    #                   = -(1/2) sqrt(-g) g_{ab} delta(g^{ab})
    sqrtg_term = (-1 // 2) * Tensor(metric, [idx_a, idx_b]) * expr

    var_L + sqrtg_term
end

# ── Helpers ──────────────────────────────────────────────────────────

"""Check if an expression is the Ricci scalar (bare Tensor(:RicScalar, []))."""
function _is_ricci_scalar(expr::TensorExpr)
    expr isa Tensor && expr.name == :RicScalar && isempty(expr.indices)
end

"""Construct the Einstein tensor G_{ab} = Ric_{ab} - (1/2) g_{ab} R."""
function _einstein_tensor(metric::Symbol)
    einstein_expr(down(:a), down(:b), metric)
end

"""
    extract_eom_and_theta(lagrangian, field, metric;
                          covd=:D, dim=4,
                          registry=current_registry()) -> (eom, theta)

Convenience wrapper: given a Lagrangian expression (not wrapped in
LagrangianDensity), compute the EOM. Theta is currently not extracted
(returns nothing).

Returns `(eom::TensorExpr, theta::Nothing)`.

# Arguments
- `lagrangian::TensorExpr` -- the Lagrangian density
- `field::Symbol` -- field to vary with respect to
- `metric::Symbol` -- metric tensor name
- `covd::Symbol=:D` -- covariant derivative name (default `:D`)
- `dim::Int=4` -- spacetime dimension
- `registry::TensorRegistry` -- tensor registry

# Examples

```julia
eom, theta = extract_eom_and_theta(
    Tensor(:RicScalar, TIndex[]), :g, :g; covd=:D, dim=4, registry=reg)
# eom == G_{ab}, theta == nothing (to be implemented in TGR-s50.3)
```
"""
function extract_eom_and_theta(lagrangian::TensorExpr, field::Symbol,
                                metric::Symbol;
                                covd::Symbol=:D, dim::Int=4,
                                registry::TensorRegistry=current_registry())
    L = LagrangianDensity(lagrangian, [field], metric, covd, dim)
    result = eom_extract(L, field; registry=registry)
    (result.eom, result.theta)
end
