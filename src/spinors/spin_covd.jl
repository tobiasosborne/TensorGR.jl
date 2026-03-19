# Spin covariant derivative nabla_{AA'}.
#
# The spinor covariant derivative is defined as:
#   nabla_{AA'} phi_B = sigma^a_{AA'} nabla_a phi_B
#
# Key properties:
#   - Leibniz rule on spinor products
#   - Metric compatibility: nabla_{AA'} epsilon_{BC} = 0
#   - Commutes with complex conjugation
#
# Reference: Penrose & Rindler Vol 1 (1984), Eq 4.4.1.

"""
    spin_covd(expr::TensorExpr, undotted::Symbol, dotted::Symbol;
              covd_name::Symbol=:D,
              registry::TensorRegistry=current_registry()) -> TensorExpr

Apply the spin covariant derivative nabla_{AA'} to `expr`, producing
a TDeriv with a soldering-form decomposition.

The spin derivative is represented as a standard TDeriv on the Tangent
bundle, composed with the soldering form:
  nabla_{AA'} T = sigma^a_{AA'} nabla_a T

# Arguments
- `expr`: the spinor expression to differentiate
- `undotted`: name for the undotted SL2C index (e.g., :A)
- `dotted`: name for the dotted SL2C_dot index (e.g., :Ap)
- `covd_name`: the registered covariant derivative (default `:D`)

# Returns
A product `sigma^a_{AA'} * nabla_a(expr)` with a fresh dummy Tangent index `a`.

# Reference
Penrose & Rindler Vol 1 (1984), Eq 4.4.1.
"""
function spin_covd(expr::TensorExpr, undotted::Symbol, dotted::Symbol;
                   covd_name::Symbol=:D,
                   registry::TensorRegistry=current_registry())
    # Find the soldering form
    sigma_name = _find_soldering_form(registry)
    sigma_name === nothing && error("No soldering form registered; call define_soldering_form! first")

    # Generate a fresh Tangent index for the intermediate derivative
    all_idxs = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idxs)
    push!(used, undotted)
    push!(used, dotted)
    tangent_dummy = fresh_index(used; vbundle=:Tangent)

    # Build nabla_a(expr)
    deriv = TDeriv(TIndex(tangent_dummy, Down, :Tangent), expr, covd_name)

    # Build sigma^a_{AA'}
    sig = Tensor(sigma_name, [
        TIndex(tangent_dummy, Up, :Tangent),
        TIndex(undotted, Down, :SL2C),
        TIndex(dotted, Down, :SL2C_dot)
    ])

    tproduct(1 // 1, TensorExpr[sig, deriv])
end

"""
    spin_covd_expr(expr::TensorExpr;
                   covd_name::Symbol=:D,
                   registry::TensorRegistry=current_registry()) -> TensorExpr

Apply spin covariant derivative with automatically generated fresh spinor
index names. Returns the product `sigma^a_{AA'} nabla_a(expr)`.
"""
function spin_covd_expr(expr::TensorExpr;
                        covd_name::Symbol=:D,
                        registry::TensorRegistry=current_registry())
    all_idxs = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idxs)
    undotted = fresh_index(used; vbundle=:SL2C)
    push!(used, undotted)
    dotted = fresh_index(used; vbundle=:SL2C_dot)

    spin_covd(expr, undotted, dotted; covd_name=covd_name, registry=registry)
end
