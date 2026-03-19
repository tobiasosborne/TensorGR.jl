#= Scalar curvature spinor Lambda.
#
# Lambda = R/24 where R is the Ricci scalar. This appears in the
# spinor decomposition of the Riemann tensor:
#   R_{abcd} = Psi_{ABCD} epsilon_{A'B'} epsilon_{C'D'} + c.c.
#            + Phi_{ABA'B'} epsilon_{CD} epsilon_{C'D'}
#            + Lambda (epsilon_{AC} epsilon_{BD} epsilon_{A'C'} epsilon_{B'D'} - ...)
#
# Ground truth: Penrose & Rindler Vol 1 (1984), Eq 4.6.26.
=#

"""
    define_lambda_spinor!(reg::TensorRegistry; manifold::Symbol=:M4)

Register the scalar curvature spinor Lambda = R/24 and its simplification rule.

Lambda has no indices (it is a scalar). The rule Lambda -> (1/24) RicScalar
is registered for automatic simplification.

Ground truth: Penrose & Rindler Vol 1 (1984), Eq 4.6.26.
"""
function define_lambda_spinor!(reg::TensorRegistry; manifold::Symbol=:M4)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    lambda_name = :Lambda_spin

    if !has_tensor(reg, lambda_name)
        register_tensor!(reg, TensorProperties(
            name=lambda_name, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_curvature_spinor => true,
                :spinor_type => :scalar_curvature,
                :definition => "R/24")))
    end

    # Simplification rule: Lambda_spin -> (1/24) RicScalar
    if has_tensor(reg, :RicScalar)
        R = Tensor(:RicScalar, TIndex[])
        make_rule(Tensor(lambda_name, TIndex[]),
                  tproduct(1 // 24, TensorExpr[R]);
                  registry=reg)
    end

    nothing
end

"""
    lambda_spinor_expr(; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the expression Lambda = (1/24) R as a TensorExpr.
"""
function lambda_spinor_expr(; registry::TensorRegistry=current_registry())
    R = Tensor(:RicScalar, TIndex[])
    tproduct(1 // 24, TensorExpr[R])
end
