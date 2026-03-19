#= Curvature spinors: Weyl (Psi), Ricci (Phi), and scalar (Lambda).
#
# The Riemann tensor decomposes into three irreducible spinor parts:
#   R_{abcd} = Psi_{ABCD} epsilon_{A'B'} epsilon_{C'D'} + c.c.
#            + Phi_{ABA'B'} epsilon_{CD} epsilon_{C'D'} + ...
#            + Lambda (epsilon_{AC} epsilon_{BD} epsilon_{A'C'} epsilon_{B'D'} - ...)
#
# - Psi_{ABCD}: Weyl spinor, totally symmetric, 5 complex components (Psi_0..Psi_4)
# - Phi_{ABA'B'}: Ricci spinor, Hermitian, 9 real components (trace-free Ricci)
# - Lambda = R/24: scalar curvature spinor
#
# Ground truth: Penrose & Rindler Vol 1 (1984), Eqs 4.6.24, 4.6.26, 4.6.41.
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

# ── Weyl spinor Psi_{ABCD} ──────────────────────────────────────────────

"""
    define_weyl_spinor!(reg::TensorRegistry; manifold::Symbol=:M4)

Register the Weyl spinor `Psi_{ABCD}` and its conjugate `Psi_bar_{A'B'C'D'}`.

Psi_{ABCD} is totally symmetric in all 4 undotted spinor indices and represents
the self-dual part of the Weyl tensor. It has 5 independent complex components
(the Newman-Penrose scalars Psi_0 ... Psi_4).

Psi_bar_{A'B'C'D'} is the complex conjugate with 4 dotted indices, also
totally symmetric.

Ground truth: Penrose & Rindler Vol 1 (1984), Eq 4.6.41.
"""
function define_weyl_spinor!(reg::TensorRegistry; manifold::Symbol=:M4)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    has_vbundle(reg, :SL2C) || error("SL2C bundle not registered; call define_spinor_bundles! first")
    has_vbundle(reg, :SL2C_dot) || error("SL2C_dot bundle not registered; call define_spinor_bundles! first")

    # Undotted Weyl spinor: Psi_{ABCD}, totally symmetric S_4
    if !has_tensor(reg, :Psi)
        register_tensor!(reg, TensorProperties(
            name=:Psi, manifold=manifold, rank=(0, 4),
            symmetries=SymmetrySpec[FullySymmetric(1, 2, 3, 4)],
            options=Dict{Symbol,Any}(
                :is_curvature_spinor => true,
                :spinor_type => :weyl,
                :vbundle => :SL2C,
                :index_vbundles => [:SL2C, :SL2C, :SL2C, :SL2C])))
    end

    # Conjugate: Psi_bar_{A'B'C'D'}, totally symmetric S_4
    if !has_tensor(reg, :Psi_bar)
        register_tensor!(reg, TensorProperties(
            name=:Psi_bar, manifold=manifold, rank=(0, 4),
            symmetries=SymmetrySpec[FullySymmetric(1, 2, 3, 4)],
            options=Dict{Symbol,Any}(
                :is_curvature_spinor => true,
                :spinor_type => :weyl_conjugate,
                :vbundle => :SL2C_dot,
                :index_vbundles => [:SL2C_dot, :SL2C_dot, :SL2C_dot, :SL2C_dot])))
    end

    nothing
end

# ── Ricci spinor Phi_{ABA'B'} ────────────────────────────────────────────

"""
    define_ricci_spinor!(reg::TensorRegistry; manifold::Symbol=:M4)

Register the Ricci spinor `Phi_{ABA'B'}`: Hermitian, 9 real independent
components, representing the trace-free Ricci tensor in spinor form.

Symmetries: `Phi_{ABA'B'} = Phi_{(AB)(A'B')}` — symmetric in the undotted
pair (slots 1,2) and symmetric in the dotted pair (slots 3,4).

Ground truth: Penrose & Rindler Vol 1 (1984), Eq 4.6.24.
"""
function define_ricci_spinor!(reg::TensorRegistry; manifold::Symbol=:M4)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    has_vbundle(reg, :SL2C) || error("SL2C bundle not registered; call define_spinor_bundles! first")
    has_vbundle(reg, :SL2C_dot) || error("SL2C_dot bundle not registered; call define_spinor_bundles! first")

    if !has_tensor(reg, :Phi_Ricci)
        register_tensor!(reg, TensorProperties(
            name=:Phi_Ricci, manifold=manifold, rank=(0, 4),
            symmetries=SymmetrySpec[Symmetric(1, 2), Symmetric(3, 4)],
            options=Dict{Symbol,Any}(
                :is_curvature_spinor => true,
                :spinor_type => :ricci,
                :vbundle => :mixed,
                :index_vbundles => [:SL2C, :SL2C, :SL2C_dot, :SL2C_dot])))
    end

    nothing
end

"""
    define_curvature_spinors!(reg::TensorRegistry; manifold::Symbol=:M4)

Register all three curvature spinors: Weyl (Psi), Ricci (Phi), and
scalar (Lambda). Convenience function that calls the individual
`define_weyl_spinor!`, `define_ricci_spinor!`, and `define_lambda_spinor!`.
"""
function define_curvature_spinors!(reg::TensorRegistry; manifold::Symbol=:M4)
    define_weyl_spinor!(reg; manifold=manifold)
    define_ricci_spinor!(reg; manifold=manifold)
    define_lambda_spinor!(reg; manifold=manifold)
    nothing
end
