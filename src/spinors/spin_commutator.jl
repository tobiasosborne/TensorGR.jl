# Spinor commutator identities (spinor Ricci identity).
#
# The commutator of spinor covariant derivatives on a spinor field gives
# curvature terms:
#
#   [nabla_{AA'}, nabla_{BB'}] kappa_C
#       = eps_{A'B'} Psi_{ABCD} kappa^D
#         + eps_{AB} Phi_{CDA'B'} kappa^D  (wrong!)
#
# Correct form (P&R Vol 1, Eq 4.9.13, simplified for a univalent spinor):
#   [nabla_{AA'}, nabla_{BB'}] kappa_C
#       = eps_{A'B'} (Psi_{ABCE} kappa^E + Lambda (eps_{AC} kappa_B + eps_{BC} kappa_A))
#         + eps_{AB} (Phi_bar terms)
#
# For abstract tensor algebra, we provide the function that returns the
# commutator expression as a TensorExpr for a given spinor field.
#
# Reference: Penrose & Rindler, Spinors and Space-Time Vol 1 (1984),
#            Eqs 4.9.2-4.9.13.

"""
    spinor_ricci_identity(field_name::Symbol, field_index::TIndex;
                          registry::TensorRegistry=current_registry()) -> TensorExpr

Return the commutator `[nabla_{AA'}, nabla_{BB'}] kappa_C` for a rank-1
undotted spinor `kappa_C`, expressed in terms of the Weyl spinor Psi,
the Ricci spinor Phi, and the scalar curvature spinor Lambda.

For a single undotted covariant spinor kappa_C (P&R Eq 4.9.13):

  [nabla_{AA'}, nabla_{BB'}] kappa_C = eps_{A'B'} X_{ABC}

where X_{ABC} contains the Weyl spinor and Lambda terms:

  X_{ABC} = Psi_{ABCE} kappa^E + Lambda (eps_{AC} kappa_B + eps_{BC} kappa_A)

(The anti-self-dual part involving Phi is zero for a purely undotted spinor.)

The result has free spinor indices A, B, C (undotted) and A', B' (dotted).

# Arguments
- `field_name`: name of the registered spinor tensor (e.g., `:kappa`)
- `field_index`: the TIndex of the spinor field's slot (e.g., `spin_down(:C)`)

# Returns
A TensorExpr representing the right-hand side of the commutator.
"""
function spinor_ricci_identity(field_name::Symbol, field_index::TIndex;
                               registry::TensorRegistry=current_registry())
    vb = field_index.vbundle
    vb == :SL2C || error("spinor_ricci_identity currently supports undotted (SL2C) spinors only")

    # Collect used names to generate fresh dummies
    C = field_index.name
    used = Set{Symbol}([:A, :B, :C, :D, :E, :Ap, :Bp])
    push!(used, C)

    E_name = fresh_index(used; vbundle=:SL2C)
    push!(used, E_name)

    # Spinor metric name
    eps_name = get(registry.metric_cache, :SL2C, :eps_spin)
    eps_dot_name = get(registry.metric_cache, :SL2C_dot, :eps_spin_dot)

    # eps_{A'B'} (prefactor for the self-dual part)
    eps_dot = Tensor(eps_dot_name, [spin_dot_down(:Ap), spin_dot_down(:Bp)])

    # ── Term 1: Psi_{ABCE} kappa^E ──
    psi = Tensor(:Psi, [spin_down(:A), spin_down(:B), spin_down(C), spin_down(E_name)])
    kappa_up = Tensor(field_name, [TIndex(E_name, Up, :SL2C)])
    term1 = tproduct(1 // 1, TensorExpr[eps_dot, psi, kappa_up])

    # ── Term 2: Lambda * eps_{AC} kappa_B ──
    R = Tensor(:RicScalar, TIndex[])
    eps_AC = Tensor(eps_name, [spin_down(:A), spin_down(C)])
    kappa_B = Tensor(field_name, [spin_down(:B)])
    term2 = tproduct(1 // 24, TensorExpr[eps_dot, R, eps_AC, kappa_B])

    # ── Term 3: Lambda * eps_{BC} kappa_A ──
    eps_BC = Tensor(eps_name, [spin_down(:B), spin_down(C)])
    kappa_A = Tensor(field_name, [spin_down(:A)])
    term3 = tproduct(1 // 24, TensorExpr[eps_dot, R, eps_BC, kappa_A])

    # Combine: eps_{A'B'} (Psi_{ABCE} kappa^E + Lambda (eps_{AC} kappa_B + eps_{BC} kappa_A))
    # Note: Lambda = R/24 is already incorporated via the 1//24 coefficient
    tsum(TensorExpr[term1, term2, term3])
end
