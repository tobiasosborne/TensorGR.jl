# Spinor decomposition of the Riemann tensor.
#
# The Riemann tensor decomposes as (P&R Vol 1, Eq 4.6.41):
#   R_{abcd} = Psi_{ABCD} eps_{A'B'} eps_{C'D'}           (Weyl, self-dual)
#            + Psi_bar_{A'B'C'D'} eps_{AB} eps_{CD}        (Weyl, anti-self-dual)
#            + Phi_{ACA'C'} eps_{BD} eps_{B'D'}            (trace-free Ricci)
#            - Phi_{ADA'D'} eps_{BC} eps_{B'C'}
#            - Phi_{BCA'C'} ... (antisymmetrized)           (not needed abstractly)
#            + 2 Lambda (eps_{AC} eps_{BD} eps_{A'C'} eps_{B'D'}
#                       - eps_{AD} eps_{BC} eps_{A'D'} eps_{B'C'})  (scalar)
#
# For abstract work, we provide the three irreducible pieces as functions
# that build the corresponding TensorExpr contributions.
#
# Reference: Penrose & Rindler, Spinors and Space-Time Vol 1 (1984),
#            Eqs 4.6.24, 4.6.26, 4.6.41.

"""
    weyl_spinor_expr(; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the Weyl spinor `Psi_{ABCD}` with 4 fresh undotted spinor indices.
"""
function weyl_spinor_expr(; registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    idxs = Symbol[]
    for _ in 1:4
        n = fresh_index(used; vbundle=:SL2C)
        push!(used, n)
        push!(idxs, n)
    end
    Tensor(:Psi, [spin_down(s) for s in idxs])
end

"""
    weyl_spinor_bar_expr(; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the conjugate Weyl spinor `Psi_bar_{A'B'C'D'}` with 4 fresh dotted spinor indices.
"""
function weyl_spinor_bar_expr(; registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    idxs = Symbol[]
    for _ in 1:4
        n = fresh_index(used; vbundle=:SL2C_dot)
        push!(used, n)
        push!(idxs, n)
    end
    Tensor(:Psi_bar, [spin_dot_down(s) for s in idxs])
end

"""
    ricci_spinor_expr(; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the Ricci spinor `Phi_{ABA'B'}` with 2 undotted + 2 dotted fresh spinor indices.
"""
function ricci_spinor_expr(; registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    u1 = fresh_index(used; vbundle=:SL2C)
    push!(used, u1)
    u2 = fresh_index(used; vbundle=:SL2C)
    push!(used, u2)
    d1 = fresh_index(used; vbundle=:SL2C_dot)
    push!(used, d1)
    d2 = fresh_index(used; vbundle=:SL2C_dot)
    Tensor(:Phi_Ricci, [spin_down(u1), spin_down(u2), spin_dot_down(d1), spin_dot_down(d2)])
end

"""
    riemann_spinor_parts(; registry::TensorRegistry=current_registry()) -> NamedTuple

Return the three irreducible spinor parts of the Riemann tensor as a NamedTuple:
- `weyl`: Psi_{ABCD} (self-dual Weyl)
- `weyl_bar`: Psi_bar_{A'B'C'D'} (anti-self-dual Weyl)
- `ricci`: Phi_{ABA'B'} (trace-free Ricci)
- `scalar`: Lambda = R/24 (scalar curvature)

These are the abstract tensors; the full decomposition formula involves
epsilon contractions that depend on the specific index slots of R_{abcd}.
"""
function riemann_spinor_parts(; registry::TensorRegistry=current_registry())
    (
        weyl = weyl_spinor_expr(; registry=registry),
        weyl_bar = weyl_spinor_bar_expr(; registry=registry),
        ricci = ricci_spinor_expr(; registry=registry),
        scalar = lambda_spinor_expr(; registry=registry),
    )
end
