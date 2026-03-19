#= Noether charge (potential) extraction for the covariant phase space framework.
#
# Given a Noether current J^a that is a divergence on-shell,
#   J^a = nabla_b Q^{ab}
# extract the antisymmetric Noether charge Q^{ab}.
#
# For Einstein-Hilbert gravity with Killing vector xi:
#   Q^{ab} = nabla^a xi^b - nabla^b xi^a   (Komar 2-form)
#
# Reference: Iyer & Wald (1994), PRD 50, 846, Eq 3.3.
=#

"""
    NoetherCharge(expr, xi, current)

Container for the Noether charge Q^{ab} (antisymmetric potential).

Fields:
- `expr::TensorExpr` -- Q^{ab} as a TensorExpr with two free upper indices
- `xi::Symbol` -- the vector field generating the diffeomorphism
- `current::NoetherCurrent` -- the originating Noether current
"""
struct NoetherCharge
    expr::TensorExpr
    xi::Symbol
    current::NoetherCurrent
end

"""
    noether_charge(J::NoetherCurrent, covd::Symbol;
                    registry::TensorRegistry=current_registry()) -> NoetherCharge

Extract the Noether charge Q^{ab} from a Noether current J^a.

On-shell (equations of motion satisfied), the Noether current is a total
divergence: J^a = nabla_b Q^{ab}.  This function extracts Q^{ab} by:

1. Setting the equations of motion to zero (on-shell)
2. Simplifying J^a to its divergence form
3. Extracting the antisymmetric tensor Q^{ab} from nabla_b Q^{ab}

The result satisfies nabla_b Q^{ab} = J^a (on-shell).

# Examples

```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g
define_curvature_tensors!(reg, :M4, :g)
define_covd!(reg, :D; manifold=:M4, metric=:g)
R = Tensor(:RicScalar, TIndex[])
L = LagrangianDensity(R, [:g], :g, :D, 4)
register_tensor!(reg, TensorProperties(
    name=:xi, manifold=:M4, rank=(1, 0),
    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
nc = noether_current(L, :g, :xi; registry=reg)
Q = noether_charge(nc, :D; registry=reg)
# Q.expr is nabla^a xi^b - nabla^b xi^a (the Komar 2-form)
```
"""
function noether_charge(J::NoetherCurrent, covd::Symbol;
                         registry::TensorRegistry=current_registry())
    with_registry(registry) do
        # Go on-shell: set EOM tensor to zero
        # For EH gravity, this means Ein = 0
        set_vanishing!(registry, :Ein)

        # Simplify the current on-shell to get pure divergence
        J_onshell = simplify(J.expr; registry=registry)

        # Extract the divergence: J^a = nabla_b V^{ab}
        # The divergence index b contracts with an up index in the argument
        ok, Q_expr = extract_divergence(J_onshell, covd; registry=registry)

        if !ok
            error("Noether current is not a divergence on-shell; " *
                  "cannot extract Noether charge")
        end

        NoetherCharge(Q_expr, J.xi, J)
    end
end

"""
    noether_charge_eh(xi::Symbol, covd::Symbol;
                       registry::TensorRegistry=current_registry()) -> TensorExpr

Return the explicit Einstein-Hilbert Noether charge (Komar 2-form):

    Q^{ab} = nabla^a xi^b - nabla^b xi^a

This is the antisymmetric tensor whose divergence gives the on-shell
Noether current: nabla_b Q^{ab} = J^a_{on-shell}.

For a Killing vector xi, integrating Q over a codimension-2 surface
gives the Komar integral for conserved charges (mass, angular momentum).

Reference: Iyer & Wald (1994), Eq 3.3; Komar (1959).

# Examples

```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g
define_curvature_tensors!(reg, :M4, :g)
define_covd!(reg, :D; manifold=:M4, metric=:g)
register_tensor!(reg, TensorProperties(
    name=:xi, manifold=:M4, rank=(1, 0),
    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
Q = noether_charge_eh(:xi, :D; registry=reg)
# Q = nabla^a xi^b - nabla^b xi^a
```
"""
function noether_charge_eh(xi::Symbol, covd::Symbol;
                            registry::TensorRegistry=current_registry())
    with_registry(registry) do
        # Q^{ab} = nabla^a xi^b - nabla^b xi^a
        # Use free indices :a, :b for the two antisymmetric slots
        grad_xi_ab = TDeriv(up(:a), Tensor(xi, [up(:b)]), covd)
        grad_xi_ba = TDeriv(up(:b), Tensor(xi, [up(:a)]), covd)
        grad_xi_ab - grad_xi_ba
    end
end
