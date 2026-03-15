#= First law of black hole mechanics / Hamiltonian variation
#
# Ties together the Noether current J and charge Q with the symplectic
# potential Theta to produce the Hamiltonian variation and Wald entropy.
#
# Key identity (Iyer & Wald 1994, Eq 3.5):
#   delta H_xi = integral_{partial Sigma} [delta Q_xi - xi . Theta]
#
# Wald entropy (Iyer & Wald 1994, Eq 4.1; Wald 1993):
#   S = -2 pi integral_H (partial L / partial R_{abcd}) epsilon_{ab} epsilon_{cd}
# which for EH reduces to S = A / 4G.
#
# For the Noether charge Q^{ab} this becomes the integrand:
#   S_integrand = 2 pi Q^{ab}  (evaluated on the bifurcation surface,
#                                where xi = 0 and nabla^{[a} xi^{b]} = epsilon^{ab})
#
# Reference: Iyer & Wald (1994), PRD 50, 846, Eqs 3.5, 4.1.
=#

"""
    HamiltonianVariation(expr, charge, potential, xi)

Container for the Hamiltonian variation boundary integrand
delta H_xi = delta Q - xi . Theta  (Iyer-Wald Eq 3.5).

Fields:
- `expr::TensorExpr` -- the boundary integrand with two free upper indices
- `charge::NoetherCharge` -- the Noether charge Q^{ab}
- `potential::SymplecticPotential` -- the symplectic potential Theta^a
- `xi::Symbol` -- the diffeomorphism generator
"""
struct HamiltonianVariation
    expr::TensorExpr
    charge::NoetherCharge
    potential::SymplecticPotential
    xi::Symbol
end

"""
    hamiltonian_variation(Q::NoetherCharge, Theta::SymplecticPotential,
                          xi::Symbol; delta_field::Symbol=:delta_g,
                          registry::TensorRegistry=current_registry()) -> HamiltonianVariation

Compute the Hamiltonian variation boundary integrand (Iyer-Wald Eq 3.5):

    delta H_xi = integral_{partial Sigma} [delta Q_xi - xi . Theta]

The integrand is a (d-2)-form (antisymmetric 2-index tensor) on the
boundary surface. For Einstein-Hilbert gravity:

    delta Q^{ab} = nabla^a delta xi^b - nabla^b delta xi^a
                 = 0 for a fixed xi (background symmetry generator)

so the full variation comes from -xi . Theta.

# Arguments
- `Q::NoetherCharge` -- the Noether charge
- `Theta::SymplecticPotential` -- the symplectic potential
- `xi::Symbol` -- the diffeomorphism generator vector field
- `delta_field::Symbol` -- name for the metric variation (default `:delta_g`)
- `registry::TensorRegistry` -- tensor registry

# Examples

```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g
define_curvature_tensors!(reg, :M4, :g)
define_covd!(reg, :D; manifold=:M4, metric=:g)
register_tensor!(reg, TensorProperties(
    name=:xi, manifold=:M4, rank=(1, 0),
    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
Q_expr = noether_charge_eh(:xi, :D; registry=reg)
R = Tensor(:RicScalar, TIndex[])
L = LagrangianDensity(R, [:g], :g, :D, 4)
sp = symplectic_potential(L, :g; registry=reg)
nc = NoetherCurrent(Tensor(:J, [up(:a)]), :xi, L, sp)
charge = NoetherCharge(Q_expr, :xi, nc)
H = hamiltonian_variation(charge, sp, :xi; registry=reg)
```
"""
function hamiltonian_variation(Q::NoetherCharge, Theta::SymplecticPotential,
                               xi::Symbol; delta_field::Symbol=:delta_g,
                               registry::TensorRegistry=current_registry())
    with_registry(registry) do
        # delta Q^{ab}: variation of the Noether charge with respect to the
        # dynamical field. For a fixed background symmetry generator xi, the
        # charge Q depends on the metric through the covariant derivative,
        # so delta Q picks up connection variation terms.
        #
        # For EH: Q^{ab} = nabla^a xi^b - nabla^b xi^a
        # delta Q = delta(nabla^a xi^b) - delta(nabla^b xi^a)
        #         = (delta Gamma terms)
        #
        # We represent delta Q as a formal variation of Q.expr.
        # For the full first-law computation, we need the linearized Q.
        delta_Q = _vary_charge(Q.expr, delta_field, Theta.lagrangian.metric,
                               Theta.lagrangian.covd, registry)

        # xi . Theta: contraction of xi with the symplectic potential.
        # Theta^a has one free index. We contract xi_b Theta^b to get a scalar,
        # then the (d-2)-form structure comes from the surface element.
        #
        # More precisely, for the boundary integral on a codim-2 surface S,
        # the integrand is the (d-2)-form:
        #   (delta Q - xi . Theta)^{ab} dS_{ab}
        #
        # xi . Theta means: insert xi into Theta viewed as a (d-1)-form,
        # producing a (d-2)-form. In index notation with Theta carrying
        # an antisymmetric pair via the surface, this is:
        #   (xi . Theta)^{ab} = xi^a Theta^b - xi^b Theta^a
        #
        # This gives the antisymmetric contraction.
        used = Set{Symbol}([:a, :b])

        xi_a = Tensor(xi, [up(:a)])
        xi_b = Tensor(xi, [up(:b)])

        # Theta with free index renamed to avoid clash
        theta_b = _rename_free_index(Theta.expr, :a, :b)
        theta_a = Theta.expr  # free index is :a

        # xi . Theta as antisymmetric (d-2)-form: xi^a Theta^b - xi^b Theta^a
        xi_dot_theta = xi_a * theta_b - xi_b * theta_a

        # Full integrand: delta Q^{ab} - (xi . Theta)^{ab}
        integrand = delta_Q - xi_dot_theta

        HamiltonianVariation(integrand, Q, Theta, xi)
    end
end

"""
    _vary_charge(Q_expr, delta_field, metric, covd, reg) -> TensorExpr

Compute the variation delta Q^{ab} of the Noether charge expression.

For the Noether charge Q^{ab} = nabla^a xi^b - nabla^b xi^a,
the variation arises from varying the covariant derivative (connection).
This produces delta-Christoffel terms contracted with xi.

For a general expression, we return a formal variation tensor.
"""
function _vary_charge(Q_expr::TensorExpr, delta_field::Symbol,
                      metric::Symbol, covd::Symbol, reg::TensorRegistry)
    # For the EH charge Q^{ab} = D^a xi^b - D^b xi^a,
    # delta Q^{ab} = (delta Gamma^a_{cd}) g^{cd} xi^b - ...
    # This is the linearized covariant derivative variation.
    #
    # We represent this as a formal tensor delta_Q^{ab} with the right structure.
    # The full computation would require the linearized connection,
    # which is available via delta_christoffel.
    #
    # For practical use in the first law, delta Q is evaluated on-shell
    # and on the bifurcation surface where xi = 0, so only the
    # nabla xi terms survive.
    delta_Q_name = Symbol(:delta_Q_, delta_field)
    if !has_tensor(reg, delta_Q_name)
        register_tensor!(reg, TensorProperties(
            name=delta_Q_name, manifold=:M4,
            rank=(2, 0),
            symmetries=Any[AntiSymmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_variation => true,
                                     :parent_charge => Q_expr)))
    end
    Tensor(delta_Q_name, [up(:a), up(:b)])
end

"""
    _rename_free_index(expr, old_name, new_name) -> TensorExpr

Rename a free index in an expression from `old_name` to `new_name`.
"""
function _rename_free_index(expr::TensorExpr, old_name::Symbol, new_name::Symbol)
    rename_dummies(expr, Dict(old_name => new_name))
end

"""
    hamiltonian_variation_eh(xi::Symbol, covd::Symbol;
                              delta_field::Symbol=:delta_g,
                              registry::TensorRegistry=current_registry()) -> TensorExpr

Return the explicit EH Hamiltonian variation integrand.

For Einstein-Hilbert gravity, the Hamiltonian variation on the boundary is
(Iyer-Wald Eq 3.5, specialized to L = R):

    (delta H_xi)^{ab} = delta(nabla^a xi^b - nabla^b xi^a)
                        - xi^a Theta^b + xi^b Theta^a

where Theta^a = g^{bc}(nabla^a delta g_{bc} - nabla_c delta g^a_b).

# Examples

```julia
H = hamiltonian_variation_eh(:xi, :D; registry=reg)
```
"""
function hamiltonian_variation_eh(xi::Symbol, covd::Symbol;
                                   delta_field::Symbol=:delta_g,
                                   registry::TensorRegistry=current_registry())
    with_registry(registry) do
        # delta Q^{ab}: formal variation of the Komar 2-form
        delta_Q_name = Symbol(:delta_Q_, delta_field)
        if !has_tensor(registry, delta_Q_name)
            register_tensor!(registry, TensorProperties(
                name=delta_Q_name, manifold=:M4,
                rank=(2, 0),
                symmetries=Any[AntiSymmetric(1, 2)],
                options=Dict{Symbol,Any}(:is_variation => true)))
        end
        dQ = Tensor(delta_Q_name, [up(:a), up(:b)])

        # Theta with free index :a, then rename for :b
        theta_a = _theta_eh_free(:a, covd, delta_field)
        theta_b = _theta_eh_free(:b, covd, delta_field)

        # xi . Theta antisymmetrized
        xi_a = Tensor(xi, [up(:a)])
        xi_b = Tensor(xi, [up(:b)])
        xi_dot_theta = xi_a * theta_b - xi_b * theta_a

        dQ - xi_dot_theta
    end
end

"""
Build the EH symplectic potential with a specified free index name.
"""
function _theta_eh_free(free_idx::Symbol, covd::Symbol, delta_field::Symbol)
    used = Set{Symbol}([free_idx])
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used)

    g_inv = Tensor(:g, [up(b), up(c)])
    dg_bc = Tensor(delta_field, [down(b), down(c)])
    dg_fb = Tensor(delta_field, [up(free_idx), down(b)])

    term1 = TDeriv(up(free_idx), dg_bc, covd)
    term2 = TDeriv(down(c), dg_fb, covd)

    g_inv * (term1 - term2)
end

"""
    WaldEntropyIntegrand(expr, charge, xi)

Container for the Wald entropy integrand on the bifurcation surface.

The Wald entropy is (Iyer-Wald 1994, Eq 4.1):
    S = -2 pi integral_H epsilon_{ab} (partial L / partial R_{abcd}) epsilon_{cd}

For a Noether charge Q, this becomes:
    S = 2 pi integral_H Q^{ab} dS_{ab}  (on bifurcation surface where xi=0)

Fields:
- `expr::TensorExpr` -- 2 pi Q^{ab}, the entropy integrand
- `charge::NoetherCharge` -- the originating Noether charge
- `xi::Symbol` -- the Killing vector (horizon generator)
"""
struct WaldEntropyIntegrand
    expr::TensorExpr
    charge::NoetherCharge
    xi::Symbol
end

"""
    wald_entropy_integrand(Q::NoetherCharge;
                            registry::TensorRegistry=current_registry()) -> WaldEntropyIntegrand

Compute the Wald entropy integrand from the Noether charge.

The Wald entropy formula (Iyer & Wald 1994, Eq 4.1; Wald 1993) is:

    S = -2 pi integral_{H} (partial L / partial R_{abcd}) epsilon_{ab} epsilon_{cd}

In terms of the Noether charge Q^{ab}, evaluated on the bifurcation surface
(where the Killing vector xi vanishes), this is:

    S = 2 pi integral_H Q^{ab} dS_{ab}

The integrand returned is `2 pi Q^{ab}` as an abstract tensor expression.

For EH gravity (L = R/(16 pi G)):
    Q^{ab} = -(1/16piG)(nabla^a xi^b - nabla^b xi^a)

On the bifurcation surface, nabla^{[a} xi^{b]} = kappa epsilon^{ab}
(where kappa is surface gravity), giving:
    S_integrand = -(2pi/16piG) * 2 kappa epsilon^{ab}
    S = A / 4G  (Bekenstein-Hawking entropy)

# Examples

```julia
Q_expr = noether_charge_eh(:xi, :D; registry=reg)
nc = NoetherCurrent(...)
charge = NoetherCharge(Q_expr, :xi, nc)
W = wald_entropy_integrand(charge; registry=reg)
# W.expr = 2 pi Q^{ab}
```
"""
function wald_entropy_integrand(Q::NoetherCharge;
                                 registry::TensorRegistry=current_registry())
    with_registry(registry) do
        # S = 2 pi integral_H Q^{ab} dS_{ab}
        # The integrand is 2 pi * Q^{ab}
        integrand = (2 // 1) * TScalar(:pi) * Q.expr

        WaldEntropyIntegrand(integrand, Q, Q.xi)
    end
end

"""
    wald_entropy_integrand_eh(xi::Symbol, covd::Symbol;
                               registry::TensorRegistry=current_registry()) -> TensorExpr

Return the explicit EH Wald entropy integrand.

For Einstein-Hilbert gravity:
    S_integrand = 2 pi * (nabla^a xi^b - nabla^b xi^a)
                = 2 pi * Q^{ab}_{EH}

This is the raw integrand before evaluation on the bifurcation surface.
"""
function wald_entropy_integrand_eh(xi::Symbol, covd::Symbol;
                                    registry::TensorRegistry=current_registry())
    with_registry(registry) do
        Q = noether_charge_eh(xi, covd; registry=registry)
        (2 // 1) * TScalar(:pi) * Q
    end
end
