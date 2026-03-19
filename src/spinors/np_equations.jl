# NP commutator relations and directional derivatives.
#
# The four directional derivatives are:
#   D = l^a nabla_a,  Delta = n^a nabla_a
#   delta = m^a nabla_a,  deltabar = mbar^a nabla_a
#
# The NP commutator equations (NP 1962, Eq 4.1):
#   [D, Delta] = (gamma + gamma_bar)D + (epsilon + epsilon_bar)Delta
#                - (tau_bar + pi)delta - (tau + pi_bar)deltabar
#   ... (3 more)
#
# Reference: Newman & Penrose, J. Math. Phys. 3, 566 (1962), Eq 4.1.

"""
    np_directional_derivative(tetrad_vec::Symbol, expr::TensorExpr;
                              covd_name::Symbol=:D) -> TensorExpr

Apply the NP directional derivative `v^a nabla_a` to `expr`.

Tetrad vector must be one of `:np_l` (D), `:np_n` (Delta),
`:np_m` (delta), `:np_mbar` (deltabar).
"""
function np_directional_derivative(tetrad_vec::Symbol, expr::TensorExpr;
                                   covd_name::Symbol=:D)
    all_idxs = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idxs)
    a = fresh_index(used; vbundle=:Tangent)

    deriv = TDeriv(TIndex(a, Down, :Tangent), expr, covd_name)
    v = Tensor(tetrad_vec, [TIndex(a, Up, :Tangent)])
    tproduct(1 // 1, TensorExpr[v, deriv])
end

"""
    NPCommutatorRelation

Stores one NP commutator relation: `[D1, D2] f` equals a linear combination
of directional derivatives of f with spin-coefficient-valued coefficients.

Fields:
- `d1`, `d2`: tetrad vector names for the two directional derivatives
- `coefficients`: Dict mapping tetrad vector name to its coefficient
  (a TensorExpr built from spin coefficients and their conjugates)
"""
struct NPCommutatorRelation
    d1::Symbol
    d2::Symbol
    coefficients::Dict{Symbol, Rational{Int}}
end

"""
    np_commutator_table() -> Vector{NamedTuple}

Return the 4 NP commutator relations as named tuples describing
which spin coefficients appear as coefficients of each directional
derivative in the commutator.

Format: `(d1, d2, D_coeff, Delta_coeff, delta_coeff, deltabar_coeff)`
where each `*_coeff` is a vector of `(spin_coeff_name, sign)` pairs.

Reference: NP (1962) Eq 4.1a-4.1d.
"""
function np_commutator_table()
    # [D, Delta] = (gamma + gamma_bar)D + (epsilon + epsilon_bar)Delta
    #              - (tau_bar + pi)delta - (tau + pi_bar)deltabar
    comm1 = (
        d1 = :np_l, d2 = :np_n,
        D_coeffs = [(:gamma_np, 1), (:gamma_np, 1)],     # gamma + gamma_bar
        Delta_coeffs = [(:epsilon_np, 1), (:epsilon_np, 1)], # epsilon + epsilon_bar
        delta_coeffs = [(:tau_np, -1), (:pi_np, -1)],     # -(tau_bar + pi)
        deltabar_coeffs = [(:tau_np, -1), (:pi_np, -1)],  # -(tau + pi_bar)
    )

    # [D, delta] = (alpha_bar + beta - pi_bar)D + kappa Delta
    #              - sigma delta - (rho + epsilon - epsilon_bar)deltabar  (... simplified)
    comm2 = (
        d1 = :np_l, d2 = :np_m,
        D_coeffs = [(:alpha_np, 1), (:beta_np, 1), (:pi_np, -1)],
        Delta_coeffs = [(:kappa, 1)],
        delta_coeffs = [(:sigma_np, -1)],
        deltabar_coeffs = [(:rho_np, -1), (:epsilon_np, -1), (:epsilon_np, 1)],
    )

    # [Delta, delta] = -nu_bar D + (tau - alpha_bar - beta)Delta
    #                  + (mu - gamma + gamma_bar)delta + lambda_bar deltabar
    comm3 = (
        d1 = :np_n, d2 = :np_m,
        D_coeffs = [(:nu_np, -1)],
        Delta_coeffs = [(:tau_np, 1), (:alpha_np, -1), (:beta_np, -1)],
        delta_coeffs = [(:mu_np, 1), (:gamma_np, -1), (:gamma_np, 1)],
        deltabar_coeffs = [(:lambda_np, 1)],
    )

    # [delta, deltabar] = (mu_bar - mu)D + (rho_bar - rho)Delta
    #                     + (alpha - beta_bar)delta - (alpha_bar - beta)deltabar  (... simplified)
    comm4 = (
        d1 = :np_m, d2 = :np_mbar,
        D_coeffs = [(:mu_np, 1), (:mu_np, -1)],
        Delta_coeffs = [(:rho_np, 1), (:rho_np, -1)],
        delta_coeffs = [(:alpha_np, 1), (:beta_np, -1)],
        deltabar_coeffs = [(:alpha_np, -1), (:beta_np, 1)],
    )

    [comm1, comm2, comm3, comm4]
end
