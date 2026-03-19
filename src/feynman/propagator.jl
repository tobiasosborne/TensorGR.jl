#= Graviton propagator in harmonic (de Donder) gauge.
#
# The graviton propagator in Feynman-de Donder gauge (xi=1) is:
#
#   D^{abcd}(k) = (1/k^2) P^{abcd}
#
# where the numerator tensor structure is:
#
#   P^{abcd} = (1/2)(eta^{ac} eta^{bd} + eta^{ad} eta^{bc} - eta^{ab} eta^{cd})
#
# This follows from inverting the gauge-fixed kinetic operator
#   K_{abcd} = -(1/2) k^2 (eta_{ac} eta_{bd} + eta_{ad} eta_{bc} - eta_{ab} eta_{cd})
#
# with the gauge-fixing Lagrangian:
#   L_gf = -(1/2)(partial_a h^a_b - (1/2) partial_b h)^2
#
# Symmetries of P^{abcd}:
#   - P^{abcd} = P^{bacd}   (symmetry in first pair)
#   - P^{abcd} = P^{abdc}   (symmetry in second pair)
#   - P^{abcd} = P^{cdab}   (pair exchange symmetry)
#
# Trace:
#   eta_{ac} P^{abcd} = (1/2)(d * eta^{bd} + eta^{bd} - eta^{bd} * d)
#                      = (1/2)(d+1-d) eta^{bd} = (1/2) eta^{bd}
#   (where d = eta_{ac} eta^{ac} is the spacetime dimension)
#   More precisely: eta_{ac} P^{abcd} = (1/2)(delta^b_c eta^{cd} + delta^d_c eta^{cb} - eta^{ab} delta^d_a)
#   ... which simplifies appropriately with contracted metrics.
#
# Idempotency (approximate):
#   P^{ab}_{ef} P^{efcd} is related to P^{abcd} via dimension-dependent corrections.
#
# References:
#   - DeWitt, Phys. Rev. 162 (1967) 1195
#   - Veltman, in "Methods in Field Theory" (Les Houches 1975)
#   - Donoghue, Phys. Rev. D 50 (1994) 3874, arXiv:gr-qc/9405057
=#

"""
    propagator_numerator(indices::Vector{TIndex}, reg::TensorRegistry;
                         metric::Symbol=:eta) -> TensorExpr

Construct the graviton propagator numerator tensor:

    P^{abcd} = (1/2)(eta^{ac} eta^{bd} + eta^{ad} eta^{bc} - eta^{ab} eta^{cd})

`indices` must be a vector of exactly 4 `TIndex` values. The metric tensor
`metric` (default `:eta`) is used for the flat background.

The numerator satisfies:
- Symmetry under exchange of first pair: P^{abcd} = P^{bacd}
- Symmetry under exchange of second pair: P^{abcd} = P^{abdc}
- Pair exchange symmetry: P^{abcd} = P^{cdab}
"""
function propagator_numerator(indices::Vector{TIndex}, reg::TensorRegistry;
                               metric::Symbol=:eta)
    length(indices) == 4 ||
        error("propagator_numerator requires exactly 4 indices, got $(length(indices))")

    a, b, c, d = indices

    # Build metric tensors for each pairing
    eta_ac = Tensor(metric, [a, c])
    eta_bd = Tensor(metric, [b, d])
    eta_ad = Tensor(metric, [a, d])
    eta_bc = Tensor(metric, [b, c])
    eta_ab = Tensor(metric, [a, b])
    eta_cd = Tensor(metric, [c, d])

    # P^{abcd} = (1/2)(eta^{ac} eta^{bd} + eta^{ad} eta^{bc} - eta^{ab} eta^{cd})
    tproduct(1 // 2, TensorExpr[eta_ac * eta_bd + eta_ad * eta_bc - eta_ab * eta_cd])
end

"""
    graviton_propagator(reg::TensorRegistry;
                        gauge::Symbol=:harmonic,
                        metric::Symbol=:eta,
                        momentum::Symbol=:k,
                        k_sq::Symbol=:k2) -> TensorPropagator

Construct the graviton propagator in harmonic (de Donder) gauge.

The propagator is:

    D^{abcd}(k) = (1/k^2) P^{abcd}

where P^{abcd} = (1/2)(eta^{ac} eta^{bd} + eta^{ad} eta^{bc} - eta^{ab} eta^{cd}).

Returns a `TensorPropagator` with:
- `name = :D_graviton`
- `indices_left = [up(a), up(b)]` (indices at one end)
- `indices_right = [up(c), up(d)]` (indices at the other end)
- `momentum = momentum`
- `expr` = the full propagator expression (numerator / k^2)
- `gauge_param = :harmonic`

Only `:harmonic` gauge is currently supported.
"""
function graviton_propagator(reg::TensorRegistry;
                              gauge::Symbol=:harmonic,
                              metric::Symbol=:eta,
                              momentum::Symbol=:k,
                              k_sq::Symbol=:k2)
    gauge == :harmonic ||
        error("Only :harmonic gauge is currently supported, got :$gauge")

    # Generate fresh indices for the propagator
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    indices_left = [up(a), up(b)]
    indices_right = [up(c), up(d)]

    # Build the numerator P^{abcd}
    all_indices = [up(a), up(b), up(c), up(d)]
    P = propagator_numerator(all_indices, reg; metric=metric)

    # Full propagator: (1/k^2) * P^{abcd}
    # Use TScalar with inv_k2 symbol to represent 1/k^2
    prop_expr = tproduct(1 // 1, TensorExpr[TScalar(Symbol(:inv_, k_sq)), P])

    TensorPropagator(:D_graviton, indices_left, indices_right, momentum, prop_expr;
                     gauge_param=gauge)
end
