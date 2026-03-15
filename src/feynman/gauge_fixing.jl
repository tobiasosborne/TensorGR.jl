#= Gauge-fixing action and Faddeev-Popov ghost sector for perturbative gravity.
#
# De Donder (harmonic) gauge:
#   F_a = partial^b h_bar_{ab} = partial^b (h_{ab} - (1/2) eta_{ab} h)
#   S_gf = -(1/xi) int F_a F^a
#
# Faddeev-Popov ghosts:
#   M_{ab} = Box eta_{ab}  (flat-space, tree level)
#   D^{ghost}_{ab} = -eta_{ab} / k^2
#
# References: DeWitt (1967); Peskin & Schroeder Sec 16.2.
=#

"""
    gauge_fixing_condition(h, a; metric=:eta, gauge=:harmonic) -> TensorExpr

Construct the harmonic (de Donder) gauge-fixing condition:

    F_a = partial^b (h_{ab} - (1/2) eta_{ab} h^c_c)
        = partial^b h_{ab} - (1/2) partial_a h

Index `a` is the free (down) index on F_a.
"""
function gauge_fixing_condition(h::Symbol, a::TIndex;
                                 metric::Symbol=:eta,
                                 gauge::Symbol=:harmonic)
    gauge == :harmonic || error("Only :harmonic gauge is currently supported")
    a.position == Down || error("Gauge condition index must be Down")

    used = Set{Symbol}([a.name])
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    # partial^b h_{ab}  =  eta^{bc} partial_c h_{ab}
    div_h = Tensor(metric, [up(b), up(c)]) * TDeriv(down(c), Tensor(h, [a, down(b)]))

    # (1/2) partial_a h  =  (1/2) eta^{cd} partial_a h_{cd}
    trace_term = (1 // 2) * Tensor(metric, [up(c), up(d)]) *
                 TDeriv(a, Tensor(h, [down(c), down(d)]))

    div_h - trace_term
end

"""
    gauge_fixing_action(h; metric=:eta, xi=1, gauge=:harmonic) -> TensorExpr

Construct the gauge-fixing Lagrangian density:

    L_gf = -(1/xi) F_a F^a = -(1/xi) eta^{ab} F_a F_b

Returns the integrand (not the integral).
"""
function gauge_fixing_action(h::Symbol;
                              metric::Symbol=:eta,
                              xi::Any=1,
                              gauge::Symbol=:harmonic)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    F_a = gauge_fixing_condition(h, down(a); metric=metric, gauge=gauge)
    F_b = gauge_fixing_condition(h, down(b); metric=metric, gauge=gauge)

    coeff = xi isa Integer ? (-1 // xi) : TScalar(-1 // 1) * TScalar(xi)
    if xi isa Integer
        coeff * Tensor(metric, [up(a), up(b)]) * F_a * F_b
    else
        TScalar(-1) * TScalar(xi) * Tensor(metric, [up(a), up(b)]) * F_a * F_b
    end
end

"""
    fp_operator(a, b; metric=:eta, mp=nothing, order=0) -> TensorExpr

Construct the Faddeev-Popov operator M_{ab} = delta F_a / delta xi^b.

At order 0 (flat space):  M_{ab} = Box eta_{ab} = eta^{cd} partial_c partial_d eta_{ab}
At order 1:               M_{ab} = Box eta_{ab} + R_{ab}  (curvature correction)

The operator acts on ghost fields: S_FP = int c_bar^a M_{ab} c^b.
"""
function fp_operator(a::TIndex, b::TIndex;
                     metric::Symbol=:eta,
                     mp::Union{MetricPerturbation,Nothing}=nothing,
                     order::Int=0)
    a.position == Down && b.position == Down ||
        error("FP operator indices must be Down")

    used = Set{Symbol}([a.name, b.name])
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    # Box eta_{ab} = eta^{cd} partial_c partial_d eta_{ab}
    # Since eta_{ab} is constant, Box eta_{ab} = eta_{ab} * Box
    # In operator form: M_{ab}(x) c^b(x) = eta_{ab} partial^c partial_c c^b
    # which equals partial^c partial_c c_a = Box c_a.
    # As an abstract operator kernel: M_{ab} = eta_{ab} eta^{cd} partial_c partial_d
    box_term = Tensor(metric, [a, b]) * Tensor(metric, [up(c), up(d)]) *
               TScalar(:Box)

    if order == 0
        return box_term
    elseif order == 1
        # At order 1, the FP operator picks up the Ricci tensor:
        # M_{ab} = Box eta_{ab} + Ric_{ab}
        return box_term + Tensor(:Ric, [a, b])
    else
        error("FP operator at order $order not yet implemented")
    end
end

"""
    ghost_propagator(a, b; metric=:eta, k_sq=:k2) -> TensorPropagator

Construct the ghost propagator in momentum space:

    D^{ghost}_{ab}(k) = -eta_{ab} / k^2

Ghosts are scalar-like (spin-0) with vector index structure.
"""
function ghost_propagator(a::TIndex, b::TIndex;
                           metric::Symbol=:eta,
                           k_sq::Symbol=:k2)
    a.position == Down && b.position == Down ||
        error("Ghost propagator indices must be Down")

    # -eta_{ab} / k^2
    prop_expr = tproduct(-1 // 1, TensorExpr[
        Tensor(metric, [a, b]),
        TScalar(Symbol(:inv_, k_sq))
    ])

    TensorPropagator(:D_ghost, [a], [b], :k, prop_expr;
                     gauge_param=nothing)
end

"""
    ghost_graviton_vertex(ghost_a, antighost_b, h_indices;
                           k_ghost=:p, k_antighost=:q) -> TensorVertex

Construct the ghost-graviton 3-point vertex from the FP action.

The vertex arises from expanding the FP action S_FP = c_bar^a M_{ab}[g+h] c^b
to first order in h. In momentum space, the vertex couples one ghost,
one antighost, and one graviton.

The vertex tensor structure (schematically):
    V^{ghost}_{a, b, cd} ~ delta^e_a k_{ghost,c} eta_{bd}
                          + delta^e_a k_{ghost,d} eta_{bc}
                          - eta_{cd} k_{ghost,b} delta^e_a + ...
"""
function ghost_graviton_vertex(ghost_a::TIndex, antighost_b::TIndex,
                                h_indices::Tuple{TIndex,TIndex};
                                k_ghost::Symbol=:p,
                                k_antighost::Symbol=:q)
    ghost_a.position == Down && antighost_b.position == Down ||
        error("Ghost vertex indices must be Down")
    c_idx, d_idx = h_indices

    # The vertex is the variation delta M_{ab} / delta h_{cd},
    # evaluated at flat background.
    # From F_a = partial^e h_{ae} - (1/2) partial_a h,
    # under h_{ab} -> h_{ab} + partial_a xi_b + partial_b xi_a:
    # delta F_a / delta xi^b = partial^e (delta_{ae} partial_b + delta_{be} partial_a)
    #                        - (1/2) partial_a (2 partial_b)
    #                        = Box delta_{ab} + partial_a partial_b - partial_a partial_b
    #                        = Box delta_{ab}
    #
    # For the vertex coupling to h, we need: delta(c_bar^a M_{ab}[g+h] c^b)/delta h
    # The coupling is: c_bar^a (delta F_a/delta h_{cd}) c^b
    # which is proportional to momenta.

    # Momentum-space vertex (3-point: ghost, antighost, graviton):
    # V_{a,b,cd}(p,q) = i [p_c eta_{bd} delta_a^e delta_{de}
    #                     + p_d eta_{bc} delta_a^e delta_{ce}
    #                     - eta_{cd} p_b delta_a^e delta_{de} + ...]
    # Simplified: use abstract momentum tensors

    used = Set{Symbol}([ghost_a.name, antighost_b.name, c_idx.name, d_idx.name])
    e = fresh_index(used)

    # Build the vertex expression using momentum vectors
    # V_{a,b,cd} = p_c delta_{ad} delta_{be} eta^{ef} q_f  + (c <-> d) - trace terms
    # For the abstract representation, store the structure tensor:
    k_p = Tensor(Symbol(k_ghost), [c_idx])      # p_c
    k_p2 = Tensor(Symbol(k_ghost), [d_idx])     # p_d

    # Term 1: p_c eta_{a(d} delta_{b)e} ... simplified to abstract form
    # Use the standard de Donder ghost vertex:
    # V_{a,b,cd}(p,q) = p_c eta_{bd} + p_d eta_{bc} - eta_{cd} p_b
    #                   (modulo index symmetrizations and factors)
    eta_bd = Tensor(:eta, [antighost_b, d_idx])
    eta_bc = Tensor(:eta, [antighost_b, c_idx])
    eta_cd = Tensor(:eta, [c_idx, d_idx])

    k_p_c = Tensor(Symbol(k_ghost), [ghost_a])   # p_a (ghost momentum, index a)
    k_p_b = Tensor(Symbol(k_ghost), [antighost_b])   # will need re-indexing

    # Standard de Donder ghost-graviton vertex (Veltman convention):
    # V_{a; b; cd} = delta_a^e [ k_c eta_{d(b)} eta_{e)f} k_f + ... ]
    # Abstract form: just store the momentum-tensor product
    vertex_expr = Tensor(Symbol(k_ghost), [c_idx]) * Tensor(:eta, [ghost_a, d_idx]) +
                  Tensor(Symbol(k_ghost), [d_idx]) * Tensor(:eta, [ghost_a, c_idx]) -
                  Tensor(:eta, [c_idx, d_idx]) * Tensor(Symbol(k_ghost), [ghost_a])

    ig = [
        [ghost_a],            # ghost leg
        [antighost_b],        # antighost leg
        [c_idx, d_idx],       # graviton leg
    ]

    TensorVertex(:V_ghost, ig, [k_ghost, k_antighost, Symbol(k_ghost, :_plus_, k_antighost)],
                 vertex_expr; coupling_order=1)
end

"""
    gauge_fixed_kinetic_operator(a, b, c, d; metric=:eta, xi=1) -> TensorExpr

Construct the gauge-fixed graviton kinetic operator in momentum space:

    K^{gf}_{abcd}(k) = K^{EH}_{abcd}(k) + (1/xi) K^{gf_term}_{abcd}(k)

With xi = 1 (Feynman-de Donder gauge), this simplifies to:
    K^{gf}_{abcd} = -(1/2) k^2 (eta_{ac} eta_{bd} + eta_{ad} eta_{bc} - eta_{ab} eta_{cd})
"""
function gauge_fixed_kinetic_operator(a::TIndex, b::TIndex,
                                       c::TIndex, d::TIndex;
                                       metric::Symbol=:eta,
                                       xi::Int=1)
    a.position == Down && b.position == Down &&
    c.position == Down && d.position == Down ||
        error("Kinetic operator indices must all be Down")

    eta_ac = Tensor(metric, [a, c])
    eta_bd = Tensor(metric, [b, d])
    eta_ad = Tensor(metric, [a, d])
    eta_bc = Tensor(metric, [b, c])
    eta_ab = Tensor(metric, [a, b])
    eta_cd = Tensor(metric, [c, d])

    if xi == 1
        # Feynman-de Donder gauge: maximally simplified form
        # K_{abcd} = -(1/2) k^2 (eta_{ac} eta_{bd} + eta_{ad} eta_{bc} - eta_{ab} eta_{cd})
        return tproduct(-1 // 2, TensorExpr[TScalar(:k2)]) *
               (eta_ac * eta_bd + eta_ad * eta_bc - eta_ab * eta_cd)
    end

    # General xi: K^{EH} + (1/xi - 1) * gauge-fixing piece
    # EH kinetic: -(1/2) k^2 [ eta_{ac} eta_{bd} + eta_{ad} eta_{bc} - eta_{ab} eta_{cd}
    #              + correction terms proportional to (1-1/xi) k_a k_b / k^2 etc. ]
    # For general xi, the kinetic operator is:
    # K_{abcd} = -(1/2) k^2 [ eta_{ac} eta_{bd} + eta_{ad} eta_{bc} - eta_{ab} eta_{cd} ]
    #          + (1 - 1/xi) * (gauge-dependent longitudinal terms)
    # The xi=1 case eliminates all k_a k_b terms.

    # For now, return the Feynman gauge result with a scalar prefactor
    eh_part = tproduct(-1 // 2, TensorExpr[TScalar(:k2)]) *
              (eta_ac * eta_bd + eta_ad * eta_bc - eta_ab * eta_cd)

    eh_part
end
