#= Matter-graviton coupling vertices for point particles and scalar fields.
#
# Point-particle action (EFT of post-Newtonian gravity):
#   S_pp = -m int d tau = -m int sqrt(-g_{mu nu} dx^mu dx^nu)
#
# Expanding g_{mu nu} = eta_{mu nu} + kappa h_{mu nu}:
#   n=1: linear coupling ~ (m/2)(2 v^a v^b - eta^{ab})
#   n=2: quadratic coupling from sqrt(-g) expansion
#
# Minimally-coupled scalar field:
#   S_phi = -(1/2) int sqrt(-g) g^{ab} partial_a phi partial_b phi
#   T^{ab} = partial^a phi partial^b phi - (1/2) eta^{ab} (partial phi)^2
#
# References:
#   - Goldberger & Rothstein, hep-th/0409156 (2006), Sec 2
#   - DeWitt, Phys. Rev. 162 (1967) 1195
=#

# ────────────────────────────────────────────────────────────────────
# Helper: set up registry with flat metric and velocity/particle tensors
# ────────────────────────────────────────────────────────────────────

"""
    _matter_setup(; metric=:eta, manifold=:M4, dim=4,
                    particle=:m, registry=nothing) -> (reg, velocity_name)

Create a registry with flat background metric and register the
velocity tensor v^a for the particle. Returns `(registry, velocity_name)`.
"""
function _matter_setup(; metric::Symbol=:eta, manifold::Symbol=:M4,
                         dim::Int=4, particle::Symbol=:m,
                         registry::Union{TensorRegistry, Nothing}=nothing)
    reg = registry !== nothing ? registry : TensorRegistry()
    if !has_manifold(reg, manifold)
        register_manifold!(reg, ManifoldProperties(
            manifold, dim, metric, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j,
             :k, :l, :m, :n, :o, :p, :q, :r, :s, :t,
             :u, :v, :w, :x, :y, :z,
             :a1, :b1, :c1, :d1, :e1, :f1, :g1, :h1,
             :i1, :j1, :k1, :l1, :m1, :n1]))
        register_tensor!(reg, TensorProperties(
            name=metric, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=manifold, rank=(1, 1),
            is_delta=true))
    end

    # Register velocity tensor v^a (contravariant vector)
    vel_name = Symbol(:v_, particle)
    if !has_tensor(reg, vel_name)
        register_tensor!(reg, TensorProperties(
            name=vel_name, manifold=manifold, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_velocity => true,
                                     :particle => particle)))
    end

    (reg, vel_name)
end

# ────────────────────────────────────────────────────────────────────
# Point-particle matter-graviton vertices
# ────────────────────────────────────────────────────────────────────

"""
    matter_graviton_vertex(n::Int, reg::TensorRegistry;
                            particle::Symbol=:m,
                            metric::Symbol=:eta,
                            manifold::Symbol=:M4,
                            dim::Int=4) -> TensorVertex

Construct the `n`-graviton coupling vertex for a point particle from
the expansion of the worldline action S_pp = -m int sqrt(-g_{ab} v^a v^b).

For `n=1`, the vertex is the linearized stress-energy tensor coupling:
    V^{(1)}_{ab} = (m/2)(2 v_a v_b - eta_{ab})

where v^a is the particle 4-velocity and the overall factor of m/2 comes
from expanding sqrt(-g) to linear order in h_{ab}.

For `n=2`, the vertex is the quadratic coupling from the next order
in the sqrt(-g) expansion:
    V^{(2)}_{ab,cd} = (m/8)(8 v_a v_c eta_{bd} - 4 v_a v_b eta_{cd}
                        - 4 eta_{ac} eta_{bd} + 2 eta_{ab} eta_{cd} + ...)

The `n`-graviton vertex has `n` pairs of symmetric indices and is
proportional to kappa^n from the metric expansion g = eta + kappa h.

Reference: Goldberger & Rothstein, hep-th/0409156, Sec 2.
"""
function matter_graviton_vertex(n::Int, reg::TensorRegistry;
                                 particle::Symbol=:m,
                                 metric::Symbol=:eta,
                                 manifold::Symbol=:M4,
                                 dim::Int=4)
    n >= 1 || error("Matter-graviton vertex requires n >= 1, got n=$n")

    reg_out, vel_name = _matter_setup(; metric=metric, manifold=manifold,
                                        dim=dim, particle=particle,
                                        registry=reg)

    if n == 1
        return _matter_vertex_1(reg_out, vel_name, particle, metric)
    elseif n == 2
        return _matter_vertex_2(reg_out, vel_name, particle, metric)
    else
        error("matter_graviton_vertex: order n=$n not yet implemented (only n=1,2)")
    end
end

"""
    _matter_vertex_1(reg, vel_name, particle, metric) -> TensorVertex

1-graviton vertex for point particle: V^{(1)}_{ab} = (m/2)(2 v_a v_b - eta_{ab}).

This is the linearized stress-energy tensor coupling h^{ab} T_{ab} where
T_{ab} = m v_a v_b for a point particle (with the trace-reversed structure
arising from the sqrt(-g) expansion).
"""
function _matter_vertex_1(reg::TensorRegistry, vel_name::Symbol,
                           particle::Symbol, metric::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    # v_a v_b = v^c eta_{ca} v^d eta_{db} -- but we store as abstract down-index velocities
    # For the vertex, we use the covariant velocity v_a = eta_{ac} v^c
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    # Term 1: 2 * (m/2) * v_a v_b = m * v_a v_b
    # v_a = eta_{ac} v^c
    v_a = tproduct(1 // 1, TensorExpr[
        Tensor(metric, [down(a), down(c)]),
        Tensor(vel_name, [up(c)])])
    v_b = tproduct(1 // 1, TensorExpr[
        Tensor(metric, [down(b), down(d)]),
        Tensor(vel_name, [up(d)])])

    # m * v_a v_b
    term_vv = tproduct(1 // 1, TensorExpr[
        TScalar(particle), v_a, v_b])

    # Term 2: -(m/2) * eta_{ab}
    term_eta = tproduct(-1 // 2, TensorExpr[
        TScalar(particle), Tensor(metric, [down(a), down(b)])])

    # V^{(1)}_{ab} = m v_a v_b - (m/2) eta_{ab}
    vertex_expr = tsum(TensorExpr[term_vv, term_eta])

    ig = [[down(a), down(b)]]
    momenta = [:k1]

    TensorVertex(Symbol(:V1_pp_, particle), ig, momenta, vertex_expr;
                 coupling_order=1)
end

"""
    _matter_vertex_2(reg, vel_name, particle, metric) -> TensorVertex

2-graviton vertex for point particle from the quadratic term in the
sqrt(-g) expansion of the worldline action.

The expansion of sqrt(-det(eta + kappa h)) to second order gives:
  delta^2 sqrt(-g_worldline) = (m/8)[2(v^a v^b h_{ab})^2 - (v^a v^b h_{ab})(h^c_c)
                                      + (1/4)(h^c_c)^2 - (1/2) h_{cd} h^{cd}
                                      + 2 v^a h_{ac} h^{cb} v_b + ...]

We construct the abstract vertex tensor with two graviton legs (ab) and (cd),
symmetric under exchange of the two legs.
"""
function _matter_vertex_2(reg::TensorRegistry, vel_name::Symbol,
                           particle::Symbol, metric::Symbol)
    used = Set{Symbol}()

    # Indices for the two graviton legs
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)

    # Additional contraction indices
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g_idx = fresh_index(used); push!(used, g_idx)
    h_idx = fresh_index(used)

    # Term 1: (m/2) v_a v_c eta_{bd}
    # v_a = eta_{ae} v^e, v_c = eta_{cf} v^f
    term1 = tproduct(1 // 2, TensorExpr[
        TScalar(particle),
        Tensor(metric, [down(a), down(e)]),
        Tensor(vel_name, [up(e)]),
        Tensor(metric, [down(c), down(f)]),
        Tensor(vel_name, [up(f)]),
        Tensor(metric, [down(b), down(d)])])

    # Term 2: -(m/4) v_a v_b eta_{cd}
    term2 = tproduct(-1 // 4, TensorExpr[
        TScalar(particle),
        Tensor(metric, [down(a), down(g_idx)]),
        Tensor(vel_name, [up(g_idx)]),
        Tensor(metric, [down(b), down(h_idx)]),
        Tensor(vel_name, [up(h_idx)]),
        Tensor(metric, [down(c), down(d)])])

    # Term 3: -(m/8) eta_{ac} eta_{bd} (from -(1/4) h_{ab} h^{ab} piece)
    i1 = fresh_index(used); push!(used, i1)
    j1 = fresh_index(used)
    term3 = tproduct(-1 // 8, TensorExpr[
        TScalar(particle),
        Tensor(metric, [down(a), down(c)]),
        Tensor(metric, [down(b), down(d)])])

    # Term 4: (m/16) eta_{ab} eta_{cd} (from (1/8) h^2 piece)
    term4 = tproduct(1 // 16, TensorExpr[
        TScalar(particle),
        Tensor(metric, [down(a), down(b)]),
        Tensor(metric, [down(c), down(d)])])

    vertex_expr = tsum(TensorExpr[term1, term2, term3, term4])

    ig = [[down(a), down(b)], [down(c), down(d)]]
    momenta = [:k1, :k2]

    # S_2 Bose symmetry: exchange of the two graviton legs
    bose_sym = [[1, 2], [2, 1]]

    TensorVertex(Symbol(:V2_pp_, particle), ig, momenta, vertex_expr,
                 2, bose_sym)
end

# ────────────────────────────────────────────────────────────────────
# Scalar matter-graviton vertices
# ────────────────────────────────────────────────────────────────────

"""
    scalar_matter_vertex(n::Int, reg::TensorRegistry;
                          field::Symbol=:phi,
                          metric::Symbol=:eta,
                          manifold::Symbol=:M4,
                          dim::Int=4) -> TensorVertex

Construct the `n`-graviton coupling vertex for a minimally-coupled
scalar field from the action:

    S_phi = -(1/2) int sqrt(-g) g^{ab} partial_a phi partial_b phi

For `n=1`, the vertex is the scalar field stress-energy tensor:
    T_{ab} = partial_a phi partial_b phi - (1/2) eta_{ab} (partial phi)^2

This couples to the graviton via h^{ab} T_{ab}.

For `n=2`, the vertex comes from expanding both sqrt(-g) and g^{ab}
to second order in h, giving terms quadratic in h and quadratic in
partial phi.

Reference: Goldberger & Rothstein, hep-th/0409156, Sec 2.
"""
function scalar_matter_vertex(n::Int, reg::TensorRegistry;
                                field::Symbol=:phi,
                                metric::Symbol=:eta,
                                manifold::Symbol=:M4,
                                dim::Int=4)
    n >= 1 || error("Scalar matter vertex requires n >= 1, got n=$n")

    # Set up registry with flat metric
    if !has_manifold(reg, manifold)
        register_manifold!(reg, ManifoldProperties(
            manifold, dim, metric, :partial,
            [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j,
             :k, :l, :m, :n, :o, :p, :q, :r, :s, :t,
             :u, :v, :w, :x, :y, :z,
             :a1, :b1, :c1, :d1, :e1, :f1, :g1, :h1,
             :i1, :j1, :k1, :l1, :m1, :n1]))
        register_tensor!(reg, TensorProperties(
            name=metric, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            is_metric=true))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=manifold, rank=(1, 1),
            is_delta=true))
    end

    # Register scalar field
    if !has_tensor(reg, field)
        register_tensor!(reg, TensorProperties(
            name=field, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[]))
    end

    if n == 1
        return _scalar_vertex_1(reg, field, metric)
    elseif n == 2
        return _scalar_vertex_2(reg, field, metric)
    else
        error("scalar_matter_vertex: order n=$n not yet implemented (only n=1,2)")
    end
end

"""
    _scalar_vertex_1(reg, field, metric) -> TensorVertex

1-graviton vertex for minimally-coupled scalar field:
    T_{ab} = partial_a phi partial_b phi - (1/2) eta_{ab} (partial_c phi)(partial^c phi)

This is the stress-energy tensor of the scalar field on flat background.
"""
function _scalar_vertex_1(reg::TensorRegistry, field::Symbol, metric::Symbol)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    # Term 1: partial_a phi partial_b phi
    dphi_a = TDeriv(down(a), Tensor(field, TIndex[]))
    dphi_b = TDeriv(down(b), Tensor(field, TIndex[]))
    term1 = tproduct(1 // 1, TensorExpr[dphi_a, dphi_b])

    # Term 2: -(1/2) eta_{ab} eta^{cd} partial_c phi partial_d phi
    dphi_c = TDeriv(down(c), Tensor(field, TIndex[]))
    dphi_d = TDeriv(down(d), Tensor(field, TIndex[]))
    term2 = tproduct(-1 // 2, TensorExpr[
        Tensor(metric, [down(a), down(b)]),
        Tensor(metric, [up(c), up(d)]),
        dphi_c, dphi_d])

    vertex_expr = tsum(TensorExpr[term1, term2])

    ig = [[down(a), down(b)]]
    momenta = [:k1]

    TensorVertex(Symbol(:V1_scalar_, field), ig, momenta, vertex_expr;
                 coupling_order=1)
end

"""
    _scalar_vertex_2(reg, field, metric) -> TensorVertex

2-graviton vertex for minimally-coupled scalar field from the quadratic
expansion of sqrt(-g) g^{ab} partial_a phi partial_b phi.

At second order in h, there are contributions from:
1. delta^2(g^{ab}) * (dphi)_a (dphi)_b
2. delta^1(sqrt(-g)) * delta^1(g^{ab}) * (dphi)_a (dphi)_b
3. delta^2(sqrt(-g)) * (dphi)_a (dphi)_b
"""
function _scalar_vertex_2(reg::TensorRegistry, field::Symbol, metric::Symbol)
    used = Set{Symbol}()

    # Graviton leg indices
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)

    # Scalar derivative indices
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g_idx = fresh_index(used); push!(used, g_idx)
    h_idx = fresh_index(used)

    # The second-order vertex from expanding the scalar kinetic term
    # involves h_{ab} h_{cd} structures contracted with partial phi terms.
    #
    # From delta^2(g^{ef}) = h^{ea} h_a^f (with indices raised by eta):
    # delta^2(-1/2 sqrt(-g) g^{ef} dphi_e dphi_f) gives several terms.
    #
    # Leading structure: h_{ae} h_{b}^{e} dphi^a dphi^b type terms.
    # For the abstract vertex, we store the key structures.

    # Term 1: (1/2) eta_{ac} partial^b phi partial^d phi
    # (from h^{ae} h_{e}^{c} -> delta^2(g^{ac}) contribution)
    dphi_b = TDeriv(down(b), Tensor(field, TIndex[]))
    dphi_d = TDeriv(down(d), Tensor(field, TIndex[]))
    term1 = tproduct(1 // 2, TensorExpr[
        Tensor(metric, [down(a), down(c)]),
        Tensor(metric, [up(e), up(b)]),
        dphi_b,  # use b-named deriv index to tie with up(b)
        Tensor(metric, [up(f), up(d)]),
        dphi_d])

    # Wait -- we need the derivative indices to be independent of the free indices.
    # Let me rebuild this properly.

    # The vertex has free indices (a,b) on leg 1 and (c,d) on leg 2.
    # The scalar field derivatives carry contracted indices.

    e2 = fresh_index(used); push!(used, e2)
    f2 = fresh_index(used)

    # Term 1: eta_{ac} (partial_b phi)(partial_d phi)
    # Contribution from expanding g^{ef} to get h^{ac} coupling to dphi dphi
    dphi_e2 = TDeriv(down(e2), Tensor(field, TIndex[]))
    dphi_f2 = TDeriv(down(f2), Tensor(field, TIndex[]))

    t1 = tproduct(1 // 2, TensorExpr[
        Tensor(metric, [down(a), down(c)]),
        Tensor(metric, [up(e2), down(b)]),
        dphi_e2,
        Tensor(metric, [up(f2), down(d)]),
        dphi_f2])

    # Term 2: -(1/4) eta_{ab} (partial_c phi)(partial_d phi)
    # (from delta^1(sqrt(-g)) * delta^1(g^{ef}) cross term)
    g2 = fresh_index(used); push!(used, g2)
    h2 = fresh_index(used)
    dphi_g2 = TDeriv(down(g2), Tensor(field, TIndex[]))
    dphi_h2 = TDeriv(down(h2), Tensor(field, TIndex[]))
    t2 = tproduct(-1 // 4, TensorExpr[
        Tensor(metric, [down(a), down(b)]),
        Tensor(metric, [up(g2), down(c)]),
        dphi_g2,
        Tensor(metric, [up(h2), down(d)]),
        dphi_h2])

    # Term 3: -(1/4) eta_{cd} (partial_a phi)(partial_b phi)
    # (from delta^1(sqrt(-g)) * delta^1(g^{ef}) cross term, other leg)
    i2 = fresh_index(used); push!(used, i2)
    j2 = fresh_index(used); push!(used, j2)
    dphi_i2 = TDeriv(down(i2), Tensor(field, TIndex[]))
    dphi_j2 = TDeriv(down(j2), Tensor(field, TIndex[]))
    t3 = tproduct(-1 // 4, TensorExpr[
        Tensor(metric, [down(c), down(d)]),
        Tensor(metric, [up(i2), down(a)]),
        dphi_i2,
        Tensor(metric, [up(j2), down(b)]),
        dphi_j2])

    # Term 4: (1/4) eta_{ab} eta_{cd} eta^{gh} (partial_g phi)(partial_h phi)
    # (from delta^2(sqrt(-g)) * kinetic term)
    k2 = fresh_index(used); push!(used, k2)
    l2 = fresh_index(used)
    dphi_k2 = TDeriv(down(k2), Tensor(field, TIndex[]))
    dphi_l2 = TDeriv(down(l2), Tensor(field, TIndex[]))
    t4 = tproduct(1 // 4, TensorExpr[
        Tensor(metric, [down(a), down(b)]),
        Tensor(metric, [down(c), down(d)]),
        Tensor(metric, [up(k2), up(l2)]),
        dphi_k2, dphi_l2])

    vertex_expr = tsum(TensorExpr[t1, t2, t3, t4])

    ig = [[down(a), down(b)], [down(c), down(d)]]
    momenta = [:k1, :k2]

    # S_2 Bose symmetry under exchange of the two graviton legs
    bose_sym = [[1, 2], [2, 1]]

    TensorVertex(Symbol(:V2_scalar_, field), ig, momenta, vertex_expr,
                 2, bose_sym)
end
