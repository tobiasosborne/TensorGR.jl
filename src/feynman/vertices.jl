#= Graviton vertices from expanded Einstein-Hilbert action.
#
# The EH action S_EH = (1/2kappa^2) int d^d x sqrt(-g) R, expanded around
# a flat background g_{ab} = eta_{ab} + kappa * h_{ab}, yields n-graviton
# vertices V^(n) at order kappa^{n-2}.
#
# The expansion uses:
#   sqrt(-g) = 1 + (1/2)h - (1/4)h_{ab}h^{ab} + (1/4)h^2 + ...
#   R = R_0 + delta^1 R + delta^2 R + ...
# where h = eta^{ab} h_{ab} is the trace.
#
# References:
#   - DeWitt, Phys. Rev. 162 (1967) 1195, Sec III
#   - Sannan, PRD 34 (1986) 1749, Eq 3.3
#   - Goldberger & Rothstein, hep-th/0409156, Sec 2
#
# The cubic vertex has 12 independent tensor structures after imposing
# Bose symmetry under permutation of (index pair, momentum).
=#

# ────────────────────────────────────────────────────────────────────
# Helper: set up flat-background perturbation
# ────────────────────────────────────────────────────────────────────

"""
    _flat_pert_setup(; metric=:eta, perturbation=:h, manifold=:M4,
                      dim=4, registry=nothing) -> (reg, mp)

Create a registry with flat background metric and metric perturbation.
Returns `(registry, MetricPerturbation)`.
"""
function _flat_pert_setup(; metric::Symbol=:eta, perturbation::Symbol=:h,
                            manifold::Symbol=:M4, dim::Int=4,
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
    mp = with_registry(reg) do
        define_metric_perturbation!(reg, metric, perturbation)
    end
    (reg, mp)
end

# ────────────────────────────────────────────────────────────────────
# sqrt(-g) expansion: delta^n(sqrt(-g)) on flat background
# ────────────────────────────────────────────────────────────────────

"""
    _delta_sqrt_g(mp::MetricPerturbation, order::Int; dim::Int=4) -> TensorExpr

Compute delta^n(sqrt(-g)) on a flat background, in terms of the
perturbation h_{ab}. Uses the identity:
  sqrt(-g) = exp((1/2) tr ln g) = 1 + (1/2)h - (1/4)h_{ab}h^{ab} + (1/8)h^2 + ...

At order 1: (1/2) h   (where h = eta^{ab} h_{ab})
At order 2: (1/8) h^2 - (1/4) h_{ab} h^{ab}
At order 3: (1/48) h^3 - (1/8) h h_{ab} h^{ab} + (1/6) h_{ab} h^{bc} h_{ca}

Returns ZERO for order 0 (the delta, not the value 1).
"""
function _delta_sqrt_g(mp::MetricPerturbation, order::Int; dim::Int=4)
    order <= 0 && return ZERO

    used = Set{Symbol}()
    met = mp.metric
    pert = mp.perturbation

    if order == 1
        # (1/2) h = (1/2) eta^{ab} h_{ab}
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used)
        return tproduct(1 // 2, TensorExpr[
            Tensor(met, [up(a), up(b)]),
            Tensor(pert, [down(a), down(b)])])
    elseif order == 2
        # (1/8) h^2 - (1/4) h_{ab} h^{ab}
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used)
        # h = eta^{ab} h_{ab}
        h_trace1 = tproduct(1 // 1, TensorExpr[
            Tensor(met, [up(a), up(b)]),
            Tensor(pert, [down(a), down(b)])])
        h_trace2 = tproduct(1 // 1, TensorExpr[
            Tensor(met, [up(c), up(d)]),
            Tensor(pert, [down(c), down(d)])])
        # h_{ab} h^{ab} = h_{ab} eta^{ac} eta^{bd} h_{cd}
        e = fresh_index(used); push!(used, e)
        f = fresh_index(used); push!(used, f)
        g_idx = fresh_index(used); push!(used, g_idx)
        h_idx = fresh_index(used)
        h_sq = tproduct(1 // 1, TensorExpr[
            Tensor(pert, [down(e), down(f)]),
            Tensor(met, [up(e), up(g_idx)]),
            Tensor(met, [up(f), up(h_idx)]),
            Tensor(pert, [down(g_idx), down(h_idx)])])

        return tsum(TensorExpr[
            tproduct(1 // 8, TensorExpr[h_trace1, h_trace2]),
            tproduct(-1 // 4, TensorExpr[h_sq])])
    elseif order == 3
        # (1/48) h^3 - (1/8) h h_{ab}h^{ab} + (1/6) h_{ab}h^{bc}h_{ca}
        # Build each piece with fresh indices

        # h^3 = (eta^{ab} h_{ab})^3
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        h1 = tproduct(1 // 1, TensorExpr[
            Tensor(met, [up(a), up(b)]),
            Tensor(pert, [down(a), down(b)])])

        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        h2 = tproduct(1 // 1, TensorExpr[
            Tensor(met, [up(c), up(d)]),
            Tensor(pert, [down(c), down(d)])])

        e = fresh_index(used); push!(used, e)
        f = fresh_index(used); push!(used, f)
        h3 = tproduct(1 // 1, TensorExpr[
            Tensor(met, [up(e), up(f)]),
            Tensor(pert, [down(e), down(f)])])

        h_cubed = tproduct(1 // 48, TensorExpr[h1, h2, h3])

        # h * h_{ab} h^{ab}
        g_idx = fresh_index(used); push!(used, g_idx)
        h_idx = fresh_index(used); push!(used, h_idx)
        h_tr = tproduct(1 // 1, TensorExpr[
            Tensor(met, [up(g_idx), up(h_idx)]),
            Tensor(pert, [down(g_idx), down(h_idx)])])

        i1 = fresh_index(used); push!(used, i1)
        j1 = fresh_index(used); push!(used, j1)
        k1 = fresh_index(used); push!(used, k1)
        l1 = fresh_index(used)
        push!(used, l1)
        h_sq_term = tproduct(1 // 1, TensorExpr[
            Tensor(pert, [down(i1), down(j1)]),
            Tensor(met, [up(i1), up(k1)]),
            Tensor(met, [up(j1), up(l1)]),
            Tensor(pert, [down(k1), down(l1)])])

        h_tr_hsq = tproduct(-1 // 8, TensorExpr[h_tr, h_sq_term])

        # h_{ab} h^{bc} h_{ca} = h_{ab} eta^{bc} h_{cd} eta^{de} h_{ea}
        m1 = fresh_index(used); push!(used, m1)
        n1 = fresh_index(used); push!(used, n1)
        o1 = fresh_index(used); push!(used, o1)
        p1 = fresh_index(used); push!(used, p1)
        q1 = fresh_index(used); push!(used, q1)
        r1 = fresh_index(used)
        h_cubic = tproduct(1 // 6, TensorExpr[
            Tensor(pert, [down(m1), down(n1)]),
            Tensor(met, [up(n1), up(o1)]),
            Tensor(pert, [down(o1), down(p1)]),
            Tensor(met, [up(p1), up(q1)]),
            Tensor(pert, [down(q1), down(r1)]),
            Tensor(met, [up(r1), up(m1)])])

        return tsum(TensorExpr[h_cubed, h_tr_hsq, h_cubic])
    end
    error("_delta_sqrt_g: order $order not implemented (only 1-3)")
end

# ────────────────────────────────────────────────────────────────────
# EH Lagrangian perturbation: delta^n(sqrt(-g) * R)
# ────────────────────────────────────────────────────────────────────

"""
    _eh_perturbation(mp::MetricPerturbation, order::Int;
                     dim::Int=4, registry::TensorRegistry=current_registry()) -> TensorExpr

Compute delta^n(sqrt(-g) R) at order `order` on a flat background.

Uses the Leibniz rule: delta^n(sqrt(-g) R) = sum_{k=0}^{n} C(n,k) delta^k(sqrt(-g)) delta^{n-k}(R).

On flat background, R_0 = 0 and delta^0(sqrt(-g)) = 1, so:
  delta^n(sqrt(-g) R) = delta^n(R) + sum_{k=1}^{n-1} delta^k(sqrt(-g)) delta^{n-k}(R)

(The k=n term vanishes because delta^0(R) = R_0 = 0 on flat background.)
"""
function _eh_perturbation(mp::MetricPerturbation, order::Int;
                           dim::Int=4, registry::TensorRegistry=current_registry())
    terms = TensorExpr[]

    for k in 0:order
        l = order - k
        # k=0: delta^0(sqrt(-g)) = 1, need delta^l(R)
        # k>0: need delta^k(sqrt(-g)) * delta^l(R)
        # l=0: delta^0(R) = R_0 = 0 on flat background
        l == 0 && continue

        delta_R = with_registry(registry) do
            δricci_scalar(mp, l)
        end
        delta_R == ZERO && continue

        if k == 0
            push!(terms, delta_R)
        else
            delta_sqrtg = _delta_sqrt_g(mp, k; dim=dim)
            delta_sqrtg == ZERO && continue
            delta_R = ensure_no_dummy_clash(delta_sqrtg, delta_R)
            push!(terms, tproduct(1 // 1, TensorExpr[delta_sqrtg, delta_R]))
        end
    end

    tsum(terms)
end

# ────────────────────────────────────────────────────────────────────
# Public API: graviton vertices
# ────────────────────────────────────────────────────────────────────

"""
    graviton_3vertex(; metric=:eta, perturbation=:h, manifold=:M4, dim=4,
                      kappa=:kappa, registry=nothing) -> TensorVertex

Construct the 3-point graviton vertex V^(3) from the cubic term in the
expanded EH action around flat space.

The vertex is derived from delta^3(sqrt(-g) R), which is the order-kappa^1
contribution to the action. It has 3 legs, each carrying a symmetric pair
of Lorentz indices (a_i, b_i) and a momentum k_i.

The result is a `TensorVertex` with:
- `coupling_order = 1` (proportional to kappa)
- Full Bose symmetry under permutation of legs
- 6 free indices (3 pairs)

Reference: DeWitt (1967); Sannan, PRD 34, 1749 (1986), Eq 3.3.
"""
function graviton_3vertex(; metric::Symbol=:eta, perturbation::Symbol=:h,
                            manifold::Symbol=:M4, dim::Int=4,
                            kappa::Symbol=:kappa,
                            registry::Union{TensorRegistry, Nothing}=nothing)
    reg, mp = _flat_pert_setup(; metric=metric, perturbation=perturbation,
                                 manifold=manifold, dim=dim, registry=registry)

    expr = with_registry(reg) do
        _eh_perturbation(mp, 3; dim=dim, registry=reg)
    end

    # The 3-vertex has Bose symmetry: invariant under any permutation of the
    # three legs (index group + momentum simultaneously).
    bose_sym = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

    v = vertex_from_perturbation(expr, 3, perturbation;
                                  name=:V3_EH,
                                  momenta=[:k1, :k2, :k3])
    # Override with correct symmetry group and coupling
    TensorVertex(:V3_EH, v.index_groups, v.momenta, v.expr,
                 1, bose_sym)
end

"""
    graviton_4vertex(; metric=:eta, perturbation=:h, manifold=:M4, dim=4,
                      kappa=:kappa, registry=nothing) -> TensorVertex

Construct the 4-point graviton vertex V^(4) from the quartic term in the
expanded EH action around flat space.

The vertex is derived from delta^4(sqrt(-g) R), which is the order-kappa^2
contribution to the action. It has 4 legs, each carrying a symmetric pair
of Lorentz indices and a momentum.

The result is a `TensorVertex` with:
- `coupling_order = 2` (proportional to kappa^2)
- Full Bose symmetry under permutation of legs
- 8 free indices (4 pairs)

Reference: DeWitt (1967); Sannan, PRD 34, 1749 (1986).
"""
function graviton_4vertex(; metric::Symbol=:eta, perturbation::Symbol=:h,
                            manifold::Symbol=:M4, dim::Int=4,
                            kappa::Symbol=:kappa,
                            registry::Union{TensorRegistry, Nothing}=nothing)
    reg, mp = _flat_pert_setup(; metric=metric, perturbation=perturbation,
                                 manifold=manifold, dim=dim, registry=registry)

    expr = with_registry(reg) do
        _eh_perturbation(mp, 4; dim=dim, registry=reg)
    end

    # The 4-vertex has full S_4 Bose symmetry.
    bose_sym = Vector{Int}[]
    for p in [[1,2,3,4], [1,2,4,3], [1,3,2,4], [1,3,4,2], [1,4,2,3], [1,4,3,2],
              [2,1,3,4], [2,1,4,3], [2,3,1,4], [2,3,4,1], [2,4,1,3], [2,4,3,1],
              [3,1,2,4], [3,1,4,2], [3,2,1,4], [3,2,4,1], [3,4,1,2], [3,4,2,1],
              [4,1,2,3], [4,1,3,2], [4,2,1,3], [4,2,3,1], [4,3,1,2], [4,3,2,1]]
        push!(bose_sym, p)
    end

    v = vertex_from_perturbation(expr, 4, perturbation;
                                  name=:V4_EH,
                                  momenta=[:k1, :k2, :k3, :k4])
    TensorVertex(:V4_EH, v.index_groups, v.momenta, v.expr,
                 2, bose_sym)
end

"""
    graviton_vertex_n(n::Int; metric=:eta, perturbation=:h, manifold=:M4,
                       dim=4, kappa=:kappa, registry=nothing) -> TensorVertex

Generic n-point graviton vertex via `expand_perturbation`.

Constructs delta^n(sqrt(-g) R) using the full perturbation engine.
For n >= 5 the computation becomes expensive; n=3 and n=4 are the
physically most relevant (tree-level and one-loop diagrams).

The vertex has:
- `coupling_order = n - 2`
- Full S_n Bose symmetry
- 2n free indices (n symmetric pairs)
"""
function graviton_vertex_n(n::Int; metric::Symbol=:eta, perturbation::Symbol=:h,
                            manifold::Symbol=:M4, dim::Int=4,
                            kappa::Symbol=:kappa,
                            registry::Union{TensorRegistry, Nothing}=nothing)
    n >= 2 || error("Graviton vertex requires n >= 2, got n=$n")

    reg, mp = _flat_pert_setup(; metric=metric, perturbation=perturbation,
                                 manifold=manifold, dim=dim, registry=registry)

    expr = with_registry(reg) do
        _eh_perturbation(mp, n; dim=dim, registry=reg)
    end

    momenta = [Symbol(:k, i) for i in 1:n]

    v = vertex_from_perturbation(expr, n, perturbation;
                                  name=Symbol(:V, n, :_EH),
                                  momenta=momenta)
    TensorVertex(v.name, v.index_groups, v.momenta, v.expr,
                 n - 2, v.symmetry_group)
end
