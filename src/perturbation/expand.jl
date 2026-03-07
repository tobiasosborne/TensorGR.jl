#= Higher-order perturbation expansion of curvature tensors.

Partition-based recursion for Christoffel, Riemann, Ricci, and Ricci scalar
perturbations at arbitrary order n, following the xPert algorithm:

  dⁿG = Sum over partitions of n distributed among sub-expressions.

Relies on:
  - MetricPerturbation, perturb, dinverse_metric  (metric_perturbation.jl)
  - all_compositions                               (partitions.jl)
  - TensorExpr AST, tproduct, tsum, ZERO           (types.jl, arithmetic.jl)
  - fresh_index, indices                            (ast/indices.jl)
  - TDeriv, walk                                    (types.jl, ast/walk.jl)
=#

# ────────────────────────────────────────────────────────────────────
# Helper: collect all index names appearing in a set of TIndex values
# ────────────────────────────────────────────────────────────────────

function _collect_used(idxs::TIndex...)
    s = Set{Symbol}()
    for idx in idxs
        push!(s, idx.name)
    end
    s
end

function _collect_used(expr::TensorExpr)
    Set{Symbol}(idx.name for idx in indices(expr))
end

function _collect_used(exprs::Vector{<:TensorExpr})
    s = Set{Symbol}()
    for e in exprs
        for idx in indices(e)
            push!(s, idx.name)
        end
    end
    s
end

# ────────────────────────────────────────────────────────────────────
# Helper: get Christoffel at order k (returns Γ₀ for k=0 on curved bg)
# ────────────────────────────────────────────────────────────────────

"""Return δᵏΓ for k≥1, or the background Christoffel Γ₀ for k=0 on curved background."""
function _get_christoffel_order(mp::MetricPerturbation,
                                 a::TIndex, b::TIndex, c::TIndex, k::Int)
    if k == 0
        if mp.curved && mp.background_christoffel !== nothing
            return Tensor(mp.background_christoffel, [a, b, c])
        else
            return ZERO
        end
    end
    δchristoffel(mp, a, b, c, k)
end

# ────────────────────────────────────────────────────────────────────
# dchristoffel: perturbation of the Christoffel symbol at order n
# ────────────────────────────────────────────────────────────────────

"""
    δchristoffel(mp::MetricPerturbation, a::TIndex, b::TIndex, c::TIndex, order::Int)

Perturbation of the Christoffel symbol Gamma^a_{bc} at the given order.

At order 1:
  δΓ^a_{bc} = (1/2) g^{ad} (∂_b h_{cd} + ∂_c h_{bd} - ∂_d h_{bc})

At order n (partition-based recursion):
  δⁿΓ^a_{bc} = (1/2) Σ_{k+l=n, k>=0, l>=1}
      δᵏ(g^{ad}) (∂_b δˡg_{cd} + ∂_c δˡg_{bd} - ∂_d δˡg_{bc})

Index `a` must be Up; `b` and `c` must be Down.
"""
function δchristoffel(mp::MetricPerturbation, a::TIndex, b::TIndex, c::TIndex, order::Int)
    @assert a.position == Up  "First index of Christoffel must be Up"
    @assert b.position == Down "Second index of Christoffel must be Down"
    @assert c.position == Down "Third index of Christoffel must be Down"
    order <= 0 && return ZERO

    used = _collect_used(a, b, c)
    terms = TensorExpr[]

    for k in 0:order
        l = order - k
        # On flat background, skip l=0 (∂g₀=0 so the term vanishes).
        # On curved background, l=0 contributes via ∂g₀ ≠ 0.
        if l < 1 && !mp.curved
            continue
        end
        l < 0 && continue

        # Fresh dummy index d for the metric contraction g^{ad}
        d = fresh_index(used)
        push!(used, d)

        # δᵏ(g^{ad})
        δk_ginv = δinverse_metric(mp, a, up(d), k)
        δk_ginv == ZERO && continue

        # δˡ(g_{cd}), δˡ(g_{bd}), δˡ(g_{bc})
        δl_gcd = perturb(Tensor(mp.metric, [TIndex(c.name, Down, c.vbundle), down(d)]), mp, l)
        δl_gbd = perturb(Tensor(mp.metric, [TIndex(b.name, Down, b.vbundle), down(d)]), mp, l)
        δl_gbc = perturb(Tensor(mp.metric, [TIndex(b.name, Down, b.vbundle), TIndex(c.name, Down, c.vbundle)]), mp, l)

        # For l>=2, perturb returns ZERO for the metric itself (only order 1 is h).
        # If all three are zero, skip.
        all_zero = (δl_gcd == ZERO && δl_gbd == ZERO && δl_gbc == ZERO)
        all_zero && continue

        # Build the three derivative terms: ∂_b δˡg_{cd} + ∂_c δˡg_{bd} - ∂_d δˡg_{bc}
        deriv_terms = TensorExpr[]
        if δl_gcd != ZERO
            push!(deriv_terms, TDeriv(b, δl_gcd))
        end
        if δl_gbd != ZERO
            push!(deriv_terms, TDeriv(c, δl_gbd))
        end
        if δl_gbc != ZERO
            push!(deriv_terms, -TDeriv(down(d), δl_gbc))
        end

        isempty(deriv_terms) && continue
        bracket = tsum(deriv_terms)

        # (1/2) δᵏ(g^{ad}) * bracket
        # Ensure no dummy clashes between δk_ginv and bracket
        bracket = ensure_no_dummy_clash(δk_ginv, bracket)
        term = tproduct(1 // 2, TensorExpr[δk_ginv, bracket])
        push!(terms, term)
    end

    tsum(terms)
end

# ────────────────────────────────────────────────────────────────────
# driemann: perturbation of the Riemann tensor at order n
# ────────────────────────────────────────────────────────────────────

"""
    δriemann(mp::MetricPerturbation, a::TIndex, b::TIndex, c::TIndex, d::TIndex, order::Int)

Perturbation of the Riemann tensor R^a_{bcd} at the given order.

Uses the standard formula:
  R^a_{bcd} = ∂_c Γ^a_{db} - ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} - Γ^a_{de} Γ^e_{cb}

At order n, expand using the Leibniz rule on partitions:
  δⁿR^a_{bcd} = ∂_c δⁿΓ^a_{db} - ∂_d δⁿΓ^a_{cb}
              + Σ_{k+l=n, k>=1, l>=1} (δᵏΓ^a_{ce} δˡΓ^e_{db} - δᵏΓ^a_{de} δˡΓ^e_{cb})

Index `a` must be Up; `b`, `c`, `d` must be Down.
"""
function δriemann(mp::MetricPerturbation, a::TIndex, b::TIndex,
                   c::TIndex, d::TIndex, order::Int)
    @assert a.position == Up   "First Riemann index must be Up"
    @assert b.position == Down "Second Riemann index must be Down"
    @assert c.position == Down "Third Riemann index must be Down"
    @assert d.position == Down "Fourth Riemann index must be Down"
    order <= 0 && return ZERO

    used = _collect_used(a, b, c, d)
    terms = TensorExpr[]

    # --- Linear part: ∂_c δⁿΓ^a_{db} - ∂_d δⁿΓ^a_{cb} ---
    δnΓ_adb = δchristoffel(mp, a, d, b, order)
    if δnΓ_adb != ZERO
        push!(terms, TDeriv(c, δnΓ_adb))
    end

    δnΓ_acb = δchristoffel(mp, a, c, b, order)
    if δnΓ_acb != ZERO
        push!(terms, -TDeriv(d, δnΓ_acb))
    end

    # --- Quadratic part: Σ_{k+l=n} δᵏΓ^a_{ce} δˡΓ^e_{db} - δᵏΓ^a_{de} δˡΓ^e_{cb} ---
    # On flat background: k≥1, l≥1 (Γ₀=0 so k=0 and l=0 vanish).
    # On curved background: k≥0, l≥0 but skip (k=0,l=0) which is the background R₀.
    k_start = mp.curved ? 0 : 1
    for k in k_start:order-k_start
        l = order - k
        (k == 0 && l == 0) && continue  # background Riemann, not a perturbation

        # Fresh dummy index e for each (k,l) pair
        e = fresh_index(used)
        push!(used, e)

        # δᵏΓ or Γ₀ when k=0 / l=0
        δkΓ_ace = _get_christoffel_order(mp, a, c, down(e), k)
        δlΓ_edb = _get_christoffel_order(mp, up(e), d, b, l)
        if δkΓ_ace != ZERO && δlΓ_edb != ZERO
            δlΓ_edb = ensure_no_dummy_clash(δkΓ_ace, δlΓ_edb)
            push!(terms, tproduct(1 // 1, TensorExpr[δkΓ_ace, δlΓ_edb]))
        end

        # Need another fresh e for the second bilinear term to avoid clash
        e2 = fresh_index(used)
        push!(used, e2)

        δkΓ_ade = _get_christoffel_order(mp, a, d, down(e2), k)
        δlΓ_ecb = _get_christoffel_order(mp, up(e2), c, b, l)
        if δkΓ_ade != ZERO && δlΓ_ecb != ZERO
            δlΓ_ecb = ensure_no_dummy_clash(δkΓ_ade, δlΓ_ecb)
            push!(terms, tproduct(-1 // 1, TensorExpr[δkΓ_ade, δlΓ_ecb]))
        end
    end

    tsum(terms)
end

# ────────────────────────────────────────────────────────────────────
# dricci: perturbation of the Ricci tensor at order n
# ────────────────────────────────────────────────────────────────────

"""
    δricci(mp::MetricPerturbation, a::TIndex, b::TIndex, order::Int)

Perturbation of the Ricci tensor Ric_{ab} at the given order.

The Ricci tensor is the trace of the Riemann tensor:
  Ric_{ab} = R^c_{acb}

So δⁿRic_{ab} = δⁿR^c_{acb}, with `c` a fresh dummy index.
Both `a` and `b` must be Down.
"""
function δricci(mp::MetricPerturbation, a::TIndex, b::TIndex, order::Int)
    @assert a.position == Down "First Ricci index must be Down"
    @assert b.position == Down "Second Ricci index must be Down"
    order <= 0 && return ZERO

    used = _collect_used(a, b)
    c = fresh_index(used)

    δriemann(mp, up(c), a, down(c), b, order)
end

# ────────────────────────────────────────────────────────────────────
# dricci_scalar: perturbation of the Ricci scalar at order n
# ────────────────────────────────────────────────────────────────────

"""
    δricci_scalar(mp::MetricPerturbation, order::Int)

Perturbation of the Ricci scalar R = g^{ab} Ric_{ab} at the given order.

At order n, uses the Leibniz rule:
  δⁿR = Σ_{k+l=n} δᵏ(g^{ab}) δˡ(Ric_{ab})
"""
function δricci_scalar(mp::MetricPerturbation, order::Int)
    order <= 0 && return ZERO

    used = Set{Symbol}()
    terms = TensorExpr[]

    for k in 0:order
        l = order - k
        l < 1 && k < 1 && continue  # need at least one perturbation

        # Fresh indices for this partition
        a = fresh_index(used)
        push!(used, a)
        b = fresh_index(used)
        push!(used, b)

        # δᵏ(g^{ab})
        δk_ginv = δinverse_metric(mp, up(a), up(b), k)
        δk_ginv == ZERO && continue

        # δˡ(Ric_{ab})
        if l == 0
            # Background Ricci tensor
            δl_ric = Tensor(:Ric, [down(a), down(b)])
        else
            δl_ric = δricci(mp, down(a), down(b), l)
        end
        δl_ric == ZERO && continue

        # Ensure no dummy clashes
        δl_ric = ensure_no_dummy_clash(δk_ginv, δl_ric)
        push!(terms, tproduct(1 // 1, TensorExpr[δk_ginv, δl_ric]))
    end

    tsum(terms)
end

# ────────────────────────────────────────────────────────────────────
# expand_perturbation: walk an expression and expand curvature tensors
# ────────────────────────────────────────────────────────────────────

"""
    expand_perturbation(expr::TensorExpr, mp::MetricPerturbation, order::Int)

Walk a tensor expression and expand all curvature tensors at the given
perturbation order.

Dispatches on tensor name:
  - `:Riem`      -> `δriemann`
  - `:Ric`       -> `δricci`
  - `:RicScalar`  -> `δricci_scalar`
  - `:Christoffel` or `:Gamma` -> `δchristoffel`
  - general tensors -> `perturb`

For products, uses the Leibniz rule (partition over factors).
For sums, distributes linearly.
For derivatives, commutes the perturbation through.
"""
function expand_perturbation(expr::TensorExpr, mp::MetricPerturbation, order::Int)
    _expand_pert(expr, mp, order)
end

function _expand_pert(t::Tensor, mp::MetricPerturbation, order::Int)
    if t.name == :Riem
        # Riemann R^a_{bcd} or R_{abcd}: need first index Up
        idxs = t.indices
        length(idxs) == 4 || error("Riemann tensor must have 4 indices, got $(length(idxs))")
        a, b, c, d = idxs
        if a.position == Up
            return δriemann(mp, a, b, c, d, order)
        else
            # All-down Riemann: lower with metric
            # R_{abcd} = g_{ae} R^e_{bcd}
            used = _collect_used(idxs...)
            e = fresh_index(used)
            δR_ebcd = δriemann(mp, up(e), b, c, d, order)
            if δR_ebcd == ZERO
                return ZERO
            end
            # Need the metric contraction g_{ae} at all orders
            metric_terms = TensorExpr[]
            for k in 0:order
                l = order - k
                δk_g = perturb(Tensor(mp.metric, [a, down(e)]), mp, k)
                δk_g == ZERO && continue
                δl_R = l == order ? δR_ebcd : δriemann(mp, up(e), b, c, d, l)
                δl_R == ZERO && continue
                if k == 0 && l == order
                    δl_R = ensure_no_dummy_clash(δk_g, δl_R)
                    push!(metric_terms, tproduct(1 // 1, TensorExpr[δk_g, δl_R]))
                elseif k + l == order
                    δl_R = ensure_no_dummy_clash(δk_g, δl_R)
                    push!(metric_terms, tproduct(1 // 1, TensorExpr[δk_g, δl_R]))
                end
            end
            return tsum(metric_terms)
        end

    elseif t.name == :Ric
        idxs = t.indices
        length(idxs) == 2 || error("Ricci tensor must have 2 indices, got $(length(idxs))")
        a, b = idxs
        if a.position == Down && b.position == Down
            return δricci(mp, a, b, order)
        else
            # Ric with raised index: Ric^a_b = g^{ac} Ric_{cb}
            # Expand via Leibniz on the metric contraction
            used = _collect_used(idxs...)
            if a.position == Up
                c = fresh_index(used)
                push!(used, c)
                terms = TensorExpr[]
                for k in 0:order
                    l = order - k
                    δk_ginv = δinverse_metric(mp, a, up(c), k)
                    δk_ginv == ZERO && continue
                    δl_ric = l > 0 ? δricci(mp, down(c), b, l) : Tensor(:Ric, [down(c), b])
                    δl_ric == ZERO && continue
                    δl_ric = ensure_no_dummy_clash(δk_ginv, δl_ric)
                    push!(terms, tproduct(1 // 1, TensorExpr[δk_ginv, δl_ric]))
                end
                return tsum(terms)
            else
                # b is Up: Ric_{a}^b = g^{bc} Ric_{ac}
                c = fresh_index(used)
                push!(used, c)
                terms = TensorExpr[]
                for k in 0:order
                    l = order - k
                    δk_ginv = δinverse_metric(mp, b, up(c), k)
                    δk_ginv == ZERO && continue
                    δl_ric = l > 0 ? δricci(mp, a, down(c), l) : Tensor(:Ric, [a, down(c)])
                    δl_ric == ZERO && continue
                    δl_ric = ensure_no_dummy_clash(δk_ginv, δl_ric)
                    push!(terms, tproduct(1 // 1, TensorExpr[δk_ginv, δl_ric]))
                end
                return tsum(terms)
            end
        end

    elseif t.name == :RicScalar
        return δricci_scalar(mp, order)

    elseif t.name in (:Christoffel, :Gamma)
        idxs = t.indices
        length(idxs) == 3 || error("Christoffel symbol must have 3 indices, got $(length(idxs))")
        return δchristoffel(mp, idxs[1], idxs[2], idxs[3], order)

    else
        # General tensor: use basic perturb
        return perturb(t, mp, order)
    end
end

function _expand_pert(s::TScalar, ::MetricPerturbation, order::Int)
    order == 0 ? s : ZERO
end

function _expand_pert(s::TSum, mp::MetricPerturbation, order::Int)
    tsum(TensorExpr[_expand_pert(t, mp, order) for t in s.terms])
end

function _expand_pert(p::TProduct, mp::MetricPerturbation, order::Int)
    # Leibniz rule: distribute perturbation order among factors
    factors = p.factors
    nf = length(factors)
    comps = all_compositions(order, nf)

    terms = TensorExpr[]
    for comp in comps
        parts = TensorExpr[]
        valid = true
        for (i, fi) in enumerate(factors)
            pi = _expand_pert(fi, mp, comp[i])
            if pi == ZERO
                valid = false
                break
            end
            push!(parts, pi)
        end
        valid || continue
        # Ensure no dummy clashes between the parts
        resolved = TensorExpr[parts[1]]
        for j in 2:length(parts)
            combined_expr = tproduct(1 // 1, resolved)
            pj = ensure_no_dummy_clash(combined_expr, parts[j])
            push!(resolved, pj)
        end
        push!(terms, tproduct(p.scalar, resolved))
    end
    tsum(terms)
end

function _expand_pert(d::TDeriv, mp::MetricPerturbation, order::Int)
    # Perturbation commutes with partial derivatives
    inner = _expand_pert(d.arg, mp, order)
    inner == ZERO ? ZERO : TDeriv(d.index, inner, d.covd)
end
