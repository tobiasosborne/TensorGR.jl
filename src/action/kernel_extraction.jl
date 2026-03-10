#= Kinetic kernel extraction for rank-2 bilinear actions.

Given δ²S bilinear in a symmetric rank-2 field h, decompose into
per-term bilinear data: each term = coefficient × h(left) × h(right).

This decomposition enables spin projection via Barnes-Rivers operators
without requiring a single canonical-position 4-index kernel tensor,
which would be awkward when h appears at mixed index positions across terms.
=#

"""
    KineticKernel

Bilinear decomposition of a quadratic action δ²S into terms of the form
`coefficient × h(left_indices) × h(right_indices)`.

Used by [`spin_project`](@ref) to compute spin-sector form factors.
"""
struct KineticKernel
    field::Symbol
    terms::Vector{@NamedTuple{coeff::TensorExpr, left::Vector{TIndex}, right::Vector{TIndex}}}
end

function Base.show(io::IO, K::KineticKernel)
    println(io, "KineticKernel(:$(K.field), $(length(K.terms)) bilinear terms)")
end

"""
    extract_kernel(expr, field; registry=current_registry()) -> KineticKernel

Decompose a bilinear expression into per-term `(coefficient, h₁_indices, h₂_indices)`.

The expression should be a TSum (or single term) that is quadratic in `field`.
Each TProduct term is split into the two field factors and everything else.

# Example
```julia
K = extract_kernel(fourier_δ2S, :h)
result = spin_project(K, :spin2; registry=reg)
```
"""
function extract_kernel(expr::TensorExpr, field::Symbol;
                        registry = current_registry())
    expanded = expand_products(expr)
    raw_terms = expanded isa TSum ? expanded.terms : TensorExpr[expanded]

    bilinears = @NamedTuple{coeff::TensorExpr, left::Vector{TIndex}, right::Vector{TIndex}}[]

    for term in raw_terms
        sc, factors = _kernel_term_parts(term)
        h_pos = findall(f -> f isa Tensor && f.name == field, factors)
        length(h_pos) == 2 || continue

        h1, h2 = factors[h_pos[1]], factors[h_pos[2]]
        coeff_factors = TensorExpr[factors[i] for i in eachindex(factors) if i ∉ h_pos]
        coeff = isempty(coeff_factors) ? TScalar(sc) : tproduct(sc, coeff_factors)

        push!(bilinears, (coeff = coeff, left = collect(h1.indices), right = collect(h2.indices)))
    end

    KineticKernel(field, bilinears)
end

"""
    spin_project(K::KineticKernel, spin; dim=4, metric=:g, k_name=:k, k_sq=:k²,
                 registry=current_registry()) -> TensorExpr

Project the kinetic kernel onto a spin sector using Barnes-Rivers projectors.

For each bilinear term, standardizes h indices to all-down position with fresh
names (to prevent projector self-contraction from shared indices), builds the
projector P^J, contracts with the coefficient via inserted metric tensors,
and sums. Returns the scalar form factor (function of k²).

`spin` is one of: `:spin2`, `:spin1`, `:spin0s`, `:spin0w`.
"""
function spin_project(K::KineticKernel, spin::Symbol;
                      dim::Int = 4, metric::Symbol = :g,
                      k_name::Symbol = :k, k_sq = :k²,
                      registry = current_registry())
    projections = TensorExpr[]

    for bt in K.terms
        # Standardize h indices: lower all to Down with fresh names.
        # This prevents projector self-contraction when left/right share names.
        new_left, new_right, metric_factors = _standardize_h_indices(
            bt.left, bt.right, metric)

        μ, ν = new_left[1], new_left[2]
        ρ, σ = new_right[1], new_right[2]

        P = _kernel_build_projector(spin, μ, ν, ρ, σ; dim, metric, k_name, k_sq)

        # Combine projector + metric connectors + coefficient
        all_factors = TensorExpr[P]
        append!(all_factors, metric_factors)
        combined = tproduct(1 // 1, all_factors)
        combined = ensure_no_dummy_clash(bt.coeff, combined)
        push!(projections, combined * bt.coeff)
    end

    with_registry(registry) do
        expr = tsum(projections)
        # Iterate: simplify exposes momentum pairs, contract_momenta exposes
        # scalar cancellations, until stable.
        for _ in 1:5
            expr = expand_products(expr)
            expr = contract_momenta(expr; k_name, k_sq)
            next = simplify(expr; registry = registry, maxiter = 40)
            next == expr && break
            expr = next
        end
        expr
    end
end

"""
    _standardize_h_indices(left, right, metric) -> (new_left, new_right, metric_factors)

Lower all h indices to Down position, inserting metric tensors to preserve
contractions. Returns fresh all-Down indices and the metric connectors.

After lowering, left and right are guaranteed to have disjoint index names
(since Up indices get fresh names, and originally-Down indices are untouched).
"""
function _standardize_h_indices(left::Vector{TIndex}, right::Vector{TIndex},
                                metric::Symbol)
    # Collect all index names to avoid when generating fresh names
    all_names = Set{Symbol}()
    for idx in left
        push!(all_names, idx.name)
    end
    for idx in right
        push!(all_names, idx.name)
    end

    metric_factors = TensorExpr[]
    new_left = TIndex[]
    new_right = TIndex[]

    # Lower left indices: Up → fresh Down, with metric connector
    for idx in left
        if idx.position == Up
            fn = fresh_index(all_names)
            push!(all_names, fn)
            push!(new_left, TIndex(fn, Down, idx.vbundle))
            # g^{old, fresh} connects original Up index to new Down index
            push!(metric_factors, Tensor(metric, [idx, TIndex(fn, Up, idx.vbundle)]))
        else
            push!(new_left, idx)
        end
    end

    # Lower right indices: Up → fresh Down, with metric connector
    for idx in right
        if idx.position == Up
            fn = fresh_index(all_names)
            push!(all_names, fn)
            push!(new_right, TIndex(fn, Down, idx.vbundle))
            push!(metric_factors, Tensor(metric, [idx, TIndex(fn, Up, idx.vbundle)]))
        else
            push!(new_right, idx)
        end
    end

    # Handle case where left and right still share Down index names
    # (from same-position pairs that fix_dummy_positions didn't catch)
    left_names = Set(i.name for i in new_left)
    for (j, idx) in enumerate(new_right)
        if idx.name in left_names
            fn = fresh_index(all_names)
            push!(all_names, fn)
            # Connect old Down name to fresh Down name via g^{old, fresh}
            push!(metric_factors, Tensor(metric,
                [TIndex(idx.name, Up, idx.vbundle), TIndex(fn, Up, idx.vbundle)]))
            new_right[j] = TIndex(fn, Down, idx.vbundle)
        end
    end

    new_left, new_right, metric_factors
end

function _kernel_build_projector(spin::Symbol, μ, ν, ρ, σ; dim, metric, k_name, k_sq)
    kw = (; metric, k_name, k_sq)
    if spin == :spin2
        spin2_projector(μ, ν, ρ, σ; dim, kw...)
    elseif spin == :spin1
        spin1_projector(μ, ν, ρ, σ; kw...)
    elseif spin == :spin0s
        spin0s_projector(μ, ν, ρ, σ; dim, kw...)
    elseif spin == :spin0w
        spin0w_projector(μ, ν, ρ, σ; kw...)
    else
        error("Unknown spin sector: $spin. Use :spin2, :spin1, :spin0s, or :spin0w.")
    end
end

# ─── contract_momenta ────────────────────────────────────────────────

"""
    contract_momenta(expr; k_name=:k, k_sq=:k²) -> TensorExpr

Contract momentum pairs `k_a k^a` → `TScalar(k²)` in product terms.
Also simplifies `TScalar(1/k²) × TScalar(k²) → TScalar(1)`.
"""
function contract_momenta(expr::TensorExpr; k_name::Symbol = :k, k_sq = :k²)
    _contract_momenta(expr, k_name, k_sq)
end

function _contract_momenta(s::TSum, k_name, k_sq)
    tsum(TensorExpr[_contract_momenta(t, k_name, k_sq) for t in s.terms])
end

function _contract_momenta(p::TProduct, k_name, k_sq)
    factors = collect(p.factors)
    scalar = p.scalar
    changed = true

    while changed
        changed = false
        for i in eachindex(factors)
            fi = factors[i]
            fi isa Tensor && fi.name == k_name && length(fi.indices) == 1 || continue
            for j in (i+1):length(factors)
                fj = factors[j]
                fj isa Tensor && fj.name == k_name && length(fj.indices) == 1 || continue
                if fi.indices[1].name == fj.indices[1].name &&
                   fi.indices[1].position != fj.indices[1].position &&
                   fi.indices[1].vbundle == fj.indices[1].vbundle
                    # Contracted pair k_a k^a → k²
                    factors[i] = TScalar(k_sq)
                    deleteat!(factors, j)
                    changed = true
                    break
                end
            end
            changed && break
        end
    end

    # Simplify TScalar(1/k²) × TScalar(k²) pairs
    _simplify_k_sq_pairs!(factors, k_sq)

    tproduct(scalar, factors)
end

_contract_momenta(t::Tensor, _, _) = t
_contract_momenta(s::TScalar, _, _) = s
function _contract_momenta(d::TDeriv, k_name, k_sq)
    TDeriv(d.index, _contract_momenta(d.arg, k_name, k_sq), d.covd)
end

function _simplify_k_sq_pairs!(factors, k_sq)
    k_sq isa Symbol || return  # non-Symbol k_sq handled by scalar simplification
    inv_expr = :(1 / $k_sq)
    i = 1
    while i <= length(factors)
        fi = factors[i]
        if fi isa TScalar && fi.val == k_sq
            j = findfirst(factors) do fj
                fj isa TScalar && _is_inverse_k_sq(fj.val, k_sq)
            end
            if j !== nothing
                factors[i] = TScalar(1)
                deleteat!(factors, j > i ? j : (i = i; j))
                continue
            end
        end
        i += 1
    end
end

_is_inverse_k_sq(val, k_sq) = val == :(1 / $k_sq)

# ─── Helpers ─────────────────────────────────────────────────────────

function _kernel_term_parts(t::TProduct)
    (t.scalar, collect(t.factors))
end
function _kernel_term_parts(t::Tensor)
    (1 // 1, TensorExpr[t])
end
function _kernel_term_parts(t::TScalar)
    (1 // 1, TensorExpr[t])
end
function _kernel_term_parts(t::TSum)
    # Single-term sum (shouldn't happen, but handle gracefully)
    length(t.terms) == 1 ? _kernel_term_parts(t.terms[1]) : (1 // 1, TensorExpr[t])
end

# ─── Direct momentum-space kernel builders ──────────────────────────
# These construct bilinear kernels directly in Fourier space using known
# linearized curvature formulas, avoiding the position-space perturbation
# engine and its index normalization issues.

"""
    build_FP_momentum_kernel(reg) -> KineticKernel

Fierz-Pauli EH quadratic Lagrangian in Fourier space:
L_FP = (1/2)k² h_{ab}h^{ab} - k_bk_c h^{ab}h^c_a + k_ak_b h^{ab}h - (1/2)k²h²
"""
function build_FP_momentum_kernel(reg)
    t1 = (1//2) * TScalar(:k²) * Tensor(:h, [down(:a), down(:b)]) * Tensor(:h, [up(:a), up(:b)])
    t2 = (-1//1) * Tensor(:k, [down(:b)]) * Tensor(:k, [down(:c)]) * Tensor(:h, [up(:a), up(:b)]) * Tensor(:h, [up(:c), down(:a)])
    t3 = (1//1) * Tensor(:k, [down(:a)]) * Tensor(:k, [down(:b)]) * Tensor(:h, [up(:a), up(:b)]) * Tensor(:h, [up(:c), down(:c)])
    t4 = (-1//2) * TScalar(:k²) * Tensor(:h, [up(:a), down(:a)]) * Tensor(:h, [up(:b), down(:b)])
    extract_kernel(t1 + t2 + t3 + t4, :h; registry = reg)
end

"""
    build_R2_momentum_kernel(reg) -> KineticKernel

(δR)² in Fourier space on flat background.
δR = k^a k^b h_{ab} - k² h, so (δR)² = 3 bilinear terms.
All h indices are Down with disjoint names (a,b for left; c,d for right).
"""
function build_R2_momentum_kernel(reg)
    a, b, c, d = down(:a), down(:b), down(:c), down(:d)
    t1 = tproduct(1 // 1, TensorExpr[
        Tensor(:k, [up(:a)]), Tensor(:k, [up(:b)]),
        Tensor(:k, [up(:c)]), Tensor(:k, [up(:d)]),
        Tensor(:h, [a, b]), Tensor(:h, [c, d])])
    t2 = tproduct(-2 // 1, TensorExpr[
        TScalar(:k²), Tensor(:g, [up(:a), up(:b)]),
        Tensor(:k, [up(:c)]), Tensor(:k, [up(:d)]),
        Tensor(:h, [a, b]), Tensor(:h, [c, d])])
    t3 = tproduct(1 // 1, TensorExpr[
        TScalar(:k²), TScalar(:k²),
        Tensor(:g, [up(:a), up(:b)]), Tensor(:g, [up(:c), up(:d)]),
        Tensor(:h, [a, b]), Tensor(:h, [c, d])])
    extract_kernel(t1 + t2 + t3, :h; registry = reg)
end

"""
    build_Ric2_momentum_kernel(reg) -> KineticKernel

(δRic)² = g^{μα}g^{νβ} δRic_{αβ} δRic_{μν} in Fourier space on flat background.
δRic_{μν} = (1/2)(k^ρ k_μ h_{νρ} + k^ρ k_ν h_{μρ} - k² h_{μν} - k_μ k_ν g^{ρσ} h_{ρσ}).
Produces 4×4 = 16 bilinear terms with all-Down h indices and disjoint names.
"""
function build_Ric2_momentum_kernel(reg)
    kup(x) = Tensor(:k, [up(x)])
    kdn(x) = Tensor(:k, [down(x)])
    guu(x, y) = Tensor(:g, [up(x), up(y)])
    h_dd(x, y) = Tensor(:h, [down(x), down(y)])
    ksq = TScalar(:k²)

    # δRic_{αβ} (copy 1, α→e, β→f, dummy→a,b), h uses {a,b,e,f}
    T = [
        (1 // 4, [kup(:a), kdn(:e)],             (:f, :a)),
        (1 // 4, [kup(:a), kdn(:f)],             (:e, :a)),
        (-1 // 4, [ksq],                          (:e, :f)),
        (-1 // 4, [kdn(:e), kdn(:f), guu(:a, :b)], (:a, :b)),
    ]
    # δRic_{μν} (copy 2, μ→i, ν→j, dummy→c,d), h uses {c,d,i,j}
    U = [
        (1 // 1, [kup(:c), kdn(:i)],             (:j, :c)),
        (1 // 1, [kup(:c), kdn(:j)],             (:i, :c)),
        (-1 // 1, [ksq],                          (:i, :j)),
        (-1 // 1, [kdn(:i), kdn(:j), guu(:c, :d)], (:c, :d)),
    ]
    common = [guu(:i, :e), guu(:j, :f)]

    terms = TensorExpr[]
    for (s_t, f_t, (l1, l2)) in T
        for (s_u, f_u, (r1, r2)) in U
            push!(terms, tproduct(s_t * s_u,
                TensorExpr[common..., f_t..., f_u..., h_dd(l1, l2), h_dd(r1, r2)]))
        end
    end
    extract_kernel(tsum(terms), :h; registry = reg)
end

# ─── Numerical evaluation of spin projection results ────────────────

"""
    _eval_spin_scalar(expr, k2_val) -> Float64

Evaluate a fully-contracted spin projection result (TScalar/TProduct/TSum
tree containing :k² symbols) at a numeric k² value.
"""
function _eval_spin_scalar(expr::TScalar, k2)
    _eval_ksq_val(expr.val, k2)
end
function _eval_spin_scalar(expr::TProduct, k2)
    Float64(expr.scalar) * prod(_eval_spin_scalar(f, k2) for f in expr.factors)
end
function _eval_spin_scalar(expr::TSum, k2)
    sum(_eval_spin_scalar(t, k2) for t in expr.terms)
end
function _eval_spin_scalar(expr::Tensor, k2)
    error("Uncontracted tensor in spin projection result: $expr")
end

function _eval_ksq_val(v, k2)
    v isa Rational && return Float64(v)
    v isa Integer && return Float64(v)
    v isa AbstractFloat && return Float64(v)
    v === :k² && return Float64(k2)
    v isa Expr || return Float64(v)
    if v.head == :call
        op = v.args[1]
        args = [_eval_ksq_val(a, k2) for a in v.args[2:end]]
        op === :* && return prod(args)
        op === :+ && return sum(args)
        op === :- && return length(args) == 1 ? -args[1] : args[1] - args[2]
        op === :/ && return args[1] / args[2]
        op === :^ && return args[1]^args[2]
    end
    error("Cannot evaluate TScalar value: $v (type=$(typeof(v)))")
end
