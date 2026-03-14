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

# ─── Position-space kernel extraction (two-momentum correct) ─────────
#
# In a bilinear integral ∫dx h₁(x) K(∂) h₂(x), the two fields carry
# opposite momenta: h₁ → +k, h₂ → -k.  Converting ∂ → momentum:
#   ∂ on h₁: ∂_a → +ik_a        ∂ on h₂: ∂_a → -ik_a
#
# With n_L derivatives on h₁ and n_R on h₂, the physical coefficient is:
#   (ik)^{n_L} (-ik)^{n_R} = i^n (-1)^{n_R} k^n
# vs the uniform-k convention (dropping all i's): k^n.
#
# For even n (real actions), i^n = (-1)^{n/2}, giving the correction:
#   phase = (-1)^{n/2 + n_R}
#
# Verified against FP ground truth, analytic R²/Ric² kernels, and the
# full 4-derivative flat spectrum (Buoninfante 2012.11829 Eq.2.13).

"""
    _distribute_derivs_sums(expr::TensorExpr) -> TensorExpr

Recursively distribute TDeriv over TSum at every level:
    ∂(A + B) → ∂A + ∂B

Does NOT apply the Leibniz rule to products — only linearizes
derivatives over sums so that after `expand_products` we get a flat
sum of product terms.
"""
_distribute_derivs_sums(expr::Tensor) = expr
_distribute_derivs_sums(expr::TScalar) = expr

function _distribute_derivs_sums(expr::TSum)
    tsum(TensorExpr[_distribute_derivs_sums(t) for t in expr.terms])
end

function _distribute_derivs_sums(expr::TProduct)
    TProduct(expr.scalar, TensorExpr[_distribute_derivs_sums(f) for f in expr.factors])
end

function _distribute_derivs_sums(expr::TDeriv)
    inner = _distribute_derivs_sums(expr.arg)
    if inner isa TSum
        terms = TensorExpr[_distribute_derivs_sums(TDeriv(expr.index, t, expr.covd))
                           for t in inner.terms]
        tsum(terms)
    else
        TDeriv(expr.index, inner, expr.covd)
    end
end

# ─── Field counting ──────────────────────────────────────────────────

"""Count how many times a tensor named `field` appears at any nesting depth."""
function _count_fields_in(expr::Tensor, field::Symbol)
    expr.name == field ? 1 : 0
end
_count_fields_in(::TScalar, ::Symbol) = 0

function _count_fields_in(expr::TProduct, field::Symbol)
    sum(_count_fields_in(f, field) for f in expr.factors; init=0)
end

function _count_fields_in(expr::TSum, field::Symbol)
    isempty(expr.terms) ? 0 : maximum(_count_fields_in(t, field) for t in expr.terms)
end

function _count_fields_in(expr::TDeriv, field::Symbol)
    _count_fields_in(expr.arg, field)
end

# ─── Field unwrapping from derivative chains ─────────────────────────

"""
    _unwrap_field_chain(expr, field) -> NamedTuple or nothing

Determine if `expr` is a "field unit" — a single occurrence of `field`
possibly wrapped in derivative chains and multiplied by constant tensors.

Returns `nothing` if the expression doesn't contain the field, or contains
it in an unrecognizable structure.

Returns `(deriv_indices, field_indices, extra_factors)` where:
- `deriv_indices`: derivative indices acting on the field
- `field_indices`: the field tensor's own indices
- `extra_factors`: non-field factors extracted from derivative chains
  (e.g. metric tensors in ∂(g × ∂h) on flat background)
"""
function _unwrap_field_chain(expr::Tensor, field::Symbol)
    expr.name != field && return nothing
    (deriv_indices=TIndex[], field_indices=collect(expr.indices),
     extra_factors=TensorExpr[])
end

_unwrap_field_chain(::TScalar, ::Symbol) = nothing
_unwrap_field_chain(::TSum, ::Symbol) = nothing

function _unwrap_field_chain(expr::TDeriv, field::Symbol)
    nf = _count_fields_in(expr.arg, field)
    nf == 0 && return nothing
    nf >= 2 && return nothing  # ∂(bilinear) → vanishes under ∫dx

    inner = _unwrap_field_chain(expr.arg, field)
    if inner !== nothing
        return (deriv_indices=vcat(TIndex[expr.index], inner.deriv_indices),
                field_indices=inner.field_indices,
                extra_factors=inner.extra_factors)
    end

    # TDeriv wrapping a product: derivative passes through non-field factors
    if expr.arg isa TProduct
        return _unwrap_deriv_product(expr.index, expr.arg, field)
    end
    nothing
end

function _unwrap_field_chain(expr::TProduct, field::Symbol)
    nf = _count_fields_in(expr, field)
    nf == 0 && return nothing
    nf >= 2 && return nothing

    field_idx = findfirst(f -> _count_fields_in(f, field) > 0, expr.factors)
    field_idx === nothing && return nothing

    inner = _unwrap_field_chain(expr.factors[field_idx], field)
    inner === nothing && return nothing

    extras = TensorExpr[expr.factors[i] for i in eachindex(expr.factors) if i != field_idx]
    expr.scalar != 1 // 1 && pushfirst!(extras, TScalar(expr.scalar))

    (deriv_indices=inner.deriv_indices, field_indices=inner.field_indices,
     extra_factors=vcat(extras, inner.extra_factors))
end

"""Unwrap ∂(product) where the product has exactly 1 field factor."""
function _unwrap_deriv_product(deriv_idx::TIndex, prod::TProduct, field::Symbol)
    nf = count(f -> _count_fields_in(f, field) > 0, prod.factors)
    nf != 1 && return nothing

    fi = findfirst(f -> _count_fields_in(f, field) > 0, prod.factors)
    inner = _unwrap_field_chain(prod.factors[fi], field)
    inner === nothing && return nothing

    extras = TensorExpr[prod.factors[i] for i in eachindex(prod.factors) if i != fi]
    prod.scalar != 1 // 1 && pushfirst!(extras, TScalar(prod.scalar))

    (deriv_indices=vcat(TIndex[deriv_idx], inner.deriv_indices),
     field_indices=inner.field_indices,
     extra_factors=vcat(extras, inner.extra_factors))
end

# ─── Single-term bilinear extraction ─────────────────────────────────

"""
Extract bilinear data from a single flat product term.

Finds two field units (h or ∂ⁿh), converts derivative chains to k-factors
with the correct two-momentum phase `(-1)^{n/2 + n_R}`, and returns
`(coeff, left, right)` or `nothing`.
"""
function _extract_bilinear_direct(term::TensorExpr, field::Symbol, k_name::Symbol)
    sc, factors = _kernel_term_parts(term)

    field_units = []
    coeff_factors = TensorExpr[]

    for (i, f) in enumerate(factors)
        info = _unwrap_field_chain(f, field)
        if info !== nothing
            push!(field_units, (idx=i, info=info))
        elseif _count_fields_in(f, field) >= 2
            return nothing  # ∂(bilinear) → vanishes under ∫dx
        elseif _count_fields_in(f, field) == 1
            return nothing  # field in unrecognizable structure
        else
            push!(coeff_factors, f)
        end
    end

    length(field_units) != 2 && return nothing

    fu1, fu2 = field_units[1].info, field_units[2].info
    n_L = length(fu1.deriv_indices)
    n_R = length(fu2.deriv_indices)
    n = n_L + n_R

    # Odd total derivatives → imaginary, vanishes in real action
    isodd(n) && return nothing

    # Two-momentum phase: (-1)^{n/2 + n_R}
    phase = iseven(div(n, 2) + n_R) ? (1 // 1) : (-1 // 1)

    # Build k-factors from derivative indices
    k_factors = TensorExpr[Tensor(k_name, [idx])
                           for idx in vcat(fu1.deriv_indices, fu2.deriv_indices)]

    all_parts = vcat(coeff_factors, fu1.extra_factors, fu2.extra_factors, k_factors)
    coeff = isempty(all_parts) ? TScalar(sc * phase) : tproduct(sc * phase, all_parts)

    (coeff = coeff, left = fu1.field_indices, right = fu2.field_indices)
end

# ─── Main entry point ────────────────────────────────────────────────

"""
    extract_kernel_direct(expr, field; k_name=:k, registry=current_registry()) -> KineticKernel

Extract the kinetic kernel directly from a position-space bilinear expression,
correctly handling two-momentum physics of quadratic forms under ∫dx.

Unlike `to_fourier` → `extract_kernel` (uniform-k, wrong for asymmetric
derivatives), this function:

1. Distributes all derivatives over sums: ∂(A+B) → ∂A + ∂B
2. Expands products to get a flat sum of product terms
3. Identifies the two field factors and their derivative chains per term
4. Converts derivatives to momentum factors with correct phase:
   `(ik)^{n_L}(-ik)^{n_R} = (-1)^{n/2 + n_R} k^n`
5. Drops total-derivative bilinear terms (∂(h₁h₂) = 0 under ∫dx)

**Important**: For complex expressions, pre-simplify each *factor* individually
before forming the bilinear product. Do NOT simplify the full bilinear expression
as `simplify` can merge terms into ∂(bilinear) structures that break extraction.

# Example
```julia
# EH kernel via linearized Einstein tensor:
d1R_ab = simplify(δricci(mp, down(:a), down(:b), 1); registry=reg)
d1R    = simplify(δricci_scalar(mp, 1); registry=reg)
EH_bilinear = h_up * d1R_ab - (1//2) * trh * d1R
K = extract_kernel_direct(EH_bilinear, :h; registry=reg)
# → spin-2=2.5, spin-0s=-1.0 (matches FP)

# R² kernel (pre-simplified factors):
K_R2 = extract_kernel_direct(d1R * d1R, :h; registry=reg)
# → spin-2=0, spin-0s=3.0
```
"""
function extract_kernel_direct(expr::TensorExpr, field::Symbol;
                               k_name::Symbol=:k,
                               registry = current_registry())
    # Distribute ∂(TSum) and expand products in a single pass.
    # Do NOT iterate: further rounds change derivative assignments
    # (n_L, n_R) and corrupt the phase calculation.
    expanded = expand_products(_distribute_derivs_sums(expr))
    raw_terms = expanded isa TSum ? expanded.terms : TensorExpr[expanded]

    bilinears = @NamedTuple{coeff::TensorExpr, left::Vector{TIndex}, right::Vector{TIndex}}[]

    for term in raw_terms
        bt = _extract_bilinear_direct(term, field, k_name)
        bt === nothing && continue
        push!(bilinears, bt)
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

        # Combine projector + metric connectors + coefficient.
        # ensure_no_dummy_clash only renames dummies that clash with OTHER
        # dummies, but the coefficient may have internal dummies that clash
        # with the projector's FREE indices (μ,ν,ρ,σ). Rename those too.
        all_factors = TensorExpr[P]
        append!(all_factors, metric_factors)
        combined = tproduct(1 // 1, all_factors)

        # Collect ALL index names used by the projector side (free + dummy)
        proj_all_names = Set{Symbol}(idx.name for idx in indices(combined))

        # Rename any coefficient dummies that clash with projector names
        coeff_fixed = bt.coeff
        _, _, coeff_pairs = _analyze_indices(coeff_fixed)
        coeff_all_names = Set{Symbol}(idx.name for idx in indices(coeff_fixed))
        all_names = union(proj_all_names, coeff_all_names)
        for (up_idx, _) in coeff_pairs
            if up_idx.name in proj_all_names
                new_name = fresh_index(all_names)
                push!(all_names, new_name)
                coeff_fixed = rename_dummy(coeff_fixed, up_idx.name, new_name)
            end
        end

        combined = ensure_no_dummy_clash(coeff_fixed, combined)
        push!(projections, combined * coeff_fixed)
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
function _kernel_term_parts(t::TDeriv)
    (1 // 1, TensorExpr[t])
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
function _eval_spin_scalar(expr::Tensor, k2; dim::Int=4)
    # Handle contracted metric/delta traces: g^a_a = delta^a_a = dim
    if length(expr.indices) == 2
        i1, i2 = expr.indices
        if i1.name == i2.name && i1.position != i2.position && i1.vbundle == i2.vbundle
            props = try
                reg = current_registry()
                has_tensor(reg, expr.name) ? get_tensor(reg, expr.name) : nothing
            catch; nothing end
            if props !== nothing && (props.is_metric || props.is_delta)
                return Float64(dim)
            end
        end
    end
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
        # Handle both symbol ops (:*) and function-ref ops (*)
        (op === :* || op === *) && return prod(args)
        (op === :+ || op === +) && return sum(args)
        (op === :- || op === -) && return length(args) == 1 ? -args[1] : args[1] - args[2]
        (op === :/ || op === /) && return args[1] / args[2]
        (op === :^ || op === ^) && return args[1]^args[2]
    end
    error("Cannot evaluate TScalar value: $v (type=$(typeof(v)))")
end

# ─── Bueno-Cano dS spectrum for 6-derivative gravity ─────────────────
# Reference: Bueno & Cano, "Einsteinian cubic gravity" (2016)
#   arXiv: 1607.06463, Eqs. (6), (13)-(14), (17)-(19)
#
# Convention: Λ is TGR's cosmological constant (R̄_μν = Λ g_μν, D=4).
# Bueno-Cano uses Λ_BC = Λ/(D-1) = Λ/3.

"""
    BuenoCanoParams

Bueno-Cano parameters (a, b, c, e) characterizing the linearized field
equations of a gravity theory on a maximally symmetric background.

From these, the physical spectrum is (Eqs. 17-19 of 1607.06463):
- `κ_eff⁻¹ = 4e − 8Λ_BC a`
- `m²_g = (−e + 2Λ_BC a) / (2a + c)`  (massive spin-2)
- `m²_s = (2e − 4Λ_BC(a + 4b + c)) / (2a + 4c + 12b)`  (spin-0)
"""
struct BuenoCanoParams{T}
    a::T
    b::T
    c::T
    e::T
end

# Additive composition
function Base.:+(p1::BuenoCanoParams, p2::BuenoCanoParams)
    BuenoCanoParams(p1.a + p2.a, p1.b + p2.b, p1.c + p2.c, p1.e + p2.e)
end

function Base.show(io::IO, p::BuenoCanoParams)
    print(io, "BC(a=$(p.a), b=$(p.b), c=$(p.c), e=$(p.e))")
end

# ── BC parameters for each Lagrangian term ──
# Each bc_* function returns BuenoCanoParams(a, b, c, e) for the given
# Lagrangian term on a maximally symmetric background with Ric = Λ g.
# Convention: Λ_BC = Λ/3 (Bueno-Cano's Λ vs TGR's Λ).
# Reference: Bueno & Cano (1607.06463), Table 1 and Eqs. (6), (13)-(14).

"""
    bc_EH(κ, Λ) -> BuenoCanoParams

BC parameters for Einstein-Hilbert term `κR`. Only `e = κ`; all others zero.
"""
bc_EH(κ, Λ) = BuenoCanoParams(zero(Λ), zero(Λ), zero(Λ), oftype(Λ, κ))

"""
    bc_R2(α₁, Λ) -> BuenoCanoParams

BC parameters for `α₁ R²`. Gives `b = 2α₁`, `e = 8α₁Λ`.
"""
bc_R2(α₁, Λ) = BuenoCanoParams(zero(Λ), oftype(Λ, 2α₁), zero(Λ), 8α₁*Λ)

"""
    bc_RicSq(α₂, Λ) -> BuenoCanoParams

BC parameters for `α₂ R_{μν}R^{μν}`. Gives `c = 2α₂`, `e = 2α₂Λ`.
"""
bc_RicSq(α₂, Λ) = BuenoCanoParams(zero(Λ), zero(Λ), oftype(Λ, 2α₂), 2α₂*Λ)

"""
    bc_R3(γ, Λ) -> BuenoCanoParams

BC parameters for cubic invariant I₁ = `γ R³`. Gives `b = 6γΛ`, `e = 24γΛ²`.
"""
bc_R3(γ, Λ) = BuenoCanoParams(zero(Λ), 6γ*Λ, zero(Λ), 24γ*Λ^2)

"""
    bc_RRicSq(γ, Λ) -> BuenoCanoParams

BC parameters for I₂ = `γ R·R_{μν}R^{μν}`. Gives `b = γΛ`, `c = 2γΛ`, `e = 6γΛ²`.
"""
bc_RRicSq(γ, Λ) = BuenoCanoParams(zero(Λ), γ*Λ, 2γ*Λ, 6γ*Λ^2)

"""
    bc_Ric3(γ, Λ) -> BuenoCanoParams

BC parameters for I₃ = `γ R_{μν}R^{νρ}R_{ρ}^{μ}`. Gives `c = 3γΛ/2`, `e = 3γΛ²/2`.
"""
bc_Ric3(γ, Λ) = BuenoCanoParams(zero(Λ), zero(Λ), 3γ*Λ/2, 3γ*Λ^2/2)

"""
    bc_RRiem2(γ, Λ) -> BuenoCanoParams

BC parameters for I₄ = `γ R·R_{μνρσ}R^{μνρσ}`. Gives `a = 4γΛ`, `b = 2γΛ/3`, `e = 4γΛ²`.
"""
bc_RRiem2(γ, Λ) = BuenoCanoParams(4γ*Λ, 2γ*Λ/3, zero(Λ), 4γ*Λ^2)

"""
    bc_RicRiem2(γ, Λ) -> BuenoCanoParams

BC parameters for I₅ = `γ R_{μν}R^{μρσδ}R^{ν}_{ρσδ}`. Gives `a = γΛ`, `c = 2γΛ/3`, `e = γΛ²`.
"""
bc_RicRiem2(γ, Λ) = BuenoCanoParams(γ*Λ, zero(Λ), 2γ*Λ/3, γ*Λ^2)

"""
    bc_Riem3(γ, Λ) -> BuenoCanoParams

BC parameters for I₆ = `γ R_{μν}^{ρσ}R_{ρσ}^{δε}R_{δε}^{μν}`. Gives `a = 2γΛ`, `e = 2γΛ²/3`.
"""
bc_Riem3(γ, Λ) = BuenoCanoParams(2γ*Λ, zero(Λ), zero(Λ), 2γ*Λ^2/3)

"""
    dS_spectrum_6deriv(; κ, α₁=0, α₂=0, β₁=0, β₂=0,
                        γ₁=0, γ₂=0, γ₃=0, γ₄=0, γ₅=0, γ₆=0, Λ)

Compute the particle spectrum of general 6-derivative gravity on de Sitter.

The action is:
  S = ∫d⁴x√g [κR + α₁R² + α₂Ric² + β₁R□R + β₂Ric□Ric
               + γ₁R³ + γ₂R·Ric² + γ₃Ric³ + γ₄R·Riem² + γ₅Ric·Riem² + γ₆Riem³]

Returns a NamedTuple with:
- `params`: total BuenoCanoParams
- `κ_eff_inv`: inverse effective Newton constant (Eq. 17)
- `m2_graviton`: massive spin-2 mass squared (Eq. 18), `Inf` if no massive mode
- `m2_scalar`: spin-0 mass squared (Eq. 19), `Inf` if no scalar mode
- `flat_f2`: flat-space spin-2 form factor coefficients `(c₁, c₂)` where f₂(z)=1+c₁z+c₂z²
- `flat_f0`: flat-space spin-0 form factor coefficients `(c₁, c₂)` where f₀(z)=1+c₁z+c₂z²

Note: β₁R□R and β₂Ric□Ric contribute to the flat form factors but not to the dS
Bueno-Cano parameters (since □R̄ = 0 on MSS). Their dS effect enters through the
replacement α → α − βm² in the mass formulas (implicit momentum dependence).

Reference: Bueno & Cano (1607.06463) Eqs. (17)-(19);
           Buoninfante et al. (2012.11829) Eq. (2.13).
"""
function dS_spectrum_6deriv(; κ, α₁=0, α₂=0, β₁=0, β₂=0,
                              γ₁=0, γ₂=0, γ₃=0, γ₄=0, γ₅=0, γ₆=0, Λ)
    # Total BC parameters (cubics contribute at O(Λ))
    p = bc_EH(κ, Λ) + bc_R2(α₁, Λ) + bc_RicSq(α₂, Λ) +
        bc_R3(γ₁, Λ) + bc_RRicSq(γ₂, Λ) + bc_Ric3(γ₃, Λ) +
        bc_RRiem2(γ₄, Λ) + bc_RicRiem2(γ₅, Λ) + bc_Riem3(γ₆, Λ)

    Λ_BC = Λ / 3

    # Effective Newton constant (Eq. 17)
    κ_eff = 4p.e - 8Λ_BC * p.a

    # Massive spin-2 mass (Eq. 18)
    denom_g = 2p.a + p.c
    m2_g = abs(denom_g) > 1e-15 * abs(p.e) ?
        (-p.e + 2Λ_BC * p.a) / denom_g : oftype(Λ, Inf)

    # Spin-0 mass (Eq. 19)
    denom_s = 2p.a + 4p.c + 12p.b
    m2_s = abs(denom_s) > 1e-15 * abs(p.e) ?
        (2p.e - 4Λ_BC * (p.a + 4p.b + p.c)) / denom_s : oftype(Λ, Inf)

    # Flat-space form factors (Buoninfante Eq. 2.13)
    flat_f2 = (-α₂/κ, -β₂/κ)
    flat_f0 = ((6α₁ + 2α₂)/κ, (6β₁ + 2β₂)/κ)

    (params = p, κ_eff_inv = κ_eff, m2_graviton = m2_g, m2_scalar = m2_s,
     flat_f2 = flat_f2, flat_f0 = flat_f0)
end

# ─── Scale and combine kinetic kernels ──────────────────────────────

"""
    scale_kernel(K::KineticKernel, factor::TensorExpr) -> KineticKernel

Scale all bilinear coefficients in a kernel by `factor`.
"""
function scale_kernel(K::KineticKernel, factor::TensorExpr)
    new_terms = map(K.terms) do bt
        (coeff = bt.coeff * factor, left = bt.left, right = bt.right)
    end
    KineticKernel(K.field, new_terms)
end

function scale_kernel(K::KineticKernel, s::Rational)
    s == 1 // 1 && return K
    new_terms = map(K.terms) do bt
        (coeff = tproduct(s, TensorExpr[bt.coeff]), left = bt.left, right = bt.right)
    end
    KineticKernel(K.field, new_terms)
end

"""
    combine_kernels(kernels::Vector{KineticKernel}) -> KineticKernel

Concatenate bilinear terms from multiple kernels (must share the same field).
"""
function combine_kernels(kernels::Vector{KineticKernel})
    isempty(kernels) && error("No kernels to combine")
    field = first(kernels).field
    all(K -> K.field == field, kernels) || error("All kernels must share the same field")
    combined = vcat((K.terms for K in kernels)...)
    KineticKernel(field, combined)
end

"""
    build_6deriv_flat_kernel(reg; κ=1, α₁=0, α₂=0, β₁=0, β₂=0) -> KineticKernel

Build the combined kinetic kernel for the 6-derivative gravity action on flat background.

The action is S = ∫d⁴x √g [κR + α₁R² + α₂Ric² + β₁R□R + β₂Ric□Ric].

On flat background, the kinetic kernel (bilinear form determining the propagator) is:
  K = κ·K_FP − 2(α₁ + β₁k²)·K_{R²} − 2(α₂ + β₂k²)·K_{Ric²}

The minus signs arise because the (δR)² and (δRic)² terms enter the kinetic
operator with opposite sign to the Fierz-Pauli kernel (the second variation of
√g R contains metric determinant contributions that flip the overall sign
relative to the raw curvature-squared bilinear forms).

Returns a KineticKernel ready for `spin_project`.
"""
function build_6deriv_flat_kernel(reg; κ=1//1, α₁=0//1, α₂=0//1, β₁=0//1, β₂=0//1)
    kernels = KineticKernel[]

    # EH contribution: κ·K_FP
    if κ != 0
        push!(kernels, scale_kernel(build_FP_momentum_kernel(reg), Rational{Int}(κ)))
    end

    # R² contribution: −2α₁·K_{R²}
    if α₁ != 0
        push!(kernels, scale_kernel(build_R2_momentum_kernel(reg), Rational{Int}(-2α₁)))
    end

    # Ric² contribution: −2α₂·K_{Ric²}
    if α₂ != 0
        push!(kernels, scale_kernel(build_Ric2_momentum_kernel(reg), Rational{Int}(-2α₂)))
    end

    # R□R contribution: −2β₁k²·K_{R²}
    if β₁ != 0
        K_R2 = build_R2_momentum_kernel(reg)
        push!(kernels, scale_kernel(scale_kernel(K_R2, TScalar(:k²)), Rational{Int}(-2β₁)))
    end

    # Ric□Ric contribution: −2β₂k²·K_{Ric²}
    if β₂ != 0
        K_Ric2 = build_Ric2_momentum_kernel(reg)
        push!(kernels, scale_kernel(scale_kernel(K_Ric2, TScalar(:k²)), Rational{Int}(-2β₂)))
    end

    isempty(kernels) && return KineticKernel(:h, eltype(build_FP_momentum_kernel(reg).terms)[])
    combine_kernels(kernels)
end

# ─── Analytic form factor prediction from Bueno-Cano parameters ──────

"""
    bc_to_form_factors(bc::BuenoCanoParams, k2, Λ) -> (f_spin2=..., f_spin0s=...)

Predict spin-projected form factors Tr(K·P^J) from Bueno-Cano parameters.

This is an independent algebraic cross-check that does NOT use the perturbation
engine. The prediction uses known flat-space kernel traces scaled by the
effective couplings encoded in the BC parameters.

The flat-space kernel traces (verified by Barnes-Rivers spin projection) are:
- K_FP:  Tr(K_FP · P²) = (5/2)k²,  Tr(K_FP · P⁰ˢ) = -k²
- K_R²:  Tr(K_R² · P²) = 0,         Tr(K_R² · P⁰ˢ) = 3k⁴
- K_Ric²: Tr(K_Ric² · P²) = (5/4)k⁴, Tr(K_Ric² · P⁰ˢ) = k⁴

The combined kernel K = κ·K_FP − 2α₁·K_R² − 2α₂·K_Ric² gives:
  f₂(k²) = (5/2)[κ_eff · k² − (c/2) · k⁴]
  f₀(k²) = −κ_eff · k² − (3b + c) · k⁴

where κ_eff = e − (2Λ/3)·a, and a, b, c, e are the BC parameters.

Returns a NamedTuple with `f_spin2` and `f_spin0s` values at the given k² and Λ.
"""
function bc_to_form_factors(bc::BuenoCanoParams, k2, Λ)
    Λ_BC = Λ / 3
    κ_eff = bc.e - 2Λ_BC * bc.a

    # Spin-2: f₂ = (5/2)[κ_eff·k² − (c/2)·k⁴]
    f2 = (5 // 2) * (κ_eff * k2 - (bc.c / 2) * k2^2)

    # Spin-0s: f₀ = −κ_eff·k² − (3b + c)·k⁴
    f0 = -κ_eff * k2 - (3 * bc.b + bc.c) * k2^2

    (f_spin2 = f2, f_spin0s = f0)
end

"""
    flat_6deriv_spin_projections(reg; κ=1, α₁=0, α₂=0, β₁=0, β₂=0)

Compute spin-projected form factors for the 6-derivative flat kernel.

Returns a NamedTuple with symbolic TensorExpr for each spin sector,
suitable for evaluation with `_eval_spin_scalar(result, k²_value)`.

The form factors should satisfy (Buoninfante 2012.11829 Eq. 2.13):
  f₂(z) = 1 − (α₂/κ)z − (β₂/κ)z²     (spin-2 sector)
  f₀(z) = 1 + (6α₁+2α₂)z/κ + (6β₁+2β₂)z²/κ  (spin-0 sector)
when normalized by the GR values.
"""
function flat_6deriv_spin_projections(reg; κ=1//1, α₁=0//1, α₂=0//1, β₁=0//1, β₂=0//1)
    K = build_6deriv_flat_kernel(reg; κ, α₁, α₂, β₁, β₂)
    with_registry(reg) do
        s2  = spin_project(K, :spin2;  registry = reg)
        s1  = spin_project(K, :spin1;  registry = reg)
        s0s = spin_project(K, :spin0s; registry = reg)
        s0w = spin_project(K, :spin0w; registry = reg)
        (spin2 = s2, spin1 = s1, spin0s = s0s, spin0w = s0w)
    end
end
