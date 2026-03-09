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
    spin_project(K::KineticKernel, spin; dim=4, metric=:η, k_name=:k, k_sq=:k²,
                 registry=current_registry()) -> TensorExpr

Project the kinetic kernel onto a spin sector using Barnes-Rivers projectors.

For each bilinear term, builds the projector P^J with the h factors' actual
index labels (ensuring correct contraction), multiplies by the coefficient,
and sums. Returns the scalar form factor (function of k²).

`spin` is one of: `:spin2`, `:spin1`, `:spin0s`, `:spin0w`.
"""
function spin_project(K::KineticKernel, spin::Symbol;
                      dim::Int = 4, metric::Symbol = :η,
                      k_name::Symbol = :k, k_sq::Symbol = :k²,
                      registry = current_registry())
    projections = TensorExpr[]

    for bt in K.terms
        μ, ν = bt.left[1], bt.left[2]
        ρ, σ = bt.right[1], bt.right[2]

        P = _kernel_build_projector(spin, μ, ν, ρ, σ; dim, metric, k_name, k_sq)
        P = ensure_no_dummy_clash(bt.coeff, P)
        push!(projections, P * bt.coeff)
    end

    with_registry(registry) do
        simplify(tsum(projections); registry = registry, maxiter = 40)
    end
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
function contract_momenta(expr::TensorExpr; k_name::Symbol = :k, k_sq::Symbol = :k²)
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
