#= Automatic extraction of quadratic form from Lagrangian density.

Given a Lagrangian that is quadratic in a set of fields, this module:
1. Fourier-transforms derivatives → momenta (via to_fourier)
2. Expands to a flat sum of product terms
3. Identifies field bilinears in each term
4. Contracts momentum/structural tensors to scalar coefficients
5. Collects coefficients into a QuadraticForm matrix
=#

"""
    extract_quadratic_form(expr, fields; fourier=true) -> QuadraticForm

Extract the quadratic form matrix from a Lagrangian expression.

The expression should be quadratic in the given `fields`. Each term is
decomposed into (coefficient) × (field_i) × (field_j), and the coefficients
are collected into a matrix M such that L = Φ_i M_{ij} Φ_j.

Momentum indices from Fourier-transformed derivatives are contracted:
`k_{_i} k_{_i}` → `:k²`, `k_{_0} k_{_0}` → `:ω²`.
"""
function extract_quadratic_form(expr::TensorExpr, fields::Vector{Symbol};
                                 fourier::Bool=true)
    # Step 1: Fourier transform (replace ∂ → k)
    fexpr = fourier ? to_fourier(expr) : expr

    # Step 2: Expand to flat sum of products
    fexpr = expand_products(expand_derivatives(fexpr))
    fexpr = expand_products(fexpr)

    # Step 3: Collect bilinear coefficients
    field_set = Set(fields)
    entries = Dict{Tuple{Symbol,Symbol}, Any}()
    _collect_bilinears!(entries, fexpr, field_set)

    # Step 4: Build QuadraticForm
    quadratic_form(entries, fields)
end

# ─── Bilinear collection ─────────────────────────────────────────────

function _collect_bilinears!(entries::Dict, expr::TSum, fields::Set{Symbol})
    for t in expr.terms
        _collect_bilinears!(entries, t, fields)
    end
end

function _collect_bilinears!(entries::Dict, p::TProduct, fields::Set{Symbol})
    field_indices = Int[]
    field_names = Symbol[]
    for (i, f) in enumerate(p.factors)
        name = _base_field_name(f)
        if name in fields
            push!(field_indices, i)
            push!(field_names, name)
        end
    end

    length(field_indices) == 2 || return  # not a bilinear

    f1, f2 = field_names[1], field_names[2]
    key = f1 <= f2 ? (f1, f2) : (f2, f1)

    # Coefficient = scalar * contracted momentum/structural factors
    coeff_factors = TensorExpr[p.factors[i] for i in eachindex(p.factors)
                                if i ∉ field_indices]
    coeff = _eval_coefficient(p.scalar, coeff_factors)

    entries[key] = _sym_add(get(entries, key, 0), coeff)
end

function _collect_bilinears!(::Dict, ::TensorExpr, ::Set{Symbol})
    # Non-product, non-sum terms: ignore (scalars, single tensors, etc.)
end

# ─── Extract base field name from a tensor or momentum·field product ──

"""Get the field name from a Tensor, or nothing if not a simple field."""
_base_field_name(t::Tensor) = t.name
_base_field_name(::TensorExpr) = :_none_

# ─── Evaluate coefficient: contract momenta and structural tensors ────

"""
Evaluate the scalar coefficient from momentum and structural tensor factors.

Handles both abstract indices (k_a k^a → p²) and component indices
(k_{_0} k_{_0} → ω², k_{_i} k_{_i} → k²_component).
"""
function _eval_coefficient(scalar::Rational{Int}, factors::Vector{TensorExpr})
    isempty(factors) && return scalar == 1 ? 1 : scalar

    # Separate momentum tensors from structural tensors
    k_factors = Tuple{TIndex, Int}[]  # (index, position_in_factors)
    other_factors = TensorExpr[]

    for (i, f) in enumerate(factors)
        if f isa Tensor && f.name == :k && length(f.indices) == 1
            push!(k_factors, (f.indices[1], i))
        elseif f isa Tensor && f.name == :δ && length(f.indices) == 2
            i1, i2 = f.indices
            cv1, cv2 = _component_value(i1), _component_value(i2)
            if cv1 !== nothing && cv2 !== nothing
                cv1 != cv2 && return 0  # δ_{_i _j} with i≠j = 0
                # δ_{_i _i} = 1, factor of 1
            else
                push!(other_factors, f)
            end
        else
            push!(other_factors, f)
        end
    end

    coeff::Any = scalar == 1 ? 1 : scalar

    # Pair up momentum factors by contracting dummy pairs
    paired = Set{Int}()
    for i in eachindex(k_factors)
        i in paired && continue
        idx_i = k_factors[i][1]
        for j in (i+1):length(k_factors)
            j in paired && continue
            idx_j = k_factors[j][1]
            if idx_i.name == idx_j.name && idx_i.position != idx_j.position
                # Contracted pair: k_a k^a
                push!(paired, i)
                push!(paired, j)
                cv = _component_value(idx_i)
                if cv !== nothing && cv == 0
                    coeff = _sym_mul(coeff, :ω²)
                elseif cv !== nothing && cv > 0
                    coeff = _sym_mul(coeff, :k²)
                else
                    # Abstract index: full 4-momentum squared
                    coeff = _sym_mul(coeff, :p²)
                end
                break
            end
        end
    end

    # Unpaired momentum factors
    for i in eachindex(k_factors)
        i in paired && continue
        idx = k_factors[i][1]
        cv = _component_value(idx)
        if cv !== nothing && cv == 0
            coeff = _sym_mul(coeff, :ω)
        elseif cv !== nothing && cv > 0
            coeff = _sym_mul(coeff, :k_spatial)
        else
            coeff = _sym_mul(coeff, :k_unpaired)
        end
    end

    if !isempty(other_factors)
        coeff = _sym_mul(coeff, Symbol("_uncontracted"))
    end

    coeff
end

"""Extract component value from a component marker index like :_0, :_1."""
function _component_value(idx::TIndex)
    s = string(idx.name)
    startswith(s, "_") && length(s) > 1 || return nothing
    tryparse(Int, s[2:end])
end
