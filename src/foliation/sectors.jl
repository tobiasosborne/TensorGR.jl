#= SO(3) sector collection for SVT decomposition.

After applying SVT substitution rules, the expression is a sum of terms
containing scalar (Φ, ψ), vector (S/V), and tensor (hTT) fields.
This module groups terms by their SO(3) spin sector.
=#

"""
    collect_sectors(expr::TensorExpr, fields::SVTFields=DEFAULT_SVT) -> Dict{Symbol, TensorExpr}

Group terms in a sum by which SVT fields they contain:
- `:scalar` → terms with only scalar fields (Φ, B, ψ, E)
- `:vector` → terms with only vector fields (S, F)
- `:tensor` → terms with only tensor fields (hTT)
- `:mixed`  → cross-sector terms (should vanish by Schur orthogonality)
- `:pure_scalar` → terms with no SVT fields (pure numbers/momenta)

Returns a Dict mapping sector labels to their collected expressions.
"""
function collect_sectors(expr::TensorExpr, fields::SVTFields=DEFAULT_SVT)
    scalar_names = Set{Symbol}([fields.ϕ, fields.B, fields.ψ, fields.E])
    vector_names = Set{Symbol}([fields.S, fields.F])
    tensor_names = Set{Symbol}([fields.hTT])
    # Structural tensors that don't determine the sector
    ignore_names = Set{Symbol}([:δ, :g, :η, :k, :p])

    # Flatten any nested TSums so each leaf term gets classified individually
    raw_terms = expr isa TSum ? expr.terms : TensorExpr[expr]
    terms = TensorExpr[]
    _flatten_sum_terms!(terms, raw_terms)

    sector_terms = Dict{Symbol, Vector{TensorExpr}}(
        :scalar => TensorExpr[],
        :vector => TensorExpr[],
        :tensor => TensorExpr[],
        :mixed => TensorExpr[],
        :pure_scalar => TensorExpr[]
    )

    for term in terms
        field_names = setdiff(_extract_field_names(term), ignore_names)

        has_scalar = !isempty(intersect(field_names, scalar_names))
        has_vector = !isempty(intersect(field_names, vector_names))
        has_tensor = !isempty(intersect(field_names, tensor_names))

        n_sectors = has_scalar + has_vector + has_tensor

        if n_sectors == 0
            push!(sector_terms[:pure_scalar], term)
        elseif n_sectors > 1
            push!(sector_terms[:mixed], term)
        elseif has_scalar
            push!(sector_terms[:scalar], term)
        elseif has_vector
            push!(sector_terms[:vector], term)
        else
            push!(sector_terms[:tensor], term)
        end
    end

    result = Dict{Symbol, TensorExpr}()
    for (sector, terms_list) in sector_terms
        isempty(terms_list) && continue
        result[sector] = tsum(terms_list)
    end
    result
end

"""Extract all tensor names appearing in an expression."""
function _extract_field_names(expr::TensorExpr)
    names = Set{Symbol}()
    _collect_tensor_names!(names, expr)
    names
end

function _collect_tensor_names!(names::Set{Symbol}, t::Tensor)
    push!(names, t.name)
end

function _collect_tensor_names!(names::Set{Symbol}, p::TProduct)
    for f in p.factors
        _collect_tensor_names!(names, f)
    end
end

function _collect_tensor_names!(names::Set{Symbol}, s::TSum)
    for t in s.terms
        _collect_tensor_names!(names, t)
    end
end

function _collect_tensor_names!(names::Set{Symbol}, d::TDeriv)
    _collect_tensor_names!(names, d.arg)
end

function _collect_tensor_names!(::Set{Symbol}, ::TScalar)
    # scalars have no tensor names
end

"""Recursively flatten nested TSum terms into a flat list."""
function _flatten_sum_terms!(out::Vector{TensorExpr}, terms)
    for t in terms
        if t isa TSum
            _flatten_sum_terms!(out, t.terms)
        else
            push!(out, t)
        end
    end
end
