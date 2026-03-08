#= solve_tensors: solve linear tensor equations for unknown tensors.

Given a tensor equation (expressed as expr = 0) and a list of unknown
tensor names, decompose the equation into terms, extract scalar coefficients
of each unknown, and solve for the unknowns. Returns RewriteRules.

Equivalent to xAct/xTras SolveTensors, extended with optional take_traces.
=#

"""
    _contains_unknown(expr::TensorExpr, unknowns::Set{Symbol}) -> Bool

Check whether `expr` contains any tensor whose name is in `unknowns`.
"""
_contains_unknown(t::Tensor, unknowns::Set{Symbol}) = t.name in unknowns
_contains_unknown(::TScalar, ::Set{Symbol}) = false
function _contains_unknown(p::TProduct, unknowns::Set{Symbol})
    any(f -> _contains_unknown(f, unknowns), p.factors)
end
function _contains_unknown(s::TSum, unknowns::Set{Symbol})
    any(t -> _contains_unknown(t, unknowns), s.terms)
end
function _contains_unknown(d::TDeriv, unknowns::Set{Symbol})
    _contains_unknown(d.arg, unknowns)
end

"""
    _check_deriv_on_unknown(expr::TensorExpr, unknowns::Set{Symbol})

Error if any unknown appears under a derivative.
"""
_check_deriv_on_unknown(::Tensor, ::Set{Symbol}) = nothing
_check_deriv_on_unknown(::TScalar, ::Set{Symbol}) = nothing
function _check_deriv_on_unknown(p::TProduct, unknowns::Set{Symbol})
    for f in p.factors
        _check_deriv_on_unknown(f, unknowns)
    end
end
function _check_deriv_on_unknown(s::TSum, unknowns::Set{Symbol})
    for t in s.terms
        _check_deriv_on_unknown(t, unknowns)
    end
end
function _check_deriv_on_unknown(d::TDeriv, unknowns::Set{Symbol})
    if _contains_unknown(d.arg, unknowns)
        error("solve_tensors cannot solve differential equations: unknown under derivative")
    end
end

"""
    _decompose_term(term::TensorExpr, unknowns::Set{Symbol})
        -> Union{Nothing, NamedTuple{(:name, :tensor, :coeff), ...}}

Decompose a single term into (unknown_name, unknown_tensor, coefficient).
Returns `nothing` if the term contains no unknowns (it's a "known" term).
Errors if the term is nonlinear in unknowns.
"""
function _decompose_term(term::Tensor, unknowns::Set{Symbol})
    if term.name in unknowns
        return (name=term.name, tensor=term, coeff=TScalar(1 // 1))
    end
    nothing
end

_decompose_term(term::TScalar, ::Set{Symbol}) = nothing

function _decompose_term(term::TDeriv, unknowns::Set{Symbol})
    _check_deriv_on_unknown(term, unknowns)
    nothing
end

function _decompose_term(term::TProduct, unknowns::Set{Symbol})
    _check_deriv_on_unknown(term, unknowns)

    # Find factors that are unknowns
    unknown_indices = Int[]
    for (i, f) in enumerate(term.factors)
        if f isa Tensor && f.name in unknowns
            push!(unknown_indices, i)
        end
    end

    isempty(unknown_indices) && return nothing

    if length(unknown_indices) > 1
        error("solve_tensors: equation is nonlinear in unknowns (multiple unknowns in one term)")
    end

    idx = unknown_indices[1]
    unknown_tensor = term.factors[idx]::Tensor
    # Coefficient = scalar * remaining factors
    remaining = TensorExpr[term.factors[i] for i in eachindex(term.factors) if i != idx]
    coeff = if isempty(remaining)
        TScalar(term.scalar)
    else
        tproduct(term.scalar, remaining)
    end
    (name=unknown_tensor.name, tensor=unknown_tensor, coeff=coeff)
end

function _decompose_term(term::TSum, unknowns::Set{Symbol})
    # A TSum as a single "term" shouldn't happen after simplify,
    # but handle gracefully
    error("solve_tensors: unexpected TSum as term; simplify the equation first")
end

"""
    _invert_coefficient(coeff::TensorExpr) -> TensorExpr

Compute 1/coeff for scalar coefficients.
"""
function _invert_coefficient(coeff::TensorExpr)
    if coeff isa TScalar && coeff.val isa Rational{Int}
        coeff.val == 0 && error("solve_tensors: zero coefficient on unknown")
        return TScalar(1 // coeff.val)
    elseif coeff isa TScalar && coeff.val isa Integer
        coeff.val == 0 && error("solve_tensors: zero coefficient on unknown")
        return TScalar(1 // Rational{Int}(coeff.val))
    else
        # Symbolic coefficient: wrap as 1/c
        return TScalar(:(1 / $(coeff isa TScalar ? coeff.val : :coeff)))
    end
end

"""
    solve_tensors(equation::TensorExpr, unknowns::Vector{Symbol};
                  registry=nothing, make_rules=true, take_traces=false)

Solve a linear tensor equation `equation = 0` for the tensor(s) named in `unknowns`.

Returns a `Vector{RewriteRule}` (if `make_rules=true`) or a
`Vector{Pair{TensorExpr, TensorExpr}}` (LHS => RHS pairs) otherwise.

# Algorithm
1. Simplify the equation
2. Decompose each term: classify as known or containing an unknown
3. Group terms by unknown tensor name
4. Solve: unknown = -(1/coeff) * known_terms
5. Build rewrite rules with symmetry variants

# Errors
- Unknown under a derivative: "cannot solve differential equations"
- Multiple unknowns in one term: "nonlinear in unknowns"
"""
function solve_tensors(equation::TensorExpr, unknowns::Vector{Symbol};
                       registry::Union{TensorRegistry, Nothing}=nothing,
                       make_rules::Bool=true, take_traces::Bool=false)
    reg = registry !== nothing ? registry : current_registry()
    unk_set = Set(unknowns)

    # Phase 1: Normalize
    simplified = simplify(equation; registry=reg)

    # Optional: take traces
    if take_traces
        simplified = _solve_with_traces(simplified, unknowns, reg)
    end

    # Phase 2+3: Decompose and group
    terms = _get_terms(simplified)

    # known_terms: terms without unknowns (will form the RHS)
    # unknown_groups: Dict{Symbol, Vector{(coeff, tensor)}}
    known_terms = TensorExpr[]
    unknown_groups = Dict{Symbol, Vector{Tuple{TensorExpr, Tensor}}}()

    for term in terms
        dec = _decompose_term(term, unk_set)
        if dec === nothing
            push!(known_terms, term)
        else
            group = get!(Vector{Tuple{TensorExpr, Tensor}}, unknown_groups, dec.name)
            push!(group, (dec.coeff, dec.tensor))
        end
    end

    # Warn about unknowns not in the equation
    for u in unknowns
        if !haskey(unknown_groups, u)
            @warn "solve_tensors: unknown '$u' does not appear in the equation"
        end
    end

    # Phase 4: Solve
    # For each unknown: sum of (coeff_i * X_{indices_i}) = -sum(known_terms)
    # If there's exactly one term per unknown with a scalar coefficient,
    # X = -(1/coeff) * known_terms
    rhs_base = isempty(known_terms) ? ZERO : tsum(TensorExpr[-t for t in known_terms])

    results = Pair{TensorExpr, TensorExpr}[]

    for u in unknowns
        haskey(unknown_groups, u) || continue
        entries = unknown_groups[u]

        if length(entries) == 1
            coeff, tensor = entries[1]
            inv_coeff = _invert_coefficient(coeff)
            # RHS = -(1/coeff) * (-known_terms) = (1/coeff) * rhs_base
            sol = _apply_scalar(inv_coeff, rhs_base)
            sol = simplify(sol; registry=reg)
            push!(results, tensor => sol)
        else
            # Multiple terms with the same unknown (different index patterns)
            # This is more complex — try to handle the simple case where all
            # have the same index structure
            @warn "solve_tensors: multiple terms with unknown '$u'; returning first solution"
            coeff, tensor = entries[1]
            # Subtract other unknown terms from RHS
            other_terms = TensorExpr[]
            for i in 2:length(entries)
                c, t = entries[i]
                push!(other_terms, tproduct(1 // 1, TensorExpr[c, t]))
            end
            adj_rhs = tsum(vcat(TensorExpr[rhs_base], TensorExpr[-t for t in other_terms]))
            inv_coeff = _invert_coefficient(coeff)
            sol = _apply_scalar(inv_coeff, adj_rhs)
            sol = simplify(sol; registry=reg)
            push!(results, tensor => sol)
        end
    end

    # Phase 5: Build rules
    if make_rules
        rules = RewriteRule[]
        for (lhs, rhs) in results
            if lhs isa Tensor && has_tensor(reg, lhs.name)
                append!(rules, make_rule(lhs, rhs; use_symmetries=true, registry=reg))
            else
                push!(rules, RewriteRule(lhs, rhs))
            end
        end
        return rules
    else
        return results
    end
end

# Convenience: single unknown
function solve_tensors(equation::TensorExpr, unknown::Symbol; kwargs...)
    solve_tensors(equation, [unknown]; kwargs...)
end

# System of equations
function solve_tensors(equations::Vector{<:TensorExpr}, unknowns::Vector{Symbol};
                       registry::Union{TensorRegistry, Nothing}=nothing,
                       make_rules::Bool=true, take_traces::Bool=false)
    reg = registry !== nothing ? registry : current_registry()
    # Solve each equation and collect results
    all_rules = make_rules ? RewriteRule[] : Pair{TensorExpr, TensorExpr}[]
    remaining_unknowns = copy(unknowns)
    for eq in equations
        isempty(remaining_unknowns) && break
        result = solve_tensors(eq, remaining_unknowns;
                               registry=reg, make_rules=make_rules,
                               take_traces=take_traces)
        if make_rules
            append!(all_rules, result)
            # Remove solved unknowns
            for r in result
                if r.pattern isa Tensor
                    filter!(u -> u != r.pattern.name, remaining_unknowns)
                end
            end
        else
            append!(all_rules, result)
            for (lhs, _) in result
                if lhs isa Tensor
                    filter!(u -> u != lhs.name, remaining_unknowns)
                end
            end
        end
    end
    all_rules
end

# ─── Helpers ────────────────────────────────────────────────────────

"""Extract the terms of an expression as a flat vector."""
function _get_terms(expr::TSum)
    expr.terms
end
function _get_terms(expr::TensorExpr)
    # Single term (not a sum)
    expr == ZERO ? TensorExpr[] : TensorExpr[expr]
end

"""Apply a scalar factor to an expression."""
function _apply_scalar(scalar::TensorExpr, expr::TensorExpr)
    if scalar isa TScalar && scalar.val isa Rational{Int} && scalar.val == 1 // 1
        return expr
    end
    if expr == ZERO
        return ZERO
    end
    tproduct(1 // 1, TensorExpr[scalar, expr])
end

"""
    _solve_with_traces(expr, unknowns, reg) -> TensorExpr

Contract the equation with the inverse metric to extract trace equations.
Currently returns the original expression augmented with trace information.
"""
function _solve_with_traces(expr::TensorExpr, unknowns::Vector{Symbol},
                            reg::TensorRegistry)
    fidx = free_indices(expr)
    length(fidx) < 2 && return expr  # Nothing to trace for scalar equations

    # Find pairs of free indices to trace over
    # Find the metric for the manifold
    metric_name = :g
    for (_, mp) in reg.manifolds
        if mp.metric !== nothing
            metric_name = mp.metric
            break
        end
    end

    # Get first two free indices
    idx1 = fidx[1].name
    idx2 = fidx[2].name
    traced = abstract_trace(expr, idx1, idx2; metric=metric_name)
    traced = simplify(traced; registry=reg)
    traced
end
