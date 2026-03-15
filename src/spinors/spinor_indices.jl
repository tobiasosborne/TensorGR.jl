# Spinor dummy pair analysis for mixed-bundle expressions.
#
# Extends fresh_index() with vbundle-aware name generation for spinor indices,
# and provides spinor-aware dummy normalization for display.
#
# Reference: Penrose & Rindler, Spinors and Space-Time Vol 1 (1984), Ch 2.

# ── Canonical name lists ────────────────────────────────────────────────────

# Fresh index alphabets (used by fresh_index when generating new dummies)
const _SPINOR_FRESH_UNDOTTED = [:A, :B, :C, :D, :E, :F]
const _SPINOR_FRESH_DOTTED   = [:Ap, :Bp, :Cp, :Dp, :Ep, :Fp]

# Display-normalization alphabets (used by normalize_spinor_dummies)
const _SPINOR_CANONICAL_UNDOTTED = [:P, :Q, :R, :S, :T, :U]
const _SPINOR_CANONICAL_DOTTED   = [:Pp, :Qp, :Rp, :Sp, :Tp, :Up]

# ── fresh_spinor_index ──────────────────────────────────────────────────────

"""
    fresh_spinor_index(used::Set{Symbol}; dotted::Bool=false) -> Symbol

Generate a fresh spinor index name not in `used`.

For undotted (SL2C) indices: tries A,B,C,D,E,F then A1,B1,...
For dotted (SL2C_dot) indices: tries Ap,Bp,Cp,Dp,Ep,Fp then Ap1,Bp1,...

See also [`fresh_index`](@ref) for the vbundle-dispatched version.
"""
function fresh_spinor_index(used::Set{Symbol}; dotted::Bool=false)
    alphabet = dotted ? _SPINOR_FRESH_DOTTED : _SPINOR_FRESH_UNDOTTED
    for s in alphabet
        s in used || return s
    end
    # Extended names: A1, B1, ... or Ap1, Bp1, ...
    for n in 1:100
        for s in alphabet
            ext = Symbol(s, n)
            ext in used || return ext
        end
    end
    error("Could not generate fresh spinor index (exhausted 600+ names)")
end

# ── Register spinor dispatch hook for fresh_index ───────────────────────────

# Set the callback in _FRESH_SPINOR_HOOK (defined in ast/indices.jl) so that
# fresh_index(used; vbundle=:SL2C) and fresh_index(used; vbundle=:SL2C_dot)
# produce spinor-appropriate names.
_FRESH_SPINOR_HOOK[] = function(used::Set{Symbol}, vbundle::Symbol)
    if vbundle === :SL2C
        return fresh_spinor_index(used; dotted=false)
    elseif vbundle === :SL2C_dot
        return fresh_spinor_index(used; dotted=true)
    end
    return nothing  # not a spinor bundle, fall through
end

# ── spinor_dummy_pairs ──────────────────────────────────────────────────────

"""
    spinor_dummy_pairs(expr::TensorExpr) -> Vector{Tuple{TIndex, TIndex}}

Return contracted spinor index pairs from `expr`. Each pair has one Up and one
Down index with the same name and same spinor vbundle (`:SL2C` or `:SL2C_dot`).

Cross-bundle pairs (same name but different vbundles) are never matched.
This is a filtered view of `dummy_pairs(expr)`.
"""
function spinor_dummy_pairs(expr::TensorExpr)
    pairs = dummy_pairs(expr)
    filter(p -> is_spinor_index(p[1]), pairs)
end

# ── normalize_spinor_dummies ────────────────────────────────────────────────

"""
    normalize_spinor_dummies(expr::TensorExpr) -> TensorExpr

Rename spinor dummy indices to canonical names, grouped by vbundle:
- SL2C dummies -> P, Q, R, S, T, U  (then P1, Q1, ...)
- SL2C_dot dummies -> Pp, Qp, Rp, Sp, Tp, Up  (then Pp1, Qp1, ...)
- Tangent dummies are left unchanged.

Dummy ordering follows first-occurrence order in the expression's index list.
"""
function normalize_spinor_dummies(expr::TensorExpr)
    pairs = dummy_pairs(expr)
    isempty(pairs) && return expr

    all_idxs = indices(expr)

    # Build first-occurrence map for ordering
    first_occurrence = Dict{Symbol, Int}()
    for (i, idx) in enumerate(all_idxs)
        haskey(first_occurrence, idx.name) || (first_occurrence[idx.name] = i)
    end

    # Group dummy names by vbundle
    sl2c_dummies = Symbol[]
    sl2c_dot_dummies = Symbol[]
    for (up_idx, _) in pairs
        if up_idx.vbundle === :SL2C
            push!(sl2c_dummies, up_idx.name)
        elseif up_idx.vbundle === :SL2C_dot
            push!(sl2c_dot_dummies, up_idx.name)
        end
    end

    # Sort each group by first occurrence
    sort!(sl2c_dummies, by = n -> get(first_occurrence, n, 0))
    sort!(sl2c_dot_dummies, by = n -> get(first_occurrence, n, 0))

    # Nothing to do if no spinor dummies
    isempty(sl2c_dummies) && isempty(sl2c_dot_dummies) && return expr

    # Build canonical name for the i-th dummy in a given alphabet
    function _canonical_name(i::Int, alphabet::Vector{Symbol})
        if i <= length(alphabet)
            return alphabet[i]
        else
            base_idx = mod1(i - length(alphabet), length(alphabet))
            gen = div(i - length(alphabet) - 1, length(alphabet)) + 1
            return Symbol(alphabet[base_idx], gen)
        end
    end

    # Two-phase rename to avoid collisions (old -> tmp -> canonical)
    phase1 = Dict{Symbol,Symbol}()
    phase2 = Dict{Symbol,Symbol}()
    counter = 0

    for (dummies, alphabet) in [(sl2c_dummies, _SPINOR_CANONICAL_UNDOTTED),
                                 (sl2c_dot_dummies, _SPINOR_CANONICAL_DOTTED)]
        for (i, old_name) in enumerate(dummies)
            counter += 1
            tmp = Symbol("__stmp", counter)
            canon = _canonical_name(i, alphabet)
            old_name != tmp && (phase1[old_name] = tmp)
            tmp != canon && (phase2[tmp] = canon)
        end
    end

    result = isempty(phase1) ? expr : rename_dummies(expr, phase1)
    isempty(phase2) ? result : rename_dummies(result, phase2)
end
