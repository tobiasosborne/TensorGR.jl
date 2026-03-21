#= Parser for xAct/Invar's Mathematica source files.
#
# Extracts MaxIndex and MaxDualIndex tables from Invar.m, and provides
# a recursive-descent parser for the subset of Mathematica used in Invar
# database files (for future use when .m data files are obtained).
#
# Cross-checks our Invar database counts against xAct's ground truth.
#
# Reference: Garcia-Parrado & Martin-Garcia (2007), arXiv:0704.1756;
#            Martin-Garcia, Portugal & Manssur (2008), arXiv:0802.1274.
=#

# ============================================================================
# Part A: MaxIndex / MaxDualIndex extraction
# ============================================================================

"""
    extract_max_indices(invar_m_path::String) -> Dict{Vector{Int}, Int}

Parse `MaxIndex[{...}] = n` definitions from xAct's Invar.m source file.
Returns a Dict mapping case vectors to their MaxIndex values.

MaxIndex gives the total count of **non-product** canonical forms at Level 1
(permutation symmetries) for the given case.  For the algebraic case {0,...,0}
of degree d, this is the number of genuinely degree-d invariants (excluding
products of lower-degree invariants).

# Example
```julia
d = extract_max_indices("reference/xAct/xAct/Invar/Invar.m")
d[[0,0]]  # => 3  (degree 2: Ric^2, Kretschmann, R_{acbd}R^{abcd})
d[[0,0,0]]  # => 9  (degree 3: 9 non-product cubic invariants)
```
"""
function extract_max_indices(path::String)
    result = Dict{Vector{Int}, Int}()
    for line in eachline(path)
        m = match(r"^MaxIndex\[\{([0-9,\s]+)\}\]\s*[:=]+=?\s*(\d+)", line)
        if m !== nothing
            case = parse.(Int, strip.(split(m.captures[1], ",")))
            count = parse(Int, m.captures[2])
            result[case] = count
        end
    end
    result
end

"""
    extract_max_dual_indices(invar_m_path::String) -> Dict{Vector{Int}, Int}

Parse `MaxDualIndex[{...}] = n` definitions from xAct's Invar.m source file.
Returns a Dict mapping case vectors to their MaxDualIndex values.

MaxDualIndex gives the total count of dual invariants at Level 1.
"""
function extract_max_dual_indices(path::String)
    result = Dict{Vector{Int}, Int}()
    for line in eachline(path)
        m = match(r"^MaxDualIndex\[\{([0-9,\s]+)\}\]\s*[:=]+=?\s*(\d+)", line)
        if m !== nothing
            case = parse.(Int, strip.(split(m.captures[1], ",")))
            count = parse(Int, m.captures[2])
            result[case] = count
        end
    end
    result
end


# ============================================================================
# Part B: Mathematica expression parser (for future database file parsing)
# ============================================================================

# ---- AST types ----

abstract type MExpr end

struct MInt <: MExpr
    val::Int
end

struct MRational <: MExpr
    num::Int
    den::Int
end

struct MList <: MExpr
    items::Vector{MExpr}
end

struct MCall <: MExpr
    name::Symbol
    args::Vector{MExpr}
end

struct MRule <: MExpr
    lhs::MExpr
    rhs::MExpr
end

struct MPlus <: MExpr
    terms::Vector{MExpr}
end

struct MTimes <: MExpr
    factors::Vector{MExpr}
end

struct MNeg <: MExpr
    arg::MExpr
end

# ---- Tokenizer ----

struct MToken
    kind::Symbol   # :int, :ident, :lbrace, :rbrace, :lbracket, :rbracket,
                   # :lparen, :rparen, :comma, :arrow, :plus, :minus,
                   # :star, :slash, :eof
    val::String
end

function _tokenize_mathematica(s::String)
    tokens = MToken[]
    i = 1
    n = length(s)
    while i <= n
        c = s[i]
        if isspace(c)
            i += 1
            continue
        elseif c == '{'
            push!(tokens, MToken(:lbrace, "{"))
            i += 1
        elseif c == '}'
            push!(tokens, MToken(:rbrace, "}"))
            i += 1
        elseif c == '['
            push!(tokens, MToken(:lbracket, "["))
            i += 1
        elseif c == ']'
            push!(tokens, MToken(:rbracket, "]"))
            i += 1
        elseif c == '('
            push!(tokens, MToken(:lparen, "("))
            i += 1
        elseif c == ')'
            push!(tokens, MToken(:rparen, ")"))
            i += 1
        elseif c == ','
            push!(tokens, MToken(:comma, ","))
            i += 1
        elseif c == '+'
            push!(tokens, MToken(:plus, "+"))
            i += 1
        elseif c == '-'
            if i + 1 <= n && s[i+1] == '>'
                push!(tokens, MToken(:arrow, "->"))
                i += 2
            else
                push!(tokens, MToken(:minus, "-"))
                i += 1
            end
        elseif c == '*'
            push!(tokens, MToken(:star, "*"))
            i += 1
        elseif c == '/'
            push!(tokens, MToken(:slash, "/"))
            i += 1
        elseif isdigit(c)
            j = i
            while j <= n && isdigit(s[j])
                j += 1
            end
            push!(tokens, MToken(:int, s[i:j-1]))
            i = j
        elseif isletter(c) || c == '_' || c == '$'
            j = i
            while j <= n && (isletter(s[j]) || isdigit(s[j]) || s[j] == '_' || s[j] == '$')
                j += 1
            end
            push!(tokens, MToken(:ident, s[i:j-1]))
            i = j
        else
            error("Unexpected character '$(c)' at position $i in: $s")
        end
    end
    push!(tokens, MToken(:eof, ""))
    tokens
end

# ---- Recursive descent parser ----

mutable struct MParser
    tokens::Vector{MToken}
    pos::Int
end

function _peek(p::MParser)
    p.tokens[p.pos]
end

function _advance(p::MParser)
    t = p.tokens[p.pos]
    p.pos += 1
    t
end

function _expect(p::MParser, kind::Symbol)
    t = _advance(p)
    t.kind == kind || error("Expected $kind, got $(t.kind) '$(t.val)'")
    t
end

# expr = rule_expr
# rule_expr = add_expr ('->' add_expr)?
function _parse_expr(p::MParser)
    lhs = _parse_add(p)
    if _peek(p).kind == :arrow
        _advance(p)  # consume ->
        rhs = _parse_add(p)
        return MRule(lhs, rhs)
    end
    lhs
end

# add_expr = mul_expr (('+' | '-') mul_expr)*
function _parse_add(p::MParser)
    terms = MExpr[_parse_mul(p)]
    while _peek(p).kind in (:plus, :minus)
        op = _advance(p)
        t = _parse_mul(p)
        if op.kind == :minus
            push!(terms, MNeg(t))
        else
            push!(terms, t)
        end
    end
    length(terms) == 1 ? terms[1] : MPlus(terms)
end

# mul_expr = unary (('*' | implicit_multiply) unary)*
# Implicit multiplication: ident or int followed by ident or lbrace or lparen
function _parse_mul(p::MParser)
    factors = MExpr[_parse_unary(p)]
    while true
        k = _peek(p).kind
        if k == :star
            _advance(p)
            push!(factors, _parse_unary(p))
        elseif k == :slash
            _advance(p)
            den = _parse_unary(p)
            # Handle a/b as a rational if both are ints
            if length(factors) == 1 && factors[1] isa MInt && den isa MInt
                factors[1] = MRational(factors[1].val, den.val)
            elseif den isa MInt
                # Keep as division: replace nothing, append MRational(1, den)
                push!(factors, MRational(1, den.val))
            else
                error("Non-integer denominator in division")
            end
        elseif k in (:ident, :int, :lbrace, :lparen) && _can_implicit_mul(factors[end])
            # Implicit multiplication (juxtaposition)
            push!(factors, _parse_unary(p))
        else
            break
        end
    end
    length(factors) == 1 ? factors[1] : MTimes(factors)
end

function _can_implicit_mul(prev::MExpr)
    # Implicit multiply is valid after an atom-like expression
    prev isa MInt || prev isa MRational || prev isa MCall ||
        prev isa MList || prev isa MTimes || prev isa MNeg
end

# unary = '-' unary | atom
function _parse_unary(p::MParser)
    if _peek(p).kind == :minus
        _advance(p)
        return MNeg(_parse_unary(p))
    end
    _parse_atom(p)
end

# atom = INTEGER | IDENT ('[' arglist ']')? | '{' arglist '}' | '(' expr ')'
function _parse_atom(p::MParser)
    t = _peek(p)
    if t.kind == :int
        _advance(p)
        return MInt(parse(Int, t.val))
    elseif t.kind == :ident
        _advance(p)
        name = Symbol(t.val)
        if _peek(p).kind == :lbracket
            _advance(p)  # consume [
            args = _parse_arglist(p, :rbracket)
            _expect(p, :rbracket)
            return MCall(name, args)
        end
        return MCall(name, MExpr[])  # bare identifier = 0-arg call
    elseif t.kind == :lbrace
        _advance(p)  # consume {
        items = _parse_arglist(p, :rbrace)
        _expect(p, :rbrace)
        return MList(items)
    elseif t.kind == :lparen
        _advance(p)  # consume (
        e = _parse_expr(p)
        _expect(p, :rparen)
        return e
    else
        error("Unexpected token $(t.kind) '$(t.val)'")
    end
end

# arglist = expr (',' expr)*  |  (empty)
function _parse_arglist(p::MParser, close::Symbol)
    if _peek(p).kind == close
        return MExpr[]
    end
    args = MExpr[_parse_expr(p)]
    while _peek(p).kind == :comma
        _advance(p)
        push!(args, _parse_expr(p))
    end
    args
end

"""
    parse_mathematica(s::String) -> MExpr

Parse a Mathematica expression string into an MExpr AST.

Supports the subset of Mathematica used in Invar database files:
integers, identifiers, function calls `f[x,y]`, lists `{a,b,c}`,
rules `lhs -> rhs`, arithmetic `+`, `-`, `*`, `/`, parentheses,
and implicit multiplication (juxtaposition).

# Example
```julia
parse_mathematica("RInv[{0,0}, 4]")
# => MCall(:RInv, [MList([MInt(0), MInt(0)]), MInt(4)])
```
"""
function parse_mathematica(s::String)
    tokens = _tokenize_mathematica(s)
    p = MParser(tokens, 1)
    result = _parse_expr(p)
    _peek(p).kind == :eof || error("Unexpected trailing tokens: $(_peek(p).val)")
    result
end


# ============================================================================
# Part C: Cross-checker
# ============================================================================

# Product counts by degree (number of product-type canonical forms at Level 1).
# Computed from partition function of the polynomial ring generated by
# non-product independent invariants at all lower degrees.
#
# Degree 1: 0 products (only R)
# Degree 2: 1 product (R^2)
# Degree 3: 4 products (R^3, R*Ric^2, R*K, R*I4_d2)
# Degree 4: 19 products (from products of d1*d3, d2*d2, d1*d1*d2, d1^4)
# Degree 5: 84 products
# Degree 6: 457 products
# Degree 7: 3078 products
#
# Source: degree5_7.jl comments; verified by partition enumeration.
const _PRODUCT_COUNTS = Dict{Int,Int}(
    1 => 0,
    2 => 1,
    3 => 4,
    4 => 19,
    5 => 84,
    6 => 457,
    7 => 3078,
)

"""
    cross_check_counts(; invar_m="reference/xAct/xAct/Invar/Invar.m") -> NamedTuple

Compare our Invar database counts against xAct's MaxIndex values.

Returns a NamedTuple with fields:
- `matches::Vector` -- cases where our counts agree with xAct
- `mismatches::Vector` -- cases where counts disagree
- `missing_in_ours::Vector` -- cases present in xAct but not in our database
- `xact_only_cases::Vector` -- differential/mixed cases in xAct we don't track

**Convention difference:**
xAct's MaxIndex counts **non-product** canonical forms at Level 1.
Our database stores total canonical counts (product + non-product) in
`n_independent + n_dependent` at step 1.  The cross-check subtracts
known product counts to compare.
"""
function cross_check_counts(; invar_m::String="reference/xAct/xAct/Invar/Invar.m")
    xact = extract_max_indices(invar_m)

    matches = NamedTuple{(:degree, :case, :xact_count, :our_count),
                          Tuple{Int, Vector{Int}, Int, Int}}[]
    mismatches = NamedTuple{(:degree, :case, :xact_count, :our_count, :our_total),
                             Tuple{Int, Vector{Int}, Int, Int, Int}}[]
    missing_in_ours = NamedTuple{(:degree, :case, :xact_count),
                                  Tuple{Int, Vector{Int}, Int}}[]
    xact_only = NamedTuple{(:case, :xact_count), Tuple{Vector{Int}, Int}}[]

    for (case_vec, xact_count) in sort(collect(xact), by=x -> (length(x[1]), x[1]))
        degree = length(case_vec)

        # Check if this is a purely algebraic case (all zeros)
        is_algebraic = all(==(0), case_vec)

        if !is_algebraic
            # Differential/mixed cases: we don't track these in the same way
            push!(xact_only, (case=case_vec, xact_count=xact_count))
            continue
        end

        # Build the case_key our database uses (e.g., "0_0" for degree 2)
        case_key = join(string.(case_vec), "_")

        # Look up our Level 1 data
        our_data = nothing
        try
            # Use the database registry directly to avoid requiring TensorGR
            # This function is designed to work standalone too
            our_data = _lookup_our_data(degree, case_key)
        catch
            push!(missing_in_ours, (degree=degree, case=case_vec, xact_count=xact_count))
            continue
        end

        if our_data === nothing
            push!(missing_in_ours, (degree=degree, case=case_vec, xact_count=xact_count))
            continue
        end

        total_canonical = our_data.n_independent + our_data.n_dependent
        products = get(_PRODUCT_COUNTS, degree, 0)
        our_non_product = total_canonical - products

        if our_non_product == xact_count
            push!(matches, (degree=degree, case=case_vec,
                           xact_count=xact_count, our_count=our_non_product))
        else
            push!(mismatches, (degree=degree, case=case_vec,
                              xact_count=xact_count, our_count=our_non_product,
                              our_total=total_canonical))
        end
    end

    (matches=matches, mismatches=mismatches,
     missing_in_ours=missing_in_ours, xact_only_cases=xact_only)
end

"""
    _lookup_our_data(degree, case_key) -> NamedTuple or nothing

Look up our Level 1 (step=1) data for a given degree and case_key.
Returns (n_independent=..., n_dependent=...) or nothing if not found.

When running inside TensorGR, uses list_invar_cases.
When running standalone, returns nothing (caller handles missing data).
"""
function _lookup_our_data(degree::Int, case_key::String)
    # Try to use TensorGR's API if available
    if isdefined(Main, :TensorGR) || @isdefined(list_invar_cases)
        cases = list_invar_cases(; degree=degree, step=1)
        for c in cases
            if c.case_key == case_key && c.dim === nothing
                return (n_independent=c.n_independent, n_dependent=c.n_dependent)
            end
        end
    end
    nothing
end

"""
    cross_check_dual_counts(; invar_m="reference/xAct/xAct/Invar/Invar.m") -> NamedTuple

Compare our dual invariant database counts against xAct's MaxDualIndex values.

Returns a NamedTuple with fields:
- `matches::Vector` -- cases where our counts agree
- `mismatches::Vector` -- cases where counts disagree
- `missing_in_ours::Vector` -- cases present in xAct but not in our database
"""
function cross_check_dual_counts(; invar_m::String="reference/xAct/xAct/Invar/Invar.m")
    xact = extract_max_dual_indices(invar_m)

    matches = NamedTuple{(:degree, :case, :xact_count, :our_count),
                          Tuple{Int, Vector{Int}, Int, Int}}[]
    mismatches = NamedTuple{(:degree, :case, :xact_count, :our_count),
                             Tuple{Int, Vector{Int}, Int, Int}}[]
    missing_in_ours = NamedTuple{(:degree, :case, :xact_count),
                                  Tuple{Int, Vector{Int}, Int}}[]

    for (case_vec, xact_count) in sort(collect(xact), by=x -> (length(x[1]), x[1]))
        degree = length(case_vec)
        is_algebraic = all(==(0), case_vec)
        if !is_algebraic
            # Skip differential dual cases for now
            continue
        end

        case_key = "dual_" * join(string.(case_vec), "_")

        our_data = nothing
        try
            if isdefined(Main, :TensorGR) || @isdefined(list_invar_cases)
                cases = list_invar_cases(; degree=degree, step=6)
                for c in cases
                    if c.case_key == case_key
                        our_data = (n_independent=c.n_independent, n_dependent=c.n_dependent)
                        break
                    end
                end
            end
        catch
        end

        if our_data === nothing
            push!(missing_in_ours, (degree=degree, case=case_vec, xact_count=xact_count))
            continue
        end

        our_total = our_data.n_independent + our_data.n_dependent
        if our_total == xact_count || our_data.n_independent == xact_count
            push!(matches, (degree=degree, case=case_vec,
                           xact_count=xact_count, our_count=our_data.n_independent))
        else
            push!(mismatches, (degree=degree, case=case_vec,
                              xact_count=xact_count, our_count=our_data.n_independent))
        end
    end

    (matches=matches, mismatches=mismatches, missing_in_ours=missing_in_ours)
end
