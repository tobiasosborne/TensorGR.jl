# ── Spinor display helpers ──────────────────────────────────────────
# Dotted spinor indices are stored as :Ap, :Bp, ... in the AST.
# Display strips trailing 'p' and appends a prime (show/unicode) or
# wraps in \dot{} (LaTeX).  Penrose & Rindler Vol 1, Section 2.5.

"""
    _spinor_base_name(name::Symbol) -> String

For dotted spinor index names (:Ap, :Bp, ...), strip the trailing 'p' suffix
to recover the base letter.  Returns the full name as-is for names that do
not end in 'p' or are single-character (e.g. :A -> "A", :Ap -> "A").
"""
function _spinor_base_name(name::Symbol)
    s = string(name)
    if length(s) >= 2 && s[end] == 'p'
        return s[1:end-1]
    end
    return s
end

function Base.show(io::IO, idx::TIndex)
    if idx.vbundle === :SL2C_dot
        base = _spinor_base_name(idx.name)
        if idx.position == Down
            print(io, "-", base, "'")
        else
            print(io, base, "'")
        end
    else
        if idx.position == Down
            print(io, "-", idx.name)
        else
            print(io, idx.name)
        end
    end
end

function Base.show(io::IO, t::Tensor)
    print(io, t.name)
    if !isempty(t.indices)
        print(io, "[")
        join(io, t.indices, ", ")
        print(io, "]")
    end
end

function Base.show(io::IO, s::TScalar)
    print(io, s.val)
end

function Base.show(io::IO, p::TProduct)
    s = p.scalar
    if isempty(p.factors)
        print(io, s)
        return
    end
    if s == 1//1
        # no prefix
    elseif s == -1//1
        print(io, "-")
    else
        print(io, "(", s, ") * ")
    end
    for (i, f) in enumerate(p.factors)
        i > 1 && print(io, " * ")
        show(io, f)
    end
end

function Base.show(io::IO, s::TSum)
    for (i, t) in enumerate(s.terms)
        i > 1 && print(io, " + ")
        show(io, t)
    end
end

function Base.show(io::IO, d::TDeriv)
    if d.covd == :partial
        print(io, "∂")
    else
        print(io, d.covd)
    end
    print(io, "[")
    show(io, d.index)
    print(io, "](")
    show(io, d.arg)
    print(io, ")")
end

# ── LaTeX rendering ──────────────────────────────────────────────────

"""
    to_latex(expr::TensorExpr)::String

Return a LaTeX string representation of a tensor expression.
"""
function to_latex end

function to_latex(idx::TIndex)
    s = idx.vbundle === :SL2C_dot ? "\\dot{$(_spinor_base_name(idx.name))}" : string(idx.name)
    if idx.position == Down
        return "_{$s}"
    else
        return "^{$s}"
    end
end

function _latex_index_name(idx::TIndex)
    idx.vbundle === :SL2C_dot ? "\\dot{$(_spinor_base_name(idx.name))}" : string(idx.name)
end

function to_latex(t::Tensor)
    s = string(t.name)
    if isempty(t.indices)
        return s
    end
    up_indices = filter(i -> i.position == Up, t.indices)
    dn_indices = filter(i -> i.position == Down, t.indices)
    if !isempty(up_indices)
        s *= "^{" * join([_latex_index_name(i) for i in up_indices], " ") * "}"
    end
    if !isempty(dn_indices)
        s *= "_{" * join([_latex_index_name(i) for i in dn_indices], " ") * "}"
    end
    return s
end

function to_latex(s::TScalar)
    return string(s.val)
end

function _latex_coeff(s::Rational{Int})
    if denominator(s) == 1
        return string(numerator(s))
    else
        n = numerator(s)
        d = denominator(s)
        sign = n < 0 ? "-" : ""
        return sign * "\\frac{$(abs(n))}{$d}"
    end
end

function to_latex(p::TProduct)
    s = p.scalar
    if isempty(p.factors)
        return _latex_coeff(s)
    end
    factors_str = join([to_latex(f) for f in p.factors], " ")
    if s == 1//1
        return factors_str
    elseif s == -1//1
        return "-" * factors_str
    else
        return _latex_coeff(s) * " " * factors_str
    end
end

function to_latex(sm::TSum)
    if isempty(sm.terms)
        return "0"
    end
    parts = String[]
    for (i, t) in enumerate(sm.terms)
        ts = to_latex(t)
        if i == 1
            push!(parts, ts)
        else
            # Check if this term starts with a minus sign
            if startswith(ts, "-")
                push!(parts, " - " * lstrip(ts[2:end]))
            else
                push!(parts, " + " * ts)
            end
        end
    end
    return join(parts)
end

function to_latex(d::TDeriv)
    idx_str = _latex_index_name(d.index)
    return "\\partial_{$idx_str} " * to_latex(d.arg)
end

# ── Unicode rendering ────────────────────────────────────────────────

const _superscript_digits = Dict(
    '0' => '\u2070', '1' => '\u00B9', '2' => '\u00B2', '3' => '\u00B3',
    '4' => '\u2074', '5' => '\u2075', '6' => '\u2076', '7' => '\u2077',
    '8' => '\u2078', '9' => '\u2079',
)

const _subscript_digits = Dict(
    '0' => '\u2080', '1' => '\u2081', '2' => '\u2082', '3' => '\u2083',
    '4' => '\u2084', '5' => '\u2085', '6' => '\u2086', '7' => '\u2087',
    '8' => '\u2088', '9' => '\u2089',
)

"""
    to_unicode(expr::TensorExpr)::String

Return a Unicode string representation of a tensor expression suitable for terminal display.

Numeric indices use Unicode super/subscript digits. Letter indices use `^a` / `_a` notation
(actual Unicode subscript letters are avoided since they cause ParseError in Julia).
"""
function to_unicode end

function _unicode_super(s::AbstractString)
    # If all characters are digits, use Unicode superscripts
    if all(isdigit, s)
        return String([_superscript_digits[c] for c in s])
    else
        return "^" * s
    end
end

function _unicode_sub(s::AbstractString)
    # If all characters are digits, use Unicode subscripts
    if all(isdigit, s)
        return String([_subscript_digits[c] for c in s])
    else
        return "_" * s
    end
end

function to_unicode(idx::TIndex)
    if idx.vbundle === :SL2C_dot
        base = _spinor_base_name(idx.name)
        s = base * "'"
        return idx.position == Down ? _unicode_sub(s) : _unicode_super(s)
    end
    s = string(idx.name)
    if idx.position == Down
        return _unicode_sub(s)
    else
        return _unicode_super(s)
    end
end

function to_unicode(t::Tensor)
    s = string(t.name)
    if isempty(t.indices)
        return s
    end
    # Render indices in order, grouping consecutive same-position indices
    for idx in t.indices
        s *= to_unicode(idx)
    end
    return s
end

function to_unicode(s::TScalar)
    return string(s.val)
end

function to_unicode(p::TProduct)
    s = p.scalar
    if isempty(p.factors)
        return string(s)
    end
    factors_str = join([to_unicode(f) for f in p.factors], " ")
    if s == 1//1
        return factors_str
    elseif s == -1//1
        return "-" * factors_str
    else
        return "(" * string(s) * ") " * factors_str
    end
end

function to_unicode(sm::TSum)
    if isempty(sm.terms)
        return "0"
    end
    parts = String[]
    for (i, t) in enumerate(sm.terms)
        ts = to_unicode(t)
        if i == 1
            push!(parts, ts)
        else
            if startswith(ts, "-")
                push!(parts, " - " * lstrip(ts[2:end]))
            else
                push!(parts, " + " * ts)
            end
        end
    end
    return join(parts)
end

function to_unicode(d::TDeriv)
    idx_str = to_unicode(d.index)
    return "∂" * idx_str * "(" * to_unicode(d.arg) * ")"
end
