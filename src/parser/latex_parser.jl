#= LaTeX tensor expression parser.

Parse textbook-style tensor notation into TensorExpr:

    tex"R_{abcd}"                      → Tensor(:R, [down(:a),...])
    tex"g^{ab} R_{abcd}"               → g * R  (juxtaposition = product)
    tex"R^a{}_{bcd}"                   → mixed indices
    tex"R_{ab} - \frac{1}{2} g_{ab} R" → Einstein tensor
    tex"\partial_a V^b"                → TDeriv(down(:a), Tensor(:V,[up(:b)]))
    tex"\nabla_a V^b"                  → TDeriv(down(:a), Tensor(:V,[up(:b)]))

Grammar:
    expr    := sum
    sum     := product (('+' | '-') product)*
    product := unary unary*            (juxtaposition = multiplication)
    unary   := '-' unary | atom
    atom    := fraction | number | derivative | tensor | '(' expr ')'
    fraction := '\frac' '{' number '}' '{' number '}'
    derivative := ('\partial'|'\nabla') sub_index atom
    tensor   := NAME indexblock?
    indexblock := ('^' indexarg | '_' indexarg | '{' '}')*
    indexarg  := '{' index* '}' | single_name
    index     := NAME (split into chars if multi-ASCII)
=#

# ── Token types ─────────────────────────────────────────────────────

struct TexToken
    type::Symbol  # :name :number :sup :sub :lbrace :rbrace :plus :minus
                  # :slash :frac :partial :nabla :lparen :rparen :eof
    value::String
    pos::Int
end

# ── LaTeX command mappings ──────────────────────────────────────────

const _LATEX_GREEK = Dict{String,String}(
    "alpha"=>"α", "beta"=>"β", "gamma"=>"γ", "delta"=>"δ",
    "epsilon"=>"ε", "varepsilon"=>"ε", "zeta"=>"ζ", "eta"=>"η",
    "theta"=>"θ", "vartheta"=>"ϑ", "iota"=>"ι", "kappa"=>"κ",
    "lambda"=>"λ", "mu"=>"μ", "nu"=>"ν", "xi"=>"ξ", "pi"=>"π",
    "rho"=>"ρ", "sigma"=>"σ", "tau"=>"τ", "upsilon"=>"υ",
    "phi"=>"φ", "varphi"=>"φ", "chi"=>"χ", "psi"=>"ψ", "omega"=>"ω",
    "Gamma"=>"Γ", "Delta"=>"Δ", "Theta"=>"Θ", "Lambda"=>"Λ",
    "Xi"=>"Ξ", "Pi"=>"Π", "Sigma"=>"Σ", "Phi"=>"Φ",
    "Psi"=>"Ψ", "Omega"=>"Ω",
)

# ── Tokenizer ───────────────────────────────────────────────────────

function _tex_tokenize(s::AbstractString)
    chars = collect(s)
    tokens = TexToken[]
    i = 1
    n = length(chars)

    while i <= n
        c = chars[i]

        if isspace(c)
            i += 1
            continue

        elseif c == '\\'
            # LaTeX command
            pos = i
            i += 1
            cmd = ""
            while i <= n && isletter(chars[i])
                cmd *= string(chars[i])
                i += 1
            end
            if cmd == "frac"
                push!(tokens, TexToken(:frac, "\\frac", pos))
            elseif cmd == "partial"
                push!(tokens, TexToken(:partial, "∂", pos))
            elseif cmd == "nabla"
                push!(tokens, TexToken(:nabla, "∇", pos))
            elseif cmd == "cdot" || cmd == "times"
                # explicit multiplication — skip (juxtaposition handles it)
                continue
            elseif haskey(_LATEX_GREEK, cmd)
                push!(tokens, TexToken(:name, _LATEX_GREEK[cmd], pos))
            else
                # Unknown command → treat as name (e.g., \Box, \square)
                push!(tokens, TexToken(:name, cmd, pos))
            end

        elseif c == '∂'
            push!(tokens, TexToken(:partial, "∂", i))
            i += 1
        elseif c == '∇'
            push!(tokens, TexToken(:nabla, "∇", i))
            i += 1

        elseif c == '^'
            push!(tokens, TexToken(:sup, "^", i))
            i += 1
        elseif c == '_'
            push!(tokens, TexToken(:sub, "_", i))
            i += 1
        elseif c == '{'
            push!(tokens, TexToken(:lbrace, "{", i))
            i += 1
        elseif c == '}'
            push!(tokens, TexToken(:rbrace, "}", i))
            i += 1
        elseif c == '('
            push!(tokens, TexToken(:lparen, "(", i))
            i += 1
        elseif c == ')'
            push!(tokens, TexToken(:rparen, ")", i))
            i += 1
        elseif c == '+'
            push!(tokens, TexToken(:plus, "+", i))
            i += 1
        elseif c == '-'
            push!(tokens, TexToken(:minus, "-", i))
            i += 1
        elseif c == '/'
            push!(tokens, TexToken(:slash, "/", i))
            i += 1
        elseif c == '='
            error("tex parser: '=' not supported — parse each side separately")

        elseif isdigit(c)
            pos = i
            num = string(c)
            i += 1
            while i <= n && isdigit(chars[i])
                num *= string(chars[i])
                i += 1
            end
            push!(tokens, TexToken(:number, num, pos))

        elseif isascii(c) && isletter(c)
            # ASCII letters: group into multi-char name
            pos = i
            name = string(c)
            i += 1
            while i <= n && isascii(chars[i]) && (isletter(chars[i]) || isdigit(chars[i]))
                name *= string(chars[i])
                i += 1
            end
            push!(tokens, TexToken(:name, name, pos))

        elseif isletter(c)
            # Non-ASCII letter (Greek Unicode etc): individual token
            push!(tokens, TexToken(:name, string(c), i))
            i += 1

        else
            error("tex parser: unexpected character '$(c)' at position $i")
        end
    end

    push!(tokens, TexToken(:eof, "", n + 1))
    tokens
end

# ── Parser state ────────────────────────────────────────────────────

mutable struct _TexParser
    tokens::Vector{TexToken}
    pos::Int
end

function _peek(p::_TexParser)
    p.tokens[p.pos]
end

function _advance!(p::_TexParser)
    t = p.tokens[p.pos]
    p.pos += 1
    t
end

function _expect!(p::_TexParser, type::Symbol)
    t = _advance!(p)
    t.type == type || _parse_error(p, "expected $(type), got '$(t.value)'")
    t
end

function _parse_error(p::_TexParser, msg::String)
    tok = p.pos <= length(p.tokens) ? p.tokens[p.pos] : p.tokens[end]
    error("tex parser: $msg at position $(tok.pos)")
end

# ── Recursive descent ──────────────────────────────────────────────

function _parse_expr(p::_TexParser)
    _parse_sum(p)
end

function _parse_sum(p::_TexParser)
    left = _parse_product(p)

    while true
        t = _peek(p)
        if t.type == :plus
            _advance!(p)
            right = _parse_product(p)
            left = left + right
        elseif t.type == :minus
            _advance!(p)
            right = _parse_product(p)
            left = left - right
        else
            break
        end
    end

    left
end

function _parse_product(p::_TexParser)
    left = _parse_unary(p)

    # Juxtaposition: keep multiplying while next token can start a factor
    while _can_start_factor(_peek(p))
        right = _parse_unary(p)
        left = left * right
    end

    left
end

function _can_start_factor(t::TexToken)
    t.type in (:name, :number, :frac, :partial, :nabla, :lparen)
end

function _parse_unary(p::_TexParser)
    t = _peek(p)
    if t.type == :minus
        _advance!(p)
        expr = _parse_unary(p)
        return -expr
    end
    _parse_atom(p)
end

function _parse_atom(p::_TexParser)
    t = _peek(p)

    if t.type == :frac
        return _parse_fraction(p)

    elseif t.type == :number
        return _parse_number(p)

    elseif t.type == :partial || t.type == :nabla
        return _parse_derivative(p)

    elseif t.type == :name
        return _parse_tensor(p)

    elseif t.type == :lparen
        _advance!(p)
        expr = _parse_expr(p)
        _expect!(p, :rparen)
        return expr

    else
        _parse_error(p, "unexpected token '$(t.value)'")
    end
end

# ── Fraction: \frac{N}{M} ──────────────────────────────────────────

function _parse_fraction(p::_TexParser)
    _expect!(p, :frac)
    _expect!(p, :lbrace)
    num_tok = _expect!(p, :number)
    _expect!(p, :rbrace)
    _expect!(p, :lbrace)
    den_tok = _expect!(p, :number)
    _expect!(p, :rbrace)
    n = parse(Int, num_tok.value)
    d = parse(Int, den_tok.value)
    TScalar(n // d)
end

# ── Number: integer, possibly followed by /integer ──────────────────

function _parse_number(p::_TexParser)
    num_tok = _advance!(p)
    n = parse(Int, num_tok.value)

    # Check for N/M rational syntax
    if _peek(p).type == :slash
        _advance!(p)
        den_tok = _expect!(p, :number)
        d = parse(Int, den_tok.value)
        return TScalar(n // d)
    end

    TScalar(n // 1)
end

# ── Derivative: \partial_a expr  or  \nabla_a expr ─────────────────

function _parse_derivative(p::_TexParser)
    _advance!(p)  # consume \partial or \nabla

    # Parse the derivative index
    idx = _parse_single_index(p, Down)  # default Down for \partial_a

    # Parse the argument
    arg = _parse_atom(p)

    TDeriv(idx, arg)
end

function _parse_single_index(p::_TexParser, default_pos::IndexPosition)
    t = _peek(p)

    # Explicit position: _a → Down, ^a → Up
    if t.type == :sub
        _advance!(p)
        return _read_one_index(p, Down)
    elseif t.type == :sup
        _advance!(p)
        return _read_one_index(p, Up)
    else
        # No explicit _ or ^: read next name as index with default position
        return _read_one_index(p, default_pos)
    end
end

function _read_one_index(p::_TexParser, pos::IndexPosition)
    t = _peek(p)
    if t.type == :lbrace
        # {a} → single index from brace group
        _advance!(p)
        name_tok = _expect!(p, :name)
        _expect!(p, :rbrace)
        sym = Symbol(name_tok.value)
        return TIndex(sym, pos)
    elseif t.type == :name
        _advance!(p)
        # For single-char names, use directly
        # For multi-char, take first char only for derivative index
        val = t.value
        if length(val) == 1
            return TIndex(Symbol(val), pos)
        else
            # Multi-char: first char is the index, rest is a separate token
            # Push remaining chars back as a new name token
            rest = val[nextind(val, 1):end]
            if !isempty(rest)
                insert!(p.tokens, p.pos, TexToken(:name, rest, t.pos + 1))
            end
            return TIndex(Symbol(val[1]), pos)
        end
    else
        _parse_error(p, "expected index name after _ or ^")
    end
end

# ── Tensor: Name with optional index blocks ─────────────────────────

function _parse_tensor(p::_TexParser)
    name_tok = _advance!(p)
    name = Symbol(name_tok.value)

    # Parse index blocks: sequences of ^{...} and _{...}
    indices = TIndex[]

    while true
        t = _peek(p)
        if t.type == :sup
            _advance!(p)
            append!(indices, _parse_index_group(p, Up))
        elseif t.type == :sub
            _advance!(p)
            append!(indices, _parse_index_group(p, Down))
        elseif t.type == :lbrace
            # Check for empty braces {} (index separator)
            next_i = p.pos + 1
            if next_i <= length(p.tokens) && p.tokens[next_i].type == :rbrace
                _advance!(p)  # {
                _advance!(p)  # }
                continue
            else
                break
            end
        else
            break
        end
    end

    Tensor(name, indices)
end

function _parse_index_group(p::_TexParser, pos::IndexPosition)
    t = _peek(p)

    if t.type == :lbrace
        _advance!(p)  # consume {
        indices = TIndex[]

        while _peek(p).type != :rbrace
            tok = _peek(p)
            if tok.type == :name
                _advance!(p)
                # Split multi-ASCII names into individual character indices
                for c in tok.value
                    push!(indices, TIndex(Symbol(c), pos))
                end
            elseif tok.type == :eof
                _parse_error(p, "unclosed brace in index group")
            else
                _parse_error(p, "unexpected '$(tok.value)' in index group")
            end
        end

        _advance!(p)  # consume }
        return indices

    elseif t.type == :name
        _advance!(p)
        val = t.value
        if length(val) == 1 || !isascii(val[1]) || length(collect(val)) == 1
            # Single character or single Unicode: one index
            return TIndex[TIndex(Symbol(val), pos)]
        else
            # Multi-char without braces: first char is the index,
            # rest goes back as a name token (it's the next tensor name)
            rest = val[nextind(val, 1):end]
            if !isempty(rest)
                insert!(p.tokens, p.pos, TexToken(:name, rest, t.pos + 1))
            end
            return TIndex[TIndex(Symbol(val[1]), pos)]
        end
    else
        _parse_error(p, "expected index name or {indices} after ^ or _")
    end
end

# ── Public API ──────────────────────────────────────────────────────

"""
    parse_tex(s::AbstractString) -> TensorExpr

Parse a LaTeX tensor expression string into a TensorExpr.

# Examples
```julia
parse_tex("R_{abcd}")                        # Tensor(:R, [down(:a),...])
parse_tex("g^{ab} R_{abcd}")                 # product by juxtaposition
parse_tex("R_{ab} - \\frac{1}{2} g_{ab} R")  # Einstein tensor
parse_tex("\\partial_a V^b")                 # derivative
parse_tex("R^a{}_{bcd}")                     # mixed indices
```

Or use the `tex"..."` string macro (no escaping needed):
```julia
tex"R_{ab} - \\frac{1}{2} g_{ab} R"
tex"\\partial_a V^b"
```
"""
function parse_tex(s::AbstractString)
    tokens = _tex_tokenize(s)
    p = _TexParser(tokens, 1)
    result = _parse_expr(p)

    if _peek(p).type != :eof
        _parse_error(p, "unexpected trailing token '$(p.tokens[p.pos].value)'")
    end

    result
end

"""
    @tex_str -> TensorExpr

String macro for writing tensor expressions in LaTeX notation.

Subscripts `_` denote covariant (Down) indices, superscripts `^` denote
contravariant (Up) indices. Juxtaposition means multiplication.
LaTeX commands like `\\partial`, `\\nabla`, `\\frac`, and Greek letters
(`\\alpha`, `\\mu`, etc.) are supported. Unicode Greek letters work directly.

# Examples
```julia
tex"R_{abcd}"                          # Riemann tensor
tex"g^{ab} g_{bc}"                     # metric product
tex"R_{ab} - \\frac{1}{2} g_{ab} R"    # Einstein tensor
tex"\\partial_a V^b"                   # partial derivative
tex"\\nabla_a T^{bc}"                  # covariant derivative
tex"R^a{}_{bcd}"                       # mixed up/down indices
tex"\\Gamma^a_{bc}"                    # Christoffel symbol
tex"F^A_{\\mu\\nu}"                    # with Greek index names
tex"2 R_{ab}"                          # integer coefficient
```
"""
macro tex_str(s)
    :(parse_tex($s))
end
