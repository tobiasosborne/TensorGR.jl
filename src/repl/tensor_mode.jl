#= Custom REPL mode for tensor algebra.
#
# Adds a `tensor>` prompt activated by pressing `\` in the Julia REPL.
# Input is parsed as LaTeX tensor notation via parse_tex, evaluated,
# and displayed in Unicode.
#
# Features:
#   - LaTeX input:  R_{abcd} R^{abcd}  →  parsed to TensorExpr
#   - Commands:     simplify %, contract %, expand %, etc.
#   - History:      % refers to last result (like Mathematica's %)
#   - Auto-display: results shown in Unicode notation
#
# The mode is thin: it delegates ALL logic to the existing AST and
# parse_tex/to_unicode infrastructure. New features that produce
# TensorExpr are automatically supported — no contract to fulfill.
#
# Activation: call `TensorGR.init_repl_mode!()` after `using TensorGR`.
# Or set ENV["TENSORGR_REPL"] = "1" before loading to auto-activate.
=#

import REPL

# ── State ────────────────────────────────────────────────────────────

"""Global state for the tensor REPL mode."""
module TensorREPL

using ..TensorGR

# Last result (% in tensor mode)
const _last_result = Ref{Any}(nothing)

# Numbered output history: %1, %2, etc.
const _history = Any[]

# Active registry for tensor mode (set via init_repl_mode! or set_tensor_registry!)
const _registry = Ref{Union{TensorRegistry, Nothing}}(nothing)

# Command registry: name => (func, help_string)
const _commands = Dict{String, Tuple{Function, String}}()

# Named variables: "expr" => TensorExpr
const _variables = Dict{String, Any}()

"""Register a command for tensor mode."""
function register_command!(name::String, f::Function, help::String="")
    _commands[name] = (f, help)
end

end  # module TensorREPL

# ── Tab completion ──────────────────────────────────────────────────

"""Completion provider for tensor REPL mode."""
struct TensorCompletionProvider <: REPL.LineEdit.CompletionProvider end

function REPL.LineEdit.complete_line(c::TensorCompletionProvider, s; hint::Bool=false)
    partial = REPL.LineEdit.input_string(s)
    pos = position(REPL.LineEdit.buffer(s))
    # Only complete up to cursor position
    before_cursor = partial[1:min(pos, lastindex(partial))]
    completions, last_word = _tensor_completions(before_cursor)
    named = REPL.LineEdit.NamedCompletion.(completions)
    return named, last_word, !isempty(completions)
end

"""Generate completions for the given partial input."""
function _tensor_completions(partial::AbstractString)
    # Find the last word being typed
    m = match(r"(\\?[\w]*)$", partial)
    last_word = m !== nothing ? String(m[1]) : ""
    isempty(last_word) && return (String[], "")

    candidates = String[]

    # Commands
    for cmd in keys(TensorREPL._commands)
        startswith(cmd, last_word) && push!(candidates, cmd)
    end
    # Built-in commands not in _commands
    for special in ("help", "vars", "info", "registry", "define", "sub", "substitute")
        startswith(special, last_word) && push!(candidates, special)
    end

    # Variables
    for v in keys(TensorREPL._variables)
        startswith(v, last_word) && push!(candidates, v)
    end

    # Registry tensor names
    reg = TensorREPL._registry[]
    if reg !== nothing
        for tname in keys(reg.tensors)
            s = string(tname)
            startswith(s, last_word) && push!(candidates, s)
        end
    end

    # LaTeX names when typing \...
    if startswith(last_word, "\\")
        prefix = last_word[2:end]
        for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
                      "theta", "iota", "kappa", "lambda", "mu", "nu", "xi",
                      "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi",
                      "psi", "omega", "partial", "nabla", "frac",
                      "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi",
                      "Sigma", "Phi", "Psi", "Omega")
            startswith(name, prefix) && push!(candidates, "\\" * name)
        end
    end

    sort!(unique!(candidates))
    return (candidates, last_word)
end

"""
    set_tensor_registry!(reg::TensorRegistry)

Set the registry used by the tensor REPL mode. Call this after
`@manifold` to make simplify/contract work in tensor mode.
"""
function set_tensor_registry!(reg::TensorRegistry)
    TensorREPL._registry[] = reg
    nothing
end

"""Get the active tensor mode registry, falling back to current_registry()."""
function _tensor_registry()
    r = TensorREPL._registry[]
    r !== nothing ? r : current_registry()
end

# ── Result tracking ──────────────────────────────────────────────────

"""Record a result in both the numbered history and the last-result ref."""
function _record_result!(result)
    push!(TensorREPL._history, result)
    TensorREPL._last_result[] = result
    result
end

"""
Resolve a percent reference: `%` (last), `%N` (by index), `%end`, `%end-1`.
Returns `nothing` if the string is not a percent reference.
"""
function _resolve_percent_ref(s::AbstractString)
    s = strip(s)
    # % or %end → last result
    if s == "%" || s == "%end"
        isempty(TensorREPL._history) && error("No previous result (%) available")
        return TensorREPL._history[end]
    end
    # %end-N
    m = match(r"^%end-(\d+)$", s)
    if m !== nothing
        offset = parse(Int, m[1])
        idx = length(TensorREPL._history) - offset
        (idx < 1 || idx > length(TensorREPL._history)) &&
            error("History index %end-$(m[1]) out of range (1:$(length(TensorREPL._history)))")
        return TensorREPL._history[idx]
    end
    # %N
    m = match(r"^%(\d+)$", s)
    if m !== nothing
        n = parse(Int, m[1])
        (n < 1 || n > length(TensorREPL._history)) &&
            error("History index %$n out of range (1:$(length(TensorREPL._history)))")
        return TensorREPL._history[n]
    end
    return nothing  # not a percent reference
end

# ── Command registry ─────────────────────────────────────────────────

function _init_commands!()
    TensorREPL.register_command!("simplify",
        expr -> with_registry(_tensor_registry()) do; simplify(expr); end,
        "Simplify expression via the full pipeline")
    TensorREPL.register_command!("canon",
        expr -> with_registry(_tensor_registry()) do; canonicalize(expr); end,
        "Canonicalize (xperm only, no collection)")
    TensorREPL.register_command!("expand",
        expr -> with_registry(_tensor_registry()) do; expand_products(expr); end,
        "Expand products")
    TensorREPL.register_command!("contract",
        expr -> with_registry(_tensor_registry()) do; contract_metrics(expr); end,
        "Contract metrics")
    TensorREPL.register_command!("latex", expr -> (println(to_latex(expr)); expr),
        "Print LaTeX form")
    TensorREPL.register_command!("indices", expr -> (println(free_indices(expr)); expr),
        "Show free indices")
    TensorREPL.register_command!("terms",
        expr -> (println(expr isa TSum ? length(expr.terms) : 1, " term(s)"); expr),
        "Count terms")
    TensorREPL.register_command!("level2",
        expr -> with_registry(_tensor_registry()) do; simplify_level2(expr); end,
        "Simplify with Bianchi + cyclic identities (Level 2)")
    TensorREPL.register_command!("simplify_level2",
        expr -> with_registry(_tensor_registry()) do; simplify_level2(expr); end,
        "Simplify with Bianchi + cyclic identities (Level 2)")
    TensorREPL.register_command!("to_riemann",
        expr -> with_registry(_tensor_registry()) do; to_riemann(expr); end,
        "Convert to Riemann basis")
    TensorREPL.register_command!("to_ricci",
        expr -> with_registry(_tensor_registry()) do; to_ricci(expr); end,
        "Convert to Ricci basis")
    TensorREPL.register_command!("covd",
        expr -> begin
            reg = _tensor_registry()
            covd_sym = _find_active_covd(reg)
            with_registry(reg) do; covd_to_christoffel(expr, covd_sym); end
        end,
        "Expand covariant derivatives to Christoffel symbols")
    TensorREPL.register_command!("perturb",
        expr -> begin
            reg = _tensor_registry()
            metric_name = _find_metric_name(reg)
            bg = Symbol(metric_name, :_bg)
            with_registry(reg) do
                linearize(expr, metric_name => (bg, :h))
            end
        end,
        "Linearize expression (first-order metric perturbation)")
    TensorREPL.register_command!("perturbation",
        expr -> begin
            reg = _tensor_registry()
            metric_name = _find_metric_name(reg)
            bg = Symbol(metric_name, :_bg)
            with_registry(reg) do
                linearize(expr, metric_name => (bg, :h))
            end
        end,
        "Linearize expression (first-order metric perturbation)")
end

"""Find the active covariant derivative name from the registry."""
function _find_active_covd(reg::TensorRegistry)
    # Check manifold default derivative first
    for (_, mp) in reg.manifolds
        mp.derivative !== nothing && return mp.derivative
    end
    # Scan for any registered CovD
    for (name, tp) in reg.tensors
        tp.is_covd && return name
    end
    error("No covariant derivative registered. Use @covd to define one.")
end

"""Find the active CovD name, falling back to :partial if none registered."""
function _find_active_covd_or_partial(reg::TensorRegistry)
    for (_, mp) in reg.manifolds
        mp.derivative !== nothing && return mp.derivative
    end
    for (name, tp) in reg.tensors
        tp.is_covd && return name
    end
    :partial  # fallback — no CovD registered
end

"""Find the metric name from the first manifold in the registry."""
function _find_metric_name(reg::TensorRegistry)
    for (mname, _) in reg.manifolds
        if haskey(reg.metric_cache, mname)
            return reg.metric_cache[mname]
        end
    end
    error("No metric registered.")
end

# ── Input processing ─────────────────────────────────────────────────

"""
    _process_tensor_input(line::AbstractString) -> Any

Process a line of tensor-mode input. Returns the result to display,
or `nothing` for empty input.
"""
function _process_tensor_input(line::AbstractString)
    s = strip(line)
    isempty(s) && return nothing

    # Help
    if s == "help" || s == "?"
        _print_tensor_help()
        return nothing
    end

    # Workspace introspection commands
    if s == "vars"
        _print_vars()
        return nothing
    end
    if s == "registry"
        _print_registry_info()
        return nothing
    end
    if s == "info" || s == "info %"
        isempty(TensorREPL._history) && error("No expression to inspect")
        _print_info(TensorREPL._history[end])
        return nothing
    end
    if startswith(s, "info ")
        arg = strip(s[6:end])
        ref = _resolve_percent_ref(arg)
        if ref !== nothing
            _print_info(ref)
        elseif _is_variable_ref(arg)
            _print_info(TensorREPL._variables[String(arg)])
        else
            _print_info(_parse_and_resolve(arg))
        end
        return nothing
    end

    # Substitute command: sub pattern -> replacement (applied to last result)
    m_sub = match(r"^(?:sub|substitute)\s+(.+?)\s*->\s*(.+)$", s)
    if m_sub !== nothing
        isempty(TensorREPL._history) && error("No expression to substitute into (use %)")
        pattern = _parse_and_resolve(String(m_sub[1]))
        replacement = _parse_and_resolve(String(m_sub[2]))
        rule = RewriteRule(pattern, replacement)
        result = with_registry(_tensor_registry()) do
            apply_rules(TensorREPL._history[end], RewriteRule[rule])
        end
        return _record_result!(result)
    end

    # Define command: define T_{ab} [on=M4]
    m_def = match(r"^define\s+(\w+)(?:_\{([a-zA-Z]+)\})?(?:\s+on=(\w+))?$", s)
    if m_def !== nothing
        name = Symbol(m_def[1])
        idx_str = m_def[2]
        reg = _tensor_registry()
        manifold_name = m_def[3] !== nothing ? Symbol(m_def[3]) : first(keys(reg.manifolds))
        n_idx = idx_str !== nothing ? length(idx_str) : 0
        rank = (0, n_idx)
        if has_tensor(reg, name)
            println("  Tensor $name already registered")
        else
            register_tensor!(reg, TensorProperties(;
                name=name, manifold=manifold_name, rank=rank))
            println("  Registered tensor $name with rank $rank on $manifold_name")
        end
        return nothing
    end

    # Check for variable assignment: "name = expr" or "name = command expr"
    m = match(r"^([a-zA-Z_]\w*)\s*=\s*(.+)$", s)
    if m !== nothing
        varname = String(m[1])
        rhs = strip(String(m[2]))
        # Don't shadow commands
        if !haskey(TensorREPL._commands, varname)
            result = _process_tensor_rhs(rhs)
            TensorREPL._variables[varname] = result
            _record_result!(result)
            return result
        end
    end

    # Check for command prefix: "simplify expr", "simplify %", or "simplify(expr)"
    for (cmd, (func, _)) in TensorREPL._commands
        # Match "cmd arg" or bare "cmd"
        if startswith(s, cmd * " ") || s == cmd
            arg_str = strip(s[length(cmd)+1:end])
            result = _apply_command(func, arg_str)
            return result
        end
        # Match "cmd(arg)" function-call syntax
        if startswith(s, cmd * "(") && endswith(s, ")")
            arg_str = strip(s[length(cmd)+2:end-1])
            result = _apply_command(func, arg_str)
            return result
        end
    end

    # Pipe/chain syntax: expr | cmd1 | cmd2
    if occursin("|", s)
        return _process_pipe_chain(s)
    end

    # Check for bare variable reference
    if _is_variable_ref(s)
        result = TensorREPL._variables[String(s)]
        _record_result!(result)
        return result
    end

    # Check for %N history reference
    ref = _resolve_percent_ref(s)
    if ref !== nothing
        _record_result!(ref)
        return ref
    end

    # Plain LaTeX expression
    expr = _parse_and_resolve(s)
    _record_result!(expr)
    return expr
end

"""Process a pipe chain: expr | cmd1 | cmd2."""
function _process_pipe_chain(s::AbstractString)
    segments = strip.(split(s, "|"))
    filter!(!isempty, segments)
    isempty(segments) && return nothing

    first_seg = String(segments[1])
    # First segment: resolve as percent ref, variable, or LaTeX
    ref = _resolve_percent_ref(first_seg)
    if ref !== nothing
        result = ref
    elseif _is_variable_ref(first_seg)
        result = TensorREPL._variables[first_seg]
    else
        result = _parse_and_resolve(first_seg)
    end

    # Apply each subsequent command
    for i in 2:length(segments)
        cmd_name = String(strip(segments[i]))
        isempty(cmd_name) && continue
        if !haskey(TensorREPL._commands, cmd_name)
            error("Unknown command in pipe: '$cmd_name'")
        end
        func, _ = TensorREPL._commands[cmd_name]
        result = func(result)
    end

    _record_result!(result)
    return result
end

"""Process the RHS of a variable assignment (may be a command, pipe, or plain LaTeX)."""
function _process_tensor_rhs(s::AbstractString)
    s = String(strip(s))

    # RHS could be a pipe chain
    if occursin("|", s)
        return _process_pipe_chain(s)
    end

    # RHS could be a percent reference (%N, %end, etc.)
    ref = _resolve_percent_ref(s)
    ref !== nothing && return ref

    # RHS could be a command: "simplify %" or "simplify expr" or "simplify(expr)"
    for (cmd, (func, _)) in TensorREPL._commands
        if startswith(s, cmd * " ") || s == cmd
            arg_str = strip(s[length(cmd)+1:end])
            return _apply_command(func, arg_str)
        end
        if startswith(s, cmd * "(") && endswith(s, ")")
            arg_str = strip(s[length(cmd)+2:end-1])
            return _apply_command(func, arg_str)
        end
    end

    # RHS is a variable reference
    if _is_variable_ref(s)
        return TensorREPL._variables[s]
    end

    # RHS is plain LaTeX
    _parse_and_resolve(s)
end

"""Apply a command function to an argument string (shared by space and paren syntax)."""
function _apply_command(func::Function, arg_str::AbstractString)
    if isempty(arg_str)
        # Apply to last result
        isempty(TensorREPL._history) && error("No previous result (%) available")
        expr = TensorREPL._history[end]
    else
        # Try percent ref (%N, %end, etc.)
        ref = _resolve_percent_ref(arg_str)
        if ref !== nothing
            expr = ref
        elseif _is_variable_ref(arg_str)
            expr = TensorREPL._variables[String(arg_str)]
        else
            expr = _parse_and_resolve(arg_str)
        end
    end
    result = func(expr)
    _record_result!(result)
    return result
end

"""Check if a string is a reference to a stored variable."""
function _is_variable_ref(s::AbstractString)
    haskey(TensorREPL._variables, String(s))
end

"""Parse LaTeX and resolve tensor names against the active registry."""
function _parse_and_resolve(s::AbstractString)
    expr = parse_tex(s)
    with_registry(_tensor_registry()) do
        _resolve_names(expr)
    end
end

"""
    _resolve_names(expr) -> TensorExpr

Walk the AST and resolve generic names to registered tensor names.
Uses the current registry to determine:
- R with 4 indices → Riem (if registered)
- R with 2 indices → Ric (if registered)
- R with 0 indices → RicScalar (if registered)
- G with 2 indices → Ein (if registered)
- C with 4 indices → Weyl (if registered)
"""
function _resolve_names(t::Tensor)
    reg = current_registry()
    n = length(t.indices)
    resolved = _try_resolve_name(reg, t.name, n)
    resolved === t.name ? t : Tensor(resolved, t.indices)
end

function _resolve_names(p::TProduct)
    TProduct(p.scalar, TensorExpr[_resolve_names(f) for f in p.factors])
end

function _resolve_names(s::TSum)
    TSum(TensorExpr[_resolve_names(t) for t in s.terms])
end

function _resolve_names(d::TDeriv)
    new_arg = _resolve_names(d.arg)
    if d.covd == :nabla
        # Resolve \nabla to the active covariant derivative
        reg = current_registry()
        covd_sym = _find_active_covd_or_partial(reg)
        return TDeriv(d.index, new_arg, covd_sym)
    end
    TDeriv(d.index, new_arg, d.covd)
end

function _resolve_names(s::TScalar)
    s
end

function _resolve_names(d::TParamDeriv)
    TParamDeriv(d.params, _resolve_names(d.arg))
end

function _resolve_names(expr::TensorExpr)
    expr  # fallback
end

# Standard name resolution table: (latex_name, n_indices) => registry_name
const _STANDARD_NAMES = Dict{Tuple{Symbol,Int}, Symbol}(
    (:R, 4) => :Riem,
    (:R, 2) => :Ric,
    (:R, 0) => :RicScalar,
    (:G, 2) => :Ein,
    (:C, 4) => :Weyl,
    (:Gamma, 3) => :Christoffel,
)

function _try_resolve_name(reg::TensorRegistry, name::Symbol, n_indices::Int)
    # If already registered, use as-is
    has_tensor(reg, name) && return name

    # Check standard aliases
    key = (name, n_indices)
    if haskey(_STANDARD_NAMES, key)
        candidate = _STANDARD_NAMES[key]
        has_tensor(reg, candidate) && return candidate
    end

    # Check tex_aliases in registry
    tex_key = (name, n_indices)
    if haskey(reg.tex_aliases, tex_key)
        return reg.tex_aliases[tex_key]
    end

    name  # no resolution found
end

# ── Workspace introspection ──────────────────────────────────────────

"""Print all stored variables."""
function _print_vars()
    if isempty(TensorREPL._variables)
        println("  No variables defined.")
        return
    end
    printstyled("  Variables:\n"; bold=true, color=:cyan)
    for (name, val) in sort(collect(TensorREPL._variables))
        printstyled("    $name"; color=:yellow)
        print(" = ")
        if val isa TensorExpr
            printstyled(to_unicode(val); color=:white)
        else
            print(val)
        end
        println()
    end
end

"""Print info about an expression: indices, terms, tensors used."""
function _print_info(expr)
    printstyled("  Expression info:\n"; bold=true, color=:cyan)

    if expr isa TensorExpr
        # Free indices
        fi = free_indices(expr)
        idx_str = isempty(fi) ? "(scalar)" : join(map(string, fi), ", ")
        println("    Free indices: $idx_str")

        # Term count
        n_terms = expr isa TSum ? length(expr.terms) : 1
        println("    Terms: $n_terms")

        # Tensor names used
        names = Set{Symbol}()
        _collect_names_repl!(names, expr)
        println("    Tensors: ", isempty(names) ? "(none)" : join(sort(collect(names)), ", "))

        # Type
        println("    Type: ", nameof(typeof(expr)))

        # Symmetries (if single tensor)
        if expr isa Tensor
            reg = _tensor_registry()
            if has_tensor(reg, expr.name)
                tp = get_tensor(reg, expr.name)
                if !isempty(tp.symmetries)
                    println("    Symmetries: ", join(string.(tp.symmetries), ", "))
                end
            end
        end
    else
        println("    Value: $expr")
        println("    Type: ", typeof(expr))
    end
end

"""Collect tensor names from an AST."""
function _collect_names_repl!(names::Set{Symbol}, t::Tensor)
    push!(names, t.name)
end
function _collect_names_repl!(names::Set{Symbol}, p::TProduct)
    for f in p.factors; _collect_names_repl!(names, f); end
end
function _collect_names_repl!(names::Set{Symbol}, s::TSum)
    for t in s.terms; _collect_names_repl!(names, t); end
end
function _collect_names_repl!(names::Set{Symbol}, d::TDeriv)
    _collect_names_repl!(names, d.arg)
end
function _collect_names_repl!(names::Set{Symbol}, ::TScalar) end
function _collect_names_repl!(names::Set{Symbol}, ::TensorExpr) end

"""Print registry summary."""
function _print_registry_info()
    reg = _tensor_registry()
    printstyled("  Registry:\n"; bold=true, color=:cyan)

    if !isempty(reg.manifolds)
        printstyled("    Manifolds:\n"; color=:yellow)
        for (name, mp) in reg.manifolds
            println("      $name  dim=$(mp.dim)")
        end
    end

    n_tensors = length(reg.tensors)
    printstyled("    Tensors: $n_tensors\n"; color=:yellow)

    # Show a sample of tensor names
    if n_tensors > 0
        tnames = sort(collect(keys(reg.tensors)))
        shown = tnames[1:min(10, length(tnames))]
        print("      ")
        print(join(shown, ", "))
        n_tensors > 10 && print(", ... ($(n_tensors - 10) more)")
        println()
    end

    println("    Rules: $(length(reg.rules))")
end

function _print_tensor_help()
    printstyled("  Tensor Mode\n"; bold=true, color=:cyan)
    println("  Type LaTeX tensor expressions directly:")
    printstyled("    R_{abcd} R^{abcd}\n"; color=:green)
    printstyled("    \\partial_a \\phi\n"; color=:green)
    printstyled("    g^{ab} R_{ab} - \\frac{1}{2} R\n"; color=:green)
    println()
    println("  Commands (apply to last result with %):")
    for cmd in sort(collect(keys(TensorREPL._commands)))
        (_, help) = TensorREPL._commands[cmd]
        printstyled("    $cmd"; color=:yellow)
        isempty(help) || print("  — $help")
        println()
    end
    println()
    println("  Variables and pipes:")
    printstyled("    expr = R_{abcd}\n"; color=:green)
    printstyled("    result = simplify expr\n"; color=:green)
    printstyled("    g^{ab} R_{ab} | contract | simplify\n"; color=:green)
    printstyled("    x = R_{abcd} | simplify\n"; color=:green)
    println()
    println("  Workspace:")
    printstyled("    vars"; color=:yellow)
    println("      — list stored variables")
    printstyled("    info"; color=:yellow)
    println("      — inspect expression (indices, terms, tensors)")
    printstyled("    registry"; color=:yellow)
    println("  — show registered manifolds and tensors")
    println()
    println("  History:")
    printstyled("    %"; color=:yellow)
    println("   — last result")
    printstyled("    %N"; color=:yellow)
    println("  — result N (e.g. %1, %3)")
    println("  Press backspace on empty line to return to julia>")
end

# ── Display ──────────────────────────────────────────────────────────

function _display_tensor_result(io::IO, result)
    result === nothing && return
    n = length(TensorREPL._history)
    printstyled(io, "  [$n] "; color=:light_black)
    if result isa TensorExpr
        printstyled(io, to_unicode(result); color=:white, bold=true)
        println(io)
    else
        println(io, result)
    end
end

# ── REPL mode setup ─────────────────────────────────────────────────

"""
    init_repl_mode!()

Add the `tensor>` REPL mode, activated by pressing `\\` (backslash).

Call this after `using TensorGR` in an interactive session, or set
`ENV["TENSORGR_REPL"] = "1"` before loading to auto-activate.

The tensor mode accepts LaTeX-style tensor expressions and displays
results in Unicode notation. Type `help` in tensor mode for commands.
"""
const _TENSOR_MODE_INIT = Ref{Bool}(false)

function init_repl_mode!(reg::TensorRegistry=current_registry())
    # Only works in interactive REPL
    isdefined(Base, :active_repl) || return nothing
    repl = Base.active_repl

    # Store registry for tensor mode
    TensorREPL._registry[] = reg

    # Guard against double-init (calling twice corrupts keymaps)
    if _TENSOR_MODE_INIT[]
        printstyled("  Tensor mode registry updated\n"; color=:cyan)
        return nothing
    end
    _TENSOR_MODE_INIT[] = true

    _init_commands!()

    LineEdit = REPL.LineEdit

    # Get the main julia> prompt
    main_mode = repl.interface.modes[1]

    # Create the tensor prompt (follows Pkg REPLMode pattern exactly)
    tensor_prompt = LineEdit.Prompt("tensor> ";
        prompt_prefix = repl.options.hascolor ? Base.text_colors[:cyan] : "",
        prompt_suffix = "",
        complete = TensorCompletionProvider(),
        sticky = true
    )

    tensor_prompt.repl = repl
    hp = main_mode.hist
    hp.mode_mapping[:tensor] = tensor_prompt
    tensor_prompt.hist = hp

    tensor_prompt.on_done = (s, buf, ok) -> begin
        line = String(take!(buf))
        if !ok || isempty(strip(line))
            return nothing
        end
        # Record in shared history so up-arrow recall works
        _record_history(tensor_prompt, line)
        Base.@invokelatest _on_tensor_done(line)
    end

    # Build keymap: mode-switch + history + defaults (skip prefix search
    # to avoid history_move transition bug with mixed-mode history)
    search_prompt, skeymap = LineEdit.setup_search_keymap(hp)
    mk = REPL.mode_keymap(main_mode)

    # Backspace on empty line returns to julia>
    exit_keymap = Dict{Any, Any}(
        '\b' => function (s, o...)
            if isempty(s) || position(LineEdit.buffer(s)) == 0
                buf = copy(LineEdit.buffer(s))
                LineEdit.transition(s, main_mode) do
                    LineEdit.state(s, main_mode).input_buffer = buf
                end
            else
                LineEdit.edit_backspace(s)
            end
            return
        end
    )

    b = Dict{Any, Any}[
        skeymap, exit_keymap, mk,
        LineEdit.history_keymap, LineEdit.default_keymap,
        LineEdit.escape_defaults,
    ]
    tensor_prompt.keymap_dict = LineEdit.keymap(b)

    # Register the mode
    push!(repl.interface.modes, tensor_prompt)

    # Add \ trigger to main julia> mode
    trigger_keymap = Dict{Any, Any}(
        '\\' => function (s, args...)
            if isempty(s) || position(LineEdit.buffer(s)) == 0
                buf = copy(LineEdit.buffer(s))
                LineEdit.transition(s, tensor_prompt) do
                    LineEdit.state(s, tensor_prompt).input_buffer = buf
                end
            else
                LineEdit.edit_insert(s, '\\')
                LineEdit.check_show_hint(s)
            end
            return
        end
    )
    main_mode.keymap_dict = LineEdit.keymap_merge(main_mode.keymap_dict, trigger_keymap)

    printstyled("  Tensor mode activated — press \\ to enter\n"; color=:cyan)
    nothing
end

"""Record an input line in the shared REPL history provider."""
function _record_history(prompt, line::String)
    hp = prompt.hist
    hp === nothing && return
    try
        # REPLHistoryProvider stores parallel vectors: history + modes
        push!(hp.history, line)
        push!(hp.modes, :tensor)
        hp.cur_idx = length(hp.history) + 1
    catch
        # Gracefully degrade if history internals change
    end
end

"""Callback for tensor mode input (wrapped in @invokelatest for world age safety)."""
function _on_tensor_done(line::String)
    try
        result = _process_tensor_input(line)
        _display_tensor_result(stdout, result)
    catch e
        printstyled(stderr, "  Error: "; color=:red, bold=true)
        showerror(stderr, e)
        println(stderr)
    end
end
