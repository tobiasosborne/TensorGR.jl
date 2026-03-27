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

# Active registry for tensor mode (set via init_repl_mode! or set_tensor_registry!)
const _registry = Ref{Union{TensorRegistry, Nothing}}(nothing)

# Command registry: name => (func, help_string)
const _commands = Dict{String, Tuple{Function, String}}()

"""Register a command for tensor mode."""
function register_command!(name::String, f::Function, help::String="")
    _commands[name] = (f, help)
end

end  # module TensorREPL

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

    # Check for command prefix: "simplify expr" or "simplify %"
    for (cmd, (func, _)) in TensorREPL._commands
        if startswith(s, cmd * " ") || s == cmd
            arg_str = strip(s[length(cmd)+1:end])
            if isempty(arg_str) || arg_str == "%"
                # Apply to last result
                TensorREPL._last_result[] === nothing &&
                    error("No previous result (%) available")
                result = func(TensorREPL._last_result[])
                TensorREPL._last_result[] = result
                return result
            else
                # Parse argument as LaTeX, then apply command
                expr = _parse_and_resolve(arg_str)
                result = func(expr)
                TensorREPL._last_result[] = result
                return result
            end
        end
    end

    # Plain LaTeX expression
    expr = _parse_and_resolve(s)
    TensorREPL._last_result[] = expr
    return expr
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
    TDeriv(d.index, _resolve_names(d.arg), d.covd)
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
    println("  % refers to the last result")
    println("  Press backspace on empty line to return to julia>")
end

# ── Display ──────────────────────────────────────────────────────────

function _display_tensor_result(io::IO, result)
    result === nothing && return
    if result isa TensorExpr
        printstyled(io, "  "; color=:light_black)
        printstyled(io, to_unicode(result); color=:white, bold=true)
        println(io)
    else
        println(io, "  ", result)
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
function init_repl_mode!(reg::TensorRegistry=current_registry())
    # Only works in interactive REPL
    isdefined(Base, :active_repl) || return nothing
    repl = Base.active_repl

    # Store registry for tensor mode
    TensorREPL._registry[] = reg

    _init_commands!()

    LineEdit = REPL.LineEdit

    # Get the main julia> prompt
    main_mode = repl.interface.modes[1]

    # Create the tensor prompt (follows Pkg REPLMode pattern exactly)
    tensor_prompt = LineEdit.Prompt("tensor> ";
        prompt_prefix = repl.options.hascolor ? Base.text_colors[:cyan] : "",
        prompt_suffix = "",
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
        Base.@invokelatest _on_tensor_done(line)
    end

    # Build keymap: search + prefix + mode-switch + history + defaults
    search_prompt, skeymap = LineEdit.setup_search_keymap(hp)
    prefix_prompt, prefix_keymap = LineEdit.setup_prefix_keymap(hp, tensor_prompt)
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
        skeymap, exit_keymap, mk, prefix_keymap,
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
