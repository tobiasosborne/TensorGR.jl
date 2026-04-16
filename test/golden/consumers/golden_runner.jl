# golden_runner.jl — TGR-bhs5.8
#
# Dispatches a golden-master case triple (input, op, expected) through
# TensorGR and compares against the committed expected output.
#
# Contract: `run_case(case::AbstractDict; registry) -> (ok::Bool, diff)`.
#   ok=true, diff=nothing on match.
#   ok=false, diff::String containing a unified-ish JSON diff on mismatch.
#
# Relies on golden_loader.jl (load_expr) and golden_emitter.jl
# (emit_json, normalize_dummies_golden) being included first.

using TensorGR
using JSON

"""
    OP_DISPATCH :: Dict{Symbol, Function}

Maps neutral op symbols to TensorGR entry points. Extend here as new phases
add operations; unknown ops raise.
"""
const OP_DISPATCH = Dict{Symbol, Function}(
    :simplify     => (e, args, reg) -> simplify(e; registry=reg),
    :canonicalize => (e, args, reg) -> simplify(e; registry=reg),
    :to_riemann   => (e, args, reg) -> simplify(to_riemann(e;
                        metric = Symbol(get(args, "metric", "g")),
                        dim    = Int(get(args, "dim", 4)));
                        registry=reg),
    :to_ricci     => (e, args, reg) -> simplify(to_ricci(e;
                        metric = Symbol(get(args, "metric", "g")),
                        dim    = Int(get(args, "dim", 4)));
                        registry=reg),
)

"""
    apply_op(op::Symbol, input::TensorExpr, args::AbstractDict, reg) -> TensorExpr
"""
function apply_op(op::Symbol, input::TensorExpr, args::AbstractDict, reg)
    haskey(OP_DISPATCH, op) || error("golden runner: unknown op :$op")
    OP_DISPATCH[op](input, args, reg)
end

"""
    _canonicalize_for_compare(expr) -> TensorExpr

Run `simplify` (full canonicalization pipeline) then apply the
golden-suite dummy normalization, so two equivalent expressions compare byte-
equal through emit_json.
"""
function _canonicalize_for_compare(expr::TensorExpr, reg)
    s = simplify(expr; registry=reg)
    normalize_dummies_golden(s)
end

"""
    run_case(case; registry) -> (ok::Bool, diff::Union{Nothing,String})
"""
function run_case(case::AbstractDict; registry)
    name = get(case, "name", "<unnamed>")
    op = Symbol(case["op"])
    args = get(case, "op_args", Dict{String,Any}())
    input    = load_expr(case["input"])
    expected = load_expr(case["expected"])

    result = apply_op(op, input, args, registry)

    result_norm   = _canonicalize_for_compare(result,   registry)
    expected_norm = _canonicalize_for_compare(expected, registry)

    if result_norm == expected_norm
        return (true, nothing)
    end

    result_json   = emit_json(result_norm;   normalize=false)
    expected_json = emit_json(expected_norm; normalize=false)

    io = IOBuffer()
    println(io, "golden case '$name' FAILED")
    println(io, "op:       :$op")
    println(io, "input:    ", summary(input))
    println(io)
    println(io, "--- expected (canonicalized) ---")
    println(io, JSON.json(expected_json, 2))
    println(io)
    println(io, "+++ got (canonicalized) +++")
    println(io, JSON.json(result_json, 2))
    return (false, String(take!(io)))
end
