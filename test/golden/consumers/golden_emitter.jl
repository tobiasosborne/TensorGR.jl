# golden_emitter.jl — TGR-bhs5.6
#
# TensorExpr -> Dict (schema v1). Inverse of golden_loader.jl's load_expr.
#
# `normalize_dummies_golden(e)` renames dummy-pair indices to d1..dN in
# AST first-occurrence order. Each TSum term is normalized independently
# (dummies are local to a term). Free indices keep their original names.
#
# `emit_json(e; normalize=true)` serializes, optionally running
# `normalize_dummies_golden` first.

using TensorGR

# ----------------- dummy normalization -----------------

"""
    normalize_dummies_golden(e::TensorExpr) -> TensorExpr

Rename dummy pairs to `:d1, :d2, ...` in AST first-occurrence order.
For a TSum, each term is renormalized independently — dummies in different
summands are local and may collide.
"""
normalize_dummies_golden(e::TensorExpr) = _norm_term(e)

_norm_term(e::TSum) = TSum(TensorExpr[_norm_term(t) for t in e.terms])

function _norm_term(e::TensorExpr)
    pairs = dummy_pairs(e)
    isempty(pairs) && return e

    dummy_name_set = Set{Symbol}()
    for (i1, _) in pairs
        push!(dummy_name_set, i1.name)
    end

    # AST first-occurrence order for dummy names
    ordered_names = Symbol[]
    for idx in indices(e)
        if idx.name in dummy_name_set && !(idx.name in ordered_names)
            push!(ordered_names, idx.name)
        end
    end

    # Two-phase rename to dodge collisions with existing d_N names.
    phase1 = Dict{Symbol,Symbol}()
    for (i, nm) in enumerate(ordered_names)
        phase1[nm] = Symbol("__golddum", i)
    end
    step1 = rename_dummies(e, phase1)

    phase2 = Dict{Symbol,Symbol}()
    for (i, _) in enumerate(ordered_names)
        phase2[Symbol("__golddum", i)] = Symbol("d", i)
    end
    rename_dummies(step1, phase2)
end

# ----------------- serialization -----------------

"""
    emit_json(e::TensorExpr; normalize=true) -> Dict

Serialize a TensorExpr to a schema-v1 Dict. If `normalize=true`, runs
`normalize_dummies_golden` first.
"""
function emit_json(e::TensorExpr; normalize::Bool=true)
    e2 = normalize ? normalize_dummies_golden(e) : e
    _emit(e2)
end

_emit(t::Tensor) = Dict(
    "type"    => "tensor",
    "name"    => String(t.name),
    "indices" => Any[_emit_index(i) for i in t.indices],
)

_emit(p::TProduct) = Dict(
    "type" => "product",
    "coef" => Dict("num" => numerator(p.scalar), "den" => denominator(p.scalar)),
    "factors" => Any[_emit(f) for f in p.factors],
)

_emit(s::TSum) = Dict(
    "type"  => "sum",
    "terms" => Any[_emit(t) for t in s.terms],
)

_emit(d::TDeriv) = Dict(
    "type"  => "deriv",
    "covd"  => String(d.covd),
    "index" => _emit_index(d.index),
    "arg"   => _emit(d.arg),
)

function _emit(s::TScalar)
    val = s.val
    vjson = if val isa Rational
        Dict("rational" => Dict("num" => numerator(val), "den" => denominator(val)))
    elseif val isa Integer
        Dict("rational" => Dict("num" => Int(val), "den" => 1))
    elseif val isa Symbol
        Dict("symbol" => String(val))
    else
        error("unsupported scalar value for emit: $(typeof(val))")
    end
    Dict("type" => "scalar", "value" => vjson)
end

_emit_index(i::TIndex) = Dict(
    "name"    => String(i.name),
    "pos"     => i.position == Up ? "up" : "down",
    "vbundle" => String(i.vbundle),
)
