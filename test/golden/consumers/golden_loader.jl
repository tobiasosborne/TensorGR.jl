# golden_loader.jl — TGR-bhs5.5
#
# JSON (schema v1) -> TensorExpr. Counterpart to golden_emitter.jl. Pure
# translation; no canonicalization. Bail loud on anything off-schema.

using TensorGR

"""
    load_index(d::Dict) -> TIndex

Schema: {name: string, pos: "up"|"down", vbundle?: string}.
"""
function load_index(d::AbstractDict)
    name = Symbol(d["name"])
    pos  = d["pos"] == "up"   ? Up :
           d["pos"] == "down" ? Down :
           error("bad index position: $(d["pos"])")
    vb   = Symbol(get(d, "vbundle", "Tangent"))
    TIndex(name, pos, vb)
end

"""
    load_expr(d::Dict) -> TensorExpr

Dispatches on the `type` tag. Throws on malformed input.
"""
function load_expr(d::AbstractDict)
    haskey(d, "type") || error("missing 'type' tag in: $d")
    t = d["type"]
    t == "tensor"  ? _load_tensor(d)  :
    t == "product" ? _load_product(d) :
    t == "sum"     ? _load_sum(d)     :
    t == "deriv"   ? _load_deriv(d)   :
    t == "scalar"  ? _load_scalar(d)  :
    error("unknown expr type tag: $t")
end

function _load_tensor(d::AbstractDict)
    name    = Symbol(d["name"])
    indices = TIndex[load_index(i) for i in d["indices"]]
    Tensor(name, indices)
end

function _load_product(d::AbstractDict)
    coef    = d["coef"]
    scalar  = Int(coef["num"]) // Int(coef["den"])
    factors = TensorExpr[load_expr(f) for f in d["factors"]]
    TProduct(scalar, factors)
end

function _load_sum(d::AbstractDict)
    terms = TensorExpr[load_expr(t) for t in d["terms"]]
    TSum(terms)
end

function _load_deriv(d::AbstractDict)
    covd  = Symbol(d["covd"])
    index = load_index(d["index"])
    arg   = load_expr(d["arg"])
    TDeriv(index, arg, covd)
end

function _load_scalar(d::AbstractDict)
    v = d["value"]
    if haskey(v, "rational")
        r = v["rational"]
        TScalar(Int(r["num"]) // Int(r["den"]))
    elseif haskey(v, "symbol")
        TScalar(Symbol(v["symbol"]))
    else
        error("scalar value must have 'rational' or 'symbol' key: $v")
    end
end
