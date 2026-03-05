#= Algebraic operations on TensorExpr with normalization on construction.

Design: smart constructors enforce invariants:
  - TProduct: flattened (no nested TProducts), scalar absorbed
  - TSum: flattened (no nested TSums), zeros removed
  - Zero propagation: 0 * anything = TScalar(0//1)
  - Identity: 1 * x = x
=#

const ZERO = TScalar(0 // 1)

# ─── Smart constructors ──────────────────────────────────────────────

"""
    tproduct(scalar, factors) -> TensorExpr

Normalized product constructor. Flattens nested TProducts, absorbs scalars,
and eliminates identities/zeros.
"""
function tproduct(s::Rational{Int}, factors::Vector{TensorExpr})
    s == 0 && return ZERO

    # Flatten: pull out nested TProducts and TScalars
    flat = TensorExpr[]
    coeff = s
    for f in factors
        if f isa TProduct
            coeff *= f.scalar
            append!(flat, f.factors)
        elseif f isa TScalar
            v = f.val
            if v isa Rational{Int}
                coeff *= v
            elseif v isa Integer
                coeff *= v // 1
            else
                push!(flat, f)
            end
        elseif f == ZERO
            return ZERO
        else
            push!(flat, f)
        end
    end

    coeff == 0 && return ZERO
    isempty(flat) && return TScalar(coeff)
    coeff == 1 && length(flat) == 1 && return flat[1]

    TProduct(coeff, flat)
end

"""
    tsum(terms) -> TensorExpr

Normalized sum constructor. Flattens nested TSums and removes zeros.
"""
function tsum(terms::Vector{TensorExpr})
    flat = TensorExpr[]
    for t in terms
        if t isa TSum
            append!(flat, t.terms)
        elseif t == ZERO
            # skip
        else
            push!(flat, t)
        end
    end

    isempty(flat) && return ZERO
    length(flat) == 1 && return flat[1]
    TSum(flat)
end

# ─── Base operator overloads ─────────────────────────────────────────

# Multiplication: TensorExpr × TensorExpr
function Base.:*(a::TensorExpr, b::TensorExpr)
    b′ = ensure_no_dummy_clash(a, b)
    tproduct(1 // 1, TensorExpr[a, b′])
end

# Scalar × TensorExpr
Base.:*(s::Number, t::TensorExpr) = tproduct(Rational{Int}(s), TensorExpr[t])
Base.:*(t::TensorExpr, s::Number) = tproduct(Rational{Int}(s), TensorExpr[t])

# Negation
Base.:-(t::TensorExpr) = tproduct(-1 // 1, TensorExpr[t])

# Addition
Base.:+(a::TensorExpr, b::TensorExpr) = tsum(TensorExpr[a, b])

# Subtraction
Base.:-(a::TensorExpr, b::TensorExpr) = tsum(TensorExpr[a, -b])

# Power (for scalar expressions like k²)
Base.:^(t::TScalar, n::Integer) = TScalar(:($(t.val)^$n))
