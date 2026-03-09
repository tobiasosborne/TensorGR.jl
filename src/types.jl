"""
Index position: Up (contravariant) or Down (covariant).
"""
@enum IndexPosition Up Down

"""
    TIndex(name, position, vbundle=:Tangent)

A tensor index with a symbolic name, position (Up or Down), and vector bundle.
"""
struct TIndex
    name::Symbol
    position::IndexPosition
    vbundle::Symbol
    TIndex(name::Symbol, position::IndexPosition, vbundle::Symbol=:Tangent) =
        new(name, position, vbundle)
end

up(s::Symbol) = TIndex(s, Up)
down(s::Symbol) = TIndex(s, Down)
up(s::Symbol, vb::Symbol) = TIndex(s, Up, vb)
down(s::Symbol, vb::Symbol) = TIndex(s, Down, vb)

Base.:(==)(a::TIndex, b::TIndex) = a.name == b.name && a.position == b.position && a.vbundle == b.vbundle
Base.hash(a::TIndex, h::UInt) = hash(a.vbundle, hash(a.position, hash(a.name, h)))

"""
Abstract supertype for all tensor expressions.
"""
abstract type TensorExpr end

"""
    Tensor(name, indices)

A single tensor with symbolic name and index slots.
"""
struct Tensor <: TensorExpr
    name::Symbol
    indices::Vector{TIndex}
end

Base.:(==)(a::Tensor, b::Tensor) = a.name == b.name && a.indices == b.indices
Base.hash(a::Tensor, h::UInt) = hash(a.indices, hash(a.name, hash(:Tensor, h)))

"""
    TProduct(scalar, factors)

A product of tensor expressions with a rational scalar coefficient.
"""
struct TProduct <: TensorExpr
    scalar::Rational{Int}
    factors::Vector{TensorExpr}
end

Base.:(==)(a::TProduct, b::TProduct) = a.scalar == b.scalar && a.factors == b.factors
Base.hash(a::TProduct, h::UInt) = hash(a.factors, hash(a.scalar, hash(:TProduct, h)))

"""
    TSum(terms)

A sum of tensor expressions.
"""
struct TSum <: TensorExpr
    terms::Vector{TensorExpr}
end

Base.:(==)(a::TSum, b::TSum) = a.terms == b.terms
Base.hash(a::TSum, h::UInt) = hash(a.terms, hash(:TSum, h))

"""
    TDeriv(index, arg, covd=:partial)

A derivative operator applied to a tensor expression.
The `covd` field identifies which covariant derivative (:partial for ∂).
"""
struct TDeriv <: TensorExpr
    index::TIndex
    arg::TensorExpr
    covd::Symbol
end

TDeriv(index::TIndex, arg::TensorExpr) = TDeriv(index, arg, :partial)

Base.:(==)(a::TDeriv, b::TDeriv) = a.index == b.index && a.arg == b.arg && a.covd == b.covd
Base.hash(a::TDeriv, h::UInt) = hash(a.covd, hash(a.arg, hash(a.index, hash(:TDeriv, h))))

"""
    TScalar(val)

A scalar value (rational, symbolic, or expression) embedded in a tensor expression.
"""
struct TScalar <: TensorExpr
    val::Any
end

Base.:(==)(a::TScalar, b::TScalar) = isequal(a.val, b.val)
Base.hash(a::TScalar, h::UInt) = hash(a.val, hash(:TScalar, h))

# ── Symmetry types (defined early so TensorProperties can use the union) ──

struct Symmetric;          i::Int; j::Int; end
struct AntiSymmetric;      i::Int; j::Int; end
struct PairSymmetric;      i::Int; j::Int; k::Int; l::Int; end
struct RiemannSymmetry;    end
struct FullySymmetric;     slots::Vector{Int}; end
struct FullyAntiSymmetric; slots::Vector{Int}; end

FullySymmetric(slots::Int...) = FullySymmetric(collect(slots))
FullyAntiSymmetric(slots::Int...) = FullyAntiSymmetric(collect(slots))

"""Union of all symmetry specification types."""
const SymmetrySpec = Union{Symmetric, AntiSymmetric, PairSymmetric,
                           RiemannSymmetry, FullySymmetric, FullyAntiSymmetric}
