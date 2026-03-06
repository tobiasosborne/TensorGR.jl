#= Box operator and scalar covariant derivative helpers.

Convenience constructors for common differential operators on scalar fields:
  - box(φ, metric): □φ = g^{ab} ∂_a ∂_b φ
  - grad_squared(φ, metric): (∇φ)² = g^{ab} ∂_a φ ∂_b φ
  - covd_chain(φ, indices): ∂_{a₁} ∂_{a₂} ... ∂_{aₙ} φ
=#

"""
    box(field::Tensor, metric::Symbol; registry=current_registry()) -> TensorExpr

Construct the d'Alembertian (box) operator applied to a scalar field:
`□φ = g^{ab} ∂_a ∂_b φ`.

The result is a TProduct of the inverse metric and nested derivatives.
To expand using a covariant derivative, first construct, then call
`covd_to_christoffel` on the result.
"""
function box(field::Tensor, metric::Symbol;
             registry::TensorRegistry=current_registry())
    with_registry(registry) do
        used = Set{Symbol}(idx.name for idx in field.indices)
        a = fresh_index(used)
        push!(used, a)
        b = fresh_index(used)
        g_inv = Tensor(metric, [up(a), up(b)])
        g_inv * TDeriv(down(a), TDeriv(down(b), field))
    end
end

"""
    grad_squared(field::Tensor, metric::Symbol; registry=current_registry()) -> TensorExpr

Construct the gradient squared of a scalar field:
`(∇φ)² = g^{ab} ∂_a φ ∂_b φ`.
"""
function grad_squared(field::Tensor, metric::Symbol;
                      registry::TensorRegistry=current_registry())
    with_registry(registry) do
        used = Set{Symbol}(idx.name for idx in field.indices)
        a = fresh_index(used)
        push!(used, a)
        b = fresh_index(used)
        Tensor(metric, [up(a), up(b)]) * TDeriv(down(a), field) * TDeriv(down(b), field)
    end
end

"""
    covd_chain(field::TensorExpr, idxs::Vector{TIndex}) -> TensorExpr

Construct a chain of covariant/partial derivatives applied to `field`:
`∂_{a₁}(∂_{a₂}(...∂_{aₙ}(field)...))`.

The outermost derivative has index `idxs[1]`.
"""
function covd_chain(field::TensorExpr, idxs::Vector{TIndex})
    result = field
    for i in length(idxs):-1:1
        result = TDeriv(idxs[i], result)
    end
    result
end

"""
    covd_product(field::Tensor, idx_a::TIndex, idx_b::TIndex) -> TensorExpr

Construct the product of two covariant derivatives of a scalar field:
`(∂_a φ)(∂_b φ)`.
"""
function covd_product(field::Tensor, idx_a::TIndex, idx_b::TIndex)
    TDeriv(idx_a, field) * TDeriv(idx_b, field)
end
