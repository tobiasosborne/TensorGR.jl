#= Invariant Lagrangian construction.

Constructs the most general diffeomorphism-invariant Lagrangian at a given
derivative order using independent curvature invariants.

Physics ground truth:
  - Order 0 (0 derivatives): cosmological constant Lambda
  - Order 2 (2 derivatives): R (Ricci scalar) -- 1 invariant
  - Order 4 (4 derivatives): R^2, R_{ab}R^{ab}, R_{abcd}R^{abcd} -- 3 invariants
    in d >= 5; 2 independent in d=4 (Gauss-Bonnet removes Kretschner)
  - Order 6 (6 derivatives): 8 invariants in d >= 7; fewer in low d

References:
  - Nutma (2014), arXiv:1308.3493
  - Fulling, King, Wybourne & Cummins (1992), CQG 9, 1151
=#

"""
    invariant_lagrangian(order::Int;
                          dim::Union{Int,Nothing}=nothing,
                          registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the most general diffeomorphism-invariant Lagrangian at derivative
order `order`, using independent curvature invariants with undetermined
symbolic coefficients (`TScalar(:c1)`, `TScalar(:c2)`, etc.).

The derivative order must be a non-negative even integer (odd orders have no
invariants for a metric theory).

# Invariant counts (Fulling et al. 1992, Table 1)
- **order 0**: cosmological constant `Lambda` (1 term)
- **order 2**: Ricci scalar `R` (1 term)
- **order 4**: `R^2`, `R_{ab}R^{ab}`, `R_{abcd}R^{abcd}` (3 terms in d>=5;
  2 in d=4 via Gauss-Bonnet)

# Arguments
- `order::Int`: derivative order (must be non-negative and even)
- `dim::Union{Int,Nothing}=nothing`: manifold dimension. If provided,
  dimensionally-dependent identities (DDIs) reduce the basis (e.g., Gauss-Bonnet
  in d=4 eliminates Kretschner). If `nothing`, returns the generic-dimension basis.
- `registry::TensorRegistry`: the registry containing curvature tensor definitions

# Returns
A `TensorExpr` (typically `TSum`) representing the most general Lagrangian with
undetermined coefficients.

# Examples
```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g registry=reg
define_curvature_tensors!(reg, :M4, :g)

# Order 0: cosmological constant
L0 = with_registry(reg) do
    invariant_lagrangian(0; registry=reg)
end  # => TScalar(:Lambda)

# Order 2: c1 * R
L2 = with_registry(reg) do
    invariant_lagrangian(2; registry=reg)
end

# Order 4, generic dimension: c1*R^2 + c2*Ric^2 + c3*K
L4 = with_registry(reg) do
    invariant_lagrangian(4; registry=reg)
end

# Order 4, d=4: c1*R^2 + c2*Ric^2 (Gauss-Bonnet removes K)
L4_d4 = with_registry(reg) do
    invariant_lagrangian(4; dim=4, registry=reg)
end
```

See also: [`all_contractions`](@ref), [`contraction_ansatz`](@ref),
[`gauss_bonnet_ddi`](@ref)
"""
function invariant_lagrangian(order::Int;
                               dim::Union{Int,Nothing}=nothing,
                               registry::TensorRegistry=current_registry())
    order < 0 && throw(ArgumentError("Derivative order must be non-negative, got $order"))
    isodd(order) && throw(ArgumentError("Derivative order must be even (no invariants at odd order), got $order"))

    # Order 0: cosmological constant
    if order == 0
        return TScalar(:Lambda)
    end

    # Order 2: c1 * RicScalar
    if order == 2
        return tproduct(1 // 1, TensorExpr[TScalar(:c1), Tensor(:RicScalar, TIndex[])])
    end

    # Order 4: quadratic curvature invariants
    if order == 4
        return _invariant_lagrangian_order4(dim, registry)
    end

    # Higher orders not yet implemented
    throw(ArgumentError("invariant_lagrangian: order $order not yet implemented (only 0, 2, 4 supported)"))
end

"""
Build the most general quadratic curvature Lagrangian (4-derivative order).

In generic dimension (dim=nothing or dim >= 5):
  L = c1 * R^2 + c2 * R_{ab}R^{ab} + c3 * R_{abcd}R^{abcd}

In d=4 (Gauss-Bonnet: K = 4*Ric^2 - R^2):
  L = c1 * R^2 + c2 * R_{ab}R^{ab}
"""
function _invariant_lagrangian_order4(dim::Union{Int,Nothing},
                                      registry::TensorRegistry)
    # Build the three standard quadratic curvature invariants explicitly.
    # This is more robust than enumerating all contractions of Riem*Riem.

    # I1: R^2 (Ricci scalar squared)
    R1 = Tensor(:RicScalar, TIndex[])
    R2 = Tensor(:RicScalar, TIndex[])
    R_squared = tproduct(1 // 1, TensorExpr[R1, R2])

    # I2: Ric^2 = R_{ab}R^{ab} (Ricci tensor squared)
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    Ric_down = Tensor(:Ric, [down(a), down(b)])
    Ric_up = Tensor(:Ric, [up(a), up(b)])
    Ric_squared = tproduct(1 // 1, TensorExpr[Ric_down, Ric_up])

    # I3: K = R_{abcd}R^{abcd} (Kretschner scalar)
    c = fresh_index(used); push!(used, c)
    d_idx = fresh_index(used); push!(used, d_idx)
    Riem_down = Tensor(:Riem, [down(a), down(b), down(c), down(d_idx)])
    Riem_up = Tensor(:Riem, [up(a), up(b), up(c), up(d_idx)])
    Kretschner = tproduct(1 // 1, TensorExpr[Riem_down, Riem_up])

    # In d=4, Gauss-Bonnet identity: K - 4*Ric^2 + R^2 = 0
    # => K = 4*Ric^2 - R^2, so K is not independent.
    # Return only R^2 and Ric^2 with 2 coefficients.
    if dim !== nothing && dim == 4
        terms = TensorExpr[
            tproduct(1 // 1, TensorExpr[TScalar(:c1), R_squared]),
            tproduct(1 // 1, TensorExpr[TScalar(:c2), Ric_squared])
        ]
        return tsum(terms)
    end

    # Generic dimension (dim=nothing or dim >= 5): all three invariants
    terms = TensorExpr[
        tproduct(1 // 1, TensorExpr[TScalar(:c1), R_squared]),
        tproduct(1 // 1, TensorExpr[TScalar(:c2), Ric_squared]),
        tproduct(1 // 1, TensorExpr[TScalar(:c3), Kretschner])
    ]
    tsum(terms)
end
