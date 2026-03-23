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
- **order 6**: 8 cubic invariants in d>=5; 7 in d=4 (cubic DDI); 4 in d=3;
  1 in d=2 (Fulling et al. 1992, Table 1)

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

    # Order 6: cubic curvature invariants
    if order == 6
        return _invariant_lagrangian_order6(dim, registry)
    end

    # Higher orders not yet implemented
    throw(ArgumentError("invariant_lagrangian: order $order not yet implemented (only 0, 2, 4, 6 supported)"))
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

"""
Build the most general cubic curvature Lagrangian (6-derivative order).

In generic dimension (dim=nothing or dim >= 5), there are 8 independent
cubic curvature scalar invariants (Fulling et al. 1992, Table 1):

  I₁ = R³
  I₂ = R · R_{ab}R^{ab}
  I₃ = R · R_{abcd}R^{abcd}
  I₄ = R_{a}^{b} R_{b}^{c} R_{c}^{a}
  I₅ = R^{ab} R_{acde} R_b^{cde}
  I₆ = R^{ab} R^{cd} R_{acbd}
  I₇ = R_{ab}^{cd} R_{cd}^{ef} R_{ef}^{ab}
  I₈ = R_{abcd} R^{ab}_{ef} R^{cdef}

In d=4, one cubic DDI relation (Fulling et al. 1992) reduces to 7 invariants.
In d=3, Weyl vanishes, further reducing the basis.
"""
function _invariant_lagrangian_order6(dim::Union{Int,Nothing},
                                      registry::TensorRegistry)
    used = Set{Symbol}()
    fi() = (s = fresh_index(used); push!(used, s); s)

    R = Tensor(:RicScalar, TIndex[])

    # --- Product invariants (from lower-order combinations) ---

    # I₁: R³
    I1 = tproduct(1 // 1, TensorExpr[R, R, R])

    # I₂: R · Ric²
    a, b = fi(), fi()
    I2 = tproduct(1 // 1, TensorExpr[R,
         Tensor(:Ric, [down(a), down(b)]),
         Tensor(:Ric, [up(a), up(b)])])

    # I₃: R · Kretschner
    c, d = fi(), fi()
    I3 = tproduct(1 // 1, TensorExpr[R,
         Tensor(:Riem, [down(a), down(b), down(c), down(d)]),
         Tensor(:Riem, [up(a), up(b), up(c), up(d)])])

    # --- Irreducible cubic invariants ---

    # I₄: Ric³ trace = R_{a}^{b} R_{b}^{c} R_{c}^{a}
    e, f = fi(), fi()
    I4 = tproduct(1 // 1, TensorExpr[
         Tensor(:Ric, [down(a), up(b)]),
         Tensor(:Ric, [down(b), up(e)]),
         Tensor(:Ric, [down(e), up(a)])])

    # I₅: R^{ab} R_{acde} R_b^{cde}
    I5 = tproduct(1 // 1, TensorExpr[
         Tensor(:Ric, [up(a), up(b)]),
         Tensor(:Riem, [down(a), down(c), down(d), down(e)]),
         Tensor(:Riem, [down(b), up(c), up(d), up(e)])])

    # I₆: R^{ab} R^{cd} R_{acbd}
    I6 = tproduct(1 // 1, TensorExpr[
         Tensor(:Ric, [up(a), up(b)]),
         Tensor(:Ric, [up(c), up(d)]),
         Tensor(:Riem, [down(a), down(c), down(b), down(d)])])

    # I₇: R_{ab}^{cd} R_{cd}^{ef} R_{ef}^{ab}  (cyclic Riem³)
    I7 = tproduct(1 // 1, TensorExpr[
         Tensor(:Riem, [down(a), down(b), up(c), up(d)]),
         Tensor(:Riem, [down(c), down(d), up(e), up(f)]),
         Tensor(:Riem, [down(e), down(f), up(a), up(b)])])

    # I₈: R_{abcd} R^{ab}_{ef} R^{cdef}
    I8 = tproduct(1 // 1, TensorExpr[
         Tensor(:Riem, [down(a), down(b), down(c), down(d)]),
         Tensor(:Riem, [up(a), up(b), down(e), down(f)]),
         Tensor(:Riem, [up(c), up(d), up(e), up(f)])])

    invariants = TensorExpr[I1, I2, I3, I4, I5, I6, I7, I8]

    # Dimension-dependent reductions
    if dim !== nothing && dim <= 4
        # In d=4, one cubic DDI eliminates one invariant.
        # Drop I₇ (cyclic Riem³), which can be expressed via the others.
        invariants = TensorExpr[I1, I2, I3, I4, I5, I6, I8]
    end

    if dim !== nothing && dim <= 3
        # In d=3, Weyl vanishes: Riem = Ric⊗g decomposition.
        # All Riem invariants reduce to Ric/R invariants.
        # Only 4 independent cubic invariants remain: I₁, I₂, I₄, I₆
        invariants = TensorExpr[I1, I2, I4, I6]
    end

    if dim !== nothing && dim <= 2
        # In d=2, Ric = (R/2)g, so only I₁ = R³ survives
        invariants = TensorExpr[I1]
    end

    # Wrap with symbolic coefficients
    n = length(invariants)
    terms = TensorExpr[]
    for (i, inv) in enumerate(invariants)
        coeff_name = Symbol(:c, i)
        push!(terms, tproduct(1 // 1, TensorExpr[TScalar(coeff_name), inv]))
    end
    n == 1 ? terms[1] : tsum(terms)
end
