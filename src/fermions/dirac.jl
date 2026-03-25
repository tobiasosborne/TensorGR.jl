#= Dirac field registration and bilinear construction.
#
# A Dirac spinor ψ is a Grassmann-odd (anticommuting) field with implicit
# 4-component spinor indices. The Dirac conjugate is ψ̄ = ψ†γ⁰.
#
# The Dirac Lagrangian (kinetic term) is:
#   L = iψ̄γ^a∂_aψ - mψ̄ψ
#
# The Dirac equation is:
#   (iγ^a∂_a - m)ψ = 0
#
# Following the design document (fermion_design.md Sec 2), Dirac fields
# are registered as rank-(0,0) Tensor objects with is_grassmann=true in
# options and suppressed spinor indices (matching GammaMatrix convention).
#
# References:
#   Peskin & Schroeder (1995), Sec 3.2-3.6.
#   Wald, *General Relativity* (1984), Appendix B.
#   Frob (2020), arXiv:2008.12422, Sec 3.
=#

"""
    define_fermion!(reg, name; manifold=:M4, type=:dirac, mass=:m)

Register a fermion field and its Dirac conjugate in the registry.

For a Dirac field `psi`, this registers:
- `psi` (Grassmann-odd, rank (0,0), suppressed spinor indices)
- `psi_bar` (Dirac conjugate, also Grassmann-odd)
- `mass` (scalar, the mass parameter)

The two fields are linked via `:conjugate_field` in their options.

# Supported types
- `:dirac` -- 4-component Dirac spinor (default)
- `:majorana` -- self-conjugate (ψ̄ = ψᵀC)
- `:weyl_left` -- left-handed 2-component (no conjugate auto-registered)
- `:weyl_right` -- right-handed 2-component (no conjugate auto-registered)

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_fermion!(reg, :psi)
    # Now psi and psi_bar are registered
end
```
"""
function define_fermion!(reg::TensorRegistry, name::Symbol;
                          manifold::Symbol=:M4,
                          type::Symbol=:dirac,
                          mass::Symbol=:m)
    has_manifold(reg, manifold) ||
        error("define_fermion!: manifold '$manifold' not registered")

    bar_name = Symbol(name, :_bar)

    has_conjugate = type in (:dirac, :majorana)

    # Register the fermion field
    if !has_tensor(reg, name)
        opts = Dict{Symbol,Any}(:fermion_type => type, :mass => mass)
        if has_conjugate
            opts[:conjugate_field] = bar_name
        end
        register_grassmann_field!(reg, name;
            manifold=manifold, rank=(0, 0), options=opts)
    end

    # Register the conjugate field (for Dirac and Majorana types)
    if has_conjugate && !has_tensor(reg, bar_name)
        register_grassmann_field!(reg, bar_name;
            manifold=manifold, rank=(0, 0),
            options=Dict{Symbol,Any}(
                :fermion_type => type,
                :conjugate_field => name,
                :is_conjugate => true,
                :mass => mass))
    end

    # Register the mass parameter as a scalar
    if !has_tensor(reg, mass)
        register_tensor!(reg, TensorProperties(
            name=mass, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_scalar_field => true,
                                     :is_mass => true)))
    end

    nothing
end

"""
    dirac_bar(psi::Tensor; registry=current_registry()) -> Tensor

Return the Dirac conjugate field ψ̄ for a given Dirac field ψ.

Looks up the conjugate name from the `:conjugate_field` option in the registry.

# Example
```julia
psi = Tensor(:psi, TIndex[])
psi_bar = dirac_bar(psi)  # => Tensor(:psi_bar, TIndex[])
```
"""
function dirac_bar(psi::Tensor;
                    registry::TensorRegistry=current_registry())
    has_tensor(registry, psi.name) ||
        error("dirac_bar: field '$(psi.name)' not registered")
    props = get_tensor(registry, psi.name)
    bar_name = get(props.options, :conjugate_field, nothing)
    bar_name === nothing &&
        error("dirac_bar: field '$(psi.name)' has no conjugate_field")
    Tensor(bar_name, TIndex[])
end

"""
    is_fermion(reg, name) -> Bool

Check if a tensor is a registered fermion field.
"""
function is_fermion(reg::TensorRegistry, name::Symbol)
    has_tensor(reg, name) || return false
    props = get_tensor(reg, name)
    haskey(props.options, :fermion_type)
end

"""
    get_conjugate_name(reg, name) -> Symbol

Return the conjugate field name for a fermion.
"""
function get_conjugate_name(reg::TensorRegistry, name::Symbol)
    has_tensor(reg, name) || error("Field '$name' not registered")
    props = get_tensor(reg, name)
    bar = get(props.options, :conjugate_field, nothing)
    bar === nothing && error("Field '$name' has no conjugate")
    bar
end

# ── Bilinear construction ──────────────────────────────────────────────

"""
    scalar_bilinear(psi_name; registry=current_registry()) -> TProduct

Build the scalar bilinear ψ̄ψ.
"""
function scalar_bilinear(psi_name::Symbol;
                          registry::TensorRegistry=current_registry())
    bar_name = get_conjugate_name(registry, psi_name)
    TProduct(1 // 1, TensorExpr[
        Tensor(bar_name, TIndex[]),
        Tensor(psi_name, TIndex[])
    ])
end

"""
    vector_bilinear(psi_name, a::TIndex; registry=current_registry()) -> TProduct

Build the vector bilinear ψ̄γ^aψ.

The index `a` is a free spacetime index on the gamma matrix.
"""
function vector_bilinear(psi_name::Symbol, a::TIndex;
                          registry::TensorRegistry=current_registry())
    bar_name = get_conjugate_name(registry, psi_name)
    TProduct(1 // 1, TensorExpr[
        Tensor(bar_name, TIndex[]),
        GammaMatrix(a),
        Tensor(psi_name, TIndex[])
    ])
end

"""
    axial_bilinear(psi_name, a::TIndex; registry=current_registry()) -> TProduct

Build the axial vector bilinear ψ̄γ^aγ⁵ψ.
"""
function axial_bilinear(psi_name::Symbol, a::TIndex;
                         registry::TensorRegistry=current_registry())
    bar_name = get_conjugate_name(registry, psi_name)
    TProduct(1 // 1, TensorExpr[
        Tensor(bar_name, TIndex[]),
        GammaMatrix(a),
        Gamma5(),
        Tensor(psi_name, TIndex[])
    ])
end

"""
    pseudo_bilinear(psi_name; registry=current_registry()) -> TProduct

Build the pseudoscalar bilinear ψ̄γ⁵ψ.
"""
function pseudo_bilinear(psi_name::Symbol;
                          registry::TensorRegistry=current_registry())
    bar_name = get_conjugate_name(registry, psi_name)
    TProduct(1 // 1, TensorExpr[
        Tensor(bar_name, TIndex[]),
        Gamma5(),
        Tensor(psi_name, TIndex[])
    ])
end

"""
    dirac_kinetic_expr(psi_name; registry=current_registry()) -> TSum

Build the Dirac kinetic Lagrangian:

    L = iψ̄γ^a∂_aψ - mψ̄ψ

Returns a TSum with two terms. The derivative index `a` is contracted
between γ^a and ∂_a.
"""
function dirac_kinetic_expr(psi_name::Symbol;
                             registry::TensorRegistry=current_registry())
    bar_name = get_conjugate_name(registry, psi_name)
    props = get_tensor(registry, psi_name)
    mass_name = get(props.options, :mass, :m)

    # Generate a fresh Tangent index for the contraction
    a_sym = fresh_index(Set{Symbol}(); vbundle=:Tangent)

    # Term 1: i ψ̄ γ^a ∂_a ψ
    psi_field = Tensor(psi_name, TIndex[])
    d_psi = TDeriv(down(a_sym), psi_field, :partial)
    kinetic = TProduct(1 // 1, TensorExpr[
        TScalar(:im),
        Tensor(bar_name, TIndex[]),
        GammaMatrix(up(a_sym)),
        d_psi
    ])

    # Term 2: -m ψ̄ ψ
    mass_term = TProduct(-1 // 1, TensorExpr[
        TScalar(mass_name),
        Tensor(bar_name, TIndex[]),
        psi_field
    ])

    TSum(TensorExpr[kinetic, mass_term])
end

"""
    dirac_equation_expr(psi_name; registry=current_registry()) -> TSum

Build the Dirac equation (iγ^a∂_a - m)ψ = 0 as an expression.

Returns the LHS: iγ^a∂_aψ - mψ (which should vanish on-shell).
"""
function dirac_equation_expr(psi_name::Symbol;
                              registry::TensorRegistry=current_registry())
    props = get_tensor(registry, psi_name)
    mass_name = get(props.options, :mass, :m)

    a_sym = fresh_index(Set{Symbol}(); vbundle=:Tangent)
    psi_field = Tensor(psi_name, TIndex[])

    # iγ^a∂_aψ
    d_psi = TDeriv(down(a_sym), psi_field, :partial)
    kinetic = TProduct(1 // 1, TensorExpr[
        TScalar(:im),
        GammaMatrix(up(a_sym)),
        d_psi
    ])

    # -mψ
    mass_term = TProduct(-1 // 1, TensorExpr[
        TScalar(mass_name),
        psi_field
    ])

    TSum(TensorExpr[kinetic, mass_term])
end
