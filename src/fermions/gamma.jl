#= Gamma matrices and Clifford algebra.
#
# The Dirac gamma matrices γ^a satisfy the Clifford algebra:
#   {γ^a, γ^b} = 2 g^{ab} I
#
# where I is the identity in spinor space. In d=4, γ^a are 4×4 matrices.
#
# The gamma matrix has one spacetime index (Up or Down) and implicit
# spinor indices (suppressed in abstract computations).
#
# Ground truth: Wald GR Appendix B; Peskin & Schroeder (1995) Sec 3.2.
=#

"""
    GammaMatrix <: TensorExpr

A Dirac gamma matrix γ^a (or γ_a) with one spacetime index.

Spinor indices are suppressed (implicit). Products of gamma matrices
form a Clifford algebra chain.

# Fields
- `index::TIndex`  -- spacetime index (Up or Down)
"""
struct GammaMatrix <: TensorExpr
    index::TIndex
end

Base.:(==)(a::GammaMatrix, b::GammaMatrix) = a.index == b.index
Base.hash(a::GammaMatrix, h::UInt) = hash(a.index, hash(:GammaMatrix, h))

# AST integration
indices(g::GammaMatrix) = TIndex[g.index]
free_indices(g::GammaMatrix) = TIndex[g.index]
children(::GammaMatrix) = TensorExpr[]
walk(f, g::GammaMatrix) = f(g)
derivative_order(::GammaMatrix) = 0
is_constant(::GammaMatrix) = false
is_sorted_covds(::GammaMatrix) = true
rename_dummy(g::GammaMatrix, old::Symbol, new::Symbol) =
    g.index.name == old ? GammaMatrix(TIndex(new, g.index.position, g.index.vbundle)) : g
rename_dummies(g::GammaMatrix, m::Dict{Symbol,Symbol}) =
    let new_name = get(m, g.index.name, g.index.name)
        new_name == g.index.name ? g : GammaMatrix(TIndex(new_name, g.index.position, g.index.vbundle))
    end
_replace_index_name(g::GammaMatrix, old::Symbol, new::Symbol) =
    g.index.name == old ? GammaMatrix(TIndex(new, g.index.position, g.index.vbundle)) : g
to_expr(g::GammaMatrix) = Expr(:call, :GammaMatrix, to_expr(g.index))
is_well_formed(::GammaMatrix) = true
_validate_walk(::GammaMatrix, ::TensorRegistry, ::Vector{String}) = nothing

function Base.show(io::IO, g::GammaMatrix)
    print(io, "γ")
    if g.index.position == Up
        print(io, "^", g.index.name)
    else
        print(io, "_", g.index.name)
    end
end

function to_latex(g::GammaMatrix)
    if g.index.position == Up
        "\\gamma^{$(g.index.name)}"
    else
        "\\gamma_{$(g.index.name)}"
    end
end

function to_unicode(g::GammaMatrix)
    if g.index.position == Up
        "γ^$(g.index.name)"
    else
        "γ_$(g.index.name)"
    end
end

function dagger(g::GammaMatrix)
    # γ^a† = γ^0 γ^a γ^0 in the Dirac representation
    # For abstract computation, dagger just flips index position
    GammaMatrix(TIndex(g.index.name,
        g.index.position == Up ? Down : Up, g.index.vbundle))
end

# ────────────────────────────────────────────────────────────────────
# Clifford algebra
# ────────────────────────────────────────────────────────────────────

"""
    gamma5(; metric::Symbol=:g) -> TensorExpr

The chirality matrix γ⁵ = iγ⁰γ¹γ²γ³ (in d=4).

Satisfies: (γ⁵)² = I, {γ⁵, γ^a} = 0.

Represented as a product of four gamma matrices with a prefactor.
"""
function gamma5(; metric::Symbol=:g)
    # γ⁵ = i γ⁰ γ¹ γ² γ³ — abstract representation
    TProduct(1 // 1, TensorExpr[
        TScalar(:im),  # imaginary unit
        GammaMatrix(down(:_0)),
        GammaMatrix(down(:_1)),
        GammaMatrix(down(:_2)),
        GammaMatrix(down(:_3))
    ])
end

"""
    clifford_relation(a::TIndex, b::TIndex; metric::Symbol=:g) -> TensorExpr

The Clifford algebra relation:

    {γ^a, γ^b} = γ^a γ^b + γ^b γ^a = 2 g^{ab}

Returns the RHS: 2 g^{ab}.
"""
function clifford_relation(a::TIndex, b::TIndex; metric::Symbol=:g)
    tproduct(2 // 1, TensorExpr[Tensor(metric, [a, b])])
end

"""
    gamma_trace(n::Int; dim::Int=4) -> Any

Trace of a product of n gamma matrices (in d dimensions).

    Tr(I) = d_s  (spinor dimension, = 4 in d=4)
    Tr(γ^a) = 0
    Tr(γ^a γ^b) = d_s g^{ab}
    Tr(odd number) = 0

For even n ≥ 4, use recursive relation via Clifford algebra.

Ground truth: Peskin & Schroeder (1995) Sec 3.2, Eqs 3.45-3.46.
"""
function gamma_trace(n::Int; dim::Int=4)
    spinor_dim = dim  # In d dimensions, Tr(I) = 2^{d/2}; for d=4, = 4
    if n == 0
        return spinor_dim
    elseif n % 2 == 1
        return 0  # Odd traces vanish
    elseif n == 2
        return spinor_dim  # Tr(γ^a γ^b) = d_s g^{ab} (coefficient)
    else
        return nothing  # Needs recursive computation for n ≥ 4
    end
end

# ────────────────────────────────────────────────────────────────────
# γ⁵ chirality: algebraic properties and trace identities
# ────────────────────────────────────────────────────────────────────

#= γ⁵ properties (d=4):
#   (γ⁵)² = I                                           (involutory)
#   {γ⁵, γ^a} = 0                                       (anticommutes with all γ^a)
#   Tr(γ⁵) = 0
#   Tr(γ⁵ γ^a γ^b) = 0
#   Tr(γ⁵ γ^a γ^b γ^c γ^d) = -4i ε^{abcd}
#
# Ground truth: Peskin & Schroeder (1995) Eq A.30; Wald, GR, Appendix B.
=#

"""
    Gamma5 <: TensorExpr

The chirality matrix γ⁵ = iγ⁰γ¹γ²γ³ as a distinct AST node.

γ⁵ is index-free (no spacetime indices). Its algebraic properties
are encoded in trace identities and anticommutation rules.
"""
struct Gamma5 <: TensorExpr end

Base.:(==)(::Gamma5, ::Gamma5) = true
Base.hash(::Gamma5, h::UInt) = hash(:Gamma5, h)

indices(::Gamma5) = TIndex[]
free_indices(::Gamma5) = TIndex[]
children(::Gamma5) = TensorExpr[]
walk(f, g::Gamma5) = f(g)
derivative_order(::Gamma5) = 0
is_constant(::Gamma5) = false
is_sorted_covds(::Gamma5) = true
rename_dummy(g::Gamma5, ::Symbol, ::Symbol) = g
rename_dummies(g::Gamma5, ::Dict{Symbol,Symbol}) = g
_replace_index_name(g::Gamma5, ::Symbol, ::Symbol) = g
to_expr(::Gamma5) = :(Gamma5())
is_well_formed(::Gamma5) = true
_validate_walk(::Gamma5, ::TensorRegistry, ::Vector{String}) = nothing
dagger(::Gamma5) = Gamma5()  # γ⁵ is Hermitian: (γ⁵)† = γ⁵

Base.show(io::IO, ::Gamma5) = print(io, "γ⁵")
to_latex(::Gamma5) = "\\gamma^5"
to_unicode(::Gamma5) = "γ⁵"

"""
    gamma5_trace(n_gamma::Int; dim::Int=4) -> Any

Trace identities involving γ⁵ and n gamma matrices.

    Tr(γ⁵) = 0
    Tr(γ⁵ γ^a γ^b) = 0
    Tr(γ⁵ γ^a γ^b γ^c γ^d) = -4i ε^{abcd}  (coefficient: -4i)
    Tr(γ⁵ × odd number of γ^a) = 0

Ground truth: Peskin & Schroeder (1995) Eq A.30.
"""
function gamma5_trace(n_gamma::Int; dim::Int=4)
    if n_gamma == 0
        # Tr(γ⁵) = 0
        return 0
    elseif n_gamma % 2 == 1
        # Tr(γ⁵ × odd) = 0
        return 0
    elseif n_gamma == 2
        # Tr(γ⁵ γ^a γ^b) = 0
        return 0
    elseif n_gamma == 4
        # Tr(γ⁵ γ^a γ^b γ^c γ^d) = -4i ε^{abcd}
        # Return the coefficient: -4i (the ε tensor is separate)
        return :(-4im)  # symbolic imaginary
    else
        return nothing  # Higher traces need recursive computation
    end
end

"""
    gamma5_anticommutator() -> TScalar

The anticommutator {γ⁵, γ^a} = 0.

Returns TScalar(0) since the anticommutator always vanishes.
"""
gamma5_anticommutator() = TScalar(0)

"""
    gamma5_squared() -> TScalar

(γ⁵)² = I (the identity matrix in spinor space).

Returns TScalar(1) representing the identity.
"""
gamma5_squared() = TScalar(1)

"""
    slash(v::TensorExpr) -> TensorExpr

Feynman slash notation: v̸ = γ^a v_a (contraction of vector with gamma matrix).

For a vector `v` with one free Down index, returns γ^a v_a.
"""
function slash(v::TensorExpr)
    free = free_indices(v)
    length(free) == 1 || error("slash requires a vector (1 free index), got $(length(free))")
    idx = free[1]
    gamma = GammaMatrix(TIndex(idx.name, idx.position == Up ? Down : Up, idx.vbundle))
    tproduct(1 // 1, TensorExpr[gamma, v])
end
