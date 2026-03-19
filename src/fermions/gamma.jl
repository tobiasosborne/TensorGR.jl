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
