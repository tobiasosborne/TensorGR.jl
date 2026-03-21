#= Invar precomputed invariant database infrastructure (Epic 3).
#
# Provides types and lazy-loading API for precomputed algebraic relations
# between curvature invariants at each level of the Invar simplification
# pipeline. The database stores reduction rules mapping non-canonical
# RInv contractions to linear combinations of canonical (independent) ones.
#
# Data files live in src/invariants/db/ and are loaded lazily on first access.
#
# Reference: Garcia-Parrado & Martin-Garcia, Comp. Phys. Comm. 176 (2007) 246, Sec 6;
#            Fulling et al. (1992), CQG 9, 1151.
=#

"""
    InvarRelation

A single reduction relation for the Invar database: a non-canonical RInv
(given by its contraction permutation) is expressed as a linear combination
of canonical (independent) RInv contractions.

# Fields
- `lhs::Vector{Int}` -- contraction permutation of the dependent (non-canonical) invariant
- `rhs::Vector{Tuple{Rational{Int}, Vector{Int}}}` -- linear combination of canonical
  contractions: `[(coefficient, canonical_contraction), ...]`
"""
struct InvarRelation
    lhs::Vector{Int}
    rhs::Vector{Tuple{Rational{Int}, Vector{Int}}}
end

function Base.:(==)(a::InvarRelation, b::InvarRelation)
    a.lhs == b.lhs && a.rhs == b.rhs
end

function Base.hash(r::InvarRelation, h::UInt)
    hash(r.rhs, hash(r.lhs, hash(:InvarRelation, h)))
end

"""
    CaseRelations

Precomputed relations for one case (degree, case_key) at one simplification
step, optionally dimension-specific.

# Fields
- `degree::Int` -- number of Riemann factors (polynomial degree in curvature)
- `case_key::String` -- case identifier, e.g., `"0_0"` for algebraic degree-2
- `step::Int` -- simplification level (1-6)
- `dim::Union{Int,Nothing}` -- manifold dimension, `nothing` for dimension-independent
- `n_independent::Int` -- number of independent invariants at this step
- `n_dependent::Int` -- number of dependent invariants (that reduce to independent ones)
- `relations::Vector{InvarRelation}` -- the reduction relations
"""
struct CaseRelations
    degree::Int
    case_key::String
    step::Int
    dim::Union{Int,Nothing}
    n_independent::Int
    n_dependent::Int
    relations::Vector{InvarRelation}
end

function Base.:(==)(a::CaseRelations, b::CaseRelations)
    a.degree == b.degree && a.case_key == b.case_key &&
    a.step == b.step && a.dim == b.dim &&
    a.n_independent == b.n_independent && a.n_dependent == b.n_dependent &&
    a.relations == b.relations
end

# ---- Lazy-loading cache and API ------------------------------------------------

const _INVAR_DB_CACHE = Dict{Tuple{Int,String,Int,Union{Int,Nothing}}, CaseRelations}()

"""
    get_invar_relations(degree, case_key, step; dim=nothing) -> CaseRelations

Look up precomputed invariant relations for a specific case.

Returns the `CaseRelations` struct containing the reduction rules that express
dependent (non-canonical) invariants as linear combinations of independent
(canonical) ones at the given simplification step.

Lazy-loads the data on first access and caches for subsequent calls.

# Arguments
- `degree::Int` -- number of Riemann factors
- `case_key::String` -- case identifier (e.g., `"0_0"`)
- `step::Int` -- simplification level (1-6)
- `dim::Union{Int,Nothing}=nothing` -- manifold dimension (for DDI/dual steps)

# Errors
Throws `KeyError` if the requested case is not available in the database.
"""
function get_invar_relations(degree::Int, case_key::String, step::Int;
                              dim::Union{Int,Nothing}=nothing)
    key = (degree, case_key, step, dim)
    if !haskey(_INVAR_DB_CACHE, key)
        cr = _load_case(degree, case_key, step, dim)
        _INVAR_DB_CACHE[key] = cr
    end
    _INVAR_DB_CACHE[key]
end

"""
    _load_case(degree, case_key, step, dim) -> CaseRelations

Load the precomputed relations for a specific case from the database registry.
Called by `get_invar_relations` on cache miss.
"""
function _load_case(degree::Int, case_key::String, step::Int,
                     dim::Union{Int,Nothing})
    # Look up in the registered data sources
    key = (degree, case_key, step, dim)
    if haskey(_INVAR_DB_REGISTRY, key)
        return _INVAR_DB_REGISTRY[key]
    end
    throw(KeyError("Invar database: no data for degree=$degree, case=$case_key, step=$step, dim=$dim"))
end

# Registry of available data (populated by db/*.jl files)
const _INVAR_DB_REGISTRY = Dict{Tuple{Int,String,Int,Union{Int,Nothing}}, CaseRelations}()

"""
    _register_invar_case!(cr::CaseRelations)

Register a `CaseRelations` entry in the database. Called by `db/*.jl` files
at module load time.
"""
function _register_invar_case!(cr::CaseRelations)
    key = (cr.degree, cr.case_key, cr.step, cr.dim)
    _INVAR_DB_REGISTRY[key] = cr
end

"""
    list_invar_cases(; step=nothing, degree=nothing) -> Vector{NamedTuple}

List all available cases in the Invar database, optionally filtered by
simplification step and/or degree.

Returns a vector of named tuples with fields:
`(degree, case_key, step, dim, n_independent, n_dependent)`.
"""
function list_invar_cases(; step::Union{Int,Nothing}=nothing,
                            degree::Union{Int,Nothing}=nothing)
    results = NamedTuple{(:degree,:case_key,:step,:dim,:n_independent,:n_dependent),
                          Tuple{Int,String,Int,Union{Int,Nothing},Int,Int}}[]
    for ((d, ck, s, dm), cr) in _INVAR_DB_REGISTRY
        (step !== nothing && s != step) && continue
        (degree !== nothing && d != degree) && continue
        push!(results, (degree=d, case_key=ck, step=s, dim=dm,
                        n_independent=cr.n_independent, n_dependent=cr.n_dependent))
    end
    sort!(results, by=x -> (x.degree, x.case_key, x.step, something(x.dim, 0)))
    results
end

"""
    is_independent_rinv(rinv::RInv, step::Int; dim=nothing) -> Bool

Check if an RInv is independent (i.e., not reducible to other invariants)
at a given simplification step.

The RInv is canonicalized first, then checked against the database relations:
if it appears as the LHS of any relation, it is dependent; otherwise independent.

# Arguments
- `rinv::RInv` -- the invariant to check
- `step::Int` -- simplification level (1-6)
- `dim::Union{Int,Nothing}=nothing` -- manifold dimension

# Returns
`true` if the invariant is independent at this step, `false` if it reduces.
"""
function is_independent_rinv(rinv::RInv, step::Int;
                              dim::Union{Int,Nothing}=nothing)
    crinv = rinv.canonical ? rinv : canonicalize(rinv)

    # Zero contraction means vanishing invariant -- not independent
    all(==(0), crinv.contraction) && return false

    degree = crinv.degree
    case_key = _algebraic_case_key(degree)

    # Try to load the relations for this case
    key = (degree, case_key, step, dim)
    cr = nothing
    if haskey(_INVAR_DB_REGISTRY, key)
        cr = _INVAR_DB_REGISTRY[key]
    elseif haskey(_INVAR_DB_CACHE, key)
        cr = _INVAR_DB_CACHE[key]
    end

    cr === nothing && return true  # No data => assume independent

    # Check if this contraction appears as the LHS of any relation
    for rel in cr.relations
        if rel.lhs == crinv.contraction
            return false
        end
    end
    true
end

"""
    _algebraic_case_key(degree) -> String

Generate the case key for a purely algebraic (non-derivative) invariant
of the given degree. The case key is a string of zeros separated by
underscores, one zero per Riemann factor.
"""
function _algebraic_case_key(degree::Int)
    join(fill("0", degree), "_")
end
