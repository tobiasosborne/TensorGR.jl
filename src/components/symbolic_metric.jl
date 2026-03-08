#= SymbolicMetric: struct and stubs for symbolic curvature computation.

The struct lives in base (no Symbolics dependency).
Implementations are in ext/TensorGRSymbolicsExt.jl.
=#

"""
    SymbolicMetric

Holds a symbolic metric tensor and its inverse, along with coordinate variables.
Fields use `Any` to avoid a Symbolics.jl dependency in the base module.

# Fields
- `coords::Vector{Any}` — coordinate symbolic variables
- `coord_names::Vector{Symbol}` — coordinate names as symbols
- `g::Matrix{Any}` — metric components g_{μν}
- `ginv::Matrix{Any}` — inverse metric g^{μν}
- `dim::Int` — spacetime dimension
"""
struct SymbolicMetric
    coords::Vector{Any}
    coord_names::Vector{Symbol}
    g::Matrix{Any}
    ginv::Matrix{Any}
    dim::Int
end

# Stub functions — implemented in ext/TensorGRSymbolicsExt.jl

"""
    symbolic_diagonal_metric(coords, diag) -> SymbolicMetric

Create a `SymbolicMetric` from a diagonal metric specified as a vector of diagonal entries.
"""
function symbolic_diagonal_metric end

"""
    symbolic_metric(coords, g::Matrix) -> SymbolicMetric

Create a `SymbolicMetric` from a full metric matrix, computing the inverse via `inv`.
"""
function symbolic_metric end

"""
    sym_deriv(expr, coord)

Compute the symbolic derivative of `expr` with respect to `coord`.
"""
function sym_deriv end

"""
    symbolic_christoffel(sm::SymbolicMetric) -> Array{Any,3}

Compute Christoffel symbols Γ^a_{bc} from a `SymbolicMetric`.
"""
function symbolic_christoffel end

"""
    symbolic_riemann(sm::SymbolicMetric, Gamma) -> Array{Any,4}

Compute Riemann tensor R^a_{bcd} from Christoffel symbols.
"""
function symbolic_riemann end

"""
    symbolic_ricci(Riem, dim::Int) -> Matrix{Any}

Compute Ricci tensor R_{bd} = R^a_{bad} by contraction.
"""
function symbolic_ricci end

"""
    symbolic_ricci_scalar(Ric, ginv, dim::Int) -> Any

Compute Ricci scalar R = g^{bd} R_{bd}.
"""
function symbolic_ricci_scalar end

"""
    symbolic_einstein(Ric, R, g, dim::Int) -> Matrix{Any}

Compute Einstein tensor G_{ab} = R_{ab} - (1/2) g_{ab} R.
"""
function symbolic_einstein end

"""
    symbolic_kretschmann(Riem, g, ginv, dim::Int) -> Any

Compute Kretschmann scalar K = R_{abcd} R^{abcd}.
"""
function symbolic_kretschmann end

"""
    symbolic_curvature_from_metric(sm::SymbolicMetric) -> NamedTuple

Compute all curvature quantities from a `SymbolicMetric`.
Returns `(; Gamma, Riem, Ric, R, G, K)`.
"""
function symbolic_curvature_from_metric end
