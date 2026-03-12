#= Symmetry ansatz types for metric reduction.

These types describe spacetime symmetries that constrain the metric tensor.
Each ansatz represents a group of isometries (Killing vectors) that the metric
must respect, reducing the number of independent components.

Currently this file defines only the type hierarchy and storage.
Metric generation logic (producing the reduced metric from an ansatz)
will be added in a future module.
=#

"""
    SymmetryAnsatz

Abstract supertype for all spacetime symmetry ansaetze.

A `SymmetryAnsatz` encodes a set of isometries (Killing vector fields) that
a metric tensor must respect. Concrete subtypes store the manifold and any
parameters needed to specify the symmetry (e.g. axis of rotation, spatial
curvature parameter).

Subtypes:
- [`SphericalSymmetry`](@ref)  -- SO(3) rotational invariance
- [`AxialSymmetry`](@ref)      -- U(1) axial rotation
- [`StaticSymmetry`](@ref)     -- timelike Killing vector (time-independence)
- [`HomogeneousIsotropy`](@ref) -- FLRW-type spatial homogeneity and isotropy
"""
abstract type SymmetryAnsatz end

# ─────────────────────────────────────────────────────────────
# Concrete subtypes
# ─────────────────────────────────────────────────────────────

"""
    SphericalSymmetry(manifold::Symbol)

SO(3) rotational symmetry -- the manifold admits three spacelike Killing vectors
generating the rotation group.  This is the symmetry of the Schwarzschild and
Reissner-Nordstroem solutions.

# Fields
- `manifold::Symbol` -- name of the registered manifold

# Example
```julia
ans = SphericalSymmetry(:M4)
```
"""
struct SphericalSymmetry <: SymmetryAnsatz
    manifold::Symbol
end

"""
    AxialSymmetry(manifold::Symbol; axis::Symbol = :z)

U(1) axial rotation symmetry -- the manifold admits a single spacelike Killing
vector generating rotations about a distinguished axis.  This is the symmetry
of the Kerr solution (combined with stationarity).

# Fields
- `manifold::Symbol` -- name of the registered manifold
- `axis::Symbol`     -- label for the symmetry axis (default `:z`)

# Example
```julia
ans = AxialSymmetry(:M4)          # axis = :z
ans = AxialSymmetry(:M4; axis=:y)
```
"""
struct AxialSymmetry <: SymmetryAnsatz
    manifold::Symbol
    axis::Symbol
end

AxialSymmetry(manifold::Symbol; axis::Symbol = :z) = AxialSymmetry(manifold, axis)

"""
    StaticSymmetry(manifold::Symbol; time_coord::Symbol = :t)

Time-translation symmetry -- the manifold admits a timelike Killing vector
field, so the metric components are independent of the time coordinate.
Equivalently, there exists a hypersurface-orthogonal timelike Killing field.

# Fields
- `manifold::Symbol`    -- name of the registered manifold
- `time_coord::Symbol`  -- label for the time coordinate (default `:t`)

# Example
```julia
ans = StaticSymmetry(:M4)
ans = StaticSymmetry(:M4; time_coord=:tau)
```
"""
struct StaticSymmetry <: SymmetryAnsatz
    manifold::Symbol
    time_coord::Symbol
end

StaticSymmetry(manifold::Symbol; time_coord::Symbol = :t) = StaticSymmetry(manifold, time_coord)

"""
    HomogeneousIsotropy(manifold::Symbol; curvature::Symbol = :k)

Spatial homogeneity and isotropy -- the spatial sections are maximally symmetric
three-dimensional spaces of constant curvature.  This is the symmetry underlying
Friedmann-Lemaitre-Robertson-Walker (FLRW) cosmologies.

The curvature parameter `k` labels the spatial curvature:
- `k = +1` : closed (spherical) spatial sections
- `k =  0` : flat spatial sections
- `k = -1` : open (hyperbolic) spatial sections

# Fields
- `manifold::Symbol`   -- name of the registered manifold
- `curvature::Symbol`  -- symbolic name for the spatial curvature parameter (default `:k`)

# Example
```julia
ans = HomogeneousIsotropy(:M4)
ans = HomogeneousIsotropy(:M4; curvature=:K)
```
"""
struct HomogeneousIsotropy <: SymmetryAnsatz
    manifold::Symbol
    curvature::Symbol
end

HomogeneousIsotropy(manifold::Symbol; curvature::Symbol = :k) = HomogeneousIsotropy(manifold, curvature)

# ─────────────────────────────────────────────────────────────
# Show methods
# ─────────────────────────────────────────────────────────────

Base.show(io::IO, s::SphericalSymmetry) =
    print(io, "SphericalSymmetry($(s.manifold))")

Base.show(io::IO, s::AxialSymmetry) =
    print(io, "AxialSymmetry($(s.manifold), axis=$(s.axis))")

Base.show(io::IO, s::StaticSymmetry) =
    print(io, "StaticSymmetry($(s.manifold), time_coord=$(s.time_coord))")

Base.show(io::IO, s::HomogeneousIsotropy) =
    print(io, "HomogeneousIsotropy($(s.manifold), curvature=$(s.curvature))")
