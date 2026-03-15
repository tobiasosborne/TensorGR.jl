# SpinIndex convenience layer for ergonomic spinor index creation.
#
# Wraps TIndex with dotted/undotted tracking and provides short constructors
# that delegate to the existing VBundle-based infrastructure (SL2C / SL2C_dot).
#
# Reference: Penrose & Rindler, Spinors and Space-Time Vol 1 (1984), Ch 2.

"""
    SpinIndex

Convenience type wrapping `TIndex` with spinor-specific operations.
Tracks dotted/undotted status and provides ergonomic constructors.

The underlying representation uses the VBundle approach:
- Undotted indices live on `:SL2C`
- Dotted indices live on `:SL2C_dot`

`SpinIndex` is a thin wrapper that makes the dotted/undotted distinction explicit
and provides a more readable API for spinor computations.

# Fields
- `idx::TIndex` -- the underlying tensor index
- `dotted::Bool` -- `true` for dotted (primed) indices, `false` for undotted
"""
struct SpinIndex
    idx::TIndex
    dotted::Bool

    function SpinIndex(idx::TIndex, dotted::Bool)
        # Validate consistency: dotted flag must match vbundle
        if dotted && idx.vbundle !== :SL2C_dot
            error("SpinIndex: dotted=true but vbundle is $(idx.vbundle), expected :SL2C_dot")
        end
        if !dotted && idx.vbundle !== :SL2C
            error("SpinIndex: dotted=false but vbundle is $(idx.vbundle), expected :SL2C")
        end
        new(idx, dotted)
    end
end

Base.:(==)(a::SpinIndex, b::SpinIndex) = a.idx == b.idx && a.dotted == b.dotted
Base.hash(a::SpinIndex, h::UInt) = hash(a.dotted, hash(a.idx, hash(:SpinIndex, h)))

# ── Constructors ─────────────────────────────────────────────────────────────

"""
    spinor(name::Symbol; up::Bool=true) -> TIndex

Create an undotted spinor index on the SL(2,C) fundamental bundle.

Equivalent to `spin_up(name)` when `up=true`, `spin_down(name)` when `up=false`,
but with a more readable interface.

# Examples
```julia
spinor(:A)            # A^A  (upper undotted)
spinor(:B, up=false)  # A_B  (lower undotted)
```
"""
function spinor(name::Symbol; up::Bool=true)
    pos = up ? Up : Down
    TIndex(name, pos, :SL2C)
end

"""
    spinor_dot(name::Symbol; up::Bool=true) -> TIndex

Create a dotted (primed) spinor index on the SL(2,C) conjugate bundle.

Equivalent to `spin_dot_up(name)` when `up=true`, `spin_dot_down(name)` when `up=false`,
but with a more readable interface.

# Examples
```julia
spinor_dot(:A)            # A^{A'}  (upper dotted)
spinor_dot(:B, up=false)  # A_{B'}  (lower dotted)
```
"""
function spinor_dot(name::Symbol; up::Bool=true)
    pos = up ? Up : Down
    TIndex(name, pos, :SL2C_dot)
end

# ── Predicates ───────────────────────────────────────────────────────────────

"""
    is_undotted(idx::TIndex) -> Bool

Return `true` if `idx` lives in the undotted (unprimed) spinor bundle `:SL2C`.

See also [`is_dotted`](@ref) for dotted indices, [`is_spinor_index`](@ref) for either.
"""
is_undotted(idx::TIndex) = idx.vbundle === :SL2C

"""
    is_dotted_spinor(idx::TIndex) -> Bool

Return `true` if `idx` lives in the dotted (primed) spinor bundle `:SL2C_dot`.

This is functionally identical to [`is_dotted`](@ref) but named to avoid ambiguity
in contexts where "dotted" might refer to other conventions.
"""
is_dotted_spinor(idx::TIndex) = idx.vbundle === :SL2C_dot

# ── Dummy pair generation ────────────────────────────────────────────────────

"""
    spinor_dummy(name::Symbol) -> Tuple{TIndex, TIndex}

Generate a matched undotted dummy pair: one upper and one lower SL2C index
with the same name, suitable for contraction.

# Examples
```julia
A_up, A_dn = spinor_dummy(:A)
# A_up == TIndex(:A, Up, :SL2C)
# A_dn == TIndex(:A, Down, :SL2C)
```
"""
function spinor_dummy(name::Symbol)
    (TIndex(name, Up, :SL2C), TIndex(name, Down, :SL2C))
end

"""
    spinor_dot_dummy(name::Symbol) -> Tuple{TIndex, TIndex}

Generate a matched dotted dummy pair: one upper and one lower SL2C_dot index
with the same name, suitable for contraction.

# Examples
```julia
Ap_up, Ap_dn = spinor_dot_dummy(:Ap)
# Ap_up == TIndex(:Ap, Up, :SL2C_dot)
# Ap_dn == TIndex(:Ap, Down, :SL2C_dot)
```
"""
function spinor_dot_dummy(name::Symbol)
    (TIndex(name, Up, :SL2C_dot), TIndex(name, Down, :SL2C_dot))
end

# ── Pair generation for tensor-spinor decomposition ──────────────────────────

"""
    spinor_pair(name::Symbol) -> Tuple{TIndex, TIndex}

Return an (undotted upper, dotted upper) pair for tensor-spinor decomposition.

This is useful when decomposing a spacetime vector index into its spinor
components via the Infeld-van der Waerden symbol: V^a -> V^{AA'}.

# Examples
```julia
A, Adot = spinor_pair(:A)
# A    == TIndex(:A, Up, :SL2C)
# Adot == TIndex(:A, Up, :SL2C_dot)
```
"""
function spinor_pair(name::Symbol)
    (TIndex(name, Up, :SL2C), TIndex(name, Up, :SL2C_dot))
end
