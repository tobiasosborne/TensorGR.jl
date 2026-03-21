#= RiemannSimplify: Top-level orchestrator for the Invar simplification pipeline.
#
# Applies all 6 levels of the Garcia-Parrado & Martin-Garcia (2007) algorithm
# in sequence for scalar Riemann invariants:
#
#   Level 1: Permutation symmetries (Butler-Portugal / xperm)
#   Level 2: Cyclic symmetry (first Bianchi identity)
#   Level 3: Second (differential) Bianchi identity
#   Level 4: Derivative commutation
#   Level 5: Dimensionally-dependent identities (DDIs)
#   Level 6: Dual invariant product relations
#
# The function is idempotent by design: each level subsumes all lower levels,
# so a second application produces the same result.
#
# Reference: Garcia-Parrado & Martin-Garcia, Comp. Phys. Comm. 176 (2007) 246, Sec 5.
=#

"""
    riemann_simplify(expr::TensorExpr;
                      registry::TensorRegistry=current_registry(),
                      dim::Union{Int,Nothing}=nothing,
                      maxlevel::Int=6,
                      covd::Symbol=:D) -> TensorExpr

Top-level simplification of scalar Riemann invariants using the full
Invar 6-level algorithm (Garcia-Parrado & Martin-Garcia 2007, Sec 5).

Applies simplification levels 1 through `maxlevel` in sequence:

| Level | Operation                              |
|-------|----------------------------------------|
| 1     | Permutation symmetries (xperm)         |
| 2     | Cyclic symmetry (first Bianchi)        |
| 3     | Differential Bianchi identity          |
| 4     | Covariant derivative commutation       |
| 5     | Dimensionally-dependent identities     |
| 6     | Dual invariant product relations       |

Each level subsumes all previous levels, so applying level N automatically
includes levels 1 through N-1.

# Arguments
- `expr`: tensor expression to simplify (may be a scalar invariant,
  a sum of invariants, or a general curvature expression)
- `registry::TensorRegistry`: the registry containing manifold/metric/curvature
  definitions (default: `current_registry()`)
- `dim::Union{Int,Nothing}=nothing`: manifold dimension. When `nothing`,
  DDIs (level 5) are skipped since they require a specific dimension.
  When an integer is given, DDIs for that dimension are applied.
- `maxlevel::Int=6`: highest simplification level to apply (1-6).
  Levels above `maxlevel` are skipped.
- `covd::Symbol=:D`: covariant derivative name for levels 3-4 (derivative
  commutation and differential Bianchi identity)

# Returns
A simplified `TensorExpr`. The result is idempotent: applying
`riemann_simplify` to the output again produces the same expression.

# Behavior
- Scalars (`TScalar`) and expressions with no curvature content pass through
  unchanged (idempotent).
- When `dim` is `nothing` and `maxlevel >= 5`, levels 5-6 are capped at
  level 4 since DDIs require a known dimension.
- When `dim` is specified, all levels up to `maxlevel` are applied.

# Ground truth
Garcia-Parrado & Martin-Garcia (2007) Sec 5; Zakhary & McIntosh (1997).

# Examples
```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g registry=reg
define_curvature_tensors!(reg, :M4, :g)
@covd D on=M4 metric=g registry=reg

# Kretschmer scalar is already canonical
K = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)]) *
    Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
riemann_simplify(K; registry=reg)  # canonical form

# Gauss-Bonnet vanishes in d=4
gb = gauss_bonnet_ddi(; metric=:g, registry=reg)
riemann_simplify(gb; registry=reg, dim=4)  # => 0

# Stop at level 2 (only permutation + Bianchi)
riemann_simplify(K; registry=reg, maxlevel=2)
```

See also: [`simplify_level1`](@ref), [`simplify_level2`](@ref),
[`simplify_level3`](@ref), [`simplify_level4`](@ref),
[`simplify_level5`](@ref), [`simplify_level6`](@ref)
"""
function riemann_simplify(expr::TensorExpr;
                           registry::TensorRegistry=current_registry(),
                           dim::Union{Int,Nothing}=nothing,
                           maxlevel::Int=6,
                           covd::Symbol=:D)
    # Validate maxlevel
    (1 <= maxlevel <= 6) ||
        error("riemann_simplify: maxlevel must be between 1 and 6, got $maxlevel")

    # Determine the effective maximum level.
    # DDIs (level 5) require a known dimension; if dim is nothing, cap at 4.
    effective_maxlevel = maxlevel
    if dim === nothing && effective_maxlevel >= 5
        effective_maxlevel = 4
    end

    # Scalar pass-through: nothing to simplify
    expr isa TScalar && return expr

    # Dispatch to the appropriate simplification level.
    # Each level N already includes levels 1 through N-1, so we only
    # need to call the highest requested level.
    if effective_maxlevel == 1
        simplify_level1(expr; registry=registry)
    elseif effective_maxlevel == 2
        simplify_level2(expr; registry=registry)
    elseif effective_maxlevel == 3
        simplify_level3(expr; covd=covd, registry=registry)
    elseif effective_maxlevel == 4
        simplify_level4(expr; covd=covd, registry=registry)
    elseif effective_maxlevel == 5
        simplify_level5(expr; covd=covd, dim=dim, registry=registry)
    else  # effective_maxlevel == 6
        simplify_level6(expr; covd=covd, dim=dim, registry=registry)
    end
end
