# Invar Database Design — TGR-ed9.1

Research output for the precomputed invariant database (Invar Epic 3).

## Problem Statement

Store ~645K precomputed curvature invariant relations for the Invar
6-level simplification pipeline. Each relation is a linear combination
of RInv canonical forms that equals zero. The database must support:

- Fast lookup by (degree, case, step) — the "case" encodes derivative
  order and Riemann factor count
- Both algebraic (non-derivative) and differential invariants
- Dimension-dependent subsets (d=4, d=5, ...)
- Dual invariants (DualRInv with Levi-Civita factors)

## Prior Art

### xAct/Invar (Mathematica)

**Approach:** Flat files in a directory hierarchy.

```
Riemann/
  1/RInv-0_0-1.m      # Step 1, case {0,0}
  2/RInv-0_0-2.m       # Step 2, case {0,0}
  5_4/RInv-0_0-5_4.m   # Step 5, dim=4, case {0,0}
  6_4/RInv-0_0-6_4.m   # Step 6, dim=4
```

Each file contains Mathematica expressions: reduction rules mapping
non-canonical invariants to linear combinations of canonical ones.
**Lazy loading** with memoization (`ProtSet`). Files are small (KB each),
many files (hundreds).

**Pros:** Simple, human-readable, no dependencies.
**Cons:** Mathematica-specific format, many small files, no indexing.

### Integralis (DuckDB)

**Approach:** DuckDB database (679 MB for 128K integrals) with:
- S-expression serialization for math ASTs
- SHA-256 structural hashing for O(1) exact lookup
- Numerical fingerprinting (16-dim Float64) for fuzzy matching
- 3-table schema: objects + provenance + verifications
- L0–L4 verification levels

**Pros:** SQL queries, columnar analytics, single-file distribution,
proven at 128K+ entries, rich metadata.
**Cons:** DuckDB.jl is a runtime dependency (~50 MB), overkill for
static lookup tables, SQL overhead for simple key→value access.

### Abstractfeld.jl (Design Only)

**Approach:** Unified S-expression IR flowing through Julia ↔ Lean4.
Inherits Integralis schema. Not yet implemented.

**Key insight:** S-expressions are the natural serialization for
mathematical ASTs — unambiguous, deterministic, parseable in <100 LOC.

## Options Analysis

### Option A: DuckDB

Store relations in a DuckDB file shipped as a build artifact.

```julia
# Schema
CREATE TABLE invariant_relations (
    id INTEGER PRIMARY KEY,
    degree INTEGER NOT NULL,        -- number of Riemann factors
    case_key VARCHAR NOT NULL,      -- e.g., "0_0" for {0,0}
    step INTEGER NOT NULL,          -- simplification level (1-6)
    dim INTEGER,                    -- NULL for dimension-independent
    is_dual BOOLEAN DEFAULT FALSE,
    lhs_contraction INTEGER[],      -- RInv contraction (non-canonical)
    rhs_terms TEXT NOT NULL,        -- JSON: [{coeff, contraction}, ...]
);
CREATE INDEX idx_lookup ON invariant_relations(degree, case_key, step, dim);
```

| Criterion          | Score |
|--------------------|-------|
| Lookup speed       | Fast (indexed SQL)    |
| File size          | ~50-100 MB            |
| Dependencies       | DuckDB.jl (~50 MB)   |
| Regeneration       | Easy (SQL INSERT)     |
| Human-readable     | Via SQL queries       |
| Julia integration  | Moderate (SQL bridge) |

**Verdict:** Too heavy for static lookup tables. DuckDB shines for
analytics and evolving datasets, not frozen mathematical relations.

### Option B: JLD2 Binary

Store as serialized Julia Dict in JLD2 format.

```julia
using JLD2
@save "invariants.jld2" relations  # Dict{Tuple{Int,String,Int,Int}, Vector{...}}
```

| Criterion          | Score |
|--------------------|-------|
| Lookup speed       | Fast (in-memory Dict) |
| File size          | ~20-50 MB             |
| Dependencies       | JLD2.jl               |
| Regeneration       | Easy                  |
| Human-readable     | No                    |
| Julia integration  | Excellent             |

**Verdict:** Viable but adds a dependency. Not human-readable.

### Option C: Serialization.jl (stdlib)

Use Julia's built-in `Serialization` module (no external deps).

```julia
using Serialization
serialize("invariants.bin", relations)
# Later:
relations = deserialize("invariants.bin")
```

| Criterion          | Score |
|--------------------|-------|
| Lookup speed       | Fast (in-memory Dict) |
| File size          | ~20-50 MB             |
| Dependencies       | None (stdlib)         |
| Regeneration       | Easy                  |
| Human-readable     | No                    |
| Julia integration  | Excellent             |
| Version stability  | Poor (format changes) |

**Verdict:** No deps, but Serialization format is not stable across
Julia versions. Risk of breakage on Julia upgrades.

### Option D: Compiled Julia Dict Literals

Generate .jl files with `const` Dict/Vector literals, one per case.

```julia
# src/invariants/db/rinv_0_0.jl (auto-generated)
const _RINV_RELATIONS_0_0 = Dict{Int, Vector{Tuple{Rational{Int}, Vector{Int}}}}(
    # step => [(coefficient, canonical_contraction), ...]
    2 => [
        (1//1, [5,6,7,8,1,2,3,4]),   # Kretschmann
        (-4//1, [3,4,7,8,1,2,5,6]),  # -4 * Ric²
        (1//1, [2,1,4,3,6,5,8,7]),   # + R²
    ],
    # ...
)
```

| Criterion          | Score |
|--------------------|-------|
| Lookup speed       | Instant (compiled)    |
| File size          | ~30-80 MB source      |
| Dependencies       | None                  |
| Regeneration       | Code generation       |
| Human-readable     | Yes (Julia source)    |
| Julia integration  | Perfect               |
| Version stability  | Excellent             |

**Verdict:** Best option. Zero deps, version-stable, human-readable,
instant after compilation. This is what xAct does (Mathematica source
files), just in Julia.

### Option E: Lazy-loaded Julia files (xAct style)

Like Option D but with lazy loading — only `include()` a file when
that case/step is first requested.

```julia
# src/invariants/database.jl
const _LOADED_CASES = Dict{Tuple{Int,String,Int}, Any}()

function get_relations(degree, case_key, step; dim=nothing)
    key = (degree, case_key, step)
    if !haskey(_LOADED_CASES, key)
        filename = _relation_filename(degree, case_key, step, dim)
        _LOADED_CASES[key] = include(filename)
    end
    _LOADED_CASES[key]
end
```

| Criterion          | Score |
|--------------------|-------|
| Lookup speed       | Fast (cached after first load) |
| File size          | ~30-80 MB total source |
| Dependencies       | None                  |
| Startup impact     | Minimal (lazy)        |
| Regeneration       | Code generation       |
| Human-readable     | Yes                   |
| Julia integration  | Perfect               |

**Verdict:** Best overall. Combines Option D's advantages with lazy
loading to avoid bloating startup time. This is the direct Julia
equivalent of xAct's `ReadInvarPerms` + `ProtSet` pattern.

## Recommendation: Option E (Lazy-loaded Julia files)

### Architecture

```
src/invariants/
  database.jl          # Public API: get_relations(), list_cases()
  db/                  # Auto-generated relation files
    rinv_0.jl          # Degree 1, case {0}
    rinv_0_0.jl        # Degree 2, case {0,0}
    rinv_0_2.jl        # Degree 2, case {0,2}
    rinv_0_0_0.jl      # Degree 3, case {0,0,0}
    ...
    drinv_0_0.jl       # Dual invariants, degree 2
    ...
    dim4/              # Dimension-dependent (steps 5-6)
      rinv_0_0_d4.jl
      ...
scripts/
  generate_invar_db.jl # Generates db/ from algorithmic computation
```

### Data Model

```julia
"""A single reduction relation: non-canonical → linear combination of canonical."""
struct InvarRelation
    lhs::Vector{Int}                              # Non-canonical contraction
    rhs::Vector{Tuple{Rational{Int}, Vector{Int}}} # [(coeff, canonical_contraction), ...]
end

"""Relations for one case at one step."""
struct CaseRelations
    degree::Int
    case_key::String      # e.g., "0_0"
    step::Int
    dim::Union{Int,Nothing}
    n_independent::Int    # Number of independent invariants
    n_dependent::Int      # Number of dependent (reducible) invariants
    relations::Vector{InvarRelation}
end
```

### Public API

```julia
# Look up relations for a specific case
get_invar_relations(degree, case_key, step; dim=nothing) -> CaseRelations

# List all available cases
list_invar_cases(; step=nothing, degree=nothing) -> Vector{NamedTuple}

# Check if an RInv is independent at a given step
is_independent(rinv::RInv, step::Int; dim=nothing) -> Bool

# Reduce an RInv to canonical basis
reduce_rinv(rinv::RInv, step::Int; dim=nothing) -> Vector{Tuple{Rational{Int}, RInv}}
```

### Generation Pipeline

```
1. For each case (degree, derivative orders):
   a. Enumerate all distinct RInv contractions (up to symmetry)
   b. Apply step 1: xperm canonicalization → identify orbits
   c. Apply step 2: Bianchi cyclic relations → find linear dependencies
   d. Apply step 3-4: Differential Bianchi, CovD commutation
   e. Apply step 5: DDIs for each target dimension
   f. Apply step 6: Dual relations
   g. Write Julia source file with const Dict literal

2. Verify: for each relation, check that the linear combination
   actually vanishes when evaluated as a TensorExpr via simplify().
```

### File Size Estimate

Based on xAct's database structure:
- Degree 2: ~10 cases, ~50 relations each → ~5 KB
- Degree 3: ~20 cases, ~500 relations each → ~100 KB
- Degree 4: ~40 cases, ~5000 relations each → ~2 MB
- Degree 5-7: ~100 cases, ~50000 relations each → ~50 MB
- Total: ~50-80 MB of Julia source (compresses to ~5-10 MB)

### Why Not DuckDB?

Integralis uses DuckDB because its data is **live** — new integrals
are ingested, verification levels change, provenance accumulates.
Curvature invariant relations are **frozen mathematics** — they don't
change once computed. A static file is simpler, faster, and has zero
dependencies. If we later need analytics (e.g., "how many degree-5
invariants are independent in d=4?"), we can always load the data into
DuckDB ad-hoc.

### Serialization Format

Following Integralis's lesson, use a **deterministic text format** for
the generated files. RInv contractions are just `Vector{Int}`, so the
Julia Dict literal format is already unambiguous and human-readable:

```julia
# Auto-generated by scripts/generate_invar_db.jl
# Case {0,0} (degree 2, order 4), Step 2 (Bianchi cyclic)
# 3 independent, 3 dependent invariants
const _CASE_0_0_STEP2 = CaseRelations(
    2, "0_0", 2, nothing, 3, 3,
    InvarRelation[
        InvarRelation(
            [5,6,7,8,1,2,3,4],  # dependent: R_{a(cd)b} contraction
            [(1//2, [5,6,7,8,1,2,3,4]), (-1//2, [3,4,7,8,1,2,5,6])]
        ),
        # ...
    ]
)
```

## Next Steps

1. **TGR-ed9.2**: Generate degree-2 database (simplest, ~10 relations)
2. **TGR-ed9.3**: Generate degree-3 database (~100 relations)
3. **TGR-ed9.4**: Generate degree-4 database (~5000 relations)
4. **TGR-ed9.5**: Generate degrees 5-7 (~640K relations)
5. **TGR-ed9.6**: Dual invariant database
6. **TGR-ed9.7-8**: Differential invariant databases
7. **TGR-ed9.9**: Wire into `riemann_simplify` via `InvSimplify`

## References

- Garcia-Parrado & Martin-Garcia (2007), Comp. Phys. Comm. 176:246, Sec 6
- Zakhary & McIntosh (1997), GRG 29:539
- Fulling et al (1992), CQG 9:1151
- Integralis project: /home/tobiasosborne/Projects/Integralis/
- Abstractfeld.jl PRD: /home/tobiasosborne/Projects/Abstractfeld.jl/PRD.md
- xAct/Invar source: reference/xAct/xAct/Invar/Invar.m
