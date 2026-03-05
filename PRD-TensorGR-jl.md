# TensorGR.jl — Product Requirements Document

**Abstract tensor computer algebra for general relativity in Julia**

Version 0.1 — March 2026, rev 2
Status: Draft

---

## 1. Problem statement

The xAct ecosystem (Martín-García et al., 2004–2025) is the de facto standard for abstract tensor computer algebra in general relativity. It has been used in over 1000 published papers. It is also trapped behind a proprietary runtime: every package in the ecosystem — xTensor, xPert, xPand, xTras, PSALTer, FieldsX, Hamilcar — requires a Mathematica license (~€400/yr individual, more institutional).

The algorithms are all published. The computational core — `xperm.c`, implementing the Butler-Portugal algorithm for index canonicalisation — is GPL C, explicitly designed for linking from other systems. The higher-level packages are GPL Wolfram Language source, readable and translatable. The Mathematica kernel provides three things: (1) a term-rewriting engine for indexed expressions, (2) polynomial/rational scalar algebra, (3) an expression tree data structure. Julia provides all three natively: its AST is an S-expression, its metaprogramming subsystem is a term rewriter, and scalar algebra is available via multiple backends.

The porting cost is bounded. The target codebase is ~10,500 lines of Julia plus ~300 lines of `ccall` wrapper around the existing `xperm.c`. At demonstrated agent-first development rates of ~2000 lines/hour of tested code, the generation time is ~5-6 hours. With human steering at 2-3 hours/day, the project completes in roughly two weeks of calendar time.

---

## 2. Goals and non-goals

### Goals

G1. Provide a fully open-source, Mathematica-free implementation of abstract tensor algebra for GR, at feature parity with xTensor + xPert + xPand for linearised gravity applications.

G2. Reuse `xperm.c` (Butler-Portugal canonicaliser) directly via FFI — do not reimplement.

G3. Represent tensor expressions via a small typed AST hierarchy (`abstract type TensorExpr end`) that is idiomatic Julia: dispatchable, compiler-visible, and extensible by downstream packages. A convenience macro layer exploits Julia's homoiconic AST for ergonomic construction.

G4. Achieve correctness against known xAct outputs and published results, verified by an automated test suite. **TDD is the law**: tests are written before implementation, always.

G5. Enable agent-first development: the codebase should be structured so that an LLM agent (Claude Code) can generate, test, and extend components with minimal human steering.

G6. Publish as a registered Julia package with documentation, enabling community contribution.

G7. Write idiomatic Julia: use multiple dispatch, the type system, and package conventions (method extension, `show` overloads, `AbstractTrees` compatibility, etc.). **However**: idiomaticity is subordinate to getting the code running. If a typed abstraction creates a blocking design problem, fall back to a working untyped implementation and file a refactor issue. Working code with a known-ugly spot beats a beautiful type hierarchy that doesn't compute Riemann tensors.

### Non-goals

NG1. Full xAct feature parity at v0.1. Specifically: spinor calculus (Spinors package), Fierz identities (FieldsX), post-Newtonian formalism (xPPN), exterior calculus (xTerior), and component calculations (xCoba) are out of scope for initial release.

NG2. GUI or notebook interface. The package is library-only, used from the Julia REPL, scripts, or Pluto/Jupyter notebooks.

NG3. Mathematica interop. We do not attempt to call Mathematica or import `.nb` files.

NG4. GPU acceleration. All computation is single-threaded CPU at v0.1.

NG5. Formal verification. Correctness is established by testing, not by Lean/Mathlib proofs.

---

## 3. Architecture

### 3.1 Design principle: typed AST with macro ergonomics

The central architectural decision: tensor expressions are represented by a small closed type hierarchy rooted at `abstract type TensorExpr end`. This hierarchy mirrors the structure of Julia's own `Expr` type — it is an S-expression — but it is typed, dispatchable, and compiler-visible.

A convenience macro constructs these types from a LaTeX-like notation, translating Julia `Expr` at compile time into typed AST nodes at runtime:

```julia
# User writes:
@tensor R[-a, -b, -c, -d]              # → Tensor(:R, [down(:a), ...])
@tensor g[a, b] * R[-a, -c, -b, -d]    # → TProduct([Tensor(:g,...), Tensor(:R,...)])
@tensor ∂[-a, T[b, -c]]                # → TDeriv(down(:a), Tensor(:T,...))
```

This gives the ergonomic benefit of the Lisp representation (readable, writable) with the engineering benefit of the type system (dispatch, inference, extensibility).

**Escape hatch**: If a typed abstraction creates a design impasse — e.g., the type hierarchy can't express some intermediate rewriting state without a combinatorial explosion of types — the code falls back to raw `Expr` manipulation for that specific subsystem, guarded by conversion functions `to_expr(::TensorExpr)::Expr` and `from_expr(::Expr)::TensorExpr`. A `@refactor` comment marks the spot. Working code with a local escape hatch beats a clean type system that blocks progress.

The type hierarchy:

```julia
abstract type TensorExpr end

struct Tensor <: TensorExpr
    name::Symbol
    indices::Vector{TIndex}
end

struct TProduct <: TensorExpr
    scalar::Rational{Int}
    factors::Vector{TensorExpr}
end

struct TSum <: TensorExpr
    terms::Vector{TensorExpr}
end

struct TDeriv <: TensorExpr
    index::TIndex
    arg::TensorExpr
end

struct TScalar <: TensorExpr
    val::Any    # Rational, Symbol, Expr — scalar coefficient sublayer
end

struct TIndex
    name::Symbol
    position::IndexPosition  # @enum IndexPosition Up Down
end

# Convenience constructors
up(s::Symbol) = TIndex(s, Up)
down(s::Symbol) = TIndex(s, Down)
```

Index convention: `up(:a)` = contravariant, `down(:a)` = covariant. Dummy indices are paired (one up, one down, same name) and subject to automatic renaming. Free indices define the tensor character of the expression.

**Why not raw `Expr`**: Raw `Expr` at runtime is untyped. You cannot dispatch `contract(::Metric, ::Riemann)` — you'd write `if expr.args[1] == :g` cascades. Other packages can't extend your types via method addition. The compiler infers `Any` everywhere. IDE autocompletion is dead. The extra ~500 lines of type definitions save thousands of lines of defensive checking downstream and make community contribution viable.

**Why not Symbolics.jl types**: Symbolics.jl represents `x + y*z` as `Add`/`Mul`/`Sym` nodes typed by Julia's number system. A tensor expression carries index slots with identity, scope, and permutation-group structure. These are fundamentally different kinds of symbolic object. The tensor layer should own its representation; the scalar sublayer (inside `TScalar.val`) is where Symbolics.jl or SymEngine.jl optionally connects.

### 3.2 Layer diagram

```
┌─────────────────────────────────────────────────┐
│  Layer 4: Applications                          │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │ Linearise│ │   SVT    │ │  Particle       │ │
│  │ (xPert)  │ │ (xPand)  │ │  spectrum       │ │
│  └──────────┘ └──────────┘ │  (future)       │ │
│                             └─────────────────┘ │
├─────────────────────────────────────────────────┤
│  Layer 3: GR Objects                            │
│  Manifold, Metric, CovariantDerivative,         │
│  Riemann, Ricci, Einstein, Weyl, Schouten       │
│  Bianchi identities, metric compatibility       │
├─────────────────────────────────────────────────┤
│  Layer 2: Tensor Algebra Engine                 │
│  Index management, contraction, raising/lower,  │
│  Leibniz rule, product rule, symmetry handling, │
│  canonicalisation (via xperm.c), IBP            │
├─────────────────────────────────────────────────┤
│  Layer 1: Foundations                           │
│  TensorRegistry (metadata Dict),               │
│  Expr manipulation primitives,                  │
│  Scalar algebra (expand/collect/factor),        │
│  xperm.c FFI wrapper                            │
└─────────────────────────────────────────────────┘
```

### 3.3 Key data structures

**TensorRegistry** — A module-level `Dict{Symbol, TensorProperties}`. Tensors carry their name and indices; their *properties* live here, not in the type. This avoids parametric type explosion while keeping runtime data separate from algebraic identity.

```julia
struct TensorProperties
    manifold::Symbol                    # e.g. :M4
    rank::Tuple{Int,Int}               # (contravariant, covariant)
    symmetries::Vector{PermGroup}      # passed to xperm.c
    dependencies::Vector{Symbol}       # e.g. [:x, :y, :z, :t]
    weight::Int                        # density weight
    options::Dict{Symbol, Any}         # extensible
end
```

**ManifoldProperties** — dimension, metric symbol, derivative symbol, index alphabet:

```julia
struct ManifoldProperties
    dim::Int
    metric::Union{Symbol, Nothing}
    derivative::Union{Symbol, Nothing}
    indices::Vector{Symbol}            # [:a, :b, :c, :d, ...]
end
```

**Dispatch via registry lookup**: Since properties live in the registry rather than the type system, dispatch on *specific* tensors (metric vs Riemann vs user-defined) uses trait-like patterns:

```julia
# Registry lookup drives behaviour
function contract(p::TProduct)
    for (i, t) in enumerate(p.factors)
        props = registry[t.name]
        if props.options[:is_metric]
            return contract_metric(p, i)
        end
    end
    ...
end
```

This is a deliberate compromise: full dispatch on tensor identity would require parametric types like `Tensor{:Riemann}`, which creates combinatorial problems when products involve arbitrary collections of tensors. The registry pattern is how xAct handles it too — `UpValues` on symbols are morally a `Dict`.

### 3.4 Scalar sublayer: no Symbolics.jl in the tensor layer

Symbolics.jl is not a dependency of the core package. The tensor layer operates on `TensorExpr` nodes. Scalar coefficients live inside `TScalar.val` and are manipulated by a minimal built-in algebra (~300 lines: `expand`, `collect`, `factor` on `Expr` trees of polynomial arithmetic).

When the tensor layer produces a fully-contracted scalar expression (e.g., after going to Fourier space), the user may optionally convert to a Symbolics.jl expression for heavier CAS operations:

```julia
using Symbolics
@variables k² p² β
scalar_expr = to_symbolics(tensor_result)  # TScalar → Symbolics.Num
simplify(scalar_expr)
```

Similarly for SymEngine.jl (faster C++ backend):

```julia
using SymEngine
scalar_expr = to_symengine(tensor_result)  # TScalar → SymEngine.Basic
expand(scalar_expr)
```

These bridges are thin (~50 lines each), optional, and loaded via Julia's package extension mechanism (`ext/TensorGRSymbolicsExt.jl`). The core package has zero non-stdlib Julia dependencies beyond the C compiler for `xperm.c`.

---

## 4. Component specifications

### 4.1 Layer 1: xperm.c FFI wrapper

**Source**: `xperm.c` from xact.es/xPerm/, GPL, ~2000 lines of C.

**Interface**: Expose the following C functions via `ccall`:

```julia
# Core: Schreier-Sims algorithm
schreier_sims(base, gs, n) → StrongGeneratingSet

# Core: canonical permutation (Butler-Portugal)
canonical_perm(perm, base, gs, frees, dummies, n) → canonical_perm

# Auxiliary
double_coset_rep(perm, base_S, gs_S, base_D, gs_D, n) → rep
```

**Design**:
- Compile `xperm.c` to a shared library `libxperm.so` at package build time (via `BinaryBuilder.jl` or `Pkg` build hook).
- Julia-side types: `Perm` (vector of `Int32`), `SGS` (base + generating set).
- Memory: allocate Julia arrays, pass pointers to C. No manual `malloc`.

**Estimated size**: 300 lines Julia.

**Tests**: Reproduce all examples from Butler's book that are included in `ButlerExamples.nb`. Reproduce xPerm timing benchmarks.

### 4.2 Layer 1: TensorExpr primitives

Functions that operate on the typed AST representing tensor expressions.

```julia
# Tree traversal
walk(f, expr::TensorExpr)::TensorExpr   # recursive map over subtrees
substitute(expr, old => new)             # structural substitution
children(expr::TensorExpr)               # AbstractTrees.jl compatible

# Index operations (dispatch on TensorExpr subtypes)
indices(t::Tensor)::Vector{TIndex}       # indices of single tensor
indices(p::TProduct)::Vector{TIndex}     # all indices in product
free_indices(expr::TensorExpr)::Vector{TIndex}   # uncontracted
dummy_pairs(expr::TensorExpr)::Vector{Tuple{TIndex,TIndex}}

# Index management
rename_dummy(expr, old::Symbol, new::Symbol)
fresh_index(used::Set{Symbol})::Symbol
ensure_no_dummy_clash!(a::TensorExpr, b::TensorExpr)

# Algebraic operations (return TensorExpr)
tensor_add(a::TensorExpr, b::TensorExpr)::TSum
tensor_mul(a::TensorExpr, b::TensorExpr)::TProduct
tensor_contract(expr::TProduct, i::TIndex, j::TIndex)::TensorExpr
contract_metrics(expr::TProduct)::TensorExpr

# Simplification
expand_products(expr::TensorExpr)::TensorExpr    # distribute over sums
collect_terms(expr::TSum)::TSum                   # combine like terms
canonicalize(expr::TProduct)::TProduct            # via xperm.c
```

**Critical subtlety — dummy index scoping**: When multiplying two tensor expressions, dummy indices in each factor are independent. If both contain a dummy `a`, one set must be renamed before forming the product. This is the #1 source of bugs in tensor CAS implementations.

Strategy: `tensor_mul` calls `ensure_no_dummy_clash!` before constructing the `TProduct`. The function renames dummies in the second argument as needed using `fresh_index`. This is enforced at the constructor level — there is no way to create a `TProduct` with clashing dummies.

**`Base` overloads** (idiomatic Julia):

```julia
Base.:*(a::TensorExpr, b::TensorExpr) = tensor_mul(a, b)
Base.:+(a::TensorExpr, b::TensorExpr) = tensor_add(a, b)
Base.:-(a::TensorExpr) = TProduct(-1//1, [a])
Base.show(io::IO, t::Tensor) = # LaTeX-like pretty printing
```

**Estimated size**: 1500 lines Julia.

**Tests**: Unit tests for each function. Integration test: verify $(g^{ab} g_{bc} = \delta^a_c)$ and $R^a{}_{bac} = R_{bc}$.

### 4.3 Layer 1: Scalar algebra (minimal)

A lightweight polynomial/rational function manipulator operating on `Expr` trees.

```julia
scalar_expand(expr)     # distribute products over sums
scalar_collect(expr, x) # collect powers of x
scalar_factor(expr)     # basic factorisation
scalar_cancel(expr)     # cancel common factors in ratios
scalar_subst(expr, rules::Dict)  # substitute values
```

If the user needs heavy-duty simplification (trigonometric, special functions, etc.), they load Symbolics.jl or SymEngine.jl and call `to_symbolics(expr)`.

**Estimated size**: 400 lines Julia.

**Tests**: Standard polynomial identities. Verify $\det M = 8(1-3\beta)k^4 p^4$ from the fourth-derivative gravity calculation.

### 4.4 Layer 2: Tensor algebra engine

The core rewriting system. This is the xTensor equivalent.

#### 4.4.1 Registry and declarations

```julia
# Declare a manifold
@manifold M4 dim=4 indices=[a,b,c,d,e,f,g,h,i,j,k,l,m,n]

# Declare a metric
@metric g on=M4 signature=(-,+,+,+)

# Declare a tensor
@tensor R[-a,-b,-c,-d] on=M4 symmetries=RiemannSymmetry()

# Declare a covariant derivative
@derivative ∇ on=M4 metric=g
```

Macros expand to `TensorRegistry` insertions. Symmetry objects encode the permutation group generators for `xperm.c`.

#### 4.4.2 Contraction engine

Given a `TProduct`, find all contracted index pairs (one `Up`, one `Down`, same `name`) and:

1. If a pair involves a metric (checked via `registry[t.name].options[:is_metric]`), eliminate the metric factor and raise/lower the partner index on the other tensor.
2. If a pair involves a Kronecker delta, substitute.
3. Otherwise, mark as a trace to be handled by `xperm.c` canonicalisation.

The contraction engine runs to fixed point: contracting a metric may expose a new contraction.

```julia
function contract_all(p::TProduct)::TensorExpr
    result = p
    while true
        next = contract_one_metric(result)
        next == result && break
        result = next
    end
    return result
end
```

#### 4.4.3 Derivative expansion

The Leibniz rule for $\nabla_a (T \cdot S) = (\nabla_a T) \cdot S + T \cdot (\nabla_a S)$.

For the covariant derivative, the connection terms:

$$\nabla_a T^{b_1 \ldots}{}_{c_1 \ldots} = \partial_a T^{b_1 \ldots}{}_{c_1 \ldots} + \Gamma^{b_1}_{a d} T^{d \ldots}{}_{c_1 \ldots} + \ldots - \Gamma^{d}_{a c_1} T^{b_1 \ldots}{}_{d \ldots} - \ldots$$

At the linearised level (flat background), $\nabla_a = \partial_a$ and connection terms vanish. But the general implementation is needed for Layer 3 (Riemann from commuting covariant derivatives).

#### 4.4.4 Canonicalisation

Given a `TProduct`:

1. Extract the permutation representation: map each slot position to an index, encode symmetries (from `TensorRegistry`) as permutation group generators.
2. Call `xperm.c`'s `canonical_perm` via `ccall` to find the canonical form.
3. Apply the resulting permutation to reorder index slots on each `Tensor` factor.
4. Return canonicalised `TProduct` with updated scalar sign.

This is the direct translation of what xTensor does, but with `ccall` to C instead of `MathLink`.

#### 4.4.5 Integration by parts

For action-level manipulations: move derivatives from one field to another, picking up boundary terms (discarded) and signs.

```julia
ibp(expr, field)  # integrate ∂ off `field` in a Lagrangian density
```

Operates on expressions under an implicit $\int d^4x$.

**Estimated total size, Layer 2**: 2500 lines Julia.

**Tests**: $g^{ab}g_{bc} = \delta^a_c$. Bianchi identity $R_{a[bcd]} = 0$. Contracted Bianchi $\nabla^a G_{ab} = 0$. Commutator of covariant derivatives yields Riemann. Metric compatibility $\nabla_a g_{bc} = 0$.

### 4.5 Layer 3: GR objects

Definitions of standard GR tensors and their properties, registered via macros.

```julia
# Riemann tensor
@tensor Riem[-a,-b,-c,-d] on=M4 symmetries=RiemannSymmetry()

# Riemann symmetries (for xperm.c):
# R_{abcd} = -R_{bacd}  (antisym in first pair)
# R_{abcd} = -R_{abdc}  (antisym in second pair)
# R_{abcd} = R_{cdab}   (pair symmetry)
# R_{a[bcd]} = 0         (algebraic Bianchi — multi-term, handled separately)

# Ricci tensor: R_{ac} = R^b{}_{abc}
@derived Ric[-a,-c] := trace(Riem, 1, 3)  # contract 1st and 3rd index

# Ricci scalar: R = g^{ac} R_{ac}
@derived RicScalar := trace(Ric, 1, 2)

# Einstein tensor: G_{ab} = R_{ab} - (1/2) g_{ab} R
@derived Ein[-a,-b] := Ric[-a,-b] - 1//2 * g[-a,-b] * RicScalar

# Weyl tensor (4D)
@derived Weyl[-a,-b,-c,-d] := weyl_decomposition(Riem, Ric, RicScalar, g)

# Schouten tensor
@derived Sch[-a,-b] := 1//2 * (Ric[-a,-b] - 1//6 * g[-a,-b] * RicScalar)
```

Multi-term symmetry handling: The algebraic Bianchi identity $R_{a[bcd]} = 0$ is not a permutation symmetry — it's a relation among terms. Implemented as a rewrite rule:

```julia
# If expr contains R[a,b,c,d] + R[a,c,d,b] + R[a,d,b,c], replace with 0
@rule bianchi_algebraic R[a_,b_,c_,d_] + R[a_,c_,d_,b_] + R[a_,d_,b_,c_] => 0
```

Similarly, the differential Bianchi identity $\nabla_{[e} R_{ab]cd} = 0$.

**Estimated size**: 800 lines Julia.

**Tests**: Verify all standard trace identities. Weyl tensor is traceless. Contracted Bianchi. In 4D: number of independent Riemann components = 20, Weyl = 10.

### 4.6 Layer 4a: Perturbation theory (xPert equivalent)

Linearisation of GR tensors around a background.

#### Core function

```julia
linearize(expr, g => η + ε*h, order=1)
```

Given a tensor expression involving the metric `g`, substitute `g = η + ε*h` (or any background + perturbation), expand to the specified order in `ε`, and return the result.

#### Explicit formulas

At first order, the perturbation formulas are combinatorial (Brizuela, Martín-García, Mena Marugán 2009). They are hard-coded for efficiency:

```julia
# Linearised Christoffel
δΓ[a,-b,-c] := 1//2 * g[a,d] * (∂[-b, h[-c,-d]] + ∂[-c, h[-b,-d]] - ∂[-d, h[-b,-c]])

# Linearised Riemann (flat background)
δR[-a,-b,-c,-d] := 1//2 * (∂[-b,∂[-c, h[-a,-d]]] + ∂[-a,∂[-d, h[-b,-c]]]
                         - ∂[-a,∂[-c, h[-b,-d]]] - ∂[-b,∂[-d, h[-a,-c]]])

# Linearised Ricci
δRic[-a,-b] := 1//2 * (∂[c,∂[-a, h[-b,-c]]] + ∂[c,∂[-b, h[-a,-c]]]
                      - ∂[-a,∂[-b, h]] - □(h[-a,-b]))

# Linearised Ricci scalar
δR := ∂[a,∂[b, h[-a,-b]]] - □(h)
```

where `h = η^{ab} h_{ab}` and `□ = η^{ab} ∂_a ∂_b`.

Higher-order perturbation (n=2,3,...) uses the general combinatorial formula from xPert. This is a sum over partitions; the formula is explicit and documented in the xPert paper (arXiv:0807.0824). Implemented as a recursive function.

#### Gauge transformations

```julia
gauge_transform(h[-a,-b], ξ) := ∂[-a, ξ[-b]] + ∂[-b, ξ[-a]]
```

Verify gauge invariance of linearised curvature tensors.

**Estimated size**: 2000 lines Julia.

**Tests**: Reproduce linearised Riemann, Ricci, scalar for flat background. Verify gauge invariance. Reproduce second-order perturbation of Ricci scalar around Schwarzschild (published in xPert paper, Table 1). Verify against xPert timings for orders n=1..5.

### 4.7 Layer 4b: SVT decomposition (xPand equivalent)

3+1 splitting and scalar-vector-tensor decomposition for cosmological/gravitational perturbation theory.

#### 3+1 foliation

Given a 4D manifold with a timelike direction, decompose tensors into spatial and temporal components:

```julia
@foliate M4 time=:t spatial=[:x,:y,:z]

# Metric perturbation SVT decomposition (flat background)
svt_decompose(h[-μ,-ν]) →
    h₀₀ = 2ϕ
    h₀ᵢ = ∂ᵢB + Sᵢ           # Sᵢ transverse
    hᵢⱼ = 2ψ δᵢⱼ + 2∂ᵢ∂ⱼE + ∂ᵢFⱼ + ∂ⱼFᵢ + hᵢⱼᵀᵀ
```

#### Fourier space

```julia
to_fourier(expr) # ∂ᵢ → -ikᵢ, ∂₀ → -iω, ∂ᵢ∂ᵢ → -k², □ → -(ω²-k²) = -p²
```

Note the sign conventions must be configurable. This is a global setting.

#### Projection operators

```julia
# Transverse projector
Pᵀᵢⱼ(k) = δᵢⱼ - kᵢkⱼ/k²

# TT projector (3D, for symmetric rank-2)
Πᵀᵀᵢⱼₖₗ(k) = 1//2 * (Pᵀᵢₖ*Pᵀⱼₗ + Pᵀᵢₗ*Pᵀⱼₖ - Pᵀᵢⱼ*Pᵀₖₗ)
```

#### Bardeen variables

```julia
# Gauge-invariant combinations
Φ = ϕ + ∂₀B - ∂₀²E      # Bardeen potential
Ψ = ψ                     # (gauge-invariant on flat background)
Vᵢ = Sᵢ - ∂₀Fᵢ          # vector Bardeen variable
```

#### Quadratic action decomposition

```julia
# Given a quadratic Lagrangian L[h], decompose into scalar/vector/tensor sectors
svt_action(L, h) → (L_scalar, L_vector, L_tensor)

# Each sector is expressed in Fourier space as a matrix quadratic form
# L_scalar = Φ* M(k,ω) Φ  (where Φ = (Φ,Ψ,...) is the vector of scalar dofs)
```

This is the culminating functionality: feed in $R_{\mu\nu}R^{\mu\nu} - \beta R^2$, get out the propagator matrix.

**Estimated size**: 1500 lines Julia.

**Tests**: The fourth-derivative gravity calculation from the proof document serves as the primary integration test. Additionally: reproduce the linearised Einstein equations in Newtonian gauge, scalar/vector/tensor decomposition for standard $R + R^2$ gravity, verify SVT orthogonality.

---

## 5. Development plan

### 5.1 Method: Agent-first, TDD is the law, human-steered

**TDD is non-negotiable.** Every component is developed tests-first. The agent writes the test file, runs it (all tests fail), writes the implementation, runs it again (all tests pass). No implementation code is written without a failing test that motivates it. If an agent session produces implementation without corresponding tests, the implementation is discarded.

Each component follows the cycle:

1. **Human writes spec** — a concise description of the component, its interface, and 3-5 key test cases. Specifies types and method signatures.
2. **Agent generates tests** — comprehensive test file covering the spec plus edge cases. Tests run and fail.
3. **Agent generates implementation** — targeting ~2000 lines per session. Tests run and pass.
4. **Human reviews** — reads generated code, checks type design decisions, identifies semantic errors (especially index scoping, sign conventions). Checks that no raw `Expr` leaked into public API.
5. **Agent fixes** — human feeds back failures, agent corrects.
6. **Integration test** — run the end-to-end fourth-derivative gravity calculation after each layer is complete.

**When stuck on types**: If the agent cannot make tests pass with a clean typed implementation within a reasonable effort (~2 attempts), it uses the escape hatch (raw `Expr` internally, typed boundary), marks `@refactor`, and moves on. Getting the test green takes priority over getting the types perfect. The refactor happens later, with the working code as a reference implementation.

### 5.2 Milestones

| Day | Milestone | Deliverable | Integration test |
|-----|-----------|-------------|------------------|
| 1 | **M1: Foundation** | `xperm.c` compiled, `ccall` wrapper, basic `Perm`/`SGS` types | Canonicalise $R_{abcd} + R_{abdc}$ → 0 |
| 2 | **M2: Expr primitives** | Index analysis, dummy renaming, `fresh_index`, `walk`/`substitute` | $g^{ab}g_{bc} = \delta^a_c$ |
| 3 | **M3: Contraction** | Metric contraction, delta substitution, fixed-point engine | $R^a{}_{bad} = R_{bd}$ |
| 4 | **M4: Canonicalisation** | Full `xperm.c`-backed canonicaliser on tensor product `Expr` trees | Riemann symmetries hold |
| 5 | **M5: Derivatives** | Partial/covariant derivative, Leibniz rule, commutator | $[\nabla_a, \nabla_b]V^c = R^c{}_{dab}V^d$ |
| 6 | **M6: GR objects** | Riemann, Ricci, Einstein, Weyl, Bianchi rules | All standard identities |
| 7 | **M7: Linearisation** | `linearize()`, explicit first-order formulas, gauge transforms | $\delta R_{\mu\nu}$ matches Claim 3.1 |
| 8 | **M8: SVT** | 3+1 split, SVT decomposition, Fourier transform | Scalar/vector/tensor sectors decouple |
| 9 | **M9: Quadratic action** | Action decomposition, matrix extraction, propagator inversion | $\det M = 8(1-3\beta)k^4 p^4$ |
| 10 | **M10: Hardening** | Edge cases, documentation, `Pkg` registration prep | Full proof reproduced |

### 5.3 Beads issue tracking

Following the project's beads methodology: each milestone is a bead. Issues discovered during integration testing are filed as sub-beads with links to the failing test case and the xAct reference output.

---

## 6. Test strategy

### 6.1 Unit tests

Every function has unit tests. Agent generates tests before implementation (TDD). Minimum coverage: every branch, every error path.

### 6.2 Reference tests

For each GR identity or published result, a test that:
1. Constructs the relevant expression using TensorGR.jl
2. Simplifies/canonicalises
3. Compares against the known result (hard-coded)

Source of reference results:
- xAct documentation notebooks (translated to Julia)
- Published tables in xPert paper (arXiv:0807.0824)
- Published tables in xPand paper (arXiv:1302.6174)
- The fourth-derivative gravity proof document (this project's origin)

### 6.3 Numerical spot-checks

For every symbolic identity, an additional test that:
1. Substitutes random numerical values for all free parameters
2. Evaluates both sides
3. Checks equality to floating-point precision

This catches sign errors and wrong coefficients that symbolic tests might miss if both sides have the same bug.

### 6.4 Regression tests against xAct

A separate (optional, requires Mathematica) test suite that:
1. Runs the same calculation in xAct via MathLink
2. Runs it in TensorGR.jl
3. Compares outputs symbolically

This is for development validation only. It is not required for CI.

---

## 7. Dependencies

### Required

| Dependency | Role | License |
|---|---|---|
| Julia ≥ 1.10 | Runtime | MIT |
| `xperm.c` | Index canonicalisation | GPL |
| C compiler (gcc/clang) | Build `libxperm.so` | — |

### Optional

| Dependency | Role | License |
|---|---|---|
| Symbolics.jl | Heavy scalar simplification | MIT |
| SymEngine.jl | Fast scalar simplification (C++ backend) | MIT |
| BinaryBuilder.jl | Cross-platform `xperm.c` compilation | MIT |
| Test (stdlib) | Testing | MIT |

The package itself is **GPL** (inherited from `xperm.c`).

---

## 8. Risks and mitigations

### R1: Dummy index scoping bugs (HIGH probability, HIGH impact)

The most common failure mode in tensor CAS. When two subexpressions share a dummy index name, combining them silently produces wrong results.

**Mitigation**: Every function that combines expressions runs `ensure_no_dummy_clash` first. Comprehensive test suite with nested contractions. Numerical spot-checks catch silent errors.

### R2: Multi-term symmetry incompleteness (MEDIUM probability, MEDIUM impact)

The Bianchi identity $R_{a[bcd]} = 0$ is not a permutation symmetry and `xperm.c` does not handle it. It must be implemented as a separate rewriting layer. This may miss simplifications that xAct catches.

**Mitigation**: Implement multi-term symmetries as an explicit rule system (pattern → replacement). Verify completeness against xAct on a test suite of Riemann polynomial expressions. Accept that v0.1 may miss some exotic simplifications — document the gap.

### R3: Expression swell at higher perturbation orders (LOW probability at v0.1, HIGH at v0.2+)

At second or third order perturbation theory, expressions grow to tens of thousands of terms. Julia's `Expr` tree manipulation may become slow compared to Mathematica's optimised kernel.

**Mitigation**: Profile at M10. If needed, introduce a hash-consed expression representation (interned subexpressions, O(1) equality checks). The `Expr`-as-representation choice does not preclude this — it's a transparent optimisation.

### R4: Agent generates subtly wrong Wolfram→Julia translations (HIGH probability, MEDIUM impact)

The agent may mishandle Wolfram-specific evaluation semantics (Hold, UpValues, pattern priority) when translating to Julia's typed dispatch model.

**Mitigation**: TDD. Every translated rule is tested against known input/output pairs from xAct. Integration tests catch accumulated errors. Human review focuses on the 10% of rules that involve evaluation order subtlety.

### R5: Scope creep (MEDIUM probability, HIGH impact)

Temptation to add xCoba (component calculations), spinors, exterior calculus before the core is solid.

**Mitigation**: PRD non-goals are explicit. v0.1 is defined by the milestone table. Features beyond M10 are deferred to v0.2.

### R6: Type system impedance (MEDIUM probability, MEDIUM impact)

Some intermediate rewriting states during tensor simplification may resist clean representation in the typed AST. For example: an expression mid-contraction where some indices have been eliminated but the result is not yet a valid `TensorExpr` of any single subtype.

**Mitigation**: The escape hatch protocol (Section 3.1). Internally drop to raw `Expr`, enforce typed invariants at the boundary via `from_expr` + assertion. Mark with `@refactor`. Track escape hatch count — if it exceeds ~5, revisit the type hierarchy before continuing. The types serve the computation, not the other way around.

---

## 9. Future work (v0.2+)

Listed in priority order:

1. **PSALTer equivalent**: Particle spectrum analysis. Spin-projection operators, saturated propagator, unitarity conditions. Requires only the quadratic action decomposition from v0.1 plus matrix algebra.

2. **Second-order perturbation theory**: Extend `linearize()` to `n=2` using xPert's combinatorial formulas. Primary use case: second-order gravitational waves.

3. **Component calculations (xCoba equivalent)**: Given a specific metric (Schwarzschild, FLRW, etc.), evaluate all tensor components numerically or symbolically.

4. **Arbitrary background**: Currently assumes flat background for perturbation theory. Extend to curved backgrounds (FLRW, de Sitter).

5. **FieldsX equivalent**: Fermions, gamma matrices, Fierz identities. Required for supergravity.

6. **Lean 4 bridge**: Export tensor identities as Lean 4 proof obligations. The identity $\det M = 8(1-3\beta)k^4p^4$ is a polynomial identity — verifiable by `ring` tactic. The real value is in bridging CAS output to formal verification, which no existing system does.

7. **Performance**: Hash-consed expressions, parallel canonicalisation for large expressions, SIMD-accelerated polynomial arithmetic.

---

## 10. Success criteria

The project succeeds at v0.1 if:

1. The fourth-derivative gravity calculation from Section 1-7 of the proof document runs end-to-end in TensorGR.jl, producing all corrected propagators.

2. All standard GR identities (Bianchi, contracted Bianchi, Weyl trace-freeness, metric compatibility) are verified by the test suite.

3. The package installs via `Pkg.add("TensorGR")` on Linux/macOS/Windows with no Mathematica dependency.

4. A user with no prior exposure can reproduce the fourth-derivative gravity calculation by following a tutorial notebook.

5. Total codebase (excluding tests) is ≤ 10,500 lines.

6. **TDD compliance**: every exported function has tests that were written *before* the implementation. Test coverage ≥ 90% of non-trivial branches.

7. **Idiomaticity**: The public API uses multiple dispatch, `Base` overloads (`*`, `+`, `show`, `==`), and standard Julia conventions. No raw `Expr` appears in the public API. Any internal escape hatches are marked `@refactor` and tracked as issues. Aim: a Julia developer unfamiliar with GR should be able to read the type hierarchy and understand the data model without reading the xAct source.

---

## Appendix A: xperm.c API reference

From `xperm.c` (Martín-García, 2003-2011, GPL):

```c
/* Core functions to expose via ccall */

void schreier_sims(
    int *base, int bl,          // base points
    int *GS, int m,             // generating set (m permutations)
    int n,                      // degree (number of points)
    int *newbase, int *nbl,     // output: new base
    int **newGS, int *nm,       // output: new generating set
    int *num                    // output: order of group
);

int canonical_perm(
    int *perm,                  // input permutation
    int n,                      // degree
    int THEORY,                 // 0=SGS given, 1=Riem sym, etc.
    int *base, int bl,          // symmetry group base
    int *GS, int m,             // symmetry group generators
    int *freeps, int fl,        // free index slots
    int *vds, int dl,           // paired dummy index slots
    int *dummies, int ddl,      // dummy index positions
    int *result                 // output: canonical permutation
);
```

## Appendix B: Wolfram→Julia translation patterns

| Wolfram pattern | Julia equivalent |
|---|---|
| `f[x_, y_]` | Method on typed struct: `f(t::Tensor)` dispatching on `t.name` via registry |
| `x_ /; condition` | `if` guard in method body, or registry lookup |
| `ReplaceAll[expr, rule]` | `substitute(expr::TensorExpr, rule)` returning new `TensorExpr` |
| `ReplaceRepeated[expr, rules]` | `fixpoint(x -> substitute(x, rules), expr)` |
| `Module[{x}, body]` | `let x = fresh_index(used); body end` |
| `UpValues` on symbols | Entries in `TensorRegistry` dict |
| `Hold`/`HoldFirst` | `@tensor` macro (operates on `Expr` pre-eval, constructs typed AST) |
| `Map[f, expr]` | `walk(f, expr::TensorExpr)` using dispatch |
| `Cases[expr, pattern]` | `filter(pred, collect_subtrees(expr))` |
| `Expand` | `expand_products(expr::TensorExpr)` or `scalar_expand` for `TScalar` |
| `Together`/`Cancel` | `scalar_cancel(s::TScalar)` |
| `Times[a, b]` | `TProduct(1//1, [a, b])` via `Base.:*` overload |
| `Plus[a, b]` | `TSum([a, b])` via `Base.:+` overload |

### Escape hatch protocol

When translating a Wolfram function that resists clean typing — e.g., a rewrite rule that produces intermediate expressions with inconsistent index structure — use this pattern:

```julia
function problematic_rewrite(expr::TensorExpr)::TensorExpr
    # @refactor: this uses raw Expr internally because [reason]
    raw = to_expr(expr)
    # ... Expr tree surgery ...
    result = from_expr(raw)
    @assert is_well_formed(result) "problematic_rewrite produced invalid TensorExpr"
    return result
end
```

The assertion enforces the typed invariant at the boundary. The `@refactor` comment marks the spot for later cleanup. The agent should prefer typed operations and resort to this only when stuck.

## Appendix C: Reference calculation

The primary integration test is the linearised fourth-derivative gravity propagator calculation:

**Input**: $I = \int d^4x\,[R^{(1)}_{\mu\nu}R^{(1)\mu\nu} - \beta\,(R^{(1)})^2]$

**Expected outputs** (corrected):

$$\langle\Phi\Phi\rangle = \frac{3}{4k^4} + \frac{1}{2k^2p^2} + \frac{1-2\beta}{8(1-3\beta)p^4}$$

$$\langle\psi\psi\rangle = \frac{1-2\beta}{8(1-3\beta)p^4}$$

$$\langle\Phi\psi\rangle = -\frac{1}{4k^2p^2} - \frac{1-2\beta}{8(1-3\beta)p^4}$$

$$\langle V_i V_j\rangle = \frac{P^T_{ij}}{k^2 p^2}$$

$$\langle h^{TT}_{ij} h^{TT}_{kl}\rangle = \frac{2\Pi^{TT}_{ijkl}}{p^4}$$

Each propagator expression is verified both symbolically (exact match) and numerically (random momentum evaluation at 100 points).
