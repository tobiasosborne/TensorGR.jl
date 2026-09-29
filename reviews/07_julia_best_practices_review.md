# TensorGR.jl — Architecture Review Against Julia Best Practices

**Reviewer**: Claude (Opus 5.5), single reviewer, no subagents
**Date**: 2026-09-29
**Commit**: `64d5144` (master)
**Scope**: `src/` (54,433 lines, 209 files), `ext/`, `deps/`, `test/`, CI, packaging
**Baseline**: Julia manual — *Style Guide*, *Performance Tips*, *Modules* (`__init__`/precompilation);
community tooling norms (Aqua.jl, JET.jl, JLL/Yggdrasil binaries, TermInterface.jl, ScopedValues)

**Evidence**: every finding marked **[verified]** is reproduced by a script in
[`07_probes/`](07_probes/). Findings marked **[from reading]** were inferred from source and not executed.

| Artifact | Purpose |
|---|---|
| `07_probes/probe_correctness.jl` | Runtime probes for §1 and §2 (run: `julia --project reviews/07_probes/probe_correctness.jl`) |
| `07_probes/probe_output.txt` | Output of the above at `64d5144` |
| `07_probes/aqua_jet.jl` | Aqua.jl + JET.jl in a temp env (run: `julia reviews/07_probes/aqua_jet.jl`) |
| `07_probes/jet_report.txt` | Full JET report (41 reports) at `64d5144` |
| `07_probes/setup.jl` | Shared registry fixture (M4, g, D, symmetric S, antisymmetric A, vector V) |

**Relation to earlier reviews**: `02_architecture_review.md` (2026-03-07, 10.8k lines) §3.4 judged the
"all indices free" xperm mode *well-engineered*. §1 below shows it is the root cause of wrong results —
**this review supersedes 02 §3.4**. 02 §4.1 (hard-coded tensor names) and §6.1 (manual walk
boilerplate, "Critical") were not acted on; the codebase has since grown ~5× and both problems scaled with it.

---

## Executive summary

The most serious problems are not Julia style issues. They are **silent wrong physics produced by the
canonicalizer's design**, and a **packaging path that cannot build the C dependency from a fresh clone**.
Below those sit architecture problems that are textbook Julia antipatterns: a closed AST with open,
hand-written traversals; ambient task-local state; `objectid`-keyed global caches; identity encoded in
`Symbol` names; an `Any`-typed scalar layer using `Expr` as a CAS; and a single 1,083-export module.

What is sound: Aqua reports **no method ambiguities and no type piracy**; the AST is immutable with
consistent `==`/`hash`; CAS integrations are proper weak-dep extensions; `xperm.c` has no global C
state, so the FFI is thread-safe.

---

## 1. Silent correctness bugs rooted in the canonicalizer design

### 1a. Covariant derivatives are canonicalized as if they commute — **[verified]**

`src/algebra/canonicalize.jl:272` adds a "derivatives commute" generator for **every** imploded
derivative chain with `nderiv >= 2`, without checking `covd == :partial`. `_implode`
(`canonicalize.jl:79`) accepts any homogeneous chain, including `∇∇`.

```
A1 canonicalize D_b D_a S_cd                  => D[-a](D[-b](S[-c,-d]))      (swapped)
A2 simplify V^e(D_bD_aS_cd − D_aD_bS_cd)      => 0                           (should be Riemann terms)
```

Correct: `[∇_a, ∇_b] S_cd = −R^f_{cab} S_fd − R^f_{dab} S_cf`.

The generator is only emitted when the tensor has declared symmetries (the loop `continue`s otherwise),
so `∇∇V` is unaffected — which is why the test suite, which mostly uses symmetry-free fields, did not
catch it. **`∇∇h` with symmetric `h` is affected.** Since `canonicalize` runs *before* `commute_covds`
in `_simplify_one_pass`, any curved-background result containing `∇∇h` must be treated as suspect until
re-verified independently: the MSS `covariant_output=true` path, the 6-derivative gravity on de Sitter
spectrum, and bench_12 ground truth.

### 1b. Canonicalization changes free-index positions — **[verified]**

`canonicalize.jl:313` rebuilds indices as
`TIndex(sym, all_indices[slot].position, …)`: index *names* are permuted by xperm but *positions* stay
pinned to slots. For a symmetry that swaps an Up and a Down slot, this changes the tensor.

```
B1 free_indices(S^b_a) → canonicalize          => [-a, b]  →  [a, -b]       (free structure changed)
B2 simplify A^b_a + A_a^b   (A antisymmetric)  => A_a^b − A^a_b              (should be 0)
```

Root cause: all indices are passed to xperm as **free** (`canonicalize.jl`, "All indices treated as free
for xperm"), which discards the dummy-exchange half of Butler–Portugal and all metric-raising/lowering
symmetry. The result is not a canonical form; dummy handling is patched afterwards by
`_normalize_dummies` and `fix_dummy_positions`. This is why CLAUDE.md accumulated warnings such as
"Do NOT sort TSum terms… breaks benchmark term counts" and "Do NOT sort deriv chains… causes
non-convergence": the pinned term counts depend on accidents of representation, not on a normal form.

### 1c. Display renders every derivative as ∂ — **[verified]**

`to_latex` (`src/show.jl:189`) and `to_unicode` (`src/show.jl:306`) ignore `TDeriv.covd`:

```
C1 to_latex(D_b D_a S_cd)    => \partial_{b} \partial_{a} S_{c d}
```

(`Base.show` at `show.jl:78` is correct.) This makes 1a invisible in rendered output and in any paper
figures produced from it.

### 1d. No well-formedness validation — **[verified]**

Sums with mismatched free indices are accepted and "simplified":

```
D1 simplify V^a + V_a        => V_a + V^a       (should be an error)
D2 simplify S^b_a − S^a_b    => 0               (ill-formed input silently annihilated)
```

A `TSum` constructor-level (or `simplify`-entry) check that all terms share the same free-index
multiset would have surfaced 1b immediately.

---

## 2. Architecture vs Julia idioms

### 2.1 Closed AST, open hand-written traversals (the expression problem) — **[verified]**

There are **18 `TensorExpr` subtypes** (6 core in `types.jl`; plus `ScalarHarmonic`, 2 vector and 3 tensor
harmonics, `LaplacianS2`, `GammaMatrix`, `Gamma5`, `ChargeConjugation`, `AlgValuedForm`). Every pass
re-implements recursion per node type: **83 functions** have a `::TDeriv` method, only **9** have a
`::TParamDeriv` method, and there is no generic fallback.

```
E1 simplify(TParamDeriv(...))         => MethodError: expand_products(::TParamDeriv)
E2 simplify(Y_{2,1} + Y_{2,1})        => MethodError: expand_products(::ScalarHarmonic)
```

`tensor_harmonics.jl:111–129` papers over this with `@eval` loops generating `rename_dummy` methods.

The same generic function is also used with incompatible contracts: `canonicalize(::TensorExpr)`
returns an expression, `canonicalize(::TRInv)` returns an `(expr, sign)` tuple. JET found a **definite**
`MethodError` at `src/invariants/trinv.jl:958` and `:982`, which call
`canonicalize(trinv; registry=registry)` — `canonicalize(::TRInv)` accepts no keywords.

**Idiomatic fix**: one structural interface — `iscall`/`operation`/`arguments`/`maketerm` (TermInterface.jl,
as used by SymbolicUtils/Metatheory) or a local `children`/`rebuild` pair — with generic fallbacks
`f(x::TensorExpr) = rebuild(x, map(f, children(x)))`. New node types then work everywhere by
implementing two methods. Give functions with different contracts different names
(`canonicalize_with_sign`).

### 2.2 Ambient registry via task-local storage — **[verified / from reading]**

- 522 `current_registry()` call sites; core passes (`contraction.jl:52,129`, `canonicalize.jl:29,153`,
  `svt/fourier.jl:60`) read the registry implicitly.
- `task_local_storage` is **not inherited by spawned tasks**. `parallel=true` only works because
  `simplify.jl:318–343` manually re-wraps each `Threads.@spawn` in `with_registry`. Any future `@spawn`
  elsewhere silently runs against `_GLOBAL_REGISTRY`.
- Two sources of truth: many APIs take `registry=current_registry()` *and* are called inside
  `with_registry(reg)` (e.g. `ddi_rules.jl:121`: `with_registry(reg) do generate_ddi_rules(…; registry=reg)`).
- `_GLOBAL_REGISTRY` is a mutable global populated at runtime and shared by all users of the process.

**Idiomatic fix**: pass an explicit context through the core algebra (it is small and hot), and use
`Base.ScopedValues` (Julia ≥ 1.11; ScopedValues.jl on 1.10) for the user-facing default. Scoped
values are inherited by child tasks, removing the manual re-wrap.

### 2.3 Global mutable caches keyed by `objectid` — **[verified]**

`src/algebra/ddi_rules.jl:3` (`_DDI_REGISTERED`) and `src/invariants/simplify_levels.jl:719`
(`_DUAL_RULES_REGISTERED`) are `Dict{UInt,…}` keyed by `objectid(reg)`:

```
H1 _DDI_REGISTERED entries after 50 throwaway registries => 50
```

- **Leak**: entries are never removed.
- **Stale hits [from reading]**: `objectid` values can be reused after GC, so a fresh registry can
  report `has_ddi_rules == true` and skip registration.
- **Data race [from reading]**: `_INVAR_DB_CACHE` (`invariants/database.jl:73,100`) is a global `Dict`
  mutated without a lock; reachable from threaded `simplify`. Contradicts the "thread-safe" claim.

**Fix**: this is per-registry state — store it in the registry. If a global cache is truly needed, use a
`WeakKeyDict` guarded by a lock (or `Base.Lockable`).

### 2.4 Identity encoded in `Symbol` names ("stringly typed") — **[from reading]**, one **[verified]**

- Pattern variables are indices whose name ends in `_` (`rules.jl:57`), checked via
  `endswith(string(idx.name), "_")` — allocates a `String` per check on the rule-matching hot path, and
  forbids user index names ending in `_`.
- Derived objects are created by name concatenation: `Symbol(t.name, :_dag)` (`walk.jl`),
  `Symbol(:Γ, name)` (`covd.jl:35`), `Riem_g1` for product factors. Collisions with user names are unchecked.
- `Tensor(:Riem, …)` / `:Ric` / `:RicScalar` hard-coded at **211** sites. Consequence:
  `set_flat!(reg, m)` (`gr/metric.jl:141`) registers vanishing rules for the global `:Riem`, `:Ric`, …
  regardless of which metric they belong to — with two metrics, flattening one zeroes the other's curvature.
- `Dict{Symbol,Any}` appears **157** times; `TensorProperties` duplicates seven flags as both typed
  fields and `options` entries (`registry.jl:36–80`), kept in sync by hand in each setter
  (`metric.jl:138–139, 186–187, 201–202`, `registry.jl:293–294`).
- `TensorRegistry.rules::Vector{Any}` and `foliations/mappings::Dict{Symbol,Any}` are untyped
  "to avoid forward ref"; product-manifold data is stored in `foliations` under a mangled key.

**Fix**: make derived objects first-class (e.g. a `CurvatureSet` struct per metric holding the tensor
names, looked up from the metric), represent pattern variables as a distinct type or flag on
`TIndex`, and resolve forward references by include order or an abstract type rather than `Any`.

### 2.5 Coefficient and scalar layer — **[verified]**

`TProduct.scalar::Rational{Int}`, and `Base.:*(s::Number, t::TensorExpr) = tproduct(Rational{Int}(s), …)`
(`algebra/arithmetic.jl`):

```
F1 (1//3)^20 * ((1//3)^20 * V)   => OverflowError
F2 0.1 * V                       => (3602879701896397//36028797018963968) * V   (silent)
F3 im * V                        => InexactError: Rational(im)
F4 TScalar(:x)^2 == simplify(x*x) => false
```

JET additionally finds `tproduct(::Complex, …)` call sites (`spinors`, `ddi_rules.jl:516`) that can never
succeed. `TScalar.val::Any` holds `Rational`, `Int`, `Symbol`, Julia `Expr`, or `Symbolics.Num`;
`Base.:^(t::TScalar, n)` builds a raw `Expr` (`arithmetic.jl`), i.e. `Expr` is used as a CAS with purely
syntactic equality. There are 8 `TScalar(:(…))` construction sites and 46 `:call`-Expr manipulations.

**Fix**: pick one coefficient ring and make it a type parameter or a single concrete type
(`Rational{BigInt}`, or a small normalized polynomial/`SymbolicUtils` term). Reject floats at the API
boundary instead of converting them to binary rationals. Complex coefficients are needed for spinors/NP.

### 2.6 Invariants not enforced by constructors — **[verified]**

Smart constructors `tproduct`/`tsum` normalize (flatten, drop zeros, absorb scalars), but raw
`TProduct(` is called at **96** sites and raw `TSum(` at **38** sites, bypassing them. Zero has several
unequal representations:

```
G1 ZERO == TSum([]), ZERO == TProduct(0//1, [V])  => (false, false)
```

The structs are immutable but wrap `Vector`s, so a caller mutating an `indices`/`factors` vector after
construction silently corrupts `hash`-keyed collections (`children(p) = p.factors` returns the internal
vector). **Fix**: enforce invariants in inner constructors; use tuples or copy-on-construct; consider
cached hashes (hash-consing) — `_simplify_fixpoint` re-hashes the whole tree every iteration.

### 2.7 Symbolic dimension declared but unsupported — **[verified by JET]**

`ManifoldProperties.dim::Union{Int,Symbol}` and `VBundleProperties.dim` admit symbolic dimensions, but
JET reports ~15 sites doing `2*d`, `d+1`, `d-1` on a possible `Symbol`
(`hamiltonian/adm.jl:458,471,753,788,821–822`, `hamiltonian/dof.jl:185`, `bimetric/linearize.jl:220,227`),
`lorentzian(::Symbol)` (`gr/metric.jl:125`), and `convert(Int, ::Symbol)` into `BasisProperties`
(`components/basis.jl:70`). Either model symbolic dimension properly or restrict the type to `Int`.

### 2.8 One flat module with 1,083 exports — **[verified]**

209 files `include`d into a single `module TensorGR`, 373 `export` statements covering 1,083 names,
including three that do not exist (Aqua: `multi_horndeski_L5`, `multi_horndeski_lagrangian`,
`symplectic_current_eh`). Core tensor algebra and research applications (Horndeski/DHOST, PPN, Feynman
rules, harmonics, bimetric, Hamiltonian/Dirac, fermions) share one namespace and one precompile unit.

**Fix**: submodules at minimum (`TensorGR.Core`, `.GR`, `.Perturbation`, `.Applications.*`), or a
`TensorGRCore` package plus application packages. Use the `public` keyword (≥ 1.11) for API that
should be documented but not exported.

---

## 3. Packaging, tooling, process

### 3.1 `deps/build.jl` cannot succeed — **[verified]**

```
Warning: Assignment to `compiler` in soft scope is ambiguous … will be treated as a new local.
Warning: No C compiler found (tried gcc, cc, clang).
Warning: Failed to build xperm.c  exception = ArgumentError: `nothing` can not be interpolated into commands
```

Two classic footguns: (1) assigning a global inside a top-level `for` in a non-interactive file creates a
local (soft scope), so `compiler` stays `nothing` although gcc is installed; (2) a top-level `return`
inside `if` in a script does not stop the script. `deps/libxperm.so` was absent in this checkout
(gitignored); a fresh clone cannot canonicalize.

Further: `_libxperm_path` (`xperm/wrapper.jl:4`) is a `const` path into the package source tree, which
is read-only once installed from a registry; the `dlopen` handle in `_ensure_lib_loaded` is unused
because `ccall((:sym, path))` resolves the library itself. CI (`ci.yml`) has run once
(2026-03-06, **failed**); CompatHelper has failed daily since at least 2026-06-12.

**Fix**: Yggdrasil recipe → `xperm_jll` (tracked as TGR-byb), `ccall((:schreier_sims, libxperm), …)`.
Interim: wrap build.jl in a function.

### 3.2 `__init__` and hard dependencies — **[from reading]**

`REPL` is a hard dependency. `__init__` (`TensorGR.jl:756`) starts an `@async` task with a bare `catch`
that swallows every error. Move the REPL mode to an extension (triggered by `REPL`) or a separate package.

### 3.3 Aqua.jl — **[verified]**

8 pass / 2 fail. Pass: ambiguities, unbound type parameters, piracy, stale deps, persistent tasks.
Fail: 3 undefined exports (§2.8); no `[compat]` entries for `Libdl`, `LinearAlgebra`, `REPL`
(CLAUDE.md's "stdlib dep, no compat entry needed" is outdated).

### 3.4 JET.jl — **[verified]**

41 reports (`07_probes/jet_report.txt`). Categories: symbolic-dimension arithmetic (§2.7),
`canonicalize(::TRInv; registry)` (§2.1, definite), `tproduct(::Complex, …)` (§2.5), unchecked
`RegexMatch` captures (`String(::Nothing)`, `parse(Int, ::Nothing)` in `parser/latex_parser.jl:87`,
`repl/tensor_mode.jl:156–165,319–351`), `convert(TIndex, ::Nothing)` into `GammaMatrix`,
`convert(Symbol, ::Nothing)` into `Tensor`, `Colon(::Int, ::Nothing)` from unchecked `findfirst`.

### 3.5 Errors and convergence — **[from reading]**

450 `error("…")` calls, zero custom exception types (callers cannot distinguish "unregistered tensor"
from "ill-formed expression"), 7 bare `catch` blocks. `simplify` non-convergence only emits `@warn` and
returns a partially simplified expression (`simplify.jl:465`); default `maxiter=20`, while CLAUDE.md
states 100.

### 3.6 Tests — **[verified / from reading]**

- `runtests.jl` `include`s 226 files into one shared `Main` namespace (no SafeTestsets/TestItems isolation).
- `test/test_full_simplify.jl` exists but is **never run** (not included).
- Many tests pin term counts from the implementation itself (circular; see 1b).
- No test exercises `∇∇` on a tensor with declared symmetries, nor free-index preservation under
  `canonicalize`.

**Fix**: add property tests — `canonicalize` preserves `free_indices`, is idempotent, and agrees with
**numerical evaluation on a random metric** (the component machinery in `src/components/` already
supports this; it is the strongest available oracle and would have caught 1a and 1b).

### 3.7 Documentation drift — **[verified]**

CLAUDE.md and the memory index describe ~12,100 lines / 71 files; actual is 54,433 lines / 209 files.
Several "critical implementation notes" there are workarounds for §1b and should be removed once it is fixed.

---

## 4. Recommended order of work

1. **Record §1a and §1b as P0 bugs.** Freeze trust in curved-background results (MSS covariant output,
   dS 6-derivative spectrum, bench_12) until re-verified against component-level numerics.
2. **Fix the build** (§3.1): JLL via Yggdrasil; get CI green.
3. **Redesign canonicalization** (§1b): pass dummies to xperm's double-coset algorithm with proper
   metric/position handling so positions move with names (as xAct does); non-partial derivative chains
   get no commutation generator (§1a); add the free-index invariant check (§1d). Then delete
   `fix_dummy_positions` and re-derive pinned term counts from physics.
4. **Traversal interface + registry context** (§2.1, §2.2): TermInterface-style `children`/`maketerm`
   with generic fallbacks; `ScopedValues`; move `objectid` caches into the registry (§2.3).
5. **Coefficient/scalar types** (§2.5, §2.6).
6. **Split the package** (§2.8) and move REPL to an extension (§3.2).

---

## Appendix: metrics at `64d5144`

| Metric | Value |
|---|---|
| `src/` lines / files | 54,433 / 209 |
| Exported names | 1,083 (3 undefined) |
| `TensorExpr` subtypes | 18 |
| Functions with a `::TDeriv` method / `::TParamDeriv` method | 83 / 9 |
| `current_registry()` call sites | 522 |
| `with_registry(` call sites | 106 |
| `Dict{Symbol,Any}` occurrences | 157 |
| `::Any` / `Vector{Any}` / `Any[]` occurrences | 107 |
| Raw `TProduct(` / `tproduct(` | 96 / 449 |
| Raw `TSum(` / `tsum(` | 38 / 176 |
| Hard-coded `Tensor(:Riem|:Ric|:RicScalar, …)` | 211 |
| `error(` calls / custom exception types | 450 / 0 |
| Aqua | 8 pass, 2 fail |
| JET reports | 41 |
| Test files / included in `runtests.jl` | 233 / 226 |
| CI runs (`ci.yml`) | 1 (failed, 2026-03-06) |
