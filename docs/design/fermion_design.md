# Design: Fermion Field Types for TensorGR.jl

## References

- Penrose, R. & Rindler, W. (1984) _Spinors and Space-Time_, Vol 1, Ch 2, 4.
- Dreiner, H. K., Haber, H. E. & Martin, S. P. (2010) "Two-component spinor
  techniques and Feynman rules for quantum field theory and supersymmetry."
  _Phys. Rept._ **494**, 1--196. arXiv:0812.1594.
- Frob, M. B. (2020) "FieldsX---An extension package for the xAct tensor
  computer algebra suite to describe mixed symmetry fields in an arbitrary
  number of dimensions." arXiv:2008.12422.
- Peskin, M. E. & Schroeder, D. V. (1995) _An Introduction to Quantum
  Field Theory_, Secs 3.2--3.6.
- Freedman, D. Z. & Van Proeyen, A. (2012) _Supergravity_, Chs 2--3.
- Wald, R. M. (1984) _General Relativity_, Appendix B.

---

## 1. Grassmann Algebra Representation

### 1.1 Problem statement

Fermion fields are Grassmann-valued (anticommuting). In a product of
fields, swapping two fermionic factors produces a sign flip:

    psi * chi = -(chi * psi)

The existing TProduct reorders factors during canonicalization (via
`_factor_sort_key` in `canonicalize.jl`) without tracking signs from
anticommutation. The question is where and how to encode the
Grassmann parity (grade) of each factor.

### 1.2 Options analyzed

**Option A: Parity flag on TensorExpr.**
Add an abstract `grassmann_parity(::TensorExpr) -> Int` method returning
0 (bosonic) or 1 (fermionic). The parity of a product is the sum of
parities mod 2. Swapping two odd-parity factors flips the TProduct sign.

Pros: minimal AST change, no new wrapper types.
Cons: requires a method on every TensorExpr subtype; must be propagated
through walk/rename_dummy/etc.

**Option B: New `GradedTensor` wrapper.**
A `GradedTensor{T<:TensorExpr}` carrying `(expr::T, grade::Int)`.

Pros: clean separation of grading from the AST.
Cons: wrapping every fermion expression doubles pattern-matching
complexity; all downstream code (walk, simplify, canonicalize, rules)
must unwrap; the grading is intrinsic to the field, not an accident
of wrapping.

**Option C: Sign tracking in TProduct.**
Store a `parity::Int` field in TProduct. On factor reordering, count
transpositions of odd-grade factors and accumulate sign flips.

Pros: localizes the sign logic to TProduct.
Cons: requires a NEW field in TProduct (breaking change to the core
struct); every TProduct constructor must be audited; parity information
is then on the product, not on the individual fields, making it hard
to query "is this factor fermionic?"

**Option D: Grassmann metadata in TensorProperties.**
Add an `is_grassmann::Bool` field to `TensorProperties`. The registry
records which tensors are Grassmann-odd. The parity of a factor is
looked up from the registry at canonicalization time.

Pros: leverages the existing registry infrastructure; no AST changes;
consistent with how `is_metric`, `is_delta`, `is_covd`, `vanishing`,
`frozen` are already handled; the `options` Dict can carry additional
fermionic metadata (Dirac/Weyl/Majorana type, number of components,
mass dimension).
Cons: requires registry access during canonicalization (already the case).

### 1.3 Recommendation: Option D (registry metadata) + Option A (dispatch hook)

**Primary mechanism: `is_grassmann` flag in TensorProperties.**

This is the same pattern as `is_metric`, `is_delta`, `is_covd`, and
`vanishing`---boolean fields on TensorProperties that the algebra engine
queries at runtime. The field should be added as a hot-path boolean
(like the existing ones) to avoid Dict lookup overhead in the inner
canonicalization loop.

**Secondary mechanism: `grassmann_parity(::TensorExpr)` dispatch hook.**

A thin dispatch layer that returns 0 or 1 for any TensorExpr. For `Tensor`,
it looks up `is_grassmann` from the registry. For `GammaMatrix`, `Gamma5`,
`ChargeConjugation`, it returns 0 (these are matrices, not fields). For
`TDeriv`, it returns the parity of the inner expression (derivatives do
not change Grassmann parity). For `TProduct`, it returns the sum of
factor parities mod 2. For `TSum`, all terms must have the same parity
(validated). For `TScalar`, it returns 0.

This two-layer approach means:
1. No new AST types.
2. No changes to TProduct struct.
3. The anticommutation sign logic lives in `_canonicalize_product`, right
   next to the existing factor-sort code.
4. Registration is simple: `register_tensor!(reg, TensorProperties(...,
   is_grassmann=true))` or via a convenience macro.

### 1.4 Sign tracking in canonicalization

The critical code path is `_canonicalize_product` in `canonicalize.jl`.
Currently, it sorts factors by `_factor_sort_key` at the end:

```julia
sort!(new_factors, by=_factor_sort_key)
```

For Grassmann fields, this sort must track the parity of the permutation
restricted to Grassmann-odd factors. The algorithm:

1. Before sorting, record which factor positions are Grassmann-odd.
2. Sort the factors by `_factor_sort_key` (same as now).
3. Count the number of transpositions of Grassmann-odd factors induced
   by the sort (the parity of the permutation restricted to odd elements).
4. Multiply the TProduct scalar by `(-1)^(count)`.

This is O(n log n) where n is the number of factors, and adds negligible
cost to the existing sort.

For the xperm-based canonicalization of index slots within a single
tensor, nothing changes: xperm already handles sign bits for index
permutations. The Grassmann sign is purely about factor ordering in
a multi-tensor product.

---

## 2. Fermion Field Type Hierarchy

### 2.1 No new AST types

Fermion fields are represented as ordinary `Tensor` objects in the AST.
Their fermionic nature is encoded in the registry via `TensorProperties`.
This is the same design as differential forms (`is_form` flag in options)
and metrics (`is_metric` flag).

A Dirac spinor psi_alpha in 4D has implicit spinor indices (suppressed
in the abstract algebra, just like GammaMatrix). In the two-component
formalism, it decomposes into explicit SL2C-indexed objects (see Sec 3).

### 2.2 TensorProperties fields for fermions

```julia
TensorProperties(
    name = :psi,
    manifold = :M4,
    rank = (0, 0),           # no explicit tensor indices
    symmetries = SymmetrySpec[],
    is_grassmann = true,     # NEW hot-path boolean
    options = Dict{Symbol,Any}(
        :fermion_type => :dirac,       # or :weyl_left, :weyl_right, :majorana
        :mass_dim => 3//2,             # engineering dimension
        :spinor_rep => :four_component # or :two_component
    )
)
```

The `is_grassmann` field is a new hot-path boolean on TensorProperties,
added alongside the existing `is_metric`, `is_delta`, `is_covd`,
`is_christoffel`, `vanishing`, `frozen`, `flat`. The options Dict
carries non-hot-path metadata.

### 2.3 Registration convenience

```julia
define_fermion!(reg, :psi;
    manifold = :M4,
    type = :dirac,           # :weyl_left, :weyl_right, :majorana
    grassmann = true)        # default true for fermions
```

This function:
1. Calls `register_tensor!` with `is_grassmann=true` and appropriate
   metadata in `options`.
2. For Dirac fermions, also registers the conjugate field (see Sec 4).
3. For two-component formulations, registers the SL2C-indexed spinor
   components (see Sec 3).

---

## 3. Dirac, Weyl, and Majorana Relationships

### 3.1 Two-component (Penrose/van der Waerden) formalism

The existing spinor infrastructure provides:
- SL2C and SL2C_dot VBundles (undotted/dotted, `spinor_bundles.jl`)
- Spin metric epsilon_{AB} (antisymmetric, `spin_metric.jl`)
- Soldering form sigma^a_{AA'} (`soldering_form.jl`)
- Curvature spinors Psi_{ABCD}, Phi_{ABA'B'} (`curvature_spinors.jl`)
- SU(2) spatial spinors (`space_spinors.jl`)

In this formalism, a Weyl spinor is a single field with spinor indices:

    psi_A     (left-handed, undotted, SL2C)
    chi^{A'}  (right-handed, dotted, SL2C_dot)

These are registered as `Tensor` objects with explicit spinor indices:

```julia
# Left-handed Weyl spinor
register_tensor!(reg, TensorProperties(
    name = :psi_L, manifold = :M4, rank = (0, 1),
    is_grassmann = true,
    options = Dict(:fermion_type => :weyl_left,
                   :index_vbundles => [:SL2C])))

# Right-handed Weyl spinor
register_tensor!(reg, TensorProperties(
    name = :chi_R, manifold = :M4, rank = (1, 0),
    is_grassmann = true,
    options = Dict(:fermion_type => :weyl_right,
                   :index_vbundles => [:SL2C_dot])))
```

### 3.2 Dirac spinor = pair of Weyl spinors

A Dirac spinor in the two-component notation is:

    Psi = (psi_A, chi^{A'})

where psi_A is left-handed and chi^{A'} is right-handed. The 4-component
Dirac spinor is a convenient bookkeeping device; the fundamental objects
are the two Weyl components.

In TensorGR, a Dirac field can be represented in two ways:

1. **Abstract (4-component):** A single `Tensor(:psi, TIndex[])` with
   `fermion_type => :dirac` and no explicit spinor indices. Gamma
   matrix algebra uses the existing `GammaMatrix`, `Gamma5`, etc.

2. **Two-component decomposition:** A pair `(psi_L, chi_R)` with
   explicit SL2C/SL2C_dot indices. This is obtained via
   `dirac_to_weyl(expr)`, which decomposes a Dirac field into its
   left and right Weyl components using the chirality projectors
   P_L = (1 - gamma5)/2, P_R = (1 + gamma5)/2.

The function `weyl_to_dirac(psi_L, chi_R)` reassembles the pair.

### 3.3 Majorana spinor = self-conjugate Dirac

A Majorana spinor satisfies the Majorana condition (already encoded in
`charge_conjugation.jl`):

    psi^c = C gamma^0 psi* = psi

In two-component notation, a Majorana spinor has chi^{A'} = (psi_A)*
(the right-handed component is the complex conjugate of the left-handed
one). Thus a Majorana spinor is specified by a single Weyl spinor.

Registration:

```julia
define_fermion!(reg, :lambda;
    manifold = :M4,
    type = :majorana)
```

This registers the field with `fermion_type => :majorana` and creates
a simplification rule that replaces the conjugate with the original:

    bar(lambda) => lambda^T C

### 3.4 Relationship table

| Field type    | Components | Two-component          | Degrees of freedom |
|---------------|------------|------------------------|-------------------|
| Weyl (left)   | psi_A      | 1 undotted spinor      | 2 complex = 4 real |
| Weyl (right)  | chi^{A'}   | 1 dotted spinor        | 2 complex = 4 real |
| Dirac         | (psi_A, chi^{A'}) | 2 Weyl spinors  | 4 complex = 8 real |
| Majorana      | (psi_A, (psi_A)*) | 1 Weyl + conjugate | 2 complex = 4 real |

---

## 4. Dirac Conjugate Mechanism

### 4.1 Definition

The Dirac conjugate is:

    bar{psi} = psi^dagger gamma^0

In the two-component formalism:

    bar{psi} = (chi_A, psi^{A'})

where chi_A = (chi^{A'})* and psi^{A'} = (psi_A)* (complex conjugation
swaps dotted/undotted and raises/lowers via the spin metric).

### 4.2 Representation

The Dirac conjugate is represented as a separate tensor in the registry,
linked to the original by metadata:

```julia
define_fermion!(reg, :psi; manifold=:M4, type=:dirac)
# This automatically registers:
#   :psi       -- the Dirac field
#   :psi_bar   -- the Dirac conjugate
# with options[:conjugate_field] linking them.
```

The naming convention `_bar` for the conjugate avoids Unicode issues
(the dagger character causes Julia ParseError, as noted in CLAUDE.md).

### 4.3 Implementation: `dirac_bar` function

```julia
dirac_bar(psi::Tensor; registry=current_registry()) -> Tensor
```

Given a Dirac field `psi`, returns the conjugate field `psi_bar`.
The function:
1. Looks up the conjugate name from `options[:conjugate_field]`.
2. Returns `Tensor(conjugate_name, TIndex[])`.

For two-component expressions:
```julia
dirac_bar_2comp(psi_A, chi_Ap; registry=current_registry()) -> TensorExpr
```
Returns `(chi_A, psi^{A'})` using `conjugate_index` from
`spinor_bundles.jl` to swap SL2C <-> SL2C_dot.

### 4.4 Grassmann parity of conjugates

The conjugate bar{psi} is also Grassmann-odd. The registry entry for
`psi_bar` has `is_grassmann = true`. Bilinears like `bar{psi} psi`
are Grassmann-even (parity 0), as required for Lagrangian terms.

---

## 5. Bilinear Form Construction

### 5.1 Standard bilinears

The 16 independent Dirac bilinears are:

| Bilinear              | Lorentz type | Grassmann parity |
|-----------------------|--------------|-----------------|
| bar{psi} psi          | scalar       | 0 (even)        |
| bar{psi} gamma^a psi  | vector       | 0 (even)        |
| bar{psi} sigma^{ab} psi | tensor    | 0 (even)        |
| bar{psi} gamma^a gamma5 psi | axial vector | 0 (even) |
| bar{psi} gamma5 psi   | pseudoscalar | 0 (even)        |

All bilinears are Grassmann-even because they contain two Grassmann-odd
factors. The grade tracking handles this automatically: `grassmann_parity`
of a TProduct with two odd factors returns 0.

### 5.2 API for bilinear construction

```julia
# Scalar bilinear
scalar_bilinear(psi; registry=current_registry()) -> TensorExpr
# Returns: psi_bar * psi

# Vector bilinear (current)
vector_bilinear(psi, a::TIndex; registry=current_registry()) -> TensorExpr
# Returns: psi_bar * gamma^a * psi

# General bilinear
dirac_bilinear(psi, Gamma::TensorExpr; registry=current_registry()) -> TensorExpr
# Returns: psi_bar * Gamma * psi
# where Gamma is any element of the Clifford basis
```

### 5.3 Two-component bilinears

In the two-component formalism, bilinears decompose via the soldering
form. For example, the vector current:

    bar{psi} gamma^a psi = psi^A sigma^a_{AA'} chi^{A'}
                          + chi_A sigma_bar^{a A A'} psi_{A'}

where sigma_bar^a = epsilon^{AB} epsilon^{A'B'} sigma^a_{BB'}.

The existing soldering form infrastructure (`soldering_form.jl`) and
spin metric (`spin_metric.jl`) provide all the building blocks. The
bilinear construction functions compose these.

---

## 6. Integration with Simplify Pipeline

### 6.1 Changes to the pipeline

The simplify pipeline (in `simplify.jl`) consists of:

```
expand_products -> contract_metrics -> contract_curvature
-> canonicalize -> [commute_covds] -> collect_terms -> apply_rules
```

The only stage that requires modification is **canonicalize**, specifically
the `_canonicalize_product` function in `canonicalize.jl`. The change is
to the factor-sort step at the end:

```julia
# Current code (line ~333):
sort!(new_factors, by=_factor_sort_key)

# New code:
sign_from_sort = _grassmann_sort!(new_factors, reg)
# sign_from_sort is +1 or -1
# Multiply into the scalar coefficient
```

The function `_grassmann_sort!` performs a stable sort by
`_factor_sort_key` and counts transpositions of Grassmann-odd factors.

### 6.2 Grassmann-aware sort

```julia
function _grassmann_sort!(factors::Vector{TensorExpr},
                          reg::TensorRegistry) -> Int
    n = length(factors)
    n < 2 && return 1

    # Tag each factor with its Grassmann parity
    parities = [_is_grassmann_factor(f, reg) for f in factors]

    # Perform a stable sort, counting transpositions of odd factors
    # using a merge-sort-based parity counter
    sign = 1
    # ... (insertion sort for small n, merge sort for large n)
    # Each swap of two adjacent Grassmann-odd factors flips sign

    return sign
end
```

The function `_is_grassmann_factor` checks:
- `Tensor`: look up `is_grassmann` from registry
- `TDeriv`: recurse on inner arg (derivatives preserve Grassmann parity)
- `GammaMatrix`, `Gamma5`, `ChargeConjugation`: return false (matrices)
- `TScalar`: return false

### 6.3 collect_terms compatibility

The `collect_terms` function in `simplify.jl` merges terms with identical
structure but different coefficients. For Grassmann expressions, two
terms are "identical" if they have the same factors in the same order
(after canonicalization). Since canonicalize now produces a canonical
factor ordering that respects Grassmann signs, collect_terms works
without modification.

### 6.4 Rules engine

The rewrite rules engine (`rules.jl`) applies pattern matching and
substitution. For Grassmann expressions, rules that reorder fermion
factors must include the sign. The existing `make_rule` infrastructure
does not need changes---rules are written with explicit signs:

```julia
# Fierz identity: already correct (Fierz matrix includes signs)
# Majorana flip: bar{lambda} gamma^a lambda = -lambda^T C gamma^a lambda
#   -> encoded as a rule with explicit -1 coefficient
```

---

## 7. Interaction with Existing Infrastructure

### 7.1 Gamma matrices (`fermions/gamma.jl`)

The existing `GammaMatrix <: TensorExpr` represents gamma^a with one
spacetime index and suppressed spinor indices. This design is compatible
with fermion fields:

- A bilinear `bar{psi} gamma^a psi` is a TProduct of three factors:
  `[Tensor(:psi_bar, []), GammaMatrix(up(:a)), Tensor(:psi, [])]`.
- The GammaMatrix is bosonic (`grassmann_parity = 0`), so reordering
  it past a fermion does not flip the sign.
- The Clifford relation `{gamma^a, gamma^b} = 2 g^{ab}` is unchanged.

### 7.2 Gamma5 and chirality (`fermions/gamma.jl`)

The `Gamma5 <: TensorExpr` node is bosonic. Its anticommutation with
gamma matrices (`{gamma5, gamma^a} = 0`) is a matrix identity, not
a Grassmann anticommutation. No changes needed.

Chirality projectors P_L = (1 - gamma5)/2, P_R = (1 + gamma5)/2 can
be applied to Dirac fields to extract Weyl components. New functions:

```julia
chiral_project_left(psi)  -> P_L * psi
chiral_project_right(psi) -> P_R * psi
```

### 7.3 Fierz identities (`fermions/fierz.jl`)

The Fierz matrix and `fierz_coefficient` function encode the
rearrangement of bilinears. These already include the correct signs
from the Grassmann anticommutation of spinors (the overall -1/4 in
the Fierz identity comes from exchanging two fermion fields).

To apply Fierz identities to abstract expressions, a new function:

```julia
fierz_rearrange(expr; registry=current_registry()) -> TensorExpr
```

This detects products of two bilinears `(bar{psi1} Gamma_A psi2)
(bar{psi3} Gamma_B psi4)` and rewrites them using the Fierz matrix.
The Grassmann signs are handled by the Fierz coefficients, not by the
general anticommutation machinery.

### 7.4 Charge conjugation (`fermions/charge_conjugation.jl`)

The `ChargeConjugation <: TensorExpr` node is bosonic (it is a matrix
in spinor space). The charge-conjugation properties are already encoded:

- `C gamma^a C^{-1} = -(gamma^a)^T`
- `C^T = -C` (antisymmetric)
- `majorana_condition()` describes psi^c = psi

New functions for charge-conjugated spinors:

```julia
charge_conjugate(psi; registry=current_registry()) -> TensorExpr
# Returns C * gamma^0 * psi* (the charge-conjugated spinor)

is_majorana(psi; registry=current_registry()) -> Bool
# Checks if the field is registered as Majorana
```

### 7.5 Spinor bundles and the soldering form

For two-component expressions, the existing SL2C infrastructure is
used directly. A left-handed Weyl spinor psi_A is a `Tensor(:psi_L,
[spin_down(:A)])` with `is_grassmann = true` and `index_vbundles =>
[:SL2C]` in options.

The soldering form sigma^a_{AA'} converts between tensor and spinor
indices. The existing `to_spinor_indices` and `to_tensor_indices`
functions work without modification for fermion bilinears.

### 7.6 Covariant derivatives

Covariant derivatives of fermion fields involve the spin connection:

    D_a psi = partial_a psi + (1/4) omega_a^{bc} sigma_{bc} psi

where sigma_{bc} = (i/2)[gamma_b, gamma_c].

In TensorGR, `TDeriv(down(:a), Tensor(:psi, []), :D)` represents
D_a psi. The Grassmann parity of `TDeriv` is the same as its argument
(derivatives do not change Grassmann grade). The spin connection is
encoded as a rule that expands `D_a psi` when requested.

---

## 8. Public API Proposal

### 8.1 Registration

```julia
# Define a Dirac fermion (registers psi and psi_bar)
define_fermion!(reg, :psi; manifold=:M4, type=:dirac)

# Define a Weyl fermion (left-handed)
define_fermion!(reg, :psi_L; manifold=:M4, type=:weyl_left)

# Define a Majorana fermion
define_fermion!(reg, :lambda; manifold=:M4, type=:majorana)

# Define a generic Grassmann field (e.g., ghost field in gauge theory)
define_fermion!(reg, :c; manifold=:M4, type=:generic_grassmann)
```

### 8.2 Grassmann algebra

```julia
# Query Grassmann parity
grassmann_parity(expr; registry=current_registry()) -> Int  # 0 or 1

# Check if a tensor is Grassmann-odd
is_grassmann(name::Symbol; registry=current_registry()) -> Bool
```

### 8.3 Dirac algebra

```julia
# Dirac conjugate
dirac_bar(psi; registry=current_registry()) -> Tensor

# Charge conjugate
charge_conjugate(psi; registry=current_registry()) -> TensorExpr

# Chirality projectors
chiral_project_left(psi) -> TensorExpr    # P_L psi
chiral_project_right(psi) -> TensorExpr   # P_R psi

# Feynman slash (already exists in gamma.jl, no change)
slash(v::TensorExpr) -> TensorExpr
```

### 8.4 Bilinear construction

```julia
# General bilinear: bar{psi} Gamma chi
dirac_bilinear(psi, chi, Gamma; registry=current_registry()) -> TProduct

# Convenience for standard bilinears
scalar_bilinear(psi, chi)         # bar{psi} chi
vector_bilinear(psi, chi, a)      # bar{psi} gamma^a chi
tensor_bilinear(psi, chi, a, b)   # bar{psi} sigma^{ab} chi
axial_bilinear(psi, chi, a)       # bar{psi} gamma^a gamma5 chi
pseudo_bilinear(psi, chi)         # bar{psi} gamma5 chi
```

### 8.5 Decomposition

```julia
# Dirac to two-component Weyl decomposition
dirac_to_weyl(psi; registry=current_registry()) -> Tuple{Tensor, Tensor}
# Returns (psi_L, chi_R) with explicit SL2C/SL2C_dot indices

# Weyl to Dirac assembly
weyl_to_dirac(psi_L, chi_R; registry=current_registry()) -> Tensor
```

### 8.6 Fierz and identities

```julia
# Apply Fierz rearrangement to a product of two bilinears
fierz_rearrange(expr; registry=current_registry()) -> TensorExpr

# Check Majorana condition
is_majorana(psi; registry=current_registry()) -> Bool
```

---

## 9. Implementation Plan

### Phase 1: Core Grassmann infrastructure (src/fermions/)

**Files to modify:**
- `src/registry.jl`: Add `is_grassmann::Bool` field to TensorProperties
  (after `vanishing`), default `false`. Update the keyword constructor.
- `src/algebra/canonicalize.jl`: Add `_grassmann_sort!` function.
  Modify `_canonicalize_product` to call it after factor sorting.

**New file:**
- `src/fermions/grassmann.jl`: `grassmann_parity`, `_is_grassmann_factor`,
  `is_grassmann` query function.

**Tests:**
- Grassmann parity of TProduct with two fermionic factors is 0.
- Swapping two fermionic Tensors in a product produces -1.
- Mixed bosonic/fermionic products: swapping boson past fermion is free.
- Triple fermion product: psi * chi * lambda = -chi * psi * lambda etc.

### Phase 2: Fermion field registration (src/fermions/)

**New file:**
- `src/fermions/fields.jl`: `define_fermion!`, `dirac_bar`,
  `charge_conjugate`, `is_majorana`.

**Tests:**
- `define_fermion!` registers field with correct properties.
- `dirac_bar` returns the conjugate field.
- Majorana condition: `charge_conjugate(lambda) == lambda`.

### Phase 3: Bilinear construction (src/fermions/)

**New file:**
- `src/fermions/bilinears.jl`: `dirac_bilinear`, `scalar_bilinear`,
  `vector_bilinear`, etc.

**Tests:**
- Bilinears have Grassmann parity 0.
- Vector bilinear has correct free Lorentz index.
- Scalar bilinear simplifies correctly under Fierz.

### Phase 4: Two-component decomposition (src/fermions/)

**New file:**
- `src/fermions/two_component.jl`: `dirac_to_weyl`, `weyl_to_dirac`,
  `dirac_bar_2comp`, chiral projectors.

**Tests:**
- Dirac to Weyl roundtrip.
- Chiral projector properties: P_L^2 = P_L, P_R^2 = P_R, P_L P_R = 0.
- Vector current decomposition matches Penrose-Rindler.

### Phase 5: Fierz rearrangement on expressions (src/fermions/)

**Modify:**
- `src/fermions/fierz.jl`: Add `fierz_rearrange(expr)` that detects
  bilinear pairs and applies the Fierz matrix.

**Tests:**
- Fierz rearrangement of (bar{psi} psi)(bar{chi} chi) reproduces
  known coefficients.
- Completeness: re-rearranging gives back the original (up to sign).

### Phase 6: Spin connection and covariant derivatives

**New file:**
- `src/fermions/spin_connection.jl`: Rules for expanding D_a psi
  into partial_a psi + spin connection terms.

**Tests:**
- D_a (bar{psi} psi) = (D_a bar{psi}) psi + bar{psi} (D_a psi)
  (Leibniz rule with correct signs).
- Commutator [D_a, D_b] psi = (1/4) R_{abcd} sigma^{cd} psi.

### Dependency ordering

Phase 1 must come first (everything else depends on Grassmann signs).
Phases 2--3 can proceed in parallel. Phase 4 depends on Phase 2.
Phase 5 depends on Phase 3. Phase 6 depends on Phases 2 and 4.

### Risk assessment

The main risk is **regression in the existing simplify pipeline**. The
`_canonicalize_product` modification (Phase 1) touches a critical code
path. Mitigation:

1. The Grassmann sort is a no-op when no Grassmann-odd factors are present
   (the sign is always +1 for pure bosonic products).
2. All existing tests must pass unchanged.
3. The `is_grassmann` field defaults to `false`, so existing registered
   tensors are unaffected.

A secondary risk is **performance**. The `_is_grassmann_factor` lookup
requires registry access. This is already the case in
`_canonicalize_product` (line ~247: `has_tensor(reg, obj.tensor_name)`),
so the additional cost is one Dict lookup per factor, which is negligible.
