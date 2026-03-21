# Design: 3D Spatial Spinor Types (SU(2) Spinors on Spacelike Hypersurfaces)

## References

- Sommers, P. (1980) "Space spinors." _J. Math. Phys._ **21**, 2567.
- Sen, A. (1981) "On the existence of neutrino 'zero-modes' in vacuum spacetimes."
  _J. Math. Phys._ **22**, 1781.
- Ashtekar, A. (1991) _Lectures on Non-Perturbative Canonical Gravity_, Ch 2.
- Penrose, R. & Rindler, W. (1984) _Spinors and Space-Time_, Vol 1.
- Shaw, W. T. (1983) "Spinor fields at spacelike infinity."
  _Gen. Rel. Grav._ **15**, 1163.

---

## 1. SpaceSpinIndex Type Design

### 1.1 Problem statement

The existing 4D spinor infrastructure uses SL(2,C) spinors with two types
of 2-component indices:

- Undotted (SL2C): A, B, C, ... with vbundle `:SL2C`
- Dotted (SL2C_dot): A', B', C', ... with vbundle `:SL2C_dot`

A 4D spacetime vector V_a maps to a bispinor V_{AA'} via the soldering
form sigma^a_{AA'}. The dotted and undotted indices live in inequivalent
representations of SL(2,C).

On a spacelike hypersurface Sigma, the relevant structure group is SU(2),
the double cover of SO(3). SU(2) spinors have a single type of
2-component index (no dotted/undotted distinction) because SU(2) is the
compact real form: its fundamental and conjugate representations are
_equivalent_ via the antisymmetric spin metric epsilon_{AB}.

The question is how to represent SU(2) spatial spinor indices in the
existing TIndex/VBundle framework.

### 1.2 Design decision: separate VBundle, not a subtype of SpinIndex

**Recommendation: register a new VBundle `:SU2` with its own index alphabet.**

Rationale:

1. **SU(2) != SL(2,C) restriction.** An SU(2) spinor index A is _not_ the
   same as an SL(2,C) index restricted to a spatial slice. They live on
   different bundles: the SU(2) spin bundle is a subbundle of the
   SL(2,C) bundle selected by the foliation. Tracking this distinction
   at the VBundle level prevents invalid contractions between SU(2) and
   SL(2,C) indices.

2. **The existing VBundle mechanism is sufficient.** The `TIndex` struct
   already carries a `vbundle::Symbol` field. Creating `:SU2` as a VBundle
   (dimension 2, with its own index alphabet) slots cleanly into the
   existing infrastructure: `define_vbundle!`, `metric_cache`, `delta_cache`,
   contraction, and canonicalization all work without modification.

3. **No new types needed.** We do NOT need a new `SpaceSpinIndex` struct
   or subtype of `TIndex`. The vbundle field already provides the
   necessary discrimination. This avoids multiple-dispatch explosion
   and keeps the type hierarchy flat.

### 1.3 Index alphabet

```
SU(2) indices: :P, :Q, :R, :S, :T, :U  (uppercase, distinct from Tangent and SL2C)
```

These are chosen to avoid collision with:
- SL2C undotted: A, B, C, D, E, F
- SL2C_dot dotted: Ap, Bp, Cp, Dp, Ep, Fp
- Tangent: a, b, c, d, e, f, ...

### 1.4 Convenience constructors

```julia
space_spin_up(s::Symbol)   = TIndex(s, Up, :SU2)
space_spin_down(s::Symbol) = TIndex(s, Down, :SU2)
```

### 1.5 Predicates

```julia
is_space_spinor_index(idx::TIndex) = idx.vbundle === :SU2
```

---

## 2. Spatial Spin Metric and Soldering Form

### 2.1 Spatial spin metric epsilon_{PQ}

The SU(2) spin metric is the antisymmetric 2-form epsilon_{PQ} that raises
and lowers spatial spinor indices. It has the same algebraic structure as
the SL(2,C) spin metric (antisymmetric, 2D) but lives on the `:SU2` bundle.

**Key identity:** epsilon^{PR} epsilon_{QR} = delta^P_Q

**Registration:**

```julia
function define_space_spin_metric!(reg::TensorRegistry; manifold::Symbol=:M4)
    # Register eps_space: epsilon_{PQ}, antisymmetric
    register_tensor!(reg, TensorProperties(
        name=:eps_space, manifold=manifold, rank=(0, 2),
        symmetries=SymmetrySpec[AntiSymmetric(1, 2)],
        options=Dict{Symbol,Any}(:is_metric => true, :vbundle => :SU2, :vbundle_dim => 2)
    ))

    # Register delta_space: delta^P_Q
    register_tensor!(reg, TensorProperties(
        name=:delta_space, manifold=manifold, rank=(1, 1),
        options=Dict{Symbol,Any}(:is_delta => true, :vbundle => :SU2, :vbundle_dim => 2)
    ))

    # Populate vbundle-keyed caches
    reg.metric_cache[:SU2] = :eps_space
    reg.delta_cache[:SU2] = :delta_space
end
```

**Relation to 4D spin metric:** The SU(2) epsilon_{PQ} is _algebraically_
identical to the SL(2,C) epsilon_{AB} restricted to spatial directions.
The relationship is made precise by the time-normal spinor (Section 3.1):
if t^{AA'} is the future-pointing unit timelike spinor (normalized so
t^{AA'} t_{AA'} = 1), then the SU(2) bundle is the kernel of
t^{AA'}: the SU(2) index P corresponds to the restriction of the SL(2,C)
index A to the eigenspace where the reality condition holds.

However, at the abstract algebra level, there is no automatic conversion
rule between epsilon_{AB} (:SL2C) and epsilon_{PQ} (:SU2). They are
distinct tensors on distinct bundles. Conversion requires explicit
projection operators (Section 3).

### 2.2 Spatial soldering form tau^i_{PQ}

The spatial soldering form is the SU(2) analogue of the SL(2,C) soldering
form. It provides the isomorphism between the spatial tangent bundle and
symmetric rank-2 SU(2) spinors:

```
tau^i_{PQ} = tau^i_{(PQ)}    (symmetric in PQ, i.e., tau^i_{PQ} = tau^i_{QP})
```

This is a rank-3 tensor with:
- Slot 1: spatial tangent index (`:Spatial` or `:Tangent` restricted to 3D)
- Slots 2,3: SU(2) spinor indices, symmetric

**Key identities:**

```
tau^i_{PQ} tau_i^{RS} = delta^{(R}_P delta^{S)}_Q    (completeness)
tau^i_{PQ} tau^{j PQ}  = gamma^{ij}                   (spatial metric reconstruction)
gamma_{ij} = tau_{i PQ} tau_j^{PQ}                    (spatial metric from soldering)
```

The crucial difference from the 4D case: because SU(2) fundamental and
conjugate representations are equivalent, both spinor indices are of the
_same_ type (both `:SU2`), and the soldering form is _symmetric_ in its
spinor indices. In 4D, sigma^a_{AA'} carries one undotted and one dotted
index with no exchange symmetry.

**Registration:**

```julia
function define_space_soldering_form!(reg::TensorRegistry;
                                      manifold::Symbol=:M4,
                                      name::Symbol=:tau)
    register_tensor!(reg, TensorProperties(
        name=name, manifold=manifold, rank=(1, 2),
        symmetries=SymmetrySpec[Symmetric(2, 3)],  # symmetric in SU(2) pair
        options=Dict{Symbol,Any}(
            :is_soldering => true,
            :is_space_soldering => true,
            :index_vbundles => [:Spatial, :SU2, :SU2])))
end
```

### 2.3 Relation between 4D and 3D soldering forms

Given a foliation with unit timelike normal n^a (n_a n^a = -1), the
4D soldering form sigma^a_{AA'} and the 3D spatial soldering form
tau^i_{PQ} are related by:

```
tau^i_{AB} = (1/sqrt(2)) sigma^i_{A}^{A'} t_{BA'}
```

where t_{AA'} = (1/sqrt(2)) n_a sigma^a_{AA'} is the unit time-normal
spinor. Here the SL(2,C) index A is identified with the SU(2) index P
via the time-normal projection (see Section 3).

This relationship would be implemented as a conversion rule, not
hardcoded into the soldering form definition.

---

## 3. Sen Connection

### 3.1 Time-normal spinor

The starting point for spatial spinors is the _time-normal spinor_
(Sommers 1980). Given a spacelike hypersurface Sigma with future-directed
unit normal n^a (n_a n^a = -1), define:

```
t_{AA'} = (1/sqrt(2)) n_a sigma^a_{AA'}
```

This is the spinor equivalent of the unit normal. It satisfies:

```
t^{AA'} t_{AA'} = 1
```

The time-normal spinor selects the SU(2) subbundle of SL(2,C): SU(2)
transformations are those SL(2,C) transformations that preserve
t_{AA'}.

**Registration:**

```julia
function define_time_normal_spinor!(reg::TensorRegistry; manifold::Symbol=:M4)
    register_tensor!(reg, TensorProperties(
        name=:t_spinor, manifold=manifold, rank=(0, 2),
        symmetries=SymmetrySpec[],
        options=Dict{Symbol,Any}(
            :is_time_normal_spinor => true,
            :index_vbundles => [:SL2C, :SL2C_dot])))
end
```

### 3.2 The Sen connection D_{AA'}

The Sen connection (Sen 1981) is the unique derivative operator on spatial
spinors determined by:

1. D_{AA'} acts on SU(2) spinor fields on Sigma
2. D_{AA'} is compatible with the spatial spin metric: D_{AA'} epsilon_{PQ} = 0
3. D_{AA'} is compatible with the time-normal spinor: D_{AA'} t^{BB'} = 0

The Sen connection decomposes into spatial and temporal parts:

```
D_{AA'} = t_A^{A'} Delta + D_{A}^{(A')} D_{(spatial)}
```

where:
- Delta is the "time derivative" (evolution along n^a)
- D_{(spatial)} is the spatial covariant derivative compatible with gamma_{ij}

### 3.3 Relation to 4D spin_covd

The existing `spin_covd` implements nabla_{AA'} = sigma^a_{AA'} nabla_a
where nabla_a is the 4D spacetime covariant derivative.

The Sen connection D_{AA'} is related to the 4D spin covariant derivative by:

```
D_{AA'} phi_B = nabla_{AA'} phi_B + (1/2) K_{AA'B}^C phi_C
```

where K_{AA'BC} is the spinor form of the extrinsic curvature:

```
K_{AA'BC} = K_{ab} sigma^a_{B(A} t_{C)A'}
```

Equivalently, in terms of the spatial derivative D_i (the 3D Levi-Civita
connection of gamma_{ij}) and the extrinsic curvature K_{ij}:

```
D_{AA'} phi_B = t_A^{A'} (dot{phi}_B + N^i D_i phi_B)
              + tau^i_{AB} t^{B'A'} (D_i phi_B + (1/2) K_{ij} tau^j_{B}^C phi_C)
```

### 3.4 Implementation strategy

The Sen connection should be implemented analogously to `spin_covd` but
with the extrinsic curvature correction:

```julia
function sen_connection(expr::TensorExpr, undotted::Symbol, dotted::Symbol;
                        covd_name::Symbol=:D,
                        registry::TensorRegistry=current_registry()) -> TensorExpr
    # nabla_{AA'} phi_B + (1/2) K_{AA'B}^C phi_C
    # Build as: spin_covd + K correction term
end
```

The key difference from spin_covd is that the Sen connection:
1. Preserves the SU(2) structure (maps SU(2) spinors to SU(2) spinors)
2. Has a torsion-like correction term from the extrinsic curvature
3. Commutes with the time-normal spinor (D_{AA'} t^{BB'} = 0)

---

## 4. Interaction with Foliation

### 4.1 Current foliation infrastructure

The foliation module (`src/foliation/`) provides:

- `FoliationProperties`: stores temporal/spatial component split
- `define_foliation!`: registers a 3+1 foliation
- `split_spacetime`, `split_all_spacetime`: decompose indices into components
- `foliate_and_decompose`: end-to-end pipeline for SVT decomposition

The hypersurface module (`src/gr/hypersurface.jl`) provides:

- `define_hypersurface!`: registers unit normal, extrinsic curvature, induced metric
- `gauss_equation`, `codazzi_equation`: Gauss-Codazzi relations
- `SubmanifoldProperties`: stores embedding data

The ADM module (`src/hamiltonian/adm.jl`) provides:

- `ADMDecomposition`: lapse, shift, spatial metric
- `define_adm!`: registers ADM variables
- `adm_extrinsic_curvature_expr`, `adm_hamiltonian_constraint_expr`, etc.

### 4.2 How spatial spinors connect

Spatial spinors provide an alternative to the tensor 3+1 decomposition.
Instead of decomposing g_{ab} -> (N, N^i, gamma_{ij}), we decompose:

```
sigma^a_{AA'} -> (t_{AA'}, tau^i_{PQ})
```

The connection between the two pictures is:

| Tensor quantity    | Spinor equivalent           |
|-------------------|-----------------------------|
| n^a (normal)      | t^{AA'}                     |
| gamma_{ij}        | tau_{i PQ} tau_j^{PQ}       |
| K_{ij}            | K_{PQRS} = K_{ij} tau^i_{PQ} tau^j_{RS} |
| D_i (spatial covd)| Spatial part of D_{AA'}     |
| N (lapse)         | Scalar, unchanged           |
| N^i (shift)       | N^{PQ} = N^i tau_{i}^{PQ}  |

### 4.3 Integration approach

Spatial spinors should be layered _on top of_ the existing foliation
and hypersurface infrastructure, not replace it. The approach:

1. `define_space_spinor_structure!` requires a registered hypersurface
   (from `define_hypersurface!`) to know the unit normal.

2. Given the normal n^a and the 4D spinor structure (SL2C, sigma), it:
   - Constructs the time-normal spinor t_{AA'}
   - Registers the SU(2) VBundle and spatial spin metric
   - Registers the spatial soldering form tau^i_{PQ}
   - Registers conversion rules between 4D and spatial spinor quantities

3. The Sen connection wraps the existing covariant derivative with the
   extrinsic curvature correction.

This means a typical setup would be:

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    define_covd!(reg, :D; manifold=:M4, metric=:g)

    # 4D spinors
    define_spinor_structure!(reg; manifold=:M4, metric=:g)

    # Foliation
    define_hypersurface!(reg, :Sigma; ambient=:M4, metric=:g, signature=-1)

    # Spatial spinors (builds on both)
    define_space_spinor_structure!(reg; manifold=:M4, hypersurface=:Sigma)
end
```

---

## 5. Ashtekar Variable Connection

### 5.1 Ashtekar self-dual connection

The Ashtekar connection (Ashtekar 1986, 1991) is the central variable
in the loop quantum gravity program. It is defined as:

```
A^i_a = Gamma^i_a + beta * K^i_a
```

where:
- Gamma^i_a is the SU(2) spin connection (spatial part of the Levi-Civita
  connection, valued in the Lie algebra su(2))
- K^i_a = K_{ab} e^{bi} is the extrinsic curvature with one index
  converted to internal (su(2)) via the triad e^a_i
- beta is the Barbero-Immirzi parameter (beta = i for self-dual, beta = 1
  for real Barbero connection)

In spinor language, the SU(2) spin connection Gamma^i_a and the
extrinsic curvature K^i_a are naturally expressed using the spatial
soldering form:

```
Gamma_{a PQ} = Gamma^i_a tau_{i PQ}
K_{a PQ}     = K^i_a tau_{i PQ}
```

So the Ashtekar connection becomes:

```
A_{a PQ} = Gamma_{a PQ} + beta * K_{a PQ}
```

This is an SU(2) connection valued in symmetric rank-2 spinors
(spin-1 representation of SU(2)), acting on the spatial tangent index a.

### 5.2 Conjugate variable: densitized triad

The canonical conjugate to A^i_a is the densitized triad:

```
E^a_i = sqrt(det(gamma)) * e^a_i
```

In spinor form:

```
E^a_{PQ} = sqrt(det(gamma)) * tau^a_{PQ}
```

The fundamental Poisson bracket is:

```
{A^i_a(x), E^b_j(y)} = beta * delta^b_a delta^i_j delta^3(x-y)
```

### 5.3 Implementation design

The Ashtekar connection requires:

1. **SU(2) connection tensor:** A_{aPQ} with one Tangent (spatial) index
   and two symmetric SU(2) indices.

2. **Barbero-Immirzi parameter:** A scalar parameter beta stored in
   registry options. For the self-dual formulation, beta = i (imaginary);
   for the real Barbero formulation, beta is real (typically beta = 1).
   In abstract tensor algebra, beta is an opaque scalar (TScalar).

3. **Curvature of the Ashtekar connection:**

```
F^i_{ab} = partial_a A^i_b - partial_b A^i_a + epsilon^i_{jk} A^j_a A^k_b
```

In spinor form:

```
F_{ab PQ} = 2 D_{[a} A_{b] PQ} + A_{a P}^R A_{b RQ} - A_{b P}^R A_{a RQ}
```

This is the field strength of the SU(2) gauge connection on the spatial
manifold.

4. **Constraint algebra in Ashtekar variables:**

- Gauss constraint: G_i = D_a E^a_i = 0 (SU(2) gauge invariance)
- Vector (diffeomorphism) constraint: V_a = F^i_{ab} E^b_i = 0
- Scalar (Hamiltonian) constraint:
  H = epsilon_{ijk} F^{ij}_{ab} E^a_k / (2 sqrt(det(gamma))) = 0

These are equivalent to the ADM constraints but expressed in terms of
(A^i_a, E^a_i) instead of (gamma_{ij}, pi^{ij}).

### 5.4 Registration

```julia
function define_ashtekar_variables!(reg::TensorRegistry;
                                     manifold::Symbol=:M4,
                                     barbero_immirzi::Symbol=:beta_BI)
    # Requires: SU2 bundle, spatial soldering form, hypersurface

    # Ashtekar connection A_{a PQ}: mixed Tangent-SU2
    register_tensor!(reg, TensorProperties(
        name=:A_ashtekar, manifold=manifold, rank=(0, 3),
        symmetries=SymmetrySpec[Symmetric(2, 3)],  # symmetric in SU2 pair
        options=Dict{Symbol,Any}(
            :is_ashtekar_connection => true,
            :index_vbundles => [:Tangent, :SU2, :SU2],
            :barbero_immirzi => barbero_immirzi)))

    # Densitized triad E^a_{PQ}: mixed Tangent-SU2
    register_tensor!(reg, TensorProperties(
        name=:E_triad, manifold=manifold, rank=(1, 2),
        symmetries=SymmetrySpec[Symmetric(2, 3)],  # symmetric in SU2 pair
        options=Dict{Symbol,Any}(
            :is_densitized_triad => true,
            :index_vbundles => [:Tangent, :SU2, :SU2],
            :density_weight => 1)))

    # Barbero-Immirzi parameter (scalar)
    if !has_tensor(reg, barbero_immirzi)
        register_tensor!(reg, TensorProperties(
            name=barbero_immirzi, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[]))
    end
end
```

---

## 6. Public API Proposal

### 6.1 Setup functions

```julia
# Core: register SU(2) spinor bundle + spatial spin metric + spatial soldering form
define_space_spinor_bundles!(reg; manifold=:M4)
define_space_spin_metric!(reg; manifold=:M4)
define_space_soldering_form!(reg; manifold=:M4, name=:tau)

# All-in-one: requires registered hypersurface and 4D spinor structure
define_space_spinor_structure!(reg; manifold=:M4, hypersurface=:Sigma)

# Macro form
@space_spinor_manifold M4 hypersurface=Sigma
```

### 6.2 Index constructors and predicates

```julia
space_spin_up(s::Symbol) -> TIndex     # TIndex(s, Up, :SU2)
space_spin_down(s::Symbol) -> TIndex   # TIndex(s, Down, :SU2)
is_space_spinor_index(idx::TIndex) -> Bool
```

### 6.3 Time-normal spinor

```julia
define_time_normal_spinor!(reg; manifold=:M4, hypersurface=:Sigma)
time_normal_spinor_expr(; registry=current_registry()) -> TensorExpr
```

### 6.4 Sen connection

```julia
# Apply Sen connection D_{AA'} to a spatial spinor expression
sen_connection(expr, undotted, dotted; covd_name=:D, registry=current_registry())
sen_connection_expr(expr; covd_name=:D, registry=current_registry())

# Decompose Sen connection into spatial + temporal parts
sen_spatial_derivative(expr, idx; registry=current_registry())
sen_temporal_derivative(expr; registry=current_registry())
```

### 6.5 Conversion functions

```julia
# Convert between 4D SL(2,C) spinors and 3D SU(2) spinors
to_space_spinor(expr; hypersurface=:Sigma, registry=current_registry())
to_4d_spinor(expr; hypersurface=:Sigma, registry=current_registry())

# Convert between spatial tensor indices and SU(2) spinor pairs
to_space_spinor_indices(expr, reg)   # V_i -> V_{PQ} via tau^i_{PQ}
to_spatial_indices(expr, reg)         # V_{PQ} -> V_i via tau_{i}^{PQ}
```

### 6.6 Ashtekar variables

```julia
define_ashtekar_variables!(reg; manifold=:M4, barbero_immirzi=:beta_BI)

# Build Ashtekar connection from spin connection + extrinsic curvature
ashtekar_connection_expr(; registry=current_registry()) -> TensorExpr

# Field strength of Ashtekar connection
ashtekar_curvature_expr(; registry=current_registry()) -> TensorExpr

# Constraint expressions in Ashtekar variables
gauss_constraint_ashtekar(; registry=current_registry()) -> TensorExpr
vector_constraint_ashtekar(; registry=current_registry()) -> TensorExpr
scalar_constraint_ashtekar(; registry=current_registry()) -> TensorExpr
```

### 6.7 Irreducible decomposition (SU(2) version)

```julia
# Decompose SU(2) spinor into symmetric + trace parts
# (identical algebraic structure to SL(2,C) case, different vbundle)
irreducible_decompose_su2(expr; registry=current_registry()) -> TensorExpr
```

---

## 7. Implementation Plan

### Phase 1: Core SU(2) spinor infrastructure

**File:** `src/spinors/space_spinors.jl`

1. `define_space_spinor_bundles!(reg)` — register `:SU2` VBundle (dim=2,
   indices=[:P,:Q,:R,:S,:T,:U])

2. `define_space_spin_metric!(reg)` — register `:eps_space` (antisymmetric)
   and `:delta_space`, populate metric_cache/delta_cache for `:SU2`

3. Index constructors: `space_spin_up`, `space_spin_down`

4. Predicate: `is_space_spinor_index`

**Testing:** Verify VBundle registration, metric contraction eps^{PR}
eps_{QR} = delta^P_Q, index raising/lowering.

**Estimated effort:** Small. Directly parallels `spinor_bundles.jl` and
`spin_metric.jl`.

### Phase 2: Spatial soldering form

**File:** `src/spinors/space_soldering.jl`

1. `define_space_soldering_form!(reg)` — register `:tau` with
   Symmetric(2,3) symmetry and completeness/metric-reconstruction rules

2. Conversion functions: `to_space_spinor_indices`, `to_spatial_indices`

3. Contraction rules:
   - Completeness: tau^i_{PQ} tau_i^{RS} = delta^{(R}_P delta^{S)}_Q
   - Metric reconstruction: tau^i_{PQ} tau^{j PQ} = gamma^{ij}

**Testing:** Verify contraction rules fire correctly. Test round-trip
conversion V_i -> V_{PQ} -> V_i.

**Estimated effort:** Medium. The symmetrized completeness relation
(delta^{(R}_P delta^{S)}_Q) requires careful index handling since both
indices are of the same type, unlike the 4D case where the product
delta^B_A delta^{B'}_{A'} has no symmetrization.

### Phase 3: Time-normal spinor and SL(2,C) <-> SU(2) conversion

**File:** `src/spinors/time_normal.jl`

1. `define_time_normal_spinor!(reg)` — register t_{AA'} with
   normalization rule t^{AA'} t_{AA'} = 1

2. Projection operators:
   - Pi^A_B = t^{AA'} t_{BA'} (projects onto SU(2) part of SL2C index)
   - Relation: delta^A_B = Pi^A_B + (conjugate part)

3. `to_space_spinor(expr)` — project SL(2,C) spinor fields to SU(2)
   using t_{AA'}

**Dependency:** Requires Phase 1 + registered hypersurface.

**Estimated effort:** Medium. The main subtlety is correctly handling the
identification between SL(2,C) index A and SU(2) index P under the
time-normal projection. This identification is the content of the
Sommers (1980) construction.

### Phase 4: Sen connection

**File:** `src/spinors/sen_connection.jl`

1. `sen_connection(expr, A, Ap)` — apply D_{AA'} = nabla_{AA'} + K correction

2. Sen connection properties:
   - Metric compatibility: D_{AA'} eps_{PQ} = 0
   - Time-normal compatibility: D_{AA'} t^{BB'} = 0
   - Torsion-free on Sigma

3. `sen_spatial_derivative`, `sen_temporal_derivative` — decompose
   D_{AA'} into spatial (D_i) and temporal (dot) parts

4. Sen-Witten identity: D_{AA'} D^{BA'} phi_B = ... (the identity
   underlying Witten's positive energy proof)

**Dependency:** Requires Phase 1-3 + registered covd + hypersurface with
extrinsic curvature.

**Estimated effort:** Large. The K-correction term involves the spinor
form of the extrinsic curvature, which requires the spatial soldering
form from Phase 2. The decomposition into spatial + temporal parts
requires careful bookkeeping of lapse and shift.

### Phase 5: Ashtekar variables

**File:** `src/spinors/ashtekar.jl`

1. `define_ashtekar_variables!(reg)` — register A_{aPQ}, E^a_{PQ},
   beta_BI

2. `ashtekar_connection_expr()` — A = Gamma + beta K in spinor form

3. `ashtekar_curvature_expr()` — field strength F_{abPQ}

4. Constraint algebra:
   - Gauss: G_{PQ} = D_a E^a_{PQ} + [A_a, E^a]_{PQ} = 0
   - Vector: V_a = F_{ab PQ} E^{b PQ} = 0
   - Scalar: H = epsilon_{PQ RS} F^{PQ}_{ab} E^{a RS} / (2 sqrt(gamma)) = 0

5. Integration with existing Hamiltonian module: verify that the
   Ashtekar constraints reproduce the ADM constraint algebra under
   the variable change (gamma, pi) <-> (A, E).

**Dependency:** Requires all previous phases + ADM module.

**Estimated effort:** Large. The Ashtekar formulation introduces
SU(2) gauge structure and the Barbero-Immirzi parameter. The constraint
algebra involves structure functions (not just structure constants) due
to the Hamiltonian constraint.

### Phase 6: Convenience wrapper

**File:** `src/spinors/space_spinor_setup.jl`

1. `define_space_spinor_structure!(reg; manifold, hypersurface)` — all-in-one

2. `@space_spinor_manifold` macro

3. SU(2) irreducible decomposition (adapt from `irreducible.jl`)

**Estimated effort:** Small. Convenience wrappers calling Phase 1-4 functions.

### Dependencies between phases

```
Phase 1 (SU2 bundle + metric)
  |
  v
Phase 2 (spatial soldering form)
  |
  v
Phase 3 (time-normal spinor, SL2C <-> SU2)
  |
  v
Phase 4 (Sen connection)          Phase 6 (convenience)
  |
  v
Phase 5 (Ashtekar variables)
```

### Testing strategy

Each phase gets a corresponding test file `test/test_space_spinors_phaseN.jl`
with:

1. **Phase 1 tests:** VBundle registration, metric contraction identities,
   index algebra (raising/lowering with eps_space).

2. **Phase 2 tests:** Soldering form contraction rules (completeness,
   metric reconstruction), round-trip index conversion.

3. **Phase 3 tests:** Time-normal normalization, projection operator
   idempotency (Pi^2 = Pi), SL(2,C) -> SU(2) -> SL(2,C) round trip.

4. **Phase 4 tests:** Sen connection metric compatibility, time-normal
   compatibility, comparison with 4D spin_covd + K correction.

5. **Phase 5 tests:** Ashtekar connection = Gamma + beta K, field
   strength symmetries, Gauss constraint under gauge transformation,
   equivalence of Ashtekar and ADM constraints.

### Risk assessment

1. **Index type collision:** The `:SU2` vbundle must be kept strictly
   separate from `:SL2C` to prevent invalid contractions. The existing
   vbundle mechanism handles this naturally, but conversion functions
   must be carefully written.

2. **Symmetric soldering form:** Unlike the 4D sigma^a_{AA'} where the
   two spinor indices live on different bundles, tau^i_{PQ} has both
   indices on `:SU2`. The completeness relation involves symmetrized
   deltas, which is more complex than the 4D product of independent
   deltas. The canonicalization engine must handle this correctly.

3. **Self-dual vs real Ashtekar variables:** The original Ashtekar
   formulation uses complex (self-dual) variables (beta = i), which
   requires reality conditions. The real Barbero formulation (beta real)
   avoids this at the cost of a more complicated Hamiltonian constraint.
   The implementation should support both via the Barbero-Immirzi
   parameter, but complex scalar arithmetic may need CAS extension
   support.

4. **Density weights:** The densitized triad E^a_i has density weight 1.
   The existing TensorProperties has a `weight` field that could track
   this, but density weight arithmetic in products is not yet implemented
   in the simplify pipeline. This may need enhancement.
