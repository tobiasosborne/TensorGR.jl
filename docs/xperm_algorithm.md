# xperm.c Algorithm Analysis

This document provides a precise analysis of the canonicalization algorithm
implemented in `deps/xperm.c`, focusing on the internal data flow, the
distinction between **slot positions** and **names**, and the two notational
conventions used by the code.

---

## 1. Terminology: Slots vs Names

Throughout xperm.c, a permutation of degree `n` is stored as an array `p[0..n-1]`
in **images notation**: `p[i-1]` is the image of point `i`.

There are two competing interpretations of what a permutation *means*:

| Convention | Permutation maps ... | Used by ... |
|---|---|---|
| **"Our notation"** (xAct/Mathematica) | slot --> slot (a symmetry generator swaps slots) | `canonical_perm` interface, the Julia wrapper |
| **"Renato's notation"** (Portugal et al.) | slot --> name (the permutation tells you which index name sits in each slot) | `canonical_perm_ext`, `coset_rep`, `double_coset_rep` |

**Slot position** = the physical position in the tensor product (which argument
of which tensor). Slots are numbered 1..n, where the last two slots (n-1, n)
encode sign.

**Name** = the canonical integer label assigned to an abstract index. In the
canonical (identity) configuration, name `i` sits in slot `i`. Free indices
and dummy pairs each get distinct names.

The relationship: if `g` is a permutation in Renato's notation, then
`g[s] = name sitting in slot s`. The inverse `g^{-1}` maps names to slots:
`g^{-1}[name] = slot where that name sits`.

---

## 2. `canonical_perm` (line 2248): The Wrapper

### Signature

```c
void canonical_perm(int *PERM,
    int SGSQ, int *base, int bl, int *GS, int m, int n,
    int *freeps, int fl, int *dummyps, int dpl, int ob, int metricQ,
    int *CPERM);
```

### Input convention

In this interface, everything uses **"our notation"** (xAct convention):

- `PERM`: a permutation in images notation, mapping slots to slots. It
  represents the current index configuration as a permutation *of slots*.
- `freeps[0..fl-1]`: the **slot positions** where free indices currently sit.
- `dummyps[0..2*dpl-1]`: the **slot positions** where dummy indices currently sit,
  in pairs `[up1, down1, up2, down2, ...]`.
- `GS`: generating set for the slot symmetry group S, where each generator is a
  slot-to-slot permutation.

### The "Change to Renato's notation" block (lines 2267-2274)

```c
inverse(PERM, PERM1, n);
for (i=0; i<fl; i++) {
    frees[i] = onpoints(freeps[i], PERM1, n);
}
for (i=0; i<2*dpl; i++) {
    dummies[i] = onpoints(dummyps[i], PERM1, n);
}
```

#### What `inverse(PERM, PERM1, n)` computes

`inverse` computes the group-theoretic inverse: if `PERM` maps point `i` to
`PERM[i-1]`, then `PERM1` satisfies `PERM1[PERM[i-1] - 1] = i`, i.e.,
`PERM1 = PERM^{-1}`.

In the xAct convention, `PERM` acts on slots. So `PERM1 = PERM^{-1}` is also a
slot permutation: `PERM1[s-1]` gives the slot that maps *to* slot `s` under
`PERM`.

But there is a deeper reading. xperm uses the convention that the *identity*
permutation corresponds to the canonical index configuration. The current
configuration is described by `PERM`. In Renato's notation, we want a
permutation `g` such that `g(slot) = name in that slot`.

In the canonical configuration (identity), name `i` sits in slot `i`. Applying
`PERM` (a slot symmetry) rearranges slots. If we write the current
configuration as "the identity configuration acted on by PERM", then `PERM`
moved the content of slot `s` to slot `PERM(s)`. Therefore the name originally
in slot `s` (which was `s` in the canonical config) is now in slot `PERM(s)`.
Equivalently, slot `t` now contains name `PERM^{-1}(t)`.

So: **`PERM1 = PERM^{-1}` is exactly Renato's `g`: it maps slot --> name.**

#### What `onpoints(freeps[i], PERM1, n)` computes

`onpoints(point, p, n)` returns `p[point-1]`, i.e., the image of `point`
under permutation `p`.

So `onpoints(freeps[i], PERM1, n) = PERM1[freeps[i] - 1]`.

Since `PERM1` maps slot --> name, this computes:

```
frees[i] = PERM1(freeps[i]) = name of the index sitting in slot freeps[i]
```

**The conversion transforms slot positions of free/dummy indices into their
names.** This is what `canonical_perm_ext` expects: `frees` and `dummies`
contain **names**, not slot positions.

#### Mathematical summary

```
PERM1 = PERM^{-1}            (Renato's g: slot -> name)
frees[i] = PERM1(freeps[i])  (name of free index at slot freeps[i])
dummies[i] = PERM1(dummyps[i])  (name of dummy index at slot dummyps[i])
```

### Call to `canonical_perm_ext`

```c
canonical_perm_ext(PERM1, n, SGSQ, base, bl, GS, m,
    frees, fl, &vds, 1, dummies, 2*dpl, &mQ,
    vrs, 0, repes, 0,
    PERM2);
```

`PERM1` (= Renato's `g`) and the names-based `frees`/`dummies` are passed.
The generators `GS` are passed unchanged -- they act on slots in both conventions.

### The "Change back to our notation" (lines 2282-2284)

```c
if (PERM2[0] != 0) inverse(PERM2, CPERM, n);
else copy_list(PERM2, CPERM, n);
```

`canonical_perm_ext` returns `PERM2` in Renato's notation (slot --> name).
To convert back to xAct's convention, we need the inverse: `CPERM = PERM2^{-1}`.

The check `PERM2[0] != 0` handles the special case where the tensor is zero
(indicated by the all-zeros permutation), in which case we just copy the zeros.

---

## 3. `canonical_perm_ext` (line 2349): The Core Algorithm

### Signature

```c
void canonical_perm_ext(int *PERM, int n,
    int SGSQ, int *base, int bl, int *GS, int m,
    int *frees, int fl,
    int *vds, int vdsl, int *dummies, int dl, int *mQ,
    int *vrs, int vrsl, int *repes, int rl,
    int *CPERM);
```

### Input convention (Renato's notation)

- `PERM`: permutation mapping slot --> name. `PERM[s-1] = name` in slot `s`.
- `GS`: generating set for the slot symmetry group S (slot-to-slot permutations).
- `frees[0..fl-1]`: **names** of free indices (NOT slot positions).
- `dummies[0..dl-1]`: **names** of dummy indices, in pairs `[up1,down1, ...]`.
- `vds[0..vdsl-1]`: lengths of dummy-sets (how many names belong to each
  vbundle). For the simple `canonical_perm` wrapper, `vdsl=1` and `vds[0]=2*dpl`.
- `mQ[0..vdsl-1]`: metric symmetry signs per dummy-set (+1=symmetric, -1=antisymmetric, 0=none).
- `repes`, `vrs`: repeated-index sets (not used in the simple wrapper).

### Step 1: Compute SGS (lines 2375-2383)

```c
if (!SGSQ) { /* SGSQ=0: Compute a Strong Generating Set */
    nonstable_points(base, bl, GS, m, n, tmpbase, &tmpbl);
    schreier_sims(tmpbase, tmpbl, GS, m, n,
        newbase, &newbl, newGS, &newm, &num);
} else {     /* SGSQ=1: GS is already a SGS */
    copy_list(base, newbase, bl); newbl=bl;
    copy_list(GS, *newGS, m*n); newm=m;
}
```

**SGSQ=0**: The caller only provides a generating set (not necessarily strong).
The function first calls `nonstable_points` to extend `base` with moved points
of generators, then runs `schreier_sims` to compute a proper SGS. The result is
stored in `newbase`/`*newGS` with lengths `newbl`/`newm`.

**SGSQ=1**: The caller guarantees that `(base, GS)` is already a strong
generating set. The data is just copied.

### Step 2: Compute slots of free indices (lines 2393-2397)

```c
inverse(PERM, PERM1, n);
for (i=0; i<fl; i++) {
    freeps[i] = onpoints(frees[i], PERM1, n);
}
```

Here `PERM` maps slot --> name, so `PERM^{-1}` maps name --> slot.
`freeps[i] = PERM^{-1}(frees[i])` = the **slot position** currently
occupied by the free index with name `frees[i]`.

This converts free-index **names** back to **slot positions** for use by
`coset_rep`, which operates on slots.

### Step 3: Coset representative for free indices (lines 2399-2400)

```c
coset_rep(PERM, n, newbase, newbl, *newGS, &newm, freeps, fl, PERM1);
```

`coset_rep` finds an element `s` of the slot symmetry group S such that
`s * PERM` places the free indices in their canonical (least) positions.

Inputs:
- `PERM`: current slot --> name permutation
- `newbase`, `*newGS`: the SGS for S
- `freeps`: **slot positions** of free indices (updated in-place during the algorithm)
- Output `PERM1`: the canonicalized `s * PERM` (still slot --> name)

After this step, `PERM1` is the representative of the right coset `S * PERM`
with free indices placed canonically. The SGS is also modified in-place:
`newm` is reduced to generators that stabilize the chosen free-index slots.

### Step 4: Stabilize the SGS (lines 2412-2418)

```c
complement(newbase, newbl, freeps, fl, 1, tmpbase, &tmpbl);
copy_list(tmpbase, newbase, tmpbl); newbl=tmpbl;
stabilizer(freeps, fl, *newGS, newm, n, *newGS, &newm);
```

The base is reduced to remove the free-index positions (they are now fixed).
The SGS is reduced to generators that stabilize all free-index slots.
This residual group is the part of S that can still act on dummy slots.

### Step 5: Double-coset representative for dummies (lines 2430-2432)

```c
double_coset_rep(PERM1, n, newbase, newbl, *newGS, newm,
    vds, vdsl, dummies, dl, mQ,
    vrs, vrsl, repes, rl, PERM2);
```

This is the core Butler-Portugal algorithm. It finds the canonical
representative of the double coset `S \ {PERM1} / D`, where:
- S is the residual slot symmetry (stabilizer of free slots)
- D is the dummy-index symmetry group (generated by name permutations that
  swap dummy pairs and exchange within pairs)

Inputs:
- `PERM1`: the coset-rep from Step 3 (slot --> name)
- `newbase`, `*newGS`, `newm`: SGS for residual S
- `dummies[0..dl-1]`: **names** of dummy indices (used to construct group D)
- `mQ`: metric symmetry signs
- Output `PERM2`: the canonical double-coset representative (slot --> name)

### Step 6: Copy to output (line 2438)

```c
copy_list(PERM2, CPERM, n);
```

If there are no dummies, `PERM1` is copied directly (line 2409).

### Final composition

The overall result is: starting from `PERM` (= `g`, the original slot-->name map),
the algorithm finds `s in S` and `d in D` such that `s * g * d` is the canonical
representative. The output `CPERM` is this `s * g * d`.

---

## 4. `coset_rep` (line 1281): Free-Index Canonicalization

### Signature

```c
void coset_rep(int *p, int n,
    int *base, int bl, int *GS, int *m,
    int *freeps, int fl,
    int *cr);
```

### Inputs

- `p`: permutation to canonicalize (slot --> name, Renato's `g`)
- `base`, `GS`, `*m`: SGS for slot symmetry group S
- `freeps[0..fl-1]`: **slot positions** of free indices (mutated in-place)
- Output `cr`: the canonical coset representative

### What "frees" means here

`freeps` contains **slot numbers** (not names). The algorithm iterates over
base points `b = base[i]` (which are also slot numbers), finds the orbit of `b`
under S, intersects with `freeps` (the current free slots), then picks the one
whose current name (`onpoints(orbit1[j], PERM, n)`) is smallest according to
the base ordering.

### Algorithm outline

For each base point `b`:
1. Compute the orbit of slot `b` under the current generators.
2. Intersect with `freeps` to find which free slots can be moved to slot `b`.
3. For each candidate free slot `orbit1[j]`, look up the name currently there:
   `deltap[j] = PERM(orbit1[j])` = name in that slot.
4. Sort names by base ordering (`sortB`) to find the smallest.
5. Find which slot holds that smallest name. Call it `pp`.
6. Trace the Schreier vector to find `om in S` such that `b^{om} = pp`
   (i.e., `om` moves slot `pp` to slot `b`).
7. Update `PERM := om * PERM` (left-multiply by slot symmetry).
8. Update `freeps` by applying `om^{-1}` to track where free slots moved.
9. Stabilize generators to fix slot `b`.

After the loop, the result `cr` has free indices in their canonical slots, and
`*m` is reduced to the stabilizer subgroup.

---

## 5. `double_coset_rep` (line 1728): Dummy-Index Canonicalization

### Signature

```c
void double_coset_rep(int *g, int n,
    int *base, int bl, int *GS, int m,
    int *vds, int vdsl, int *dummies, int dl, int *mQ,
    int *vrs, int vrsl, int *repes, int rl,
    int *dcr);
```

### Inputs

- `g`: permutation (slot --> name, Renato's notation)
- `base`, `GS`, `m`: SGS for residual slot symmetry S (after free-index fixing)
- `dummies[0..dl-1]`: **names** of dummy indices, in pairs
- `mQ[0..vdsl-1]`: metric symmetry per dummy-set
- Output `dcr`: the canonical double-coset representative

### What "dummies" means here

`dummies` contains **names** (index labels), NOT slot positions. The group D is
constructed from these names: `SGSofdummyset` builds generators that permute
names (swapping pairs like `(a_up, b_up)(a_down, b_down)` and, if the metric
is (anti)symmetric, also swapping within a pair `(a_up, a_down)`).

The algorithm uses `g^{-1}` (name --> slot) to convert names to slots when
needed:

```c
inverse(g, ig, n);
for (i=0; i<dril; i++) {
    drummyslots[i] = onpoints(drummies[i], ig, n);  /* name -> slot */
}
```

### Algorithm outline (Butler-Portugal)

The algorithm searches through base points in order, at each step finding the
least name that can appear in the current slot under the combined action of S
(from the left) and D (from the right):

1. **Extend base** `bS` to cover all dummy slots.
2. **Main loop** over `i = 1..bSl`:
   a. Compute orbit of slot `bS[i-1]` under current S (Schreier vector).
   b. Compute SGS for current D (from remaining dummies/repes).
   c. Compute orbits under D.
   d. Use the ALPHA/TAB data structure to find all possible images:
      for each `(s,d)` pair stored so far, compute `s * g * d` applied to
      the current base slot, collecting all reachable names (accounting for
      D-orbits).
   e. Sort images by `bSsort` ordering, pick the least: `p[i-1]`.
   f. Find `s1 in S`, `d1 in D` achieving that image; store in ALPHA.
   g. **Consistency check**: verify no permutation appears with both signs
      (would mean the tensor is zero).
   h. Stabilize S at slot `bS[i-1]`; drop the used dummy pair from D.

3. **Result**: compose `s * g * d` from the final ALPHA entry.

The ALPHA data structure stores a tree of partial solutions `(L, s, d)` where:
- `L` is a list of intermediate slots visited so far
- `s` is the accumulated S-element
- `d` is the accumulated D-element
- The composition `s * g * d` agrees with the canonical image on all slots
  processed so far.

---

## 6. `schreier_sims` (line 980): SGS Computation

### Signature

```c
void schreier_sims(int *base, int bl, int *GS, int m, int n,
    int *newbase, int *nbl, int **newGS, int *nm, int *num);
```

### Memory contract

The critical memory requirement is stated in the comment at line 973-977:

> We assume that enough space (at least, and typically, m*n integers)
> has been already allocated for newGS. We also assume that newGS can
> be reallocated. That's why we do not send a pointer, but a pointer
> to that pointer.

The function receives `int **newGS` -- a pointer to a pointer. The inner
pointer `*newGS` must point to C-heap memory (allocated via `malloc` or
`Libc.malloc` from Julia), because `schreier_sims_step` may `realloc` it:

```c
/* line 1177 in schreier_sims_step */
*newGS = (int*)realloc(*newGS, ((*nm)+1)*n*sizeof(int));
```

### When realloc happens

`schreier_sims_step` (called from the main loop of `schreier_sims`) discovers
new Schreier generators. Whenever a new generator is found that is not already
in the subgroup H^(i), it is appended to `*newGS`, which requires reallocation:

1. Line 1177: `*newGS = realloc(*newGS, (nm+1)*n*sizeof(int))` -- grow by one
   permutation slot.
2. Lines 1014-1015: internal buffers `GS2` and `stab` are also reallocated when
   `*nm > m2`, but these are local and freed within `schreier_sims`.

**The caller MUST ensure `*newGS` is C-heap allocated (not Julia GC-managed).**
This is why the Julia wrapper uses `Libc.malloc`:

```julia
newGS_ptr = Libc.malloc(m * n * sizeof(Cint))
newGS_ref = Ref{Ptr{Cint}}(Ptr{Cint}(newGS_ptr))
```

And frees with `Libc.free(newGS_ref[])` after extracting the results.

### Recursive structure

`schreier_sims` iterates `i` from `nbl` down to 1, calling
`schreier_sims_step` at each level. `schreier_sims_step` can recursively call
itself (line 1218) when it discovers new generators that require ensuring the
SGS at deeper levels. Each recursive call may further realloc `*newGS`.

---

## 7. The Julia Wrapper: `xperm_canonical_perm` (wrapper.jl)

### What it passes

```julia
function xperm_canonical_perm(perm::Perm, base::Vector{Int32},
                              generators::Vector{Perm},
                              free_slots::Vector{Int32},
                              dummy_pairs::Vector{Int32},
                              n::Int)
```

The wrapper calls `canonical_perm` (NOT `canonical_perm_ext`), passing:

- `perm.data` as `PERM` (the slot-to-slot permutation in xAct convention)
- `SGSQ = 0` (generators are not necessarily a SGS)
- `free_slots` as `freeps` -- documented as "positions of free index slots"
- `dummy_pairs` as `dummyps` -- documented as "paired dummy slot positions"
- `ob = 1` (unused but kept for compatibility)
- `metricQ = 1` if there are dummy pairs, else `0`

### Correctness analysis

**This is correct.** The `canonical_perm` wrapper expects:

| Parameter | Expected by C | Passed by Julia | Match? |
|---|---|---|---|
| `PERM` | Slot-to-slot perm (xAct convention) | `perm.data` | Yes (assuming the caller constructs the perm correctly) |
| `freeps` | **Slot positions** of free indices | `free_slots` (documented as slot positions) | Yes |
| `dummyps` | **Slot positions** of dummy indices | `dummy_pairs` (documented as slot positions) | Yes |
| `GS` | Slot symmetry generators | `generators` flattened | Yes |
| `dpl` | Number of dummy *pairs* | `div(length(dummy_pairs), 2)` | Yes |

The `canonical_perm` wrapper handles the notation conversion internally
(computing `PERM1 = PERM^{-1}` and converting slot positions to names via
`onpoints`), and converts back at the end with `inverse(PERM2, CPERM, n)`.

**Important subtlety**: `free_slots` and `dummy_pairs` must contain the **current
slot positions** of the indices in the specific permutation `perm`, not the
"canonical" positions. The conversion `onpoints(freeps[i], PERM^{-1}, n)`
correctly maps current-slot-position to name regardless.

Actually, re-reading more carefully: the `freeps` and `dummyps` passed to
`canonical_perm` are described as "initial slots" in the comment at line 2241-2242:

> The list dummyps of dpl pairs of (initial) slots of dummies.
> The list freeps (length fl) contains the slots of the free indices.

The word "initial" here refers to the slots in the **canonical (identity)**
configuration. Since in the identity configuration, name `i` sits in slot `i`,
the "initial slot" of a free index *is* numerically equal to its name.

So `freeps` and `dummyps` are equivalently:
- The slot positions in the canonical configuration, OR
- The names of the free/dummy indices

This is why the conversion `onpoints(freeps[i], PERM^{-1}, n)` works correctly:
it maps name --> actual current slot under the given permutation, which then
gets converted back to name inside `canonical_perm_ext`.

Wait -- let's trace this more carefully. `canonical_perm` does:

```c
inverse(PERM, PERM1, n);               // PERM1 = PERM^{-1}
frees[i] = onpoints(freeps[i], PERM1, n);  // = PERM^{-1}(freeps[i])
```

If `freeps[i]` is the **slot** where free index `i` sits in the **identity**
configuration (= name of the index = `freeps[i]` numerically), and `PERM` maps
the identity config to the current config by rearranging slots, then
`PERM1(freeps[i]) = PERM^{-1}(freeps[i])`.

But `PERM` in xAct convention maps old-slot to new-slot. `PERM^{-1}` maps
new-slot to old-slot. So `PERM^{-1}(freeps[i])` would give the slot that
`freeps[i]` came from... this needs more care.

**Cleaner derivation**: Let's say `PERM` represents a slot permutation `sigma`.
The current index configuration is `sigma` applied to the identity. In the
identity config, slot `s` has name `s`. After applying `sigma`, slot `s` has
the content that was in slot `sigma^{-1}(s)`, which has name `sigma^{-1}(s)`.

So the Renato permutation `g(s) = sigma^{-1}(s) = PERM^{-1}(s)`, confirming
`PERM1 = g`.

Now: `frees[i] = g(freeps[i]) = PERM^{-1}(freeps[i])`. Since `freeps[i]`
is the *initial* slot (= name) of the `i`-th free index, and `g` maps
slot -> name, this gives `g(freeps[i])`.

But wait: `g` maps slot to name. `freeps[i]` is a name (= initial slot).
We want to pass the **name** of the free index to `canonical_perm_ext`.
The name of the `i`-th free index is `freeps[i]` itself (it's defined as
the initial slot = name). So the correct value to pass is just `freeps[i]`.

But the code computes `PERM^{-1}(freeps[i])` instead. Is this correct?

The answer is: **it is correct IF `freeps[i]` is the initial slot of the
free index AND the "name" in Renato's formalism is literally what sits at
that slot under `g`.** The names are not intrinsic labels -- they are defined
by the permutation. The name of the index at initial-slot `s` under
configuration `g` is `g(s)`. So to tell `canonical_perm_ext` which names are
free, we need `g(freeps[i]) = PERM^{-1}(freeps[i])`.

This is exactly what the code computes. The Julia wrapper passes `freeps`
as initial slot positions (= the slots in the identity config), which is
correct for the `canonical_perm` interface.

### Summary of correctness

The Julia wrapper correctly:
1. Passes `free_slots` as initial slot positions (which `canonical_perm`
   converts to names internally).
2. Passes `dummy_pairs` as initial slot positions (similarly converted).
3. Uses `SGSQ=0` since raw generators are passed (not a SGS).
4. Uses `Libc.malloc` for the SGS buffer in `xperm_schreier_sims`.
5. The `canonical_perm` function handles all notation conversion internally.

---

## 8. Summary of the Complete Data Flow

```
Julia caller
    |
    |  perm (slot-to-slot), free_slots (initial slots), dummy_pairs (initial slots)
    v
canonical_perm  [xAct notation]
    |
    |  1. PERM1 = PERM^{-1}  (now slot -> name, i.e., Renato's g)
    |  2. frees[i] = PERM1(freeps[i])  (convert initial slots to names)
    |  3. dummies[i] = PERM1(dummyps[i])  (convert initial slots to names)
    |
    v
canonical_perm_ext  [Renato's notation]
    |
    |  4. If SGSQ=0: compute SGS via schreier_sims
    |  5. Convert frees (names) back to freeps (current slots) via PERM^{-1}
    |  6. coset_rep: find s in S minimizing free-index placement
    |     -> PERM1 = s * g  (free indices canonical)
    |  7. Stabilize S at free slots
    |  8. double_coset_rep: find s' in S_stab, d in D minimizing dummy placement
    |     -> PERM2 = s' * PERM1 * d  (all indices canonical)
    |
    v
canonical_perm  [back to xAct notation]
    |
    |  9. CPERM = PERM2^{-1}  (convert slot->name back to slot->slot)
    |
    v
Julia caller receives CPERM
```

Sign information is encoded in the last two points of the permutation:
- `CPERM[n-1] = n-1, CPERM[n] = n` means positive sign (identity on sign slots)
- `CPERM[n-1] = n, CPERM[n] = n-1` means negative sign (sign slots swapped)
- `CPERM = all zeros` means the tensor expression is zero
