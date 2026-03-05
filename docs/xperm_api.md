# xperm.c API Reference

**Source**: `deps/xperm.c` -- (C) Jose M. Martin-Garcia 2003-2011, GNU GPL.

This document provides a precise specification of xperm's public API functions,
focusing on calling conventions, memory ownership, and the critical distinction
between "names" (index identities) and "slots" (tensor positions).

---

## Fundamental Conventions

### 1-Indexed Permutations in Images Notation

All permutations in xperm use **1-indexed Images notation**. A permutation of
degree `n` is stored as an `int` array of length `n` where `perm[i-1]` is the
image of point `i` (using 0-based C array indexing to represent 1-based
mathematical points).

The identity permutation of degree 3 is `{1, 2, 3}`.

The permutation `{3, 1, 2}` maps point 1 to 3, point 2 to 1, point 3 to 2.

### Sign Encoding

Sign is encoded in the **last two positions** of a degree-`n` permutation
(positions `n-1` and `n` in 1-indexed terms, i.e., C array indices `n-2` and
`n-1`):

- **Positive sign**: `perm[n-2] == n-1` and `perm[n-1] == n` (i.e., the last
  two points are fixed: the identity on those points).
- **Negative sign**: `perm[n-2] == n` and `perm[n-1] == n-1` (the last two
  points are swapped).

### Zero Result

When the canonicalization determines that the tensor expression is identically
zero (e.g., symmetric indices contracted with antisymmetric ones), the result
permutation is **all zeros**: `{0, 0, ..., 0}`.

In `canonical_perm` (the wrapper), this is detected by checking `PERM2[0] != 0`.

### The "n" Convention: Degree Includes Sign Slots

The degree `n` of the permutations **includes** the two sign-encoding points.
For a tensor with `k` actual index slots:

```
n = k + 2
```

The last two positions (points `n-1` and `n`) are reserved for sign encoding
and are not actual tensor slots.

### Names vs. Slots: The Two Conventions

This is the most critical distinction and the source of most wrapper bugs.

**xTensor convention** (used by the Mathematica package and `canonical_perm`'s
external interface):
- A permutation `g` maps **index-names to slot-numbers**.
- `p = onpoints(i, g, n)` means: "index `i` is at slot `p`."
- Equivalently: `i = onpoints(p, inverse(g), n)` means: "at slot `p` we find
  index `i`."

**Renato's convention** (used internally by xperm, in `canonical_perm_ext`,
`coset_rep`, `double_coset_rep`):
- A permutation `g` maps **slot-numbers to index-names**.
- `name = onpoints(slot, g, n)` means: "at slot `slot` we find index `name`."

These are **inverses** of each other. The `canonical_perm` wrapper function
performs the conversion at entry and exit (see "Change to Renato's notation"
below).

---

## Key Helper Functions

### `onpoints`

```c
int onpoints(int point, int *p, int n);
```

Returns the image of `point` under permutation `p` of degree `n`. If
`point <= n`, returns `p[point - 1]`. If `point > n`, returns `point`
unchanged (acts as identity on points beyond the degree).

**1-indexed**: `point` ranges from 1 to n.

### `inverse`

```c
void inverse(int *p, int *ip, int n);
```

Computes the inverse of the `n`-permutation `p`, storing the result in `ip`.
Both `p` and `ip` are arrays of length `n`. `ip` must be pre-allocated.

The inverse satisfies: `product(p, ip, result, n)` yields the identity.

### `product`

```c
void product(int *p1, int *p2, int *p, int n);
```

Computes the product of permutations `p1` and `p2` (applied **left to right**),
storing the result in `p`. That is, `p[i-1] = p2[p1[i-1] - 1]` -- first apply
`p1`, then apply `p2`.

**Important**: `p` must not alias `p1` or `p2`.

### `nonstable_points`

```c
void nonstable_points(int *list1, int l1, int *GS, int m, int n,
    int *list2, int *l2);
```

Produces a list of points (`list2`, length `*l2`) such that no permutation in
the generating set `GS` fixes all of them. The first `l1` points of `list2`
are copied from `list1` (even if they are all stable). Additional points are
appended as needed.

- `list1` (length `l1`): initial points (typically the user-supplied base hint)
- `GS`: flattened array of `m` permutations, each of degree `n`
- `list2` (pre-allocated, typically `n` ints): output list
- `*l2`: output length

This is used by `schreier_sims` to extend a partial base to cover all generators.

### `coset_rep`

```c
void coset_rep(int *p, int n,
    int *base, int bl, int *GS, int *m,
    int *freeps, int fl,
    int *cr);
```

Computes a canonical representative of the right coset `S.p` for the group `S`
described by the SGS `(base, GS)`, considering only free-index slots.

**Parameters (all in Renato's convention -- slots):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | `int[n]` | Input permutation (slot-to-name, Renato's convention) |
| `n` | `int` | Degree of permutations |
| `base` | `int[bl]` | Base of the SGS (slot positions, 1-indexed) |
| `bl` | `int` | Length of base |
| `GS` | `int[*m * n]` | Generating set (flattened). **Modified on output** -- the GS is stabilized and `*m` is updated |
| `m` | `int*` | Pointer to number of generators. **Modified on output** |
| `freeps` | `int[fl]` | Slot positions of free indices. **Modified on output** -- updated to reflect the new slot positions after the coset representative is applied |
| `fl` | `int` | Number of free indices |
| `cr` | `int[n]` | Output: canonical coset representative |

**Memory**: All arrays must be pre-allocated by the caller. `GS` is modified
in-place (stabilized); `freeps` is modified in-place (slots are permuted).
No reallocation occurs.

**Important**: `m` is a pointer (not a value) because `coset_rep` modifies it.
`GS` is also modified in-place.

### `double_coset_rep`

```c
void double_coset_rep(int *g, int n, int *base, int bl, int *GS, int m,
    int *vds, int vdsl, int *dummies, int dl, int *mQ,
    int *vrs, int vrsl, int *repes, int rl, int *dcr);
```

Computes a canonical representative of the double coset `S.g.D`.

**All in Renato's convention** (permutation maps slots to names).

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | `int[n]` | Input permutation (slot-to-name) |
| `n` | `int` | Degree |
| `base` | `int[bl]` | Base of SGS for group S (slot positions) |
| `bl` | `int` | Base length |
| `GS` | `int[m*n]` | Generating set for group S |
| `m` | `int` | Number of generators (value, not pointer) |
| `vds` | `int[vdsl]` | Lengths of each dummyset |
| `vdsl` | `int` | Number of dummysets |
| `dummies` | `int[dl]` | Pairs of dummy **names** [u1,d1, u2,d2, ...]. **Modified** (reordered internally) |
| `dl` | `int` | Total length of dummies list (sum of vds elements) |
| `mQ` | `int[vdsl]` | Metric symmetry per dummyset: 1=symmetric, -1=antisymmetric, 0=no metric |
| `vrs` | `int[vrsl]` | Lengths of each repeatedset |
| `vrsl` | `int` | Number of repeatedsets |
| `repes` | `int[rl]` | Repeated index **names**. **Modified** (reordered internally) |
| `rl` | `int` | Total length of repes list (sum of vrs elements) |
| `dcr` | `int[n]` | Output: canonical representative |

**Zero result**: If inconsistency is detected (same permutation with opposite
signs), `dcr` is set to all zeros.

**Memory**: Internally allocates and frees all temporary storage. The `dummies`
and `repes` arrays are modified in-place (reordered). All other input arrays
are not modified.

---

## Public API: `schreier_sims`

### Signature

```c
void schreier_sims(int *base, int bl, int *GS, int m, int n,
    int *newbase, int *nbl, int **newGS, int *nm, int *num);
```

### Purpose

Computes a Strong Generating Set (SGS) for the permutation group generated by
`GS`. Given an optional partial base, it extends the base and computes a
complete SGS.

### Parameters

| Parameter | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `base` | in | `int[bl]` | Initial base points (1-indexed). Can be empty (`bl=0`). These become the first points of the computed base. |
| `bl` | in | `int` | Length of initial base (0 if no hint) |
| `GS` | in | `int[m*n]` | Input generating set: `m` permutations of degree `n`, stored contiguously. `GS[i*n .. (i+1)*n-1]` is the i-th permutation (0-indexed). |
| `m` | in | `int` | Number of input generators |
| `n` | in | `int` | Degree of all permutations. Must be > 0. |
| `newbase` | out | `int[n]` | Computed base. Must be pre-allocated with space for `n` integers (the maximum possible base length). |
| `nbl` | out | `int*` | Pointer to length of computed base. Set on output. |
| `newGS` | in/out | `int**` | **Pointer to pointer** to the output generating set. See memory ownership below. |
| `nm` | out | `int*` | Pointer to number of generators in output SGS. Set on output. |
| `num` | out | `int*` | Pointer to counter of Schreier generator tests performed. Set on output. |

### Memory Ownership -- CRITICAL

**`newGS` uses `realloc` internally.** The memory pointed to by `*newGS` must
have been allocated with C's `malloc` (or equivalent), NOT with Julia's
`Array` allocator. The algorithm calls `realloc` on `*newGS` whenever it
needs to add a new generator to the SGS.

**Correct usage from Julia:**

```julia
# Allocate with Libc.malloc (C heap), NOT with Julia Array
ptr = Libc.malloc(m * n * sizeof(Cint))
newGS_ref = Ref{Ptr{Cint}}(Ptr{Cint}(ptr))

# ... ccall ...

# After use, free with Libc.free (the pointer may have changed due to realloc!)
Libc.free(newGS_ref[])
```

**Incorrect usage (will segfault or corrupt memory):**

```julia
# WRONG: Julia-allocated array will be realloc'd by C code
newGS = Vector{Cint}(undef, m * n)
newGS_ref = Ref{Ptr{Cint}}(pointer(newGS))  # SEGFAULT when realloc'd
```

**Pre-conditions:**
- `*newGS` must point to `malloc`-allocated memory of at least `m * n * sizeof(int)` bytes.
- The initial content of `*newGS` does not matter; it is overwritten with a copy
  of `GS` at the start.

**Post-conditions:**
- `*newGS` may point to different memory than on input (due to `realloc`).
- The caller must free `*newGS` (which is the potentially-new pointer) with `free`.
- `newbase` is filled with the computed base of length `*nbl`.
- `*newGS` contains `*nm` permutations of degree `n`.

### Other Allocations

- `newbase`: Caller-allocated, at least `n` ints. Written but not reallocated.
- `nbl`, `nm`, `num`: Caller-allocated `int*`. Written on output.
- Internally, `schreier_sims` allocates and frees several temporary buffers
  (`base2`, `GS2`, `stab`), and calls `schreier_sims_step` which performs
  additional `realloc` on `*newGS`.

---

## Public API: `canonical_perm`

### Signature

```c
void canonical_perm(int *perm,
    int SGSQ, int *base, int bl, int *GS, int m, int n,
    int *freeps, int fl, int *dummyps, int dpl, int ob, int metricQ,
    int *cperm);
```

### Purpose

Find a canonical representative of the permutation `perm` under the double
coset defined by slot symmetries (group S) and dummy-index symmetries (group D).

**This is the simpler, backwards-compatible wrapper** around `canonical_perm_ext`.
It handles only a single dummyset with a single metric symmetry sign. For multiple
dummysets or repeatedsets, use `canonical_perm_ext` directly.

### The "Change to Renato's notation" Transformation

**This is the most critical detail for correct usage.**

`canonical_perm` accepts `perm`, `freeps`, and `dummyps` in **xTensor's
convention** (perm maps names to slots), but internally xperm works in
**Renato's convention** (perm maps slots to names).

On entry, `canonical_perm` performs:

```c
// 1. Invert the permutation: from names-to-slots to slots-to-names
inverse(PERM, PERM1, n);

// 2. Convert free SLOT positions to free NAMES
for (i = 0; i < fl; i++) {
    frees[i] = onpoints(freeps[i], PERM1, n);
}

// 3. Convert dummy SLOT positions to dummy NAMES
for (i = 0; i < 2*dpl; i++) {
    dummies[i] = onpoints(dummyps[i], PERM1, n);
}
```

On exit, it inverts the result back:

```c
if (PERM2[0] != 0) inverse(PERM2, CPERM, n);
else copy_list(PERM2, CPERM, n);  // zero stays zero
```

### What `freeps` and `dummyps` Are (in `canonical_perm`)

**`freeps` and `dummyps` are SLOT POSITIONS** (in xTensor convention), because
`canonical_perm` will convert them to names internally.

- `freeps[i]` is the **slot number** (1-indexed) where the i-th free index sits
  in the configuration described by `perm`.
- `dummyps[i]` is the **slot number** (1-indexed) where the i-th dummy index
  sits. Dummy indices are listed as pairs: `[slot_of_u1, slot_of_d1, slot_of_u2, slot_of_d2, ...]`.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `perm` | `int[n]` | Input permutation in **xTensor convention** (names-to-slots). NOT modified. |
| `SGSQ` | `int` | 1 if `(base, GS)` is already a SGS; 0 if `GS` is just a generating set (SGS will be computed internally). |
| `base` | `int[bl]` | Base for the symmetry group (slot positions, 1-indexed). If `SGSQ=0`, these are used as initial base hints. |
| `bl` | `int` | Length of base |
| `GS` | `int[m*n]` | Generating set for slot symmetry group S. `m` permutations of degree `n`, flattened. |
| `m` | `int` | Number of generators |
| `n` | `int` | Degree of permutations (includes 2 sign slots) |
| `freeps` | `int[fl]` | **Slot positions** of free indices (1-indexed). NOT modified (a copy is made). |
| `fl` | `int` | Number of free indices |
| `dummyps` | `int[2*dpl]` | **Slot positions** of dummy indices as pairs. NOT modified (a copy is made). |
| `dpl` | `int` | Number of dummy **pairs** (NOT the length of `dummyps`, which is `2*dpl`). |
| `ob` | `int` | **Unused** (kept for backwards compatibility). Pass any value. |
| `metricQ` | `int` | Metric symmetry: 1 = symmetric, -1 = antisymmetric, 0 = no metric. Applies to all dummy pairs. |
| `cperm` | `int[n]` | Output: canonical permutation in **xTensor convention** (names-to-slots). |

### Constraint Equation

```
2 * dpl + fl + 2 == n
```

Where:
- `dpl` = number of dummy pairs
- `fl` = number of free indices
- `2` = the two sign-encoding positions (points `n-1` and `n`)
- `n` = total degree

Equivalently: the number of actual tensor slots is `n - 2 = 2*dpl + fl`.

### Return Convention

- **Normal result**: `cperm` contains the canonical permutation in xTensor
  convention (names-to-slots). The sign is encoded in positions `n-1` and `n`.
- **Zero result**: `cperm` is all zeros (`{0, 0, ..., 0}`). This means the
  tensor expression vanishes identically.

### Memory Ownership

`canonical_perm` allocates all internal buffers and frees them before returning.
The caller only needs to provide pre-allocated arrays for `perm`, `base`, `GS`,
`freeps`, `dummyps`, and `cperm`. None of the input arrays are modified.

Internally, `canonical_perm` may call `schreier_sims` (if `SGSQ=0`), which uses
`malloc`/`realloc` for the SGS computation. All such memory is freed internally.

---

## Public API: `canonical_perm_ext`

### Signature

```c
void canonical_perm_ext(int *PERM, int n,
    int SGSQ, int *base, int bl, int *GS, int m,
    int *frees, int fl,
    int *vds, int vdsl, int *dummies, int dl, int *mQ,
    int *vrs, int vrsl, int *repes, int rl,
    int *CPERM);
```

### Purpose

Extended canonicalization supporting multiple dummysets and repeatedsets.
This is the function that does the actual work; `canonical_perm` is a
backwards-compatible wrapper around it.

### CRITICAL: `frees` and `dummies` are NAMES, not slots

Unlike `canonical_perm` (which accepts slot positions and converts them
internally), **`canonical_perm_ext` expects NAMES** (i.e., index identities
in the canonical configuration).

The function computes slots from names internally:

```c
// Compute slots of free indices from their names
inverse(PERM, PERM1, n);
for (i = 0; i < fl; i++) {
    freeps[i] = onpoints(frees[i], PERM1, n);
}
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `PERM` | `int[n]` | Input permutation in **Renato's convention** (slots-to-names). |
| `n` | `int` | Degree of permutations |
| `SGSQ` | `int` | 1 if `(base, GS)` is a SGS; 0 if not (SGS will be computed). |
| `base` | `int[bl]` | Base for group S (slot positions, 1-indexed) |
| `bl` | `int` | Base length |
| `GS` | `int[m*n]` | Generating set for group S |
| `m` | `int` | Number of generators |
| `frees` | `int[fl]` | **Names** of free indices (1-indexed). These are the index identities, not slot positions. |
| `fl` | `int` | Number of free indices |
| `vds` | `int[vdsl]` | Array of lengths of each dummyset |
| `vdsl` | `int` | Number of dummysets |
| `dummies` | `int[dl]` | Dummy index **names** as pairs: `[u1,d1, u2,d2, ...]`. Modified internally (reordered). |
| `dl` | `int` | Total length of dummies list. Must equal `sum(vds)`. |
| `mQ` | `int[vdsl]` | Metric symmetry per dummyset (1, -1, or 0) |
| `vrs` | `int[vrsl]` | Array of lengths of each repeatedset. Can be `NULL` if `vrsl=0`. |
| `vrsl` | `int` | Number of repeatedsets |
| `repes` | `int[rl]` | Repeated index **names**. Can be `NULL` if `rl=0`. Modified internally. |
| `rl` | `int` | Total length of repes list. Must equal `sum(vrs)`. |
| `CPERM` | `int[n]` | Output: canonical permutation in **Renato's convention** (slots-to-names). |

### Algorithm Steps

1. If `SGSQ=0`, compute SGS from `(base, GS)` using `schreier_sims`.
2. Compute slot positions of free indices from their names.
3. Call `coset_rep` with the free-index slots to canonicalize the free part.
   This produces `PERM1` and updates the SGS (stabilizing the free slots).
4. If there are dummies or repeats (`dl + rl > 0`):
   a. Remove free-index base points from the base.
   b. Stabilize the SGS with respect to the free slots.
   c. Call `double_coset_rep` with `PERM1` and the dummy/repeat info.
   d. Result goes to `PERM2`.
5. Copy result to `CPERM`.

### Return Convention

Same as `canonical_perm`, but the result is in **Renato's convention**
(slots-to-names), not xTensor convention.

- **Normal result**: `CPERM` is the canonical slots-to-names permutation.
- **Zero result**: `CPERM` is all zeros.

### Memory Ownership

Internally allocates a C-heap buffer for the SGS (via `malloc`), which may be
`realloc`'d by `schreier_sims`. All internal allocations are freed before
return. Input arrays `frees`, `dummies`, `repes` may be modified internally
(but `canonical_perm` passes copies, so this is safe when called through the
wrapper).

---

## Calling `canonical_perm` from Julia: Correct Convention

Given the analysis above, here is how `canonical_perm` should be called:

### Input Construction

Suppose you have a tensor expression with:
- `n` = total degree (number of index slots + 2 for sign)
- A permutation `perm` mapping index names to slot positions (xTensor convention)
- Slot symmetry generators in `GS` (each is an `n`-permutation)
- `free_slots`: the slot numbers (1-indexed) where free indices are located
- `dummy_slots`: the slot numbers (1-indexed) where dummy indices are located,
  as pairs `[slot_u1, slot_d1, slot_u2, slot_d2, ...]`

Then call:

```julia
ccall((:canonical_perm, libxperm), Cvoid,
    (Ptr{Cint},                           # perm (names-to-slots)
     Cint, Ptr{Cint}, Cint,               # SGSQ, base, bl
     Ptr{Cint}, Cint, Cint,               # GS, m, n
     Ptr{Cint}, Cint,                     # freeps (SLOT positions), fl
     Ptr{Cint}, Cint,                     # dummyps (SLOT positions), dpl (num PAIRS)
     Cint, Cint,                          # ob (unused), metricQ
     Ptr{Cint}),                          # cperm (output)
    perm, SGSQ, base, bl, GS, m, n,
    free_slots, fl, dummy_slots, dpl, ob, metricQ,
    cperm)
```

### `dpl` is the Number of PAIRS

The `dummyps` array has length `2 * dpl`, but the parameter passed to
`canonical_perm` is `dpl` (the number of pairs), NOT `2 * dpl`.

### Calling `canonical_perm_ext` from Julia

If calling `canonical_perm_ext` directly, remember:

1. The input `PERM` must be in **Renato's convention** (slots-to-names), which is
   the **inverse** of xTensor convention.
2. `frees` and `dummies` are **names** (index identities), not slot positions.
3. The output `CPERM` is in Renato's convention. To convert back to xTensor
   convention, take its inverse.
4. The SGS computation uses `malloc`/`realloc` internally; the pointer buffer for
   `newGS` is allocated and freed within `canonical_perm_ext`, so the caller does
   not need to worry about it (unlike calling `schreier_sims` directly).

---

## Worked Example

Consider `T_{ab} + T_{ba}` with `T` symmetric, two free indices `a, b`.

Tensor has 2 slots. With sign encoding: `n = 4`.

The identity configuration `T_{ab}` is the permutation `{1, 2, 3, 4}` in
xTensor convention (name 1 is at slot 1, name 2 is at slot 2, signs fixed).

The symmetry generator (swap slots 1 and 2) is `{2, 1, 3, 4}`.

Free slots: `{1, 2}` (both slots have free indices). `fl = 2`.

No dummies: `dpl = 0`. Constraint: `2*0 + 2 + 2 = 4 = n`. Correct.

Calling `canonical_perm`:
- `perm = {1, 2, 3, 4}`
- `base = {1}` (or `{1, 2}`)
- `GS = {2, 1, 3, 4}`
- `freeps = {1, 2}`
- `dummyps = {}`, `dpl = 0`
- `metricQ = 0`

---

## Summary of Convention Differences

| Aspect | `canonical_perm` | `canonical_perm_ext` |
|--------|-----------------|---------------------|
| Input perm convention | xTensor (names-to-slots) | Renato (slots-to-names) |
| Output perm convention | xTensor (names-to-slots) | Renato (slots-to-names) |
| Free index argument | **Slot positions** (`freeps`) | **Names** (`frees`) |
| Dummy index argument | **Slot positions** (`dummyps`) | **Names** (`dummies`) |
| Notation conversion | Done internally (2 inverses) | Caller's responsibility |
| Multiple dummysets | No (single set only) | Yes |
| Repeatedsets | No | Yes |

---

## `SGSD` (Internal, but Important for Understanding)

```c
void SGSD(int *vds, int vdsl, int *dummies, int dl, int *mQ,
          int *vrs, int vrsl, int *repes, int rl, int n,
          int firstd, int *KD, int *KDl, int *bD, int *bDl);
```

Constructs the SGS for the dummy/repeated-index symmetry group D. Called
internally by `double_coset_rep`.

**How the dummy group D is generated:**

For a dummyset with `dpl` pairs and metric symmetry `sym`:
- **Pair exchange generators** (`dpl - 1` of them): swap consecutive pairs.
  E.g., with pairs `(a,b)` and `(c,d)`: the generator swaps `a<->c` and `b<->d`.
- **Within-pair exchange generators** (if `sym != 0`, `dpl` of them): swap the
  two elements of each pair. If `sym = -1`, the last two points are swapped
  (negative sign).
- Total generators: `2*dpl - 1` if `sym != 0`, `dpl - 1` if `sym == 0`.
- Base of D: the first element of each pair.

For a repeatedset with `rl` indices:
- **Adjacent transposition generators** (`rl - 1` of them): swap consecutive
  repeated indices.
- Base: first `rl - 1` elements.

---

## Common Pitfalls

1. **Passing Julia-allocated arrays to `schreier_sims` for `newGS`**: The `*newGS`
   pointer MUST be `Libc.malloc`-allocated because `schreier_sims_step` calls
   `realloc` on it.

2. **Confusing `dpl` with `dl`**: In `canonical_perm`, `dpl` is the number of
   **pairs**, so `dummyps` has length `2*dpl`. In `canonical_perm_ext`, `dl` is
   the total length of the `dummies` array.

3. **Confusing slots and names**: `canonical_perm` takes **slots**;
   `canonical_perm_ext` takes **names**. Mixing these up produces wrong results.

4. **Forgetting the sign slots**: `n` includes 2 extra points for sign encoding.
   The constraint `2*dpl + fl + 2 == n` must hold.

5. **Passing `fl=0` and `dpl=0`**: This is valid (no free indices, no dummies)
   but then `n` must be 2, and the permutation is just the sign marker `{1,2}`
   or `{2,1}`.

6. **Not initializing `num`**: The `num` parameter to `schreier_sims` should be
   initialized to 0 before the call (it's a counter that gets incremented).

7. **Zero-generator edge case**: If `m=0` (no symmetries), `coset_rep` returns
   `p` unchanged, and `schreier_sims` returns the trivial group.
