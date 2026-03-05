# xperm.c `canonical_perm` Calling Convention

## Summary

`canonical_perm` is a backwards-compatibility wrapper around `canonical_perm_ext`.
It performs an `inverse` on input and output to bridge between xTensor notation
(name-to-slot) and Renato Portugal's notation (slot-to-name) used internally by
the algorithm.

Due to this double inversion, `canonical_perm` works correctly with EITHER
convention for the input permutation, provided `freeps` and `dummyps` are
passed consistently:

| PERM convention        | freeps contains | dummyps contains |
|------------------------|-----------------|------------------|
| xTensor (name -> slot) | slot positions  | slot positions   |
| Renato  (slot -> name) | slot positions  | **names**        |

For free-only expressions (no dummies), freeps can always be slot positions
regardless of convention, because a double-inverse cancellation makes the
value pass through unchanged.

**TensorGR.jl currently avoids the dummy question entirely**: `canonicalize.jl`
treats ALL indices as free (passing `dummyps = Int32[]`) and handles dummy
renaming separately in `collect_terms` via `_normalize_dummies`.

---

## The C Function Signature

```c
void canonical_perm(int *PERM,
    int SGSQ, int *base, int bl, int *GS, int m, int n,
    int *freeps, int fl, int *dummyps, int dpl, int ob, int metricQ,
    int *CPERM);
```

Parameters:
- `PERM`: input permutation of degree `n` (the index configuration to canonicalize)
- `SGSQ`: 0 = GS is an ordinary generating set; 1 = GS is already a strong generating set
- `base`: base points for the SGS (length `bl`)
- `GS`: flattened array of `m` permutations of degree `n`
- `n`: degree of the permutation group (= number of index slots + 2 sign points)
- `freeps`: array of free index identifiers (length `fl`)
- `dummyps`: array of dummy pair identifiers (length `2*dpl`), format `[u1,d1, u2,d2, ...]`
- `dpl`: number of dummy pairs
- `ob`: (obsolete, kept for backwards compat) pass 1
- `metricQ`: 1 = symmetric metric, -1 = antisymmetric, 0 = no metric
- `CPERM`: output buffer for the canonical permutation (length `n`)

Constraint: `2*dpl + fl + 2 == n`

---

## What `canonical_perm` Does Internally

```c
/* Step 1: Convert to Renato's notation */
inverse(PERM, PERM1, n);
for (i=0; i<fl; i++)
    frees[i] = onpoints(freeps[i], PERM1, n);
for (i=0; i<2*dpl; i++)
    dummies[i] = onpoints(dummyps[i], PERM1, n);

/* Step 2: Run the extended algorithm */
canonical_perm_ext(PERM1, n, ..., frees, fl, ..., dummies, 2*dpl, ..., PERM2);

/* Step 3: Convert back */
if (PERM2[0] != 0) inverse(PERM2, CPERM, n);
else copy_list(PERM2, CPERM, n);  /* zero = expression vanishes */
```

Key functions:
- `inverse(p, ip, n)`: computes `ip` = inverse permutation of `p`
- `onpoints(point, p, n)`: returns `p[point]` (1-indexed array lookup)

---

## The Two Notation Conventions

### xTensor convention: `perm[name] = slot`

The permutation maps each canonical name (integer label) to the slot it
occupies. Example: `T_{ba}` with names a=1, b=2:
- name 1 (a) is in slot 2 => `perm[1] = 2`
- name 2 (b) is in slot 1 => `perm[2] = 1`
- `perm = [2, 1, 3, 4]`

### Renato convention: `perm[slot] = name`

The permutation maps each slot to the name of the index occupying it.
Example: `T_{ba}`:
- slot 1 has index b (name 2) => `perm[1] = 2`
- slot 2 has index a (name 1) => `perm[2] = 1`
- `perm = [2, 1, 3, 4]`

For this example the two conventions give the same array because `(1 2)` is
its own inverse. They differ for non-involution permutations like `R_{dcab}`:
- xTensor: `[3, 4, 2, 1, 5, 6]`
- Renato:  `[4, 3, 1, 2, 5, 6]`

---

## Why Both Conventions Work for Free Indices

When `canonical_perm` processes freeps, the transformation chain is:

1. `canonical_perm`: `frees[i] = PERM1[freeps[i]]` where `PERM1 = inverse(PERM)`
2. `canonical_perm_ext`: `freeps_internal[i] = PERM_ext_inv[frees[i]]` where `PERM_ext_inv = inverse(PERM1) = PERM`

Combining: `freeps_internal[i] = PERM[PERM1[freeps[i]]] = freeps[i]`

The double inverse cancels. Whatever integers you pass as freeps, those exact
integers reach the `coset_rep` algorithm unchanged. Since `coset_rep` treats
freeps as slot positions (base points to stabilize), passing slot positions
[1, 2, ..., k] is always correct.

---

## Why Dummyps Depends on Convention

For dummyps, there is NO second inverse. The transformation is:

1. `canonical_perm`: `dummies[i] = PERM1[dummyps[i]]` where `PERM1 = inverse(PERM)`
2. These `dummies` are passed unchanged through `canonical_perm_ext` to `double_coset_rep`.
3. `double_coset_rep` treats `dummies` as **names** (canonical integer labels).

For the result to be correct, `dummies[i]` must equal the name of the i-th
dummy index. So:

- If PERM is xTensor (name->slot): `PERM1 = slot->name`. Need `dummyps[i]` to be
  a **slot position** so that `PERM1[slot] = name`. This is the designed convention.

- If PERM is Renato (slot->name): `PERM1 = name->slot`. Need `dummyps[i]` to be
  a **name** so that `PERM1[name] = slot`. Wait -- that gives a slot, not a name!

  Actually, in Renato convention, the roles of name and slot are swapped
  throughout the algorithm (PERM1 is passed to canonical_perm_ext as if it
  were a Renato-convention perm, but it's actually name->slot). The entire
  algorithm runs "upside down" and the final inverse at the end of
  canonical_perm corrects it. For dummies, passing **names** (= values that
  appear in the PERM array) produces the correct result empirically.

### Empirical Verification

Test case: `V_a S^{ab} V_b` with S symmetric, all indices contracted.

Slot layout: slot1=down-a(V), slot2=up-a(S), slot3=up-b(S), slot4=down-b(V)

Dummy naming: up-a=1, up-b=2, down-a=3, down-b=4.
Pairs: (1,3) and (2,4). Identity perm = canonical.

Non-canonical config `V_a S^{ab} V_b`:
- Renato perm = `[3, 1, 2, 4, 5, 6]` (slot1->name3, slot2->name1, ...)
- xTensor perm = `[2, 3, 1, 4, 5, 6]` (name1->slot2, name2->slot3, ...)

Symmetry generator (S sym in slots 2,3): `[1, 3, 2, 4, 5, 6]`

```
                          dummyps     result           correct?
Renato  + names [1,3,2,4]           [1,2,3,4,5,6]     YES
Renato  + slots [2,1,3,4]           [1,2,4,3,5,6]     NO
xTensor + slots [2,1,3,4]           [1,2,3,4,5,6]     YES
xTensor + names [1,3,2,4]           [1,2,4,3,5,6]     NO
```

---

## Worked Examples

### Example 1: Free indices only -- `T_{ba}` antisymmetric

Names: a=1, b=2. Two index slots + 2 sign = n=4.

```
perm = [2, 1, 3, 4]      # T_{ba}: slot1=b=2, slot2=a=1 (same in both conventions)
gen  = [2, 1, 4, 3]      # swap slots 1,2 with sign flip
base = [1, 2]
freeps = [1, 2]           # all slots are free
dummyps = []
```

Result: `[1, 2, 4, 3]` = canonical perm with sign flip = `-T_{ab}`.

### Example 2: Free indices -- `R_{dcab}` Riemann

Names: a=1, b=2, c=3, d=4. Four slots + 2 sign = n=6.

```
perm = [3, 4, 2, 1, 5, 6]  # xTensor: name(a)->slot3, name(b)->slot4, etc.
  OR  [4, 3, 1, 2, 5, 6]  # Renato:  slot1->d=4, slot2->c=3, etc.
gen1 = [2, 1, 3, 4, 6, 5]  # antisym(a,b)
gen2 = [1, 2, 4, 3, 6, 5]  # antisym(c,d)
gen3 = [3, 4, 1, 2, 5, 6]  # pair sym
base = [1, 2, 3, 4]
freeps = [1, 2, 3, 4]
dummyps = []
```

Result (both conventions): `[1, 2, 3, 4, 6, 5]` = `-R_{abcd}`.

### Example 3: Dummy indices -- `g^{ab} T_{ab}` (symmetric g, antisymmetric T)

This expression must vanish: contracting a symmetric tensor with an
antisymmetric one gives zero.

```
n = 6
# Unique names: slot1(up-a from g)=1, slot2(up-b from g)=2,
#               slot3(down-a from T)=3, slot4(down-b from T)=4
perm = [1, 2, 3, 4, 5, 6]  # identity (canonical naming matches slot order)
g_sym    = [2, 1, 3, 4, 5, 6]  # g symmetric: swap slots 1,2
T_antisym = [1, 2, 4, 3, 6, 5]  # T antisymmetric: swap slots 3,4 with sign
base = [1, 2, 3, 4]
freeps = []
dummyps = [1, 3, 2, 4]  # pair(name1, name3), pair(name2, name4)
metricQ = 1  # symmetric metric
```

Result: `[0, 0, 0, 0, 0, 0]` = expression is zero.

---

## How TensorGR.jl Currently Calls canonical_perm

The `_canonicalize_product` function in `src/algebra/canonicalize.jl` deliberately
avoids using xperm's dummy-index machinery:

1. **Unique names**: Every slot gets a distinct name (even both halves of a
   dummy pair). Names are assigned by sorting all slots alphabetically by
   symbol, then numbering 1, 2, 3, ...

2. **All free**: `freeps = Int32.(1:nslots)`, `dummyps = Int32[]`. This means
   `canonical_perm` only runs the `coset_rep` algorithm (no `double_coset_rep`).

3. **Dummy normalization**: Handled separately by `_normalize_dummies` in the
   `collect_terms` simplification pass.

This design avoids the convention subtleties of `dummyps` entirely, at the cost
of not exploiting xperm's dummy-index canonicalization (which can find the
canonical dummy relabeling in one pass).

### Perm convention in canonicalize.jl

The perm is built as `perm_data[slot] = name` (Renato convention). Since only
free indices are used, and the double-inverse cancellation applies, this works
correctly.

The return value is interpreted via `cperm_inv = perm_inverse(cperm)` and then
`cperm_inv[slot]` gives the canonical name at each slot, from which the
original index symbol is recovered.

---

## If You Want to Use xperm's Dummy Canonicalization

To properly use `canonical_perm` with dummy indices, you must:

1. Assign **unique names** to every slot. Dummy pairs get two distinct names
   (e.g., up-a=1, down-a=2, up-b=3, down-b=4). The naming determines the
   canonical ordering: name 1 < name 2 < ... means the canonical config has
   these names in slot order.

2. Build a proper bijective permutation `perm[i]` of degree `n = nslots + 2`.

3. Choose a convention and pass dummyps accordingly:

   **Option A (xTensor convention):**
   - `perm[name] = slot`
   - `dummyps = [slot_of_up_1, slot_of_down_1, slot_of_up_2, slot_of_down_2, ...]`
   - `freeps = [slot_of_free_1, slot_of_free_2, ...]`

   **Option B (Renato convention):**
   - `perm[slot] = name`
   - `dummyps = [name_of_up_1, name_of_down_1, name_of_up_2, name_of_down_2, ...]`
   - `freeps = [slot_of_free_1, slot_of_free_2, ...]` (or `[1,2,...,nfree]` if free slots come first)

   Note: `freeps` can be slot positions in both conventions (double-inverse cancellation).
   Only `dummyps` differs between conventions.

4. The result `CPERM` is in the same convention as the input `PERM`.
   - If input was xTensor: `CPERM[name] = canonical_slot`
   - If input was Renato: `CPERM[slot] = canonical_name`

5. Sign is encoded in the last two points:
   - `CPERM[n-1] == n-1, CPERM[n] == n`: positive sign
   - `CPERM[n-1] == n, CPERM[n] == n-1`: negative sign
   - `CPERM == all zeros`: expression vanishes

---

## Mathematical Background

The `canonical_perm` function finds a canonical representative of the
permutation PERM in the double coset `S \ PERM / D`, where:
- `S` is the slot symmetry group (generated by `GS`)
- `D` is the dummy/metric index group (generated by `dummyps` + `metricQ`)

The canonical representative is the lexicographically smallest permutation
in the double coset, determined by Butler-Portugal's backtrack algorithm.

The two-step algorithm:
1. `coset_rep(PERM, S, freeps)`: find canonical rep in right coset `S * PERM`
   using free index positions to break ties.
2. `double_coset_rep(PERM1, S_stabilized, D)`: extend to the double coset
   using dummy pair information.

The `canonical_perm` wrapper adds inverse operations at start and end to
bridge between xTensor's name-to-slot convention and Renato's slot-to-name
convention used by the internal algorithms.
