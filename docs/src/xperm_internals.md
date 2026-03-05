# xperm.c Internals

TensorGR uses José Martin-Garcia's `xperm.c` library for index canonicalization via the Butler-Portugal algorithm.

See the detailed documentation files:
- [xperm API Reference](../xperm_api.md)
- [xperm Algorithm Analysis](../xperm_algorithm.md)
- [xperm Calling Convention](../xperm_calling_convention.md)

## Key Points

### Permutation Convention

xperm uses **images notation**: `perm[i]` is the image of point `i` (1-indexed).

The library internally uses two conventions:
- **xAct convention** (slot-to-slot): used by `canonical_perm`
- **Renato's convention** (slot-to-name): used by `canonical_perm_ext`

The `canonical_perm` wrapper handles conversion between conventions automatically.

### Calling from Julia

```julia
cperm = xperm_canonical_perm(perm, base, generators, freeps, dummyps, n)
```

- `perm`: the current index configuration as a permutation
- `freeps`: names of free indices (in the identity config, name = initial slot)
- `dummyps`: pairs of names for dummy indices `[up1, down1, up2, down2, ...]`
- `generators`: slot symmetry generators (e.g., Symmetric(1,2) → swap slots 1↔2)
- The perm **must** be a proper bijection (no repeated values)

### Memory Safety

`schreier_sims` may `realloc` its output buffer. When calling it directly, the buffer must be C-heap allocated via `Libc.malloc`, not Julia GC-managed.
