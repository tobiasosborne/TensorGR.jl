# Tetrad / Frame Bundle

Register a Lorentz (frame) VBundle for orthonormal or null-frame indices, separate from coordinate (Tangent) indices. The tetrad e^a_I carries one Tangent index and one Lorentz index. The frame metric eta_{IJ} and frame Kronecker delta are registered on the Lorentz bundle.

## Frame Bundle Setup

Register the Lorentz VBundle, frame metric eta_{IJ}, and frame delta on a manifold. Frame indices use capital Latin letters (I, J, K, L, M, N).

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_frame_bundle!(reg; manifold=:M4, dim=4)
    # Now use frame_up(:I), frame_down(:J) for Lorentz indices
    e = Tensor(:e, [up(:a), frame_down(:I)])  # tetrad
end
```

```@docs
define_frame_bundle!
```

## Frame Indices

Convenience constructors for frame (Lorentz) indices and predicates to distinguish them from coordinate indices.

```@docs
frame_up
frame_down
is_frame_index
```
