#= Project tensors from coordinate (Tangent) basis to frame (Lorentz) basis.
#
# The projection inserts tetrad components e^a_I for each free Tangent index:
#   T^{I...}_{J...} = e^I_a ... e^b_J ... T^{a...}_{b...}
#
# For a contravariant (Up) Tangent index a → Up Lorentz index I:
#   T^I = e^I_a T^a     (co-frame insertion)
#
# For a covariant (Down) Tangent index a → Down Lorentz index I:
#   T_I = e^a_I T_a     (tetrad insertion)
#
# Dummy (contracted) Tangent indices are left unchanged — they are
# internal contractions that do not depend on basis choice.
#
# References:
#   Chandrasekhar, *The Mathematical Theory of Black Holes* (1983), Ch 1.
#   Carroll, *Spacetime and Geometry* (2004), Sec 3.9.
=#

"""
    to_frame(expr, tetrad_name; registry=current_registry()) -> TensorExpr

Project a tensor expression from coordinate (Tangent) basis to frame
(Lorentz) basis by inserting tetrad components for each free Tangent index.

# Conversion rules
- Up Tangent index `a`: multiply by `e^I_a` (co-frame), producing Up Lorentz `I`
- Down Tangent index `a`: multiply by `e^a_I` (tetrad), producing Down Lorentz `I`
- Dummy Tangent indices are not affected

# Arguments
- `expr::TensorExpr` -- the expression to project
- `tetrad_name::Symbol` -- name of the tetrad tensor (e.g., `:e`)
- `registry::TensorRegistry` -- the active registry

# Returns
A new `TensorExpr` with all free Tangent indices replaced by Lorentz indices,
multiplied by the appropriate tetrad factors. The result is not simplified.

# Example
```julia
# Project a vector V^a to frame basis V^I = e^I_a V^a
V = Tensor(:V, [up(:a)])
V_frame = to_frame(V, :e)
# Result: TProduct with e^I_a V^a (I is a fresh Lorentz index)
```
"""
function to_frame(expr::TensorExpr, tetrad_name::Symbol;
                  registry::TensorRegistry=current_registry())
    has_tensor(registry, tetrad_name) ||
        error("to_frame: tetrad '$tetrad_name' not registered")

    fi = free_indices(expr)
    tangent_free = filter(idx -> idx.vbundle === :Tangent, fi)

    # Nothing to project if no free Tangent indices
    if isempty(tangent_free)
        return expr
    end

    # Collect all index names in the expression to avoid collisions
    all_idx = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idx)

    # Build tetrad insertion factors
    tetrad_factors = TensorExpr[]
    for idx in tangent_free
        frame_sym = fresh_index(used; vbundle=:Lorentz)
        push!(used, frame_sym)

        if idx.position === Up
            # T^a → e^I_a T^a: insert e(frame_up(I), down(a))
            push!(tetrad_factors, Tensor(tetrad_name,
                [TIndex(frame_sym, Up, :Lorentz),
                 TIndex(idx.name, Down, :Tangent)]))
        else
            # T_a → e^a_I T_a: insert e(up(a), frame_down(I))
            push!(tetrad_factors, Tensor(tetrad_name,
                [TIndex(idx.name, Up, :Tangent),
                 TIndex(frame_sym, Down, :Lorentz)]))
        end
    end

    return TProduct(1 // 1, [tetrad_factors..., expr])
end

"""
    from_frame(expr, tetrad_name; registry=current_registry()) -> TensorExpr

Project a tensor expression from frame (Lorentz) basis back to coordinate
(Tangent) basis by inserting inverse tetrad components for each free Lorentz
index.

# Conversion rules
- Up Lorentz index `I`: multiply by `e^a_I` (tetrad), producing Up Tangent `a`
- Down Lorentz index `I`: multiply by `e^I_a` (co-frame), producing Down Tangent `a`
- Dummy Lorentz indices are not affected

# Arguments
- `expr::TensorExpr` -- the expression to project
- `tetrad_name::Symbol` -- name of the tetrad tensor (e.g., `:e`)
- `registry::TensorRegistry` -- the active registry

# Returns
A new `TensorExpr` with all free Lorentz indices replaced by Tangent indices.

# Example
```julia
# Project a frame vector V^I back to coordinate basis V^a = e^a_I V^I
V = Tensor(:V, [frame_up(:I)])
V_coord = from_frame(V, :e)
```
"""
function from_frame(expr::TensorExpr, tetrad_name::Symbol;
                    registry::TensorRegistry=current_registry())
    has_tensor(registry, tetrad_name) ||
        error("from_frame: tetrad '$tetrad_name' not registered")

    fi = free_indices(expr)
    lorentz_free = filter(idx -> idx.vbundle === :Lorentz, fi)

    if isempty(lorentz_free)
        return expr
    end

    all_idx = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idx)

    tetrad_factors = TensorExpr[]
    for idx in lorentz_free
        tangent_sym = fresh_index(used; vbundle=:Tangent)
        push!(used, tangent_sym)

        if idx.position === Up
            # T^I → e^a_I T^I: insert e(up(a), frame_down(I))
            push!(tetrad_factors, Tensor(tetrad_name,
                [TIndex(tangent_sym, Up, :Tangent),
                 TIndex(idx.name, Down, :Lorentz)]))
        else
            # T_I → e^I_a T_I: insert e(frame_up(I), down(a))
            push!(tetrad_factors, Tensor(tetrad_name,
                [TIndex(idx.name, Up, :Lorentz),
                 TIndex(tangent_sym, Down, :Tangent)]))
        end
    end

    return TProduct(1 // 1, [tetrad_factors..., expr])
end
