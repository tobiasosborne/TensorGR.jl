#= Change of frame (Lorentz transformation) between tetrad choices.
#
# If two tetrads e^a_I and e'^a_{I'} are related by a Lorentz transformation
# Λ^{I'}_{J}, then:
#   e'^a_{I'} = Λ^J_{I'} e^a_J     (inverse transform on frame index)
#   Λ^{I'}_J  = e'^{I'}_a e^a_J    (transition matrix)
#
# A tensor in frame basis transforms as:
#   T'^{I'...}_{J'...} = Λ^{I'}_K ... Λ^L_{J'} ... T^{K...}_{L...}
#
# For orthonormal frames, Λ is a Lorentz matrix: Λ^T η Λ = η.
# For null frames, Λ preserves the null metric structure.
#
# References:
#   Chandrasekhar, *The Mathematical Theory of Black Holes* (1983), Ch 1.
#   Wald, *General Relativity* (1984), Sec 3.4b.
=#

"""
    define_frame_transformation!(reg, from_tetrad, to_tetrad;
                                  manifold=:M4, transform_name=nothing)

Register the Lorentz transformation tensor Λ^I_J between two tetrad frames.

The transformation is defined as:
    Λ^I_J = e'^I_a e^a_J

where e' is the `to_tetrad` and e is the `from_tetrad`.

# Arguments
- `reg::TensorRegistry` -- the registry
- `from_tetrad::Symbol` -- source tetrad name
- `to_tetrad::Symbol` -- target tetrad name
- `manifold::Symbol` -- manifold (default `:M4`)
- `transform_name::Symbol` -- name for Λ (default: `Symbol(:Lambda_, from, :_to_, to)`)

# Returns
The name of the registered transformation tensor.
"""
function define_frame_transformation!(reg::TensorRegistry,
                                       from_tetrad::Symbol, to_tetrad::Symbol;
                                       manifold::Symbol=:M4,
                                       transform_name::Union{Symbol,Nothing}=nothing)
    has_vbundle(reg, :Lorentz) ||
        error("define_frame_transformation!: :Lorentz VBundle not found")
    has_tensor(reg, from_tetrad) ||
        error("define_frame_transformation!: tetrad '$from_tetrad' not registered")
    has_tensor(reg, to_tetrad) ||
        error("define_frame_transformation!: tetrad '$to_tetrad' not registered")

    name = transform_name === nothing ?
        Symbol(:Lambda_, from_tetrad, :_to_, to_tetrad) : transform_name

    if !has_tensor(reg, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_frame_transform => true,
                :from_tetrad => from_tetrad,
                :to_tetrad => to_tetrad,
                :vbundle => :Lorentz)))
    end
    return name
end

"""
    frame_transformation_expr(from_tetrad, to_tetrad, I, J;
                               registry=current_registry()) -> TensorExpr

Build the explicit expression for the frame transformation:

    Λ^I_J = e'^I_a e^a_J

where e' = `to_tetrad` and e = `from_tetrad`.

# Arguments
- `from_tetrad::Symbol` -- source tetrad name
- `to_tetrad::Symbol` -- target tetrad name
- `I::TIndex` -- Up Lorentz index (output, in to-frame)
- `J::TIndex` -- Down Lorentz index (input, in from-frame)

# Returns
A `TProduct` expression.
"""
function frame_transformation_expr(from_tetrad::Symbol, to_tetrad::Symbol,
                                    I::TIndex, J::TIndex;
                                    registry::TensorRegistry=current_registry())
    I.vbundle === :Lorentz && I.position === Up ||
        error("frame_transformation_expr: I must be Up on :Lorentz")
    J.vbundle === :Lorentz && J.position === Down ||
        error("frame_transformation_expr: J must be Down on :Lorentz")

    # Generate fresh Tangent dummy index a
    used = Set{Symbol}([I.name, J.name])
    a_sym = fresh_index(used; vbundle=:Tangent)

    # Λ^I_J = e'^I_a e^a_J
    e_prime_Ia = Tensor(to_tetrad,
        [I, TIndex(a_sym, Down, :Tangent)])
    e_aJ = Tensor(from_tetrad,
        [TIndex(a_sym, Up, :Tangent), J])

    return TProduct(1 // 1, [e_prime_Ia, e_aJ])
end

"""
    change_frame(expr, from_tetrad, to_tetrad;
                 registry=current_registry()) -> TensorExpr

Transform a frame-indexed expression from one tetrad basis to another
by inserting Lorentz transformation matrices for each free Lorentz index.

For each free Lorentz index:
- Up index `I`: multiply by `Λ^{I'}_I = e'^{I'}_a e^a_I`
- Down index `I`: multiply by `Λ^I_{I'} = e^I_a e'^a_{I'}`

# Arguments
- `expr::TensorExpr` -- expression in `from_tetrad` frame basis
- `from_tetrad::Symbol` -- source tetrad name
- `to_tetrad::Symbol` -- target tetrad name
- `registry::TensorRegistry` -- the active registry

# Returns
A new `TensorExpr` with transformation matrices inserted. Not simplified.

# Example
```julia
V_I = Tensor(:V, [frame_up(:I)])   # vector in frame e
V_new = change_frame(V_I, :e, :e_prime)  # vector in frame e'
```
"""
function change_frame(expr::TensorExpr,
                      from_tetrad::Symbol, to_tetrad::Symbol;
                      registry::TensorRegistry=current_registry())
    has_tensor(registry, from_tetrad) ||
        error("change_frame: tetrad '$from_tetrad' not registered")
    has_tensor(registry, to_tetrad) ||
        error("change_frame: tetrad '$to_tetrad' not registered")

    fi = free_indices(expr)
    lorentz_free = filter(idx -> idx.vbundle === :Lorentz, fi)

    if isempty(lorentz_free)
        return expr
    end

    all_idx = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idx)

    transform_factors = TensorExpr[]
    for idx in lorentz_free
        # Generate a fresh Tangent dummy for the transition
        a_sym = fresh_index(used; vbundle=:Tangent)
        push!(used, a_sym)

        if idx.position === Up
            # T^I → Λ^{I'}_I T^I = e'^{I'}_a e^a_I T^I
            # Fresh Lorentz index I' for the output
            new_sym = fresh_index(used; vbundle=:Lorentz)
            push!(used, new_sym)
            # e'^{I'}_a (new up Lorentz, down Tangent)
            push!(transform_factors, Tensor(to_tetrad,
                [TIndex(new_sym, Up, :Lorentz),
                 TIndex(a_sym, Down, :Tangent)]))
            # e^a_I (up Tangent, contracts with old I)
            push!(transform_factors, Tensor(from_tetrad,
                [TIndex(a_sym, Up, :Tangent),
                 TIndex(idx.name, Down, :Lorentz)]))
        else
            # T_I → Λ^I_{I'} T_I = e^I_a e'^a_{I'} T_I
            new_sym = fresh_index(used; vbundle=:Lorentz)
            push!(used, new_sym)
            # e^I_a (old up Lorentz, down Tangent) — contracts with T_I
            push!(transform_factors, Tensor(from_tetrad,
                [TIndex(idx.name, Up, :Lorentz),
                 TIndex(a_sym, Down, :Tangent)]))
            # e'^a_{I'} (up Tangent, new down Lorentz)
            push!(transform_factors, Tensor(to_tetrad,
                [TIndex(a_sym, Up, :Tangent),
                 TIndex(new_sym, Down, :Lorentz)]))
        end
    end

    return TProduct(1 // 1, [transform_factors..., expr])
end
