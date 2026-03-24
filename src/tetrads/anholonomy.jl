#= Anholonomy (structure) coefficients for tetrad frames.
#
# The anholonomy coefficients c^I_{JK} measure the failure of a frame
# to be integrable: [e_J, e_K] = c^I_{JK} e_I, where e_I are the
# frame vectors (tetrad).
#
# For a coordinate basis, c^I_{JK} = 0 identically (holonomic frame).
# For a general tetrad e^a_I:
#   c^I_{JK} = e^I_a (e^b_J ∂_b e^a_K - e^b_K ∂_b e^a_J)
#
# The coefficients are antisymmetric in (J,K) since the Lie bracket
# is antisymmetric: [e_J, e_K] = -[e_K, e_J].
#
# References:
#   Chandrasekhar, *The Mathematical Theory of Black Holes* (1983), Ch 1.
#   Nakahara, *Geometry, Topology and Physics* (2003), Sec 7.8.
#   Carroll, *Spacetime and Geometry* (2004), Sec 3.9 (non-coordinate bases).
=#

"""
    define_anholonomy!(reg, tetrad_name; manifold=:M4,
                       anholonomy_name=:Omega)

Register the anholonomy coefficient tensor c^I_{JK} for a tetrad.

The anholonomy tensor has rank (1,2) with all indices on the `:Lorentz`
bundle, and is antisymmetric in slots (2,3):
    c^I_{JK} = -c^I_{KJ}

This follows from [e_J, e_K] = -[e_K, e_J].

# Arguments
- `reg::TensorRegistry` -- the registry
- `tetrad_name::Symbol` -- name of the tetrad tensor (e.g., `:e`)
- `manifold::Symbol` -- manifold (default `:M4`)
- `anholonomy_name::Symbol` -- name for the anholonomy tensor (default `:Omega`)

# Requires
- `:Lorentz` VBundle must exist (via `define_frame_bundle!`)
- `tetrad_name` must be registered

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_frame_bundle!(reg; manifold=:M4)
    register_tensor!(reg, TensorProperties(
        name=:e, manifold=:M4, rank=(1, 1),
        symmetries=SymmetrySpec[]))
    define_anholonomy!(reg, :e)
end
```
"""
function define_anholonomy!(reg::TensorRegistry, tetrad_name::Symbol;
                            manifold::Symbol=:M4,
                            anholonomy_name::Symbol=:Omega)
    has_vbundle(reg, :Lorentz) ||
        error("define_anholonomy!: :Lorentz VBundle not found. " *
              "Call define_frame_bundle! first.")
    has_tensor(reg, tetrad_name) ||
        error("define_anholonomy!: tetrad '$tetrad_name' not registered")

    if !has_tensor(reg, anholonomy_name)
        register_tensor!(reg, TensorProperties(
            name=anholonomy_name, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[AntiSymmetric(2, 3)],
            options=Dict{Symbol,Any}(
                :is_anholonomy => true,
                :tetrad => tetrad_name,
                :vbundle => :Lorentz)))
    end
    nothing
end

"""
    anholonomy_expr(tetrad_name, I, J, K; registry) -> TensorExpr

Build the anholonomy coefficient expression:

    c^I_{JK} = e^I_a (e^b_J ∂_b e^a_K - e^b_K ∂_b e^a_J)

where `e^a_I` is the tetrad (mixed Tangent/Lorentz tensor), and
∂_b is the partial derivative on the Tangent bundle.

# Arguments
- `tetrad_name::Symbol` -- name of the tetrad tensor
- `I::TIndex` -- contravariant `:Lorentz` output index
- `J::TIndex` -- covariant `:Lorentz` input index
- `K::TIndex` -- covariant `:Lorentz` input index (antisymmetric with J)

# Returns
A `TensorExpr` in TDeriv form (not simplified).

# Index Requirements
- `I` must be `Up` on `:Lorentz`
- `J, K` must be `Down` on `:Lorentz`

# Example
```julia
expr = anholonomy_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))
```
"""
function anholonomy_expr(tetrad_name::Symbol,
                         I::TIndex, J::TIndex, K::TIndex;
                         registry::TensorRegistry=current_registry())
    # Validate frame index positions and bundles
    I.vbundle === :Lorentz && I.position === Up ||
        error("anholonomy_expr: I must be Up on :Lorentz, got $(I.position) on $(I.vbundle)")
    J.vbundle === :Lorentz && J.position === Down ||
        error("anholonomy_expr: J must be Down on :Lorentz")
    K.vbundle === :Lorentz && K.position === Down ||
        error("anholonomy_expr: K must be Down on :Lorentz")

    # Generate fresh Tangent dummy indices to avoid collisions
    used_names = Set{Symbol}([I.name, J.name, K.name])
    a_sym = fresh_index(used_names; vbundle=:Tangent)
    push!(used_names, a_sym)
    b_sym = fresh_index(used_names; vbundle=:Tangent)

    # Build: e^I_a * e^b_J * ∂_b(e^a_K)
    #   = e(I, down(a)) * e(up(b), J) * TDeriv(down(b), e(up(a), K))
    e_Ia = Tensor(tetrad_name, [I, down(a_sym)])
    e_bJ = Tensor(tetrad_name, [up(b_sym), J])
    de_aK = TDeriv(down(b_sym), Tensor(tetrad_name, [up(a_sym), K]), :partial)
    term1 = TProduct(1 // 1, [e_Ia, e_bJ, de_aK])

    # Build: e^I_a * e^b_K * ∂_b(e^a_J)
    e_bK = Tensor(tetrad_name, [up(b_sym), K])
    de_aJ = TDeriv(down(b_sym), Tensor(tetrad_name, [up(a_sym), J]), :partial)
    term2 = TProduct(1 // 1, [e_Ia, e_bK, de_aJ])

    # c^I_{JK} = term1 - term2
    return TSum([term1, TProduct(-1 // 1, term2.factors)])
end

"""
    has_anholonomy(reg::TensorRegistry, tetrad_name::Symbol) -> Bool

Check if anholonomy coefficients are registered for the given tetrad.
"""
function has_anholonomy(reg::TensorRegistry, tetrad_name::Symbol)
    for (name, props) in reg.tensors
        if get(props.options, :is_anholonomy, false) &&
           get(props.options, :tetrad, nothing) === tetrad_name
            return true
        end
    end
    return false
end

"""
    get_anholonomy_name(reg::TensorRegistry, tetrad_name::Symbol) -> Symbol

Return the name of the anholonomy tensor registered for `tetrad_name`.
Throws if not found.
"""
function get_anholonomy_name(reg::TensorRegistry, tetrad_name::Symbol)
    for (name, props) in reg.tensors
        if get(props.options, :is_anholonomy, false) &&
           get(props.options, :tetrad, nothing) === tetrad_name
            return name
        end
    end
    error("No anholonomy registered for tetrad '$tetrad_name'")
end
