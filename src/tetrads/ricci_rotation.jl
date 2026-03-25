#= Ricci rotation coefficients for tetrad frames.
#
# The Ricci rotation coefficients γ^I_{JK} are the connection coefficients
# in the frame (tetrad) basis. For a metric-compatible, torsion-free
# connection they are determined entirely by the anholonomy coefficients:
#
#   γ_{IJK} = ½(c_{IJK} + c_{KIJ} - c_{JKI})
#
# where c_{IJK} = η_{IM} c^M_{JK} are the lowered anholonomy coefficients.
# Equivalently, with the first index raised:
#
#   γ^I_{JK} = ½(c^I_{JK} + η^{IM}η_{KN}c^N_{MJ} - η^{IM}η_{JN}c^N_{KM})
#
# The all-down γ_{IJK} is antisymmetric in its first two indices (I,J)
# when the connection is metric-compatible:
#   γ_{IJK} = -γ_{JIK}
#
# References:
#   Chandrasekhar, *The Mathematical Theory of Black Holes* (1983), Ch 1.
#   Carroll, *Spacetime and Geometry* (2004), Sec 3.9.
#   Nakahara, *Geometry, Topology and Physics* (2003), Sec 7.8.
=#

"""
    define_ricci_rotation!(reg, tetrad_name; manifold=:M4,
                           rotation_name=:gamma_rot)

Register Ricci rotation coefficient tensor γ^I_{JK} for a tetrad.

The tensor has rank (1,2) with all indices on the `:Lorentz` bundle.
No manifest slot symmetry is registered (the antisymmetry γ_{IJK} = -γ_{JIK}
involves metric lowering of the first index).

# Arguments
- `reg::TensorRegistry` -- the registry
- `tetrad_name::Symbol` -- name of the tetrad tensor (e.g., `:e`)
- `manifold::Symbol` -- manifold (default `:M4`)
- `rotation_name::Symbol` -- name for the rotation coefficient tensor
  (default `:gamma_rot`)

# Requires
- `:Lorentz` VBundle must exist (via `define_frame_bundle!`)
- `tetrad_name` must be registered
- Anholonomy coefficients for `tetrad_name` must be defined

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
    define_ricci_rotation!(reg, :e)
end
```
"""
function define_ricci_rotation!(reg::TensorRegistry, tetrad_name::Symbol;
                                manifold::Symbol=:M4,
                                rotation_name::Symbol=:gamma_rot)
    has_vbundle(reg, :Lorentz) ||
        error("define_ricci_rotation!: :Lorentz VBundle not found. " *
              "Call define_frame_bundle! first.")
    has_tensor(reg, tetrad_name) ||
        error("define_ricci_rotation!: tetrad '$tetrad_name' not registered")
    has_anholonomy(reg, tetrad_name) ||
        error("define_ricci_rotation!: anholonomy for '$tetrad_name' not defined. " *
              "Call define_anholonomy! first.")

    if !has_tensor(reg, rotation_name)
        register_tensor!(reg, TensorProperties(
            name=rotation_name, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_ricci_rotation => true,
                :tetrad => tetrad_name,
                :vbundle => :Lorentz)))
    end
    nothing
end

"""
    ricci_rotation_expr(tetrad_name, I, J, K; registry) -> TensorExpr

Build the Ricci rotation coefficient expression in terms of anholonomy:

    γ^I_{JK} = ½(c^I_{JK} + η^{IM}η_{KN}c^N_{MJ} - η^{IM}η_{JN}c^N_{KM})

where c is the anholonomy tensor and η is the frame metric.

# Arguments
- `tetrad_name::Symbol` -- name of the tetrad tensor
- `I::TIndex` -- contravariant `:Lorentz` index (position Up)
- `J::TIndex` -- covariant `:Lorentz` index (position Down)
- `K::TIndex` -- covariant `:Lorentz` index (position Down)

# Returns
A `TensorExpr` (not simplified).

# Example
```julia
expr = ricci_rotation_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))
```
"""
function ricci_rotation_expr(tetrad_name::Symbol,
                              I::TIndex, J::TIndex, K::TIndex;
                              registry::TensorRegistry=current_registry())
    # Validate index positions and bundles
    I.vbundle === :Lorentz && I.position === Up ||
        error("ricci_rotation_expr: I must be Up on :Lorentz, " *
              "got $(I.position) on $(I.vbundle)")
    J.vbundle === :Lorentz && J.position === Down ||
        error("ricci_rotation_expr: J must be Down on :Lorentz")
    K.vbundle === :Lorentz && K.position === Down ||
        error("ricci_rotation_expr: K must be Down on :Lorentz")

    # Get anholonomy tensor name
    c_name = get_anholonomy_name(registry, tetrad_name)

    # Generate fresh Lorentz dummy indices M, N
    used = Set{Symbol}([I.name, J.name, K.name])
    M_sym = fresh_index(used; vbundle=:Lorentz)
    push!(used, M_sym)
    N_sym = fresh_index(used; vbundle=:Lorentz)

    M_up   = TIndex(M_sym, Up, :Lorentz)
    M_down = TIndex(M_sym, Down, :Lorentz)
    N_up   = TIndex(N_sym, Up, :Lorentz)
    N_down = TIndex(N_sym, Down, :Lorentz)

    # Term 1: ½ c^I_{JK}
    term1 = TProduct(1 // 2, [Tensor(c_name, [I, J, K])])

    # Term 2: ½ η^{IM} η_{KN} c^N_{MJ}
    term2 = TProduct(1 // 2, [
        Tensor(:eta, [I, M_up]),
        Tensor(:eta, [K, N_down]),
        Tensor(c_name, [N_up, M_down, J])
    ])

    # Term 3: -½ η^{IM} η_{JN} c^N_{KM}
    term3 = TProduct(-1 // 2, [
        Tensor(:eta, [I, M_up]),
        Tensor(:eta, [J, N_down]),
        Tensor(c_name, [N_up, K, M_down])
    ])

    return TSum([term1, term2, term3])
end

"""
    has_ricci_rotation(reg::TensorRegistry, tetrad_name::Symbol) -> Bool

Check if Ricci rotation coefficients are registered for the given tetrad.
"""
function has_ricci_rotation(reg::TensorRegistry, tetrad_name::Symbol)
    for (_, props) in reg.tensors
        if get(props.options, :is_ricci_rotation, false) &&
           get(props.options, :tetrad, nothing) === tetrad_name
            return true
        end
    end
    return false
end

"""
    get_ricci_rotation_name(reg::TensorRegistry, tetrad_name::Symbol) -> Symbol

Return the name of the Ricci rotation tensor registered for `tetrad_name`.
Throws if not found.
"""
function get_ricci_rotation_name(reg::TensorRegistry, tetrad_name::Symbol)
    for (name, props) in reg.tensors
        if get(props.options, :is_ricci_rotation, false) &&
           get(props.options, :tetrad, nothing) === tetrad_name
            return name
        end
    end
    error("No Ricci rotation coefficients registered for tetrad '$tetrad_name'")
end
