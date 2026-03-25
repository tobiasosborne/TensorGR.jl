#= Curvature tensors in the tetrad (frame) basis.
#
# Two approaches are provided:
#
# 1. Projection approach: project coordinate-basis Riemann into frame basis
#    R^I_{JKL} = e^I_a e^b_J e^c_K e^d_L R^a_{bcd}
#
# 2. Structure equation approach: compute from Ricci rotation coefficients
#    R^I_{JKL} = e_K(γ^I_{JL}) - e_L(γ^I_{JK})
#                + γ^I_{KM}γ^M_{JL} - γ^I_{LM}γ^M_{JK}
#                - c^M_{KL}γ^I_{JM}
#
# The frame-basis Ricci tensor and scalar are obtained by contraction
# with the frame metric η:
#    R_{IJ} = η^{KL} R_{KILJ}
#    R      = η^{IJ} R_{IJ}
#
# References:
#   Chandrasekhar, *The Mathematical Theory of Black Holes* (1983), Eq 1.174.
#   Nakahara, *Geometry, Topology and Physics* (2003), Sec 7.8.
#   Carroll, *Spacetime and Geometry* (2004), Sec 3.9.
=#

"""
    frame_riemann_expr(tetrad_name, I, J, K, L;
                        registry=current_registry()) -> TensorExpr

Build the frame-basis Riemann tensor by projecting the coordinate Riemann:

    R^I_{JKL} = e^I_a e^b_J e^c_K e^d_L R^a_{bcd}

This is the standard approach using `to_frame` on the Riemann tensor.

# Arguments
- `tetrad_name::Symbol` -- tetrad name (e.g., `:e`)
- `I::TIndex` -- Up Lorentz index
- `J, K, L::TIndex` -- Down Lorentz indices

# Returns
A `TProduct` with four tetrad factors and the Riemann tensor.
"""
function frame_riemann_expr(tetrad_name::Symbol,
                             I::TIndex, J::TIndex, K::TIndex, L::TIndex;
                             registry::TensorRegistry=current_registry())
    # Validate all indices are Lorentz with correct positions
    I.vbundle === :Lorentz && I.position === Up ||
        error("frame_riemann_expr: I must be Up on :Lorentz")
    for (name, idx) in [("J", J), ("K", K), ("L", L)]
        idx.vbundle === :Lorentz && idx.position === Down ||
            error("frame_riemann_expr: $name must be Down on :Lorentz")
    end

    # Generate four fresh Tangent indices for the coordinate Riemann
    used = Set{Symbol}([I.name, J.name, K.name, L.name])
    a = fresh_index(used; vbundle=:Tangent); push!(used, a)
    b = fresh_index(used; vbundle=:Tangent); push!(used, b)
    c = fresh_index(used; vbundle=:Tangent); push!(used, c)
    d = fresh_index(used; vbundle=:Tangent); push!(used, d)

    # R^I_{JKL} = e^I_a e^b_J e^c_K e^d_L R^a_{bcd}
    return TProduct(1 // 1, [
        Tensor(tetrad_name, [I, down(a)]),                  # e^I_a
        Tensor(tetrad_name, [up(b), J]),                    # e^b_J
        Tensor(tetrad_name, [up(c), K]),                    # e^c_K
        Tensor(tetrad_name, [up(d), L]),                    # e^d_L
        Tensor(:Riem, [up(a), down(b), down(c), down(d)])   # R^a_{bcd}
    ])
end

"""
    frame_riemann_structure_expr(tetrad_name, I, J, K, L;
                                  registry=current_registry()) -> TensorExpr

Build the frame Riemann tensor from Ricci rotation coefficients via
the Cartan structure equation:

    R^I_{JKL} = e_K(γ^I_{JL}) - e_L(γ^I_{JK})
                + γ^I_{KM}γ^M_{JL} - γ^I_{LM}γ^M_{JK}
                - c^M_{KL}γ^I_{JM}

where γ are Ricci rotation coefficients and c are anholonomy coefficients.

# Arguments
- `tetrad_name::Symbol` -- tetrad name
- `I::TIndex` -- Up Lorentz
- `J, K, L::TIndex` -- Down Lorentz

# Requires
Ricci rotation coefficients and anholonomy must be defined for `tetrad_name`.
"""
function frame_riemann_structure_expr(tetrad_name::Symbol,
                                       I::TIndex, J::TIndex, K::TIndex, L::TIndex;
                                       registry::TensorRegistry=current_registry())
    I.vbundle === :Lorentz && I.position === Up ||
        error("frame_riemann_structure_expr: I must be Up on :Lorentz")
    for (name, idx) in [("J", J), ("K", K), ("L", L)]
        idx.vbundle === :Lorentz && idx.position === Down ||
            error("frame_riemann_structure_expr: $name must be Down on :Lorentz")
    end

    gamma_name = get_ricci_rotation_name(registry, tetrad_name)
    c_name = get_anholonomy_name(registry, tetrad_name)

    used = Set{Symbol}([I.name, J.name, K.name, L.name])
    M_sym = fresh_index(used; vbundle=:Lorentz)
    M_up   = TIndex(M_sym, Up, :Lorentz)
    M_down = TIndex(M_sym, Down, :Lorentz)

    # We need a Tangent dummy for the directional derivatives
    push!(used, M_sym)
    a_sym = fresh_index(used; vbundle=:Tangent)
    push!(used, a_sym)
    b_sym = fresh_index(used; vbundle=:Tangent)

    # Term 1: e_K(γ^I_{JL}) = e^a_K ∂_a γ^I_{JL}
    gamma_IJL = Tensor(gamma_name, [I, J, L])
    d_gamma_IJL = TDeriv(down(a_sym), gamma_IJL, :partial)
    term1 = TProduct(1 // 1, [Tensor(tetrad_name, [up(a_sym), K]), d_gamma_IJL])

    # Term 2: -e_L(γ^I_{JK}) = -e^a_L ∂_a γ^I_{JK}
    gamma_IJK = Tensor(gamma_name, [I, J, K])
    d_gamma_IJK = TDeriv(down(b_sym), gamma_IJK, :partial)
    term2 = TProduct(-1 // 1, [Tensor(tetrad_name, [up(b_sym), L]), d_gamma_IJK])

    # Term 3: γ^I_{KM} γ^M_{JL}
    term3 = TProduct(1 // 1, [
        Tensor(gamma_name, [I, K, M_down]),
        Tensor(gamma_name, [M_up, J, L])
    ])

    # Term 4: -γ^I_{LM} γ^M_{JK}
    term4 = TProduct(-1 // 1, [
        Tensor(gamma_name, [I, L, M_down]),
        Tensor(gamma_name, [M_up, J, K])
    ])

    # Term 5: -c^M_{KL} γ^I_{JM}
    term5 = TProduct(-1 // 1, [
        Tensor(c_name, [M_up, K, L]),
        Tensor(gamma_name, [I, J, M_down])
    ])

    return TSum([term1, term2, term3, term4, term5])
end

"""
    frame_ricci_expr(tetrad_name, I, J;
                      registry=current_registry()) -> TensorExpr

Build the frame-basis Ricci tensor by projecting the coordinate Ricci:

    R_{IJ} = e^a_I e^b_J R_{ab}

# Arguments
- `tetrad_name::Symbol` -- tetrad name
- `I, J::TIndex` -- Down Lorentz indices
"""
function frame_ricci_expr(tetrad_name::Symbol,
                           I::TIndex, J::TIndex;
                           registry::TensorRegistry=current_registry())
    I.vbundle === :Lorentz && I.position === Down ||
        error("frame_ricci_expr: I must be Down on :Lorentz")
    J.vbundle === :Lorentz && J.position === Down ||
        error("frame_ricci_expr: J must be Down on :Lorentz")

    used = Set{Symbol}([I.name, J.name])
    a = fresh_index(used; vbundle=:Tangent); push!(used, a)
    b = fresh_index(used; vbundle=:Tangent)

    return TProduct(1 // 1, [
        Tensor(tetrad_name, [up(a), I]),
        Tensor(tetrad_name, [up(b), J]),
        Tensor(:Ric, [down(a), down(b)])
    ])
end

"""
    frame_ricci_scalar_expr(tetrad_name;
                             registry=current_registry()) -> TensorExpr

The Ricci scalar is a scalar invariant, so it is the same in any basis:

    R = g^{ab} R_{ab} = η^{IJ} R_{IJ}

Returns the coordinate-basis Ricci scalar `RicScalar` (a `Tensor` with no indices),
since it is identical in all bases.
"""
function frame_ricci_scalar_expr(tetrad_name::Symbol;
                                  registry::TensorRegistry=current_registry())
    # The Ricci scalar is basis-independent
    return Tensor(:RicScalar, TIndex[])
end
