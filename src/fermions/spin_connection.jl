#= Spin connection for Dirac spinors in curved spacetime.
#
# The covariant derivative of a Dirac spinor involves the spin connection:
#
#   ∇_a ψ = ∂_a ψ + (1/4) ω_a^{IJ} γ_I γ_J ψ
#
# where ω_a^{IJ} is the spin connection (antisymmetric in I,J) with one
# spacetime (Tangent) index and two frame (Lorentz) indices.
#
# For the Dirac conjugate:
#   ∇_a ψ̄ = ∂_a ψ̄ - (1/4) ω_a^{IJ} ψ̄ γ_I γ_J
#
# The spin connection is related to the Ricci rotation coefficients by:
#   ω_a^{IJ} = e^K_a γ^{IJ}{}_K
#
# where γ^{IJ}{}_K = η^{IM}η^{JN}γ_{MNK} is the Ricci rotation coefficient
# with the first two indices raised.
#
# References:
#   Wald, *General Relativity* (1984), Appendix B.
#   Birrell & Davies, *QFT in Curved Space* (1982), Sec 3.8.
#   Fröb (2020), arXiv:2008.12422, Sec 3.2.
=#

"""
    define_spin_connection!(reg, tetrad_name; manifold=:M4,
                             connection_name=:omega_spin)

Register the spin connection tensor ω_a^{IJ} for Dirac spinors.

The spin connection has:
- One Down Tangent index `a` (spacetime derivative direction)
- Two Lorentz indices I,J (antisymmetric in I,J)

# Requires
- `:Lorentz` VBundle (via `define_frame_bundle!`)
- Tetrad registered
- Ricci rotation coefficients defined (via `define_ricci_rotation!`)
"""
function define_spin_connection!(reg::TensorRegistry, tetrad_name::Symbol;
                                  manifold::Symbol=:M4,
                                  connection_name::Symbol=:omega_spin)
    has_vbundle(reg, :Lorentz) ||
        error("define_spin_connection!: :Lorentz VBundle not found")
    has_tensor(reg, tetrad_name) ||
        error("define_spin_connection!: tetrad '$tetrad_name' not registered")
    has_ricci_rotation(reg, tetrad_name) ||
        error("define_spin_connection!: Ricci rotation for '$tetrad_name' not defined. " *
              "Call define_ricci_rotation! first.")

    if !has_tensor(reg, connection_name)
        register_tensor!(reg, TensorProperties(
            name=connection_name, manifold=manifold, rank=(2, 1),
            symmetries=SymmetrySpec[AntiSymmetric(1, 2)],
            options=Dict{Symbol,Any}(
                :is_spin_connection => true,
                :tetrad => tetrad_name,
                :vbundle => :Lorentz)))
    end
    nothing
end

"""
    spin_connection_expr(tetrad_name, a, I, J; registry) -> TProduct

Build the spin connection expression in terms of Ricci rotation coefficients:

    ω_a^{IJ} = e^K_a γ^{IJ}{}_K = e^K_a η^{IM}η^{JN}γ_{MNK}

Since γ^I_{JK} has the first index up and the rest down (γ^I_{JK} = η^{IM}γ_{MJK}),
and γ_{MNK} = -γ_{NMK}, we have:

    ω_a^{IJ} = e^K_a (η^{JN} γ^I_{NK})

which uses one eta contraction on the Ricci rotation coefficient.

# Arguments
- `tetrad_name::Symbol` -- name of the tetrad tensor
- `a::TIndex` -- Down Tangent index (derivative direction)
- `I::TIndex` -- Up Lorentz index
- `J::TIndex` -- Up Lorentz index

# Returns
A `TProduct` expression.
"""
function spin_connection_expr(tetrad_name::Symbol,
                               a::TIndex, I::TIndex, J::TIndex;
                               registry::TensorRegistry=current_registry())
    a.vbundle === :Tangent && a.position === Down ||
        error("spin_connection_expr: a must be Down on :Tangent")
    I.vbundle === :Lorentz && I.position === Up ||
        error("spin_connection_expr: I must be Up on :Lorentz")
    J.vbundle === :Lorentz && J.position === Up ||
        error("spin_connection_expr: J must be Up on :Lorentz")

    gamma_name = get_ricci_rotation_name(registry, tetrad_name)

    used = Set{Symbol}([a.name, I.name, J.name])
    K_sym = fresh_index(used; vbundle=:Lorentz)
    push!(used, K_sym)
    N_sym = fresh_index(used; vbundle=:Lorentz)

    K_up   = TIndex(K_sym, Up, :Lorentz)
    K_down = TIndex(K_sym, Down, :Lorentz)
    N_up   = TIndex(N_sym, Up, :Lorentz)
    N_down = TIndex(N_sym, Down, :Lorentz)

    # ω_a^{IJ} = e^K_a · η^{JN} · γ^I_{NK}
    # e^K_a: tetrad with [Up Lorentz K, Down Tangent a]
    # η^{JN}: frame metric raising J
    # γ^I_{NK}: Ricci rotation with [Up Lorentz I, Down Lorentz N, Down Lorentz K]
    return TProduct(1 // 1, [
        Tensor(tetrad_name, [K_up, a]),
        Tensor(:eta, [J, N_up]),
        Tensor(gamma_name, [I, N_down, K_down])
    ])
end

"""
    has_spin_connection(reg, tetrad_name) -> Bool

Check if a spin connection is registered for the given tetrad.
"""
function has_spin_connection(reg::TensorRegistry, tetrad_name::Symbol)
    for (_, props) in reg.tensors
        if get(props.options, :is_spin_connection, false) &&
           get(props.options, :tetrad, nothing) === tetrad_name
            return true
        end
    end
    false
end

"""
    get_spin_connection_name(reg, tetrad_name) -> Symbol

Return the name of the spin connection tensor for `tetrad_name`.
"""
function get_spin_connection_name(reg::TensorRegistry, tetrad_name::Symbol)
    for (name, props) in reg.tensors
        if get(props.options, :is_spin_connection, false) &&
           get(props.options, :tetrad, nothing) === tetrad_name
            return name
        end
    end
    error("No spin connection registered for tetrad '$tetrad_name'")
end

"""
    dirac_covd_expr(psi_name, a, tetrad_name; registry) -> TSum

Build the covariant derivative of a Dirac field:

    ∇_a ψ = ∂_a ψ + (1/4) ω_a^{IJ} γ_I γ_J ψ

The result has the spin connection ω_a^{IJ} as an abstract tensor
(not expanded into Ricci rotation coefficients).

# Arguments
- `psi_name::Symbol` -- name of the Dirac field
- `a::TIndex` -- Down Tangent index (derivative direction)
- `tetrad_name::Symbol` -- tetrad name (for looking up the spin connection)

# Returns
A `TSum` with two terms: the partial derivative and the spin connection term.
"""
function dirac_covd_expr(psi_name::Symbol, a::TIndex,
                          tetrad_name::Symbol;
                          registry::TensorRegistry=current_registry())
    a.vbundle === :Tangent && a.position === Down ||
        error("dirac_covd_expr: a must be Down on :Tangent")

    omega_name = get_spin_connection_name(registry, tetrad_name)
    psi = Tensor(psi_name, TIndex[])

    # Generate fresh Lorentz dummy indices
    used = Set{Symbol}([a.name])
    I_sym = fresh_index(used; vbundle=:Lorentz)
    push!(used, I_sym)
    J_sym = fresh_index(used; vbundle=:Lorentz)

    I_up   = TIndex(I_sym, Up, :Lorentz)
    I_down = TIndex(I_sym, Down, :Lorentz)
    J_up   = TIndex(J_sym, Up, :Lorentz)
    J_down = TIndex(J_sym, Down, :Lorentz)

    # Term 1: ∂_a ψ
    term1 = TDeriv(a, psi, :partial)

    # Term 2: (1/4) ω_a^{IJ} γ_I γ_J ψ
    term2 = TProduct(1 // 4, TensorExpr[
        Tensor(omega_name, [I_up, J_up, a]),
        GammaMatrix(I_down),
        GammaMatrix(J_down),
        psi
    ])

    TSum(TensorExpr[term1, term2])
end

"""
    dirac_bar_covd_expr(psi_name, a, tetrad_name; registry) -> TSum

Build the covariant derivative of the Dirac conjugate:

    ∇_a ψ̄ = ∂_a ψ̄ - (1/4) ω_a^{IJ} ψ̄ γ_I γ_J

Note the minus sign and the ordering (ψ̄ is to the left of the gammas).
"""
function dirac_bar_covd_expr(psi_name::Symbol, a::TIndex,
                              tetrad_name::Symbol;
                              registry::TensorRegistry=current_registry())
    a.vbundle === :Tangent && a.position === Down ||
        error("dirac_bar_covd_expr: a must be Down on :Tangent")

    bar_name = get_conjugate_name(registry, psi_name)
    omega_name = get_spin_connection_name(registry, tetrad_name)
    psi_bar = Tensor(bar_name, TIndex[])

    used = Set{Symbol}([a.name])
    I_sym = fresh_index(used; vbundle=:Lorentz)
    push!(used, I_sym)
    J_sym = fresh_index(used; vbundle=:Lorentz)

    I_up   = TIndex(I_sym, Up, :Lorentz)
    I_down = TIndex(I_sym, Down, :Lorentz)
    J_up   = TIndex(J_sym, Up, :Lorentz)
    J_down = TIndex(J_sym, Down, :Lorentz)

    # Term 1: ∂_a ψ̄
    term1 = TDeriv(a, psi_bar, :partial)

    # Term 2: -(1/4) ω_a^{IJ} ψ̄ γ_I γ_J
    term2 = TProduct(-1 // 4, TensorExpr[
        Tensor(omega_name, [I_up, J_up, a]),
        psi_bar,
        GammaMatrix(I_down),
        GammaMatrix(J_down)
    ])

    TSum(TensorExpr[term1, term2])
end
