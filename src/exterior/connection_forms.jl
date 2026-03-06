#= Connection and curvature forms with Cartan structure equations.

These functions build abstract TensorExpr representations of:
  - Connection 1-forms: omega^a_b = Gamma^a_{cb} dx^c
  - Curvature 2-forms: Omega^a_b = d(omega^a_b) + omega^a_c ^ omega^c_b
  - First Cartan structure equation:  T^a = d(theta^a) + omega^a_b ^ theta^b
  - Second Cartan structure equation: Omega^a_b = d(omega^a_b) + omega^a_c ^ omega^c_b
=#

"""
    connection_form(christoffel::Symbol, a::TIndex, b::TIndex, form_idx::TIndex) -> TensorExpr

Build the connection 1-form omega^a_b = Gamma^a_{c b} dx^c.

The `form_idx` plays the role of the 1-form (contracted) index c.
Returns `Tensor(christoffel, [a, form_idx, b])`, representing the
Christoffel symbol with its middle index serving as the form leg.

# Arguments
- `christoffel`: name of the connection coefficient tensor (e.g. `:Gamma`)
- `a`: upper frame index
- `b`: lower frame index
- `form_idx`: the covariant 1-form index (down)
"""
function connection_form(christoffel::Symbol, a::TIndex, b::TIndex,
                         form_idx::TIndex)
    Tensor(christoffel, [a, form_idx, b])
end

"""
    curvature_form(christoffel::Symbol, a::TIndex, b::TIndex,
                   form_idx1::TIndex, form_idx2::TIndex) -> TensorExpr

Build the curvature 2-form via the second structure equation:

    Omega^a_b = d(omega^a_b) + omega^a_c wedge omega^c_b

The two form indices (`form_idx1`, `form_idx2`) label the antisymmetric
2-form legs of the result.

# Arguments
- `christoffel`: name of the connection coefficient tensor
- `a`: upper frame index
- `b`: lower frame index
- `form_idx1`, `form_idx2`: the two covariant 2-form indices
"""
function curvature_form(christoffel::Symbol, a::TIndex, b::TIndex,
                        form_idx1::TIndex, form_idx2::TIndex)
    # Collect all index names in use
    used = Set{Symbol}([a.name, b.name, form_idx1.name, form_idx2.name])

    # omega^a_b as a 1-form in form_idx1
    omega_ab = connection_form(christoffel, a, b, form_idx1)

    # d(omega^a_b): exterior derivative adds form_idx2
    d_omega = TDeriv(form_idx2, omega_ab)

    # omega^a_c wedge omega^c_b: need a fresh dummy index c
    c = fresh_index(used)
    push!(used, c)

    omega_ac = connection_form(christoffel, a, TIndex(c, b.position, b.vbundle), form_idx1)
    omega_cb = connection_form(christoffel, TIndex(c, a.position, a.vbundle), b, form_idx2)

    # wedge of two 1-forms (p=1, q=1)
    wedge_term = wedge(omega_ac, omega_cb, 1, 1)

    # Omega^a_b = d(omega) + omega^a_c ^ omega^c_b
    d_omega + wedge_term
end

"""
    cartan_first_structure(torsion::Symbol, connection::Symbol,
                           coframe::Symbol, a::TIndex,
                           form_idx1::TIndex, form_idx2::TIndex) -> TensorExpr

Build the first Cartan structure equation (torsion 2-form):

    T^a = d(theta^a) + omega^a_b wedge theta^b

where theta^a is the coframe (vielbein 1-form) and omega^a_b is the
connection 1-form built from `connection`.

# Arguments
- `torsion`: (unused in the construction, but documents intent)
- `connection`: name of the connection coefficient tensor
- `coframe`: name of the coframe 1-form tensor
- `a`: upper frame index
- `form_idx1`, `form_idx2`: the two covariant 2-form indices
"""
function cartan_first_structure(torsion::Symbol, connection::Symbol,
                                coframe::Symbol, a::TIndex,
                                form_idx1::TIndex, form_idx2::TIndex)
    used = Set{Symbol}([a.name, form_idx1.name, form_idx2.name])

    # theta^a as a 1-form in form_idx1
    theta_a = Tensor(coframe, [a, form_idx1])

    # d(theta^a): derivative with index form_idx2
    d_theta = TDeriv(form_idx2, theta_a)

    # omega^a_b wedge theta^b: need a fresh dummy b
    b = fresh_index(used)
    push!(used, b)

    # Connection 1-form omega^a_b with form leg = form_idx1
    omega_ab = connection_form(connection, a, TIndex(b, Down, a.vbundle), form_idx1)

    # Coframe 1-form theta^b with form leg = form_idx2
    theta_b = Tensor(coframe, [TIndex(b, Up, a.vbundle), form_idx2])

    # wedge of two 1-forms
    wedge_term = wedge(omega_ab, theta_b, 1, 1)

    # T^a = d(theta^a) + omega^a_b ^ theta^b
    d_theta + wedge_term
end

"""
    cartan_second_structure(curvature::Symbol, connection::Symbol,
                            a::TIndex, b::TIndex,
                            form_idx1::TIndex, form_idx2::TIndex) -> TensorExpr

Build the second Cartan structure equation (curvature 2-form):

    Omega^a_b = d(omega^a_b) + omega^a_c wedge omega^c_b

This is mathematically identical to [`curvature_form`](@ref) but accepts
an explicit `curvature` symbol for documentation/naming purposes.

# Arguments
- `curvature`: (unused in the construction, documents the curvature tensor name)
- `connection`: name of the connection coefficient tensor
- `a`: upper frame index
- `b`: lower frame index
- `form_idx1`, `form_idx2`: the two covariant 2-form indices
"""
function cartan_second_structure(curvature::Symbol, connection::Symbol,
                                 a::TIndex, b::TIndex,
                                 form_idx1::TIndex, form_idx2::TIndex)
    curvature_form(connection, a, b, form_idx1, form_idx2)
end
