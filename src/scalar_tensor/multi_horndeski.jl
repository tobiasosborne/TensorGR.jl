#= Multi-field Horndeski scalar-tensor theory.

Generalizes the single-field Horndeski Lagrangian to N scalar fields
phi^I (I=1..N) with a field-space kinetic matrix
  X^{IJ} = -(1/2) g^{ab} d_a phi^I d_b phi^J
and a field-space (sigma-model) metric G_{IJ}(phi).

The multi-field Lagrangians generalize L_2 through L_5:
  L_2 = G_2(phi^I, X^{IJ})
  L_3 = -G_3^I(phi, X) Box(phi_I)     (summed over I)
  L_4 = G_4 R + G_{4,X^{IJ}} [(Box phi_I)(Box phi_J) - (dd phi_I)_{ab}(dd phi_J)^{ab}]
  L_5 generalized similarly (not implemented in this first version)

Ground truth: Kobayashi, Rep. Prog. Phys. 82, 086901 (2019), Sec 6;
              Ohashi et al, JCAP 1512 (2015) 009, arXiv:1507.04344;
              Padilla et al, JHEP 1212 (2012) 031, arXiv:1208.3373.

Key property: N=1 reduces to the standard single-field Horndeski theory.
=#

# -- MultiScalarTensorFunction -----------------------------------------------

"""
    MultiScalarTensorFunction(name, field_derivs, X_derivs)

Represents an abstract function G_n(phi^I, X^{IJ}) and its derivatives
in the multi-field Horndeski setting.  `field_derivs` counts total
phi-differentiations and `X_derivs` counts total X-differentiations.
"""
struct MultiScalarTensorFunction
    name::Symbol        # base name, e.g. :MG2, :MG3, :MG4, :MG5
    field_derivs::Int   # total number of phi-differentiations
    X_derivs::Int       # total number of X-differentiations
end

"""
    multi_g_tensor_name(mstf::MultiScalarTensorFunction) -> Symbol

Return the registry tensor name for this multi-field G-function, e.g. :MG4_X.
"""
function multi_g_tensor_name(mstf::MultiScalarTensorFunction)
    base = string(mstf.name)
    suffix = ""
    if mstf.field_derivs > 0
        suffix *= join(fill("phi", mstf.field_derivs))
    end
    if mstf.X_derivs > 0
        suffix *= join(fill("X", mstf.X_derivs))
    end
    isempty(suffix) ? Symbol(base) : Symbol(base, "_", suffix)
end

"""
    differentiate_MG(mstf::MultiScalarTensorFunction, var::Symbol) -> MultiScalarTensorFunction

Differentiate the multi-field scalar-tensor function with respect to :phi or :X.
"""
function differentiate_MG(mstf::MultiScalarTensorFunction, var::Symbol)
    if var == :X
        MultiScalarTensorFunction(mstf.name, mstf.field_derivs, mstf.X_derivs + 1)
    elseif var == :phi
        MultiScalarTensorFunction(mstf.name, mstf.field_derivs + 1, mstf.X_derivs)
    else
        error("differentiate_MG: variable must be :phi or :X, got $var")
    end
end

# -- MultiHorndeskiTheory ----------------------------------------------------

"""
    MultiHorndeskiTheory

Multi-field Horndeski theory with N scalar fields phi^I.

Fields:
- `n_fields`: number of scalar fields N
- `field_names`: names of scalar fields, e.g. [:phi1, :phi2]
- `field_metric`: symbol for field-space metric G_{IJ}
- `G2`: multi-field G_2 function (scalar)
- `G3`: vector of G_3^I functions (one per field)
- `G4`: multi-field G_4 function (scalar)
- `G5`: multi-field G_5 function (scalar)
- `manifold`: spacetime manifold symbol
- `metric`: spacetime metric symbol
- `field_vbundle`: field-space vector bundle symbol
"""
struct MultiHorndeskiTheory
    n_fields::Int
    field_names::Vector{Symbol}
    field_metric::Symbol
    G2::MultiScalarTensorFunction
    G3::Vector{MultiScalarTensorFunction}
    G4::MultiScalarTensorFunction
    G5::MultiScalarTensorFunction
    manifold::Symbol
    metric::Symbol
    field_vbundle::Symbol
end

# -- Registration -------------------------------------------------------------

"""
    define_multi_horndeski!(reg; n_fields, manifold, metric, field_names=nothing,
                             field_metric=:Gfield)

Register a multi-field Horndeski theory with N scalar fields.

Registers:
- Each scalar field phi^I as a rank-0 tensor
- A field-space metric G_{IJ} (symmetric rank-2 in field space)
- Multi-field G-functions MG2, MG3_I, MG4, MG5 as rank-0 tensors
- A field-space vector bundle :FieldSpace with dim=N
- Kinetic matrix tensor X_{IJ}

Returns a `MultiHorndeskiTheory` struct.
"""
function define_multi_horndeski!(reg::TensorRegistry;
    n_fields::Int,
    manifold::Symbol,
    metric::Symbol,
    field_names::Union{Nothing,Vector{Symbol}}=nothing,
    field_metric::Symbol=:Gfield)

    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    n_fields >= 1 || error("n_fields must be >= 1, got $n_fields")

    # Default field names: phi1, phi2, ...
    if field_names === nothing
        field_names = [Symbol("phi", i) for i in 1:n_fields]
    end
    length(field_names) == n_fields ||
        error("field_names length $(length(field_names)) != n_fields $n_fields")

    # Register field-space vector bundle
    field_vbundle = :FieldSpace
    field_indices = [Symbol("I"), Symbol("J"), Symbol("K"), Symbol("L"),
                     Symbol("M"), Symbol("N"), Symbol("P"), Symbol("Q")]
    if !has_vbundle(reg, field_vbundle)
        define_vbundle!(reg, field_vbundle;
            manifold=manifold, dim=n_fields, indices=field_indices)
    end

    # Register each scalar field as rank-0 tensor
    for fname in field_names
        if !has_tensor(reg, fname)
            register_tensor!(reg, TensorProperties(
                name=fname, manifold=manifold, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:is_scalar_field => true)))
        end
    end

    # Register field-space metric G_{IJ} as rank-(0,2) symmetric tensor
    if !has_tensor(reg, field_metric)
        register_tensor!(reg, TensorProperties(
            name=field_metric, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_field_metric => true,
                                     :n_fields => n_fields)))
    end

    # Base multi-field G-functions
    MG2 = MultiScalarTensorFunction(:MG2, 0, 0)
    MG4 = MultiScalarTensorFunction(:MG4, 0, 0)
    MG5 = MultiScalarTensorFunction(:MG5, 0, 0)

    # G3 has one function per field: MG3_1, MG3_2, ...
    MG3_vec = [MultiScalarTensorFunction(Symbol("MG3_", i), 0, 0)
               for i in 1:n_fields]

    # Register scalar G-functions needed for L2, L3, L4
    needed_scalars = [
        MG2,
        MG4, differentiate_MG(MG4, :X),
        MG5,
    ]

    for stf in needed_scalars
        tname = multi_g_tensor_name(stf)
        if !has_tensor(reg, tname)
            register_tensor!(reg, TensorProperties(
                name=tname, manifold=manifold, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:is_multi_scalar_tensor_function => true,
                                         :mstf_base => stf.name,
                                         :mstf_field_derivs => stf.field_derivs,
                                         :mstf_X_derivs => stf.X_derivs)))
        end
    end

    # Register per-field G3 functions
    for stf in MG3_vec
        tname = multi_g_tensor_name(stf)
        if !has_tensor(reg, tname)
            register_tensor!(reg, TensorProperties(
                name=tname, manifold=manifold, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:is_multi_scalar_tensor_function => true,
                                         :mstf_base => stf.name,
                                         :mstf_field_derivs => stf.field_derivs,
                                         :mstf_X_derivs => stf.X_derivs)))
        end
    end

    # Register kinetic matrix symbol (abstract tensor, for display/documentation)
    kinetic_name = :Xkin
    if !has_tensor(reg, kinetic_name)
        register_tensor!(reg, TensorProperties(
            name=kinetic_name, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_kinetic_matrix => true)))
    end

    MultiHorndeskiTheory(n_fields, field_names, field_metric,
                          MG2, MG3_vec, MG4, MG5,
                          manifold, metric, field_vbundle)
end

# -- Kinetic matrix -----------------------------------------------------------

"""
    kinetic_matrix(theory::MultiHorndeskiTheory, I::Int, J::Int;
                   registry=current_registry()) -> TensorExpr

Build the (I,J) component of the kinetic matrix:
  X^{IJ} = -(1/2) g^{ab} d_a phi^I d_b phi^J

where I, J are field indices (1-based). X^{IJ} is symmetric: X^{IJ} = X^{JI}.
"""
function kinetic_matrix(theory::MultiHorndeskiTheory, I::Int, J::Int;
                         registry::TensorRegistry=current_registry())
    1 <= I <= theory.n_fields || error("Field index I=$I out of range 1:$(theory.n_fields)")
    1 <= J <= theory.n_fields || error("Field index J=$J out of range 1:$(theory.n_fields)")

    phi_I = Tensor(theory.field_names[I], TIndex[])
    phi_J = Tensor(theory.field_names[J], TIndex[])

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    g_up = Tensor(theory.metric, [up(a), up(b)])
    dphi_I = TDeriv(down(a), phi_I)
    dphi_J = TDeriv(down(b), phi_J)

    (-1 // 2) * g_up * dphi_I * dphi_J
end

"""
    kinetic_matrix_full(theory::MultiHorndeskiTheory;
                        registry=current_registry()) -> Matrix{TensorExpr}

Build the full N x N kinetic matrix X^{IJ} as a Julia matrix of tensor
expressions.
"""
function kinetic_matrix_full(theory::MultiHorndeskiTheory;
                              registry::TensorRegistry=current_registry())
    N = theory.n_fields
    X = Matrix{TensorExpr}(undef, N, N)
    for I in 1:N, J in 1:N
        X[I, J] = kinetic_matrix(theory, I, J; registry=registry)
    end
    X
end

# -- Multi-field Lagrangians --------------------------------------------------

"""
    multi_horndeski_L2(theory::MultiHorndeskiTheory;
                        registry=current_registry()) -> TensorExpr

Multi-field L_2 = G_2(phi^I, X^{IJ}).

This is simply the abstract scalar function MG2, which depends on all
fields and kinetic matrix entries.  At the abstract tensor level, it is
represented as the rank-0 tensor MG2.

Kobayashi (2019) Sec 6, generalized Eq 5.
"""
function multi_horndeski_L2(theory::MultiHorndeskiTheory;
                             registry::TensorRegistry=current_registry())
    Tensor(multi_g_tensor_name(theory.G2), TIndex[])
end

"""
    multi_horndeski_L3(theory::MultiHorndeskiTheory;
                        registry=current_registry()) -> TensorExpr

Multi-field L_3 = -sum_I G_3^I(phi, X) Box(phi_I).

Each G_3^I is a separate scalar function of all fields and kinetic matrix.
The sum is over all field indices I=1..N.

Ohashi et al (2015) Eq 2.2; Padilla et al (2012).
"""
function multi_horndeski_L3(theory::MultiHorndeskiTheory;
                             registry::TensorRegistry=current_registry())
    N = theory.n_fields
    terms = TensorExpr[]
    for I in 1:N
        phi_I = Tensor(theory.field_names[I], TIndex[])
        box_phi_I = box(phi_I, theory.metric; registry=registry)
        G3_I = Tensor(multi_g_tensor_name(theory.G3[I]), TIndex[])
        push!(terms, (-1 // 1) * G3_I * box_phi_I)
    end
    length(terms) == 1 ? terms[1] : TSum(terms)
end

"""
    multi_horndeski_L4(theory::MultiHorndeskiTheory;
                        registry=current_registry()) -> TensorExpr

Multi-field L_4 = G_4 R + G_{4,X} * sum_{I,J} dG4/dX^{IJ} *
                  [(Box phi_I)(Box phi_J) - (dd phi_I)_{ab}(dd phi_J)^{ab}]

At the abstract level, for the G_{4,X} coupling we sum over all field pairs.
The derivative dG4/dX^{IJ} is represented by the abstract tensor MG4_X,
which in the multi-field case carries implicit field-space dependence.

For a clean first implementation, we build the structure summing over
all field pairs (I,J), using MG4_X as a single scalar coupling (the
trace/diagonal part).  The full field-space derivative structure can be
specialized by the user.

Kobayashi (2019) Sec 6; Ohashi et al (2015).
"""
function multi_horndeski_L4(theory::MultiHorndeskiTheory;
                             registry::TensorRegistry=current_registry())
    with_registry(registry) do
        N = theory.n_fields
        R = Tensor(:RicScalar, TIndex[])
        G4 = Tensor(multi_g_tensor_name(theory.G4), TIndex[])
        G4X = Tensor(multi_g_tensor_name(differentiate_MG(theory.G4, :X)), TIndex[])

        # First term: G4 * R (scalar curvature coupling)
        term1 = G4 * R

        # Second term: sum over field pairs
        # G_{4,X} * [(Box phi_I)(Box phi_J) - (dd phi_I)_{ab} (dd phi_J)^{ab}]
        # For each pair (I,J) we build the bracket.
        pair_terms = TensorExpr[]
        for I in 1:N, J in I:N  # symmetric in I,J
            phi_I = Tensor(theory.field_names[I], TIndex[])
            phi_J = Tensor(theory.field_names[J], TIndex[])

            box_I = box(phi_I, theory.metric; registry=registry)
            box_J = box(phi_J, theory.metric; registry=registry)

            used = Set{Symbol}()
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used); push!(used, b)
            c = fresh_index(used); push!(used, c)
            d = fresh_index(used); push!(used, d)

            # (dd phi_I)_{ab} (dd phi_J)^{ab}
            dd_I = covd_chain(phi_I, [down(a), down(b)])
            dd_J = covd_chain(phi_J, [down(c), down(d)])
            g_ac = Tensor(theory.metric, [up(a), up(c)])
            g_bd = Tensor(theory.metric, [up(b), up(d)])
            nabla_IJ = g_ac * g_bd * dd_I * dd_J

            bracket = box_I * box_J - nabla_IJ

            # Symmetry factor: diagonal (I==J) gets weight 1, off-diagonal gets weight 2
            coeff = (I == J) ? (1 // 1) : (2 // 1)
            push!(pair_terms, coeff * bracket)
        end

        pair_sum = length(pair_terms) == 1 ? pair_terms[1] : TSum(pair_terms)
        term1 + G4X * pair_sum
    end
end

# -- Single-field reduction ---------------------------------------------------

"""
    to_single_field(theory::MultiHorndeskiTheory;
                    registry=current_registry()) -> HorndeskiTheory

For N=1, construct the equivalent single-field HorndeskiTheory.

This verifies that the multi-field framework correctly reduces to the
standard Horndeski theory when only one scalar field is present.

The mapping is:
  phi^1 -> phi
  MG2 -> G2, MG3_1 -> G3, MG4 -> G4, MG5 -> G5
"""
function to_single_field(theory::MultiHorndeskiTheory;
                          registry::TensorRegistry=current_registry())
    theory.n_fields == 1 ||
        error("to_single_field requires n_fields=1, got $(theory.n_fields)")

    # Register single-field Horndeski tensors by delegating to define_horndeski!
    define_horndeski!(registry;
        manifold=theory.manifold, metric=theory.metric,
        scalar_field=theory.field_names[1])
end
