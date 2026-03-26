#= Stress-energy tensor for the Dirac field.
#
# The symmetrized (Belinfante) stress-energy tensor for a Dirac field is:
#
#   T^{ab} = (i/4)[ψ̄γ^a∇^bψ + ψ̄γ^b∇^aψ - (∇^aψ̄)γ^bψ - (∇^bψ̄)γ^aψ]
#
# This is symmetric in (a,b) by construction. On-shell (using the Dirac
# equation), it satisfies:
#   - Conservation: ∇_a T^{ab} = 0
#   - Trace: T^a_a = m ψ̄ψ (massive) or 0 (massless, conformal)
#
# The covariant derivatives ∇ here include the spin connection.
# In the abstract representation, we use D to denote the covariant
# derivative and leave the spin connection implicit.
#
# References:
#   Birrell & Davies, *QFT in Curved Space* (1982), Sec 3.8.
#   Fröb (2020), arXiv:2008.12422, Sec 3.3.
#   Parker & Toms, *QFT in Curved Spacetime* (2009), Sec 2.6.
=#

"""
    DiracStressEnergy

Stores the definition of a Dirac field stress-energy tensor.
"""
struct DiracStressEnergy
    name::Symbol
    manifold::Symbol
    metric::Symbol
    field::Symbol
    conjugate::Symbol
    covd::Symbol
end

"""
    define_dirac_stress_energy!(reg, name; manifold, metric,
                                 field, covd=:D) -> DiracStressEnergy

Register the Dirac stress-energy tensor T^{ab} (symmetric, rank (2,0)).

# Arguments
- `name::Symbol` -- name for the stress-energy tensor
- `manifold::Symbol` -- manifold name
- `metric::Symbol` -- metric name
- `field::Symbol` -- Dirac field name (must be registered via `define_fermion!`)
- `covd::Symbol` -- covariant derivative name (default `:D`)
"""
function define_dirac_stress_energy!(reg::TensorRegistry, name::Symbol;
                                      manifold::Symbol,
                                      metric::Symbol,
                                      field::Symbol,
                                      covd::Symbol=:D)
    @lock reg.lock begin
        has_manifold(reg, manifold) ||
            error("define_dirac_stress_energy!: manifold '$manifold' not registered")
        has_tensor(reg, metric) ||
            error("define_dirac_stress_energy!: metric '$metric' not registered")
        is_grassmann(reg, field) ||
            error("define_dirac_stress_energy!: '$field' is not a Grassmann field")

        bar_name = get_conjugate_name(reg, field)

        if !has_tensor(reg, name)
            register_tensor!(reg, TensorProperties(
                name=name, manifold=manifold, rank=(2, 0),
                symmetries=SymmetrySpec[Symmetric(1, 2)],
                options=Dict{Symbol,Any}(
                    :is_stress_energy => true,
                    :matter_type => :dirac,
                    :field => field,
                    :conjugate => bar_name,
                    :covd => covd,
                    :metric => metric)))
        end

        dse = DiracStressEnergy(name, manifold, metric, field, bar_name, covd)
        reg.foliations[Symbol(:dirac_stress_energy_, name)] = dse
        dse
    end
end

"""
    dirac_stress_energy_expr(a, b, dse; registry=current_registry()) -> TSum

Build the Belinfante symmetrized Dirac stress-energy tensor:

    T^{ab} = (i/4)[ψ̄γ^a D^b ψ + ψ̄γ^b D^a ψ
                    - (D^a ψ̄)γ^b ψ - (D^b ψ̄)γ^a ψ]

where D is the covariant derivative (with spin connection implicit).

# Arguments
- `a, b::TIndex` -- Up Tangent indices
- `dse::DiracStressEnergy` -- the stress-energy definition

# Returns
A `TSum` with four terms (the four symmetrized bilinear contributions).
"""
function dirac_stress_energy_expr(a::TIndex, b::TIndex,
                                   dse::DiracStressEnergy;
                                   registry::TensorRegistry=current_registry())
    a.position === Up || error("dirac_stress_energy_expr: a must be Up")
    b.position === Up || error("dirac_stress_energy_expr: b must be Up")

    psi = Tensor(dse.field, TIndex[])
    psi_bar = Tensor(dse.conjugate, TIndex[])
    covd = dse.covd

    # D^a ψ and D^b ψ (covariant derivative with raised index)
    # D^a = g^{ac} D_c, but in abstract form we use TDeriv with Up index
    Da_psi = TDeriv(a, psi, covd)
    Db_psi = TDeriv(b, psi, covd)

    # D^a ψ̄ and D^b ψ̄
    Da_psi_bar = TDeriv(a, psi_bar, covd)
    Db_psi_bar = TDeriv(b, psi_bar, covd)

    # Term 1: (i/4) ψ̄ γ^a D^b ψ
    term1 = TProduct(1 // 4, TensorExpr[
        TScalar(:im),
        psi_bar,
        GammaMatrix(a),
        Db_psi
    ])

    # Term 2: (i/4) ψ̄ γ^b D^a ψ
    term2 = TProduct(1 // 4, TensorExpr[
        TScalar(:im),
        psi_bar,
        GammaMatrix(b),
        Da_psi
    ])

    # Term 3: -(i/4) (D^a ψ̄) γ^b ψ
    term3 = TProduct(-1 // 4, TensorExpr[
        TScalar(:im),
        Da_psi_bar,
        GammaMatrix(b),
        psi
    ])

    # Term 4: -(i/4) (D^b ψ̄) γ^a ψ
    term4 = TProduct(-1 // 4, TensorExpr[
        TScalar(:im),
        Db_psi_bar,
        GammaMatrix(a),
        psi
    ])

    TSum(TensorExpr[term1, term2, term3, term4])
end

"""
    dirac_stress_trace_expr(dse; registry=current_registry()) -> TProduct

The trace of the Dirac stress-energy tensor:

    T^a_a = m ψ̄ψ

On-shell (using the Dirac equation), the trace gives the mass term.
For massless fields (m=0), T^a_a = 0 (conformal invariance).

Returns the expression `m * ψ̄ψ`.
"""
function dirac_stress_trace_expr(dse::DiracStressEnergy;
                                  registry::TensorRegistry=current_registry())
    props = get_tensor(registry, dse.field)
    mass_name = get(props.options, :mass, :m)

    TProduct(1 // 1, TensorExpr[
        TScalar(mass_name),
        Tensor(dse.conjugate, TIndex[]),
        Tensor(dse.field, TIndex[])
    ])
end

"""
    get_dirac_stress_energy(reg, name) -> DiracStressEnergy

Retrieve a previously defined Dirac stress-energy from the registry.
"""
function get_dirac_stress_energy(reg::TensorRegistry, name::Symbol)
    key = Symbol(:dirac_stress_energy_, name)
    haskey(reg.foliations, key) ||
        error("No Dirac stress-energy :$name registered")
    reg.foliations[key]::DiracStressEnergy
end
