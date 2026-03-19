# Charge conjugation matrix C.
#
# The charge conjugation matrix satisfies:
#   C γ^a C^{-1} = -(γ^a)^T
#
# Properties:
#   C^T = -C  (antisymmetric)
#   C^{-1} = C^† = -C  (unitary and antisymmetric)
#   C σ^{ab} C^{-1} = -(σ^{ab})^T
#   C γ^5 C^{-1} = (γ^5)^T
#
# Majorana condition: ψ^c = C ψ̄^T = ψ (Majorana spinor self-conjugate)
#
# Ground truth: Wald, GR, Appendix B;
#               Freedman & Van Proeyen, Supergravity (2012), Ch 3.

"""
    ChargeConjugation <: TensorExpr

The charge conjugation matrix C in spinor space.
Index-free (implicit spinor indices). Satisfies C γ^a C^{-1} = -(γ^a)^T.
"""
struct ChargeConjugation <: TensorExpr end

Base.:(==)(::ChargeConjugation, ::ChargeConjugation) = true
Base.hash(::ChargeConjugation, h::UInt) = hash(:ChargeConjugation, h)

indices(::ChargeConjugation) = TIndex[]
free_indices(::ChargeConjugation) = TIndex[]
children(::ChargeConjugation) = TensorExpr[]
walk(f, c::ChargeConjugation) = f(c)
derivative_order(::ChargeConjugation) = 0
is_constant(::ChargeConjugation) = false
is_sorted_covds(::ChargeConjugation) = true
rename_dummy(c::ChargeConjugation, ::Symbol, ::Symbol) = c
rename_dummies(c::ChargeConjugation, ::Dict{Symbol,Symbol}) = c
_replace_index_name(c::ChargeConjugation, ::Symbol, ::Symbol) = c
to_expr(::ChargeConjugation) = :(ChargeConjugation())
is_well_formed(::ChargeConjugation) = true
_validate_walk(::ChargeConjugation, ::TensorRegistry, ::Vector{String}) = nothing
dagger(::ChargeConjugation) = ChargeConjugation()  # C^† = -C, but dagger returns same type

Base.show(io::IO, ::ChargeConjugation) = print(io, "C")
to_latex(::ChargeConjugation) = "C"
to_unicode(::ChargeConjugation) = "C"

"""
    charge_conjugation_properties() -> NamedTuple

Return the algebraic properties of the charge conjugation matrix C.

- `:antisymmetric` — C^T = -C
- `:unitary` — C^{-1} = C^†
- `:gamma_conjugation` — C γ^a C^{-1} = -(γ^a)^T
- `:sigma_conjugation` — C σ^{ab} C^{-1} = -(σ^{ab})^T
- `:gamma5_conjugation` — C γ^5 C^{-1} = (γ^5)^T
"""
function charge_conjugation_properties()
    (
        antisymmetric = true,       # C^T = -C
        unitary = true,             # C^{-1} = C^†
        gamma_conjugation = -1,     # C γ^a C^{-1} = -1 × (γ^a)^T
        sigma_conjugation = -1,     # C σ^{ab} C^{-1} = -1 × (σ^{ab})^T
        gamma5_conjugation = 1,     # C γ^5 C^{-1} = +1 × (γ^5)^T
    )
end

"""
    majorana_condition() -> String

Return a description of the Majorana condition:
    ψ^c = C γ^0 ψ* = ψ
A Majorana spinor is its own charge conjugate.
"""
majorana_condition() = "ψ^c = C γ⁰ ψ* = ψ (self-conjugate)"
