#= Differential forms.

A k-form is a completely antisymmetric rank-(0,k) tensor.
We represent forms as standard TensorExpr with the `is_form` flag
and degree metadata.
=#

"""
    define_form!(reg, name; manifold, degree) -> TensorProperties

Define a k-form tensor with full antisymmetry.
"""
function define_form!(reg::TensorRegistry, name::Symbol;
                      manifold::Symbol, degree::Int)
    degree >= 0 || error("Form degree must be non-negative")

    # Build antisymmetry generators for all pairs
    syms = SymmetrySpec[]
    for i in 1:degree-1
        push!(syms, AntiSymmetric(i, i+1))
    end

    register_tensor!(reg, TensorProperties(
        name=name, manifold=manifold, rank=(0, degree),
        symmetries=syms,
        options=Dict{Symbol,Any}(:is_form => true, :degree => degree)))
end

"""
    form_degree(reg, name) -> Int

Get the degree of a registered form.
"""
function form_degree(reg::TensorRegistry, name::Symbol)
    props = get_tensor(reg, name)
    get(props.options, :degree, -1)
end
