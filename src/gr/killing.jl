#= Killing vector fields.

A Killing vector ξ satisfies ∇_{(a} ξ_{b)} = 0, equivalently
£_ξ g_{ab} = 0.

NOTE: The Killing equation is not automatically enforced as a rewrite rule.
Users must manually impose Killing symmetry via `make_rule` or simplify.
=#

"""
    define_killing!(reg, name; manifold, metric) -> TensorProperties

Define a Killing vector field. Registers the vector with an `:is_killing` flag
and records the associated metric. The Killing equation ∇_{(a} ξ_{b)} = 0 is
not automatically registered as a rule; use `make_rule` to impose it.
"""
function define_killing!(reg::TensorRegistry, name::Symbol;
                         manifold::Symbol, metric::Symbol)
    # Register the vector field
    if !has_tensor(reg, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=manifold, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_killing => true, :metric => metric)))
    end

    get_tensor(reg, name)
end
