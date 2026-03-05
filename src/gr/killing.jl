#= Killing vector fields.

A Killing vector ξ satisfies ∇_{(a} ξ_{b)} = 0, equivalently
£_ξ g_{ab} = 0. We register this as a rewrite rule.
=#

"""
    define_killing!(reg, name; manifold, metric) -> TensorProperties

Define a Killing vector field and register the Killing equation as a rule.
"""
function define_killing!(reg::TensorRegistry, name::Symbol;
                         manifold::Symbol, metric::Symbol)
    # Register the vector field
    if !has_tensor(reg, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=manifold, rank=(1, 0),
            symmetries=Any[],
            options=Dict{Symbol,Any}(:is_killing => true, :metric => metric)))
    end

    # Killing equation rule: ∇_{(a} ξ_{b)} = 0
    # This means the symmetrized covariant derivative of ξ (with index lowered) vanishes.
    # In practice, we register: Lie derivative of metric along ξ = 0
    # £_ξ g_{ab} = 0 → any expression containing £_ξ g simplifies
    register_rule!(reg, RewriteRule(
        function(expr)
            # Match: derivative of metric contracted with Killing vector
            # This is a simplified pattern; full implementation would check
            # the Killing equation structurally
            false  # Placeholder: specific patterns added as needed
        end,
        _ -> ZERO
    ))

    get_tensor(reg, name)
end
