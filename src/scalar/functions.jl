#= Scalar function handling for derivatives.

define_scalar_function!(reg, f) — register functions like sin, exp
so that derivatives follow the chain rule:
  ∂_a f(Φ) = f'(Φ) ∂_a Φ
=#

"""
    define_scalar_function!(reg, name; derivative=nothing)

Register a scalar function for chain rule in derivatives.
The derivative, if provided, is another scalar function name: f' = derivative.
"""
function define_scalar_function!(reg::TensorRegistry, name::Symbol;
                                  derivative::Union{Symbol, Nothing}=nothing)
    if !haskey(reg.tensors, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=:_scalar, rank=(0, 0),
            symmetries=Any[],
            options=Dict{Symbol,Any}(:is_scalar_function => true,
                                     :derivative => derivative)))
    else
        get_tensor(reg, name).options[:is_scalar_function] = true
        get_tensor(reg, name).options[:derivative] = derivative
    end
    nothing
end

"""
    scalar_function_derivative(reg, name) -> Union{Symbol, Nothing}

Get the derivative function name, or nothing if not set.
"""
function scalar_function_derivative(reg::TensorRegistry, name::Symbol)
    has_tensor(reg, name) || return nothing
    props = get_tensor(reg, name)
    get(props.options, :derivative, nothing)
end
