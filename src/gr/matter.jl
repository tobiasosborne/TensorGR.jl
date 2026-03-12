#= Perfect fluid stress-energy tensor.

Implements the perfect fluid stress-energy tensor:
    T^{ab} = (rho + p) u^a u^b + p g^{ab}

where rho is the energy density, p is the pressure, and u^a is the
4-velocity field with normalization g_{ab} u^a u^b = -1.

Port of xAct's PerfectFluid functionality.
=#

"""
    PerfectFluidProperties

Stores the definition of a perfect fluid stress-energy tensor.

Fields:
- `name`: stress-energy tensor name (e.g. :T)
- `manifold`: manifold name
- `metric`: metric name
- `energy_density`: scalar field name for energy density (e.g. :rho)
- `pressure`: scalar field name for pressure (e.g. :p)
- `velocity`: 4-velocity vector name (e.g. :u)
"""
struct PerfectFluidProperties
    name::Symbol
    manifold::Symbol
    metric::Symbol
    energy_density::Symbol
    pressure::Symbol
    velocity::Symbol
end

"""
    define_perfect_fluid!(reg, name; manifold, metric, rho=:rho, p=:p, u=:u)

Define a perfect fluid and register its associated tensors:
- Stress-energy tensor `T^{ab}` (symmetric, rank (2,0))
- Energy density `rho` (scalar, rank (0,0))
- Pressure `p` (scalar, rank (0,0))
- 4-velocity `u^a` (vector, rank (1,0))

Also registers the normalization rule `u_a u^a = -1`.

Returns a `PerfectFluidProperties` storing the fluid definition.

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    fp = define_perfect_fluid!(reg, :T; manifold=:M4, metric=:g)
    expr = perfect_fluid_expr(up(:a), up(:b), fp)
end
```
"""
function define_perfect_fluid!(reg::TensorRegistry, name::Symbol;
                                manifold::Symbol,
                                metric::Symbol,
                                rho::Symbol=:rho,
                                p::Symbol=:p,
                                u::Symbol=:u)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    has_tensor(reg, metric) || error("Metric $metric not registered")

    # Register the stress-energy tensor T^{ab} (symmetric, rank (2,0))
    if !has_tensor(reg, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=manifold, rank=(2, 0),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(
                :is_stress_energy => true,
                :fluid_type => :perfect,
                :metric => metric,
                :energy_density => rho,
                :pressure => p,
                :velocity => u)))
    end

    # Register energy density as a scalar (rank (0,0))
    if !has_tensor(reg, rho)
        register_tensor!(reg, TensorProperties(
            name=rho, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_scalar_field => true,
                                     :is_energy_density => true)))
    end

    # Register pressure as a scalar (rank (0,0))
    if !has_tensor(reg, p)
        register_tensor!(reg, TensorProperties(
            name=p, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_scalar_field => true,
                                     :is_pressure => true)))
    end

    # Register 4-velocity u^a (vector, rank (1,0))
    if !has_tensor(reg, u)
        register_tensor!(reg, TensorProperties(
            name=u, manifold=manifold, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_velocity => true,
                                     :metric => metric)))
    end

    # Normalization rule: u_a u^a = -1
    # After metric contraction, we look for a product containing u_a u^a
    # (one index down, one up, same name => contracted pair).
    # This follows the pattern from hypersurface.jl's n_a n^a = signature rule.
    register_rule!(reg, RewriteRule(
        function(expr)
            if !(expr isa TProduct)
                return false
            end
            u_count = count(f -> f isa Tensor && f.name == u, expr.factors)
            u_count >= 2 || return false
            u_factors = [f for f in expr.factors if f isa Tensor && f.name == u]
            length(u_factors) >= 2 || return false
            for i in 1:length(u_factors), j in i+1:length(u_factors)
                fi, fj = u_factors[i], u_factors[j]
                if length(fi.indices) == 1 && length(fj.indices) == 1
                    if fi.indices[1].name == fj.indices[1].name &&
                       fi.indices[1].position != fj.indices[1].position
                        return true
                    end
                end
            end
            false
        end,
        function(expr)
            # Replace u_a u^a with -1
            factors = copy(expr.factors)
            u_indices = Int[]
            for (i, f) in enumerate(factors)
                if f isa Tensor && f.name == u
                    push!(u_indices, i)
                end
            end
            # Find first contracting pair and remove
            for i in 1:length(u_indices), j in i+1:length(u_indices)
                fi = factors[u_indices[i]]
                fj = factors[u_indices[j]]
                if length(fi.indices) == 1 && length(fj.indices) == 1 &&
                   fi.indices[1].name == fj.indices[1].name &&
                   fi.indices[1].position != fj.indices[1].position
                    remaining = [factors[k] for k in eachindex(factors)
                                 if k != u_indices[i] && k != u_indices[j]]
                    return tproduct(expr.scalar * (-1 // 1), isempty(remaining) ?
                                    TensorExpr[TScalar(1 // 1)] : remaining)
                end
            end
            expr
        end
    ))

    # Store properties in foliations dict (general-purpose storage, same as hypersurface)
    fp = PerfectFluidProperties(name, manifold, metric, rho, p, u)
    reg.foliations[Symbol(:perfect_fluid_, name)] = fp

    fp
end

"""
    perfect_fluid_expr(a, b, fluid) -> TensorExpr

Construct the perfect fluid stress-energy tensor expression:
    T^{ab} = (rho + p) u^a u^b + p g^{ab}

Both indices `a` and `b` must be `Up` (contravariant).

# Arguments
- `a::TIndex`: first contravariant index
- `b::TIndex`: second contravariant index
- `fluid::PerfectFluidProperties`: the fluid definition

# Example
```julia
expr = perfect_fluid_expr(up(:a), up(:b), fp)
# Returns: (rho + p) * u^a * u^b + p * g^{ab}
```
"""
function perfect_fluid_expr(a::TIndex, b::TIndex, fluid::PerfectFluidProperties)
    @assert a.position == Up "First index must be Up (contravariant), got Down"
    @assert b.position == Up "Second index must be Up (contravariant), got Down"

    rho = fluid.energy_density
    p = fluid.pressure
    u_name = fluid.velocity
    g = fluid.metric

    # (rho + p) u^a u^b
    rho_plus_p = TSum(TensorExpr[TScalar(rho), TScalar(p)])
    u_a = Tensor(u_name, [a])
    u_b = Tensor(u_name, [b])
    kinetic_term = tproduct(1 // 1, TensorExpr[rho_plus_p, u_a, u_b])

    # p g^{ab}
    g_ab = Tensor(g, [a, b])
    pressure_term = tproduct(1 // 1, TensorExpr[TScalar(p), g_ab])

    # T^{ab} = (rho + p) u^a u^b + p g^{ab}
    TSum(TensorExpr[kinetic_term, pressure_term])
end

"""
    get_perfect_fluid(reg, name) -> PerfectFluidProperties

Retrieve a previously defined perfect fluid from the registry.

# Arguments
- `reg::TensorRegistry`: the registry
- `name::Symbol`: the stress-energy tensor name used in `define_perfect_fluid!`

# Throws
- `KeyError` if no perfect fluid with the given name is registered.
"""
function get_perfect_fluid(reg::TensorRegistry, name::Symbol)
    key = Symbol(:perfect_fluid_, name)
    haskey(reg.foliations, key) || error("No perfect fluid :$name registered")
    reg.foliations[key]::PerfectFluidProperties
end
