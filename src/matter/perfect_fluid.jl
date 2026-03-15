#= Perfect fluid stress-energy tensor: high-level API.
#
# Builds on PerfectFluidProperties (src/gr/matter.jl) and PerfectFluid
# (src/matter/eos.jl) to provide:
#   - stress_energy(fluid)              -> T_{ab}  (covariant form)
#   - trace_stress_energy(fluid; reg)   -> T = -rho + (d-1)*p  (scalar expression)
#   - conservation_equation(fluid, covd)-> nabla_a T^{ab} = 0  (TDeriv expression)
#
# Ground truth: Wald (1984) Sec 4.2; Hawking & Ellis (1973) Sec 3.3.
=#

"""
    stress_energy(fluid::PerfectFluid; indices=(:a, :b), registry=current_registry()) -> TensorExpr

Return the perfect fluid stress-energy tensor T_{ab} (both indices covariant):

    T_{ab} = (rho + p) u_a u_b + p g_{ab}

This is the covariant form of `perfect_fluid_expr` (which returns T^{ab}).

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    fp = define_perfect_fluid!(reg, :T; manifold=:M4, metric=:g)
    fluid = PerfectFluid(BarotropicEOS(0), fp)
    T_ab = stress_energy(fluid)
end
```
"""
function stress_energy(fluid::PerfectFluid;
                       indices::Tuple{Symbol,Symbol}=(:a, :b),
                       registry::TensorRegistry=current_registry())
    fp = fluid.properties
    a_sym, b_sym = indices

    a = down(a_sym)
    b = down(b_sym)

    rho = fp.energy_density
    p   = fp.pressure
    u_name = fp.velocity
    g   = fp.metric

    # (rho + p) u_a u_b
    rho_plus_p = TSum(TensorExpr[TScalar(rho), TScalar(p)])
    u_a = Tensor(u_name, [a])
    u_b = Tensor(u_name, [b])
    kinetic_term = tproduct(1 // 1, TensorExpr[rho_plus_p, u_a, u_b])

    # p g_{ab}
    g_ab = Tensor(g, [a, b])
    pressure_term = tproduct(1 // 1, TensorExpr[TScalar(p), g_ab])

    # T_{ab} = (rho + p) u_a u_b + p g_{ab}
    TSum(TensorExpr[kinetic_term, pressure_term])
end

"""
    trace_stress_energy(fluid::PerfectFluid; registry=current_registry()) -> TensorExpr

Return the trace T = g^{ab} T_{ab} of the perfect fluid stress-energy tensor.

In `d` spacetime dimensions with signature (-,+,...,+):

    T = g^{ab} T_{ab} = -rho + (d-1) * p

In 4D this is T = -rho + 3p.

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    fp = define_perfect_fluid!(reg, :T; manifold=:M4, metric=:g)
    fluid = PerfectFluid(BarotropicEOS(1//3), fp)
    T = trace_stress_energy(fluid)  # => -rho + 3p
end
```
"""
function trace_stress_energy(fluid::PerfectFluid;
                             registry::TensorRegistry=current_registry())
    fp = fluid.properties
    d = get_manifold(registry, fp.manifold).dim

    rho = fp.energy_density
    p   = fp.pressure

    # T = g^{ab} T_{ab} = (rho + p) * g^{ab} u_a u_b + p * g^{ab} g_{ab}
    #   = (rho + p) * (-1) + p * d          [using u^a u_a = -1 and g^{ab} g_{ab} = d]
    #   = -rho - p + d*p
    #   = -rho + (d - 1)*p
    TSum(TensorExpr[
        tproduct(-1 // 1, TensorExpr[TScalar(rho)]),
        tproduct((d - 1) // 1, TensorExpr[TScalar(p)])
    ])
end

"""
    conservation_equation(fluid::PerfectFluid, covd::Symbol;
                          free_index::Symbol=:b,
                          contract_index::Symbol=:a,
                          registry=current_registry()) -> TensorExpr

Return the expression nabla_a T^{ab} representing the divergence of the
stress-energy tensor. The conservation law is nabla_a T^{ab} = 0.

The returned expression is a TDeriv wrapping the stress-energy tensor,
suitable for further manipulation (e.g., covd_to_christoffel, simplify).

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_covd!(reg, :D; manifold=:M4, metric=:g)
    fp = define_perfect_fluid!(reg, :T; manifold=:M4, metric=:g)
    fluid = PerfectFluid(BarotropicEOS(0), fp)
    cons = conservation_equation(fluid, :D)  # => D_a T^{ab}
end
```
"""
function conservation_equation(fluid::PerfectFluid, covd::Symbol;
                                free_index::Symbol=:b,
                                contract_index::Symbol=:a,
                                registry::TensorRegistry=current_registry())
    fp = fluid.properties

    # Build T^{ab} using the existing perfect_fluid_expr
    T_ab = perfect_fluid_expr(up(contract_index), up(free_index), fp)

    # nabla_a T^{ab}  (derivative index down, contracts with the first up index)
    TDeriv(down(contract_index), T_ab, covd)
end
