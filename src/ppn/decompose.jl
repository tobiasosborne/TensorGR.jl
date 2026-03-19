#= PPN-to-component bridge.
#
# Connects abstract tensor expressions from the PPN formalism to the
# 3+1 component representation. The PPN metric is defined component-by-
# component (g_{00}, g_{0i}, g_{ij}) with explicit v/c order assignments,
# while the existing codebase works with abstract indexed tensors.
#
# Ground truth: Will (2018) Ch 4.1; MTW Ch 39.
=#

# ────────────────────────────────────────────────────────────────────
# PPN foliation setup
# ────────────────────────────────────────────────────────────────────

"""
    ppn_foliation!(reg::TensorRegistry; manifold::Symbol=:M4,
                   temporal::Int=0, spatial::Vector{Int}=Int[1,2,3])
        -> FoliationProperties

Set up the 3+1 foliation for PPN formalism and register auxiliary tensors.

Creates:
- A foliation `:ppn` with the specified temporal/spatial split
- Spatial Kronecker delta `delta_s` (rank-(0,2) symmetric on spatial indices)
- Spatial metric `gamma_s` (rank-(0,2) symmetric on spatial indices)
- Lapse function `N_ppn` (scalar)
- Shift vector `N_ppn_i` (rank-(0,1) spatial vector)

The PPN expansion is NOT a standard perturbation (no epsilon). Different
metric components have different leading orders in v/c:
  g_{00} = -1 + O(2) + O(4),  g_{0i} = O(3),  g_{ij} = delta_{ij} + O(2)

Ground truth: Will (2018) Sec 4.1.
"""
function ppn_foliation!(reg::TensorRegistry; manifold::Symbol=:M4,
                        temporal::Int=0, spatial::Vector{Int}=Int[1,2,3])
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    # Create foliation (or return existing)
    fol_name = :ppn
    if has_foliation(reg, fol_name)
        return get_foliation(reg, fol_name)
    end

    fol = define_foliation!(reg, fol_name; manifold=manifold,
                            temporal=temporal, spatial=spatial)

    # Register spatial Kronecker delta
    if !has_tensor(reg, :delta_s)
        register_tensor!(reg, TensorProperties(
            name=:delta_s, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_spatial_delta => true,
                                     :ppn_foliation => fol_name)))
    end

    # Register spatial metric (= delta_ij + 2*gamma*U*delta_ij in PPN)
    if !has_tensor(reg, :gamma_s)
        register_tensor!(reg, TensorProperties(
            name=:gamma_s, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_spatial_metric => true,
                                     :ppn_foliation => fol_name)))
    end

    # Register lapse function
    if !has_tensor(reg, :N_ppn)
        register_tensor!(reg, TensorProperties(
            name=:N_ppn, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_lapse => true,
                                     :ppn_foliation => fol_name)))
    end

    # Register shift vector
    if !has_tensor(reg, :N_ppn_i)
        register_tensor!(reg, TensorProperties(
            name=:N_ppn_i, manifold=manifold, rank=(0, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_shift => true,
                                     :ppn_foliation => fol_name)))
    end

    fol
end

# ────────────────────────────────────────────────────────────────────
# PPN metric decomposition
# ────────────────────────────────────────────────────────────────────

"""
    PPNMetricComponents

The three independent components of the PPN metric in the 3+1 split.

# Fields
- `g00::TensorExpr` -- time-time component (scalar expression)
- `g0i::TensorExpr` -- time-space component (carries one spatial index, or zero)
- `gij::TensorExpr` -- space-space component (carries spatial delta structure)
"""
struct PPNMetricComponents
    g00::TensorExpr
    g0i::TensorExpr
    gij::TensorExpr
end

function Base.show(io::IO, mc::PPNMetricComponents)
    print(io, "PPNMetricComponents(g00, g0i, gij)")
end

"""
    ppn_decompose(metric::Dict{Tuple{Symbol,Symbol}, TensorExpr})
        -> PPNMetricComponents

Extract the three metric components from a `ppn_metric_ansatz` output.

Returns a `PPNMetricComponents` with:
- `g00` -- the time-time component (scalar expression)
- `g0i` -- the time-space component (vector expression or zero)
- `gij` -- the space-space component (rank-2 spatial expression)

# Example
```julia
metric = ppn_metric_ansatz(ppn_gr(), reg; order=2)
mc = ppn_decompose(metric)
mc.g00  # => -1 + 2U - 2U²
mc.g0i  # => -(7/2)V - (1/2)W  (for GR)
mc.gij  # => (1 + 2U)δᵢⱼ
```
"""
function ppn_decompose(metric::Dict{Tuple{Symbol,Symbol}, TensorExpr})
    haskey(metric, (:time, :time)) ||
        error("Metric dict missing (:time, :time) key")
    haskey(metric, (:time, :space)) ||
        error("Metric dict missing (:time, :space) key")
    haskey(metric, (:space, :space)) ||
        error("Metric dict missing (:space, :space) key")

    PPNMetricComponents(
        metric[(:time, :time)],
        metric[(:time, :space)],
        metric[(:space, :space)]
    )
end

"""
    ppn_decompose(params::PPNParameters, reg::TensorRegistry; order::Int=2)
        -> PPNMetricComponents

Convenience: build the PPN metric ansatz and decompose in one call.
"""
function ppn_decompose(params::PPNParameters, reg::TensorRegistry; order::Int=2)
    metric = ppn_metric_ansatz(params, reg; order=order)
    ppn_decompose(metric)
end

# ────────────────────────────────────────────────────────────────────
# PPN metric composition (reconstruction)
# ────────────────────────────────────────────────────────────────────

"""
    ppn_compose(mc::PPNMetricComponents)
        -> Dict{Tuple{Symbol,Symbol}, TensorExpr}

Reconstruct a metric dict from PPNMetricComponents.
This is the inverse of `ppn_decompose`.
"""
function ppn_compose(mc::PPNMetricComponents)
    Dict{Tuple{Symbol,Symbol}, TensorExpr}(
        (:time, :time) => mc.g00,
        (:time, :space) => mc.g0i,
        (:space, :space) => mc.gij
    )
end

"""
    ppn_compose(g00::TensorExpr, g0i::TensorExpr, gij::TensorExpr)
        -> Dict{Tuple{Symbol,Symbol}, TensorExpr}

Reconstruct a metric dict from individual components.
"""
function ppn_compose(g00::TensorExpr, g0i::TensorExpr, gij::TensorExpr)
    ppn_compose(PPNMetricComponents(g00, g0i, gij))
end

# ────────────────────────────────────────────────────────────────────
# PPN Christoffel components (order-by-order)
# ────────────────────────────────────────────────────────────────────

"""
    PPNChristoffelComponents

Christoffel symbol components in the PPN 3+1 decomposition,
organized by index pattern (temporal=0, spatial=i,j,k).

Each field is a TensorExpr that may carry free spatial indices.

Ground truth: Will (2018) Eqs 4.13--4.19.
"""
struct PPNChristoffelComponents
    G000::TensorExpr    # Gamma^0_{00} — scalar
    G00j::TensorExpr    # Gamma^0_{0j} — spatial vector
    G0ij::TensorExpr    # Gamma^0_{ij} — spatial rank-2
    Gi00::TensorExpr    # Gamma^i_{00} — spatial vector
    Gi0j::TensorExpr    # Gamma^i_{0j} — spatial rank-2
    Gijk::TensorExpr    # Gamma^i_{jk} — spatial rank-3
end

function Base.show(io::IO, ::PPNChristoffelComponents)
    print(io, "PPNChristoffelComponents(Γ⁰₀₀, Γ⁰₀ⱼ, Γ⁰ᵢⱼ, Γⁱ₀₀, Γⁱ₀ⱼ, Γⁱⱼₖ)")
end

"""
    ppn_christoffel_1pn(mc::PPNMetricComponents;
                        registry::TensorRegistry=current_registry())
        -> PPNChristoffelComponents

Compute Christoffel symbols at 1PN order from PPN metric components.

At 1PN with g_{00} = -(1 - 2U), g_{0i} = 0, g_{ij} = (1 + 2γU)δ_{ij}:

    Γ⁰₀₀ = ∂₀U          (Newtonian time-time)
    Γ⁰₀ⱼ = ∂ⱼU          (Newtonian time-space)
    Γ⁰ᵢⱼ = γ(∂₀U)δᵢⱼ    (spatial through time)
    Γⁱ₀₀ = ∂ⁱU           (Newtonian gravity)
    Γⁱ₀ⱼ = γ(∂₀U)δⁱⱼ    (frame dragging seed)
    Γⁱⱼₖ = γ(δⁱⱼ∂ₖU + δⁱₖ∂ⱼU - δⱼₖ∂ⁱU)  (spatial Christoffel)

Ground truth: Will (2018) Eqs 4.13--4.15 at lowest nontrivial order.

Note: These are the Christoffel symbols of the FULL PPN metric, not just
the spatial part. The expressions are in terms of abstract PPN potential
tensors (U, V_ppn, W_ppn, etc.), not numeric components.
"""
function ppn_christoffel_1pn(mc::PPNMetricComponents;
                             registry::TensorRegistry=current_registry())
    U = Tensor(:U, TIndex[])

    # Partial derivatives of U (abstract)
    used = Set{Symbol}()
    i_name = fresh_index(used); push!(used, i_name)
    j_name = fresh_index(used); push!(used, j_name)
    k_name = fresh_index(used)

    # Temporal derivative ∂₀U (represented as TScalar placeholder)
    dU_t = TScalar(:dU_dt)

    # Spatial derivative ∂ᵢU
    dU_i = TDeriv(down(i_name), U)
    dU_j = TDeriv(down(j_name), U)
    dU_k = TDeriv(down(k_name), U)

    # Spatial upstairs derivative ∂ⁱU (at 1PN, g^{ij} ≈ δ^{ij})
    dU_up_i = TDeriv(up(i_name), U)

    # Spatial delta
    delta_ij = Tensor(:delta_s, [down(i_name), down(j_name)])
    delta_up_i_j = Tensor(:delta_s, [up(i_name), down(j_name)])
    delta_up_i_k = Tensor(:delta_s, [up(i_name), down(k_name)])
    delta_jk = Tensor(:delta_s, [down(j_name), down(k_name)])

    # Extract gamma parameter from the gij component
    # At 1PN: gij = (1 + 2γU)δᵢⱼ, so γ is the coefficient of 2U in gij
    # For now, use a scalar placeholder; the caller knows γ from PPNParameters
    gamma_ppn = TScalar(:gamma_ppn)

    # Γ⁰₀₀ = ∂₀U
    G000 = dU_t

    # Γ⁰₀ⱼ = ∂ⱼU
    G00j = dU_j

    # Γ⁰ᵢⱼ = γ(∂₀U)δᵢⱼ
    G0ij = tproduct(1 // 1, TensorExpr[gamma_ppn, dU_t, delta_ij])

    # Γⁱ₀₀ = ∂ⁱU  (raised with flat spatial metric at this order)
    Gi00 = dU_up_i

    # Γⁱ₀ⱼ = γ(∂₀U)δⁱⱼ
    Gi0j = tproduct(1 // 1, TensorExpr[gamma_ppn, dU_t, delta_up_i_j])

    # Γⁱⱼₖ = γ(δⁱⱼ∂ₖU + δⁱₖ∂ⱼU - δⱼₖ∂ⁱU)
    term1 = tproduct(1 // 1, TensorExpr[gamma_ppn, delta_up_i_j, dU_k])
    term2 = tproduct(1 // 1, TensorExpr[gamma_ppn, delta_up_i_k, dU_j])
    term3 = tproduct(-1 // 1, TensorExpr[gamma_ppn, delta_jk, dU_up_i])
    Gijk = tsum(TensorExpr[term1, term2, term3])

    PPNChristoffelComponents(G000, G00j, G0ij, Gi00, Gi0j, Gijk)
end

"""
    ppn_christoffel(params::PPNParameters, reg::TensorRegistry;
                    order::Int=1) -> PPNChristoffelComponents

Compute PPN Christoffel symbols from parameters at specified PN order.

Currently supports order=1 (1PN). Higher orders require the full
2PN metric and more involved computation.
"""
function ppn_christoffel(params::PPNParameters, reg::TensorRegistry;
                         order::Int=1)
    order == 1 || error("Only order=1 (1PN) Christoffel symbols are currently supported")
    mc = ppn_decompose(params, reg; order=order)
    ppn_christoffel_1pn(mc; registry=reg)
end
