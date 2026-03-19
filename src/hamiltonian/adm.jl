#= ADM (Arnowitt-Deser-Misner) decomposition.
#
# Decomposes the spacetime metric g_{ab} into:
#   N        -- lapse function (scalar)
#   N^i      -- shift vector (spatial vector)
#   gamma_{ij} -- spatial metric on spacelike hypersurface
#
# Line element: ds² = -N² dt² + γ_{ij}(dx^i + N^i dt)(dx^j + N^j dt)
#
# Extrinsic curvature:
#   K_{ij} = (1/2N)(∂_t γ_{ij} - D_i N_j - D_j N_i)
#
# Canonical momenta:
#   π^{ij} = (√γ/2N)(K^{ij} - γ^{ij} K)
#
# ADM Hamiltonian:
#   H = ∫ (N·H + N^i·H_i) d³x
#   H = (π^{ij}π_{ij} - ½π²)/√γ - √γ R^{(3)}    (Hamiltonian constraint)
#   H_i = -2 D_j π^j_i                             (momentum constraint)
#
# Ground truth: Arnowitt, Deser, Misner (1962); Wald Ch 10.
=#

"""
    ADMDecomposition

Result of the ADM 3+1 decomposition of a spacetime metric.

# Fields
- `lapse::Symbol`           -- lapse function tensor name
- `shift::Symbol`           -- shift vector tensor name
- `spatial_metric::Symbol`  -- spatial metric tensor name
- `foliation::Symbol`       -- associated foliation name
- `manifold::Symbol`        -- spacetime manifold
"""
struct ADMDecomposition
    lapse::Symbol
    shift::Symbol
    spatial_metric::Symbol
    foliation::Symbol
    manifold::Symbol
end

function Base.show(io::IO, adm::ADMDecomposition)
    print(io, "ADM(N=:$(adm.lapse), N^i=:$(adm.shift), γ=:$(adm.spatial_metric))")
end

"""
    define_adm!(reg::TensorRegistry; manifold::Symbol=:M4,
                lapse::Symbol=:N_adm, shift::Symbol=:Ni_adm,
                spatial_metric::Symbol=:gamma_adm) -> ADMDecomposition

Register the ADM variables on a manifold with 3+1 foliation.

Creates:
- A foliation `:adm` with temporal=0, spatial=[1,2,3]
- Lapse function N (scalar)
- Shift vector N^i (contravariant spatial vector)
- Spatial metric γ_{ij} (symmetric rank-2, spatial)
- Spatial inverse metric γ^{ij}
- Extrinsic curvature K_{ij} (symmetric rank-2, spatial)
- Trace of extrinsic curvature K (scalar)
- Conjugate momentum π^{ij} (symmetric rank-2, spatial, contravariant)

Ground truth: Arnowitt, Deser, Misner (1962); Wald (1984) Ch 10.
"""
function define_adm!(reg::TensorRegistry; manifold::Symbol=:M4,
                     lapse::Symbol=:N_adm, shift::Symbol=:Ni_adm,
                     spatial_metric::Symbol=:gamma_adm)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    fol_name = :adm
    if !has_foliation(reg, fol_name)
        define_foliation!(reg, fol_name; manifold=manifold,
                          temporal=0, spatial=Int[1,2,3])
    end

    # Lapse function N (scalar)
    if !has_tensor(reg, lapse)
        register_tensor!(reg, TensorProperties(
            name=lapse, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_lapse => true, :adm => true)))
    end

    # Shift vector N^i (contravariant spatial vector)
    if !has_tensor(reg, shift)
        register_tensor!(reg, TensorProperties(
            name=shift, manifold=manifold, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_shift => true, :adm => true)))
    end

    # Spatial metric γ_{ij} (symmetric rank-2)
    if !has_tensor(reg, spatial_metric)
        register_tensor!(reg, TensorProperties(
            name=spatial_metric, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true, :is_spatial => true, :adm => true)))
    end

    # Extrinsic curvature K_{ij} (symmetric rank-2)
    K_name = Symbol(:K_, spatial_metric)
    if !has_tensor(reg, K_name)
        register_tensor!(reg, TensorProperties(
            name=K_name, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_extrinsic_curvature => true, :adm => true)))
    end

    # Trace K (scalar)
    Ktrace_name = Symbol(:K_trace_, spatial_metric)
    if !has_tensor(reg, Ktrace_name)
        register_tensor!(reg, TensorProperties(
            name=Ktrace_name, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_extrinsic_trace => true, :adm => true)))
    end

    # Conjugate momentum π^{ij} (symmetric rank-2, contravariant)
    pi_name = Symbol(:pi_, spatial_metric)
    if !has_tensor(reg, pi_name)
        register_tensor!(reg, TensorProperties(
            name=pi_name, manifold=manifold, rank=(2, 0),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_momentum => true, :adm => true)))
    end

    ADMDecomposition(lapse, shift, spatial_metric, fol_name, manifold)
end

# ────────────────────────────────────────────────────────────────────
# ADM constraint expressions
# ────────────────────────────────────────────────────────────────────

"""
    hamiltonian_constraint(adm::ADMDecomposition;
                           registry::TensorRegistry=current_registry()) -> TensorExpr

Build the Hamiltonian constraint expression:

    H = π^{ij}π_{ij} - (1/2)π² - R^{(3)}

where π = γ_{ij}π^{ij} is the trace and R^{(3)} is the spatial Ricci scalar.
(The √γ factors are absorbed into the constraint density.)

The Hamiltonian constraint H ≈ 0 on the constraint surface.

Ground truth: Arnowitt, Deser, Misner (1962); Wald (1984) Eq 10.2.29.
"""
function hamiltonian_constraint(adm::ADMDecomposition;
                                registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    i = fresh_index(used); push!(used, i)
    j = fresh_index(used); push!(used, j)
    k = fresh_index(used); push!(used, k)
    l = fresh_index(used)

    pi_name = Symbol(:pi_, adm.spatial_metric)
    K_trace = Symbol(:K_trace_, adm.spatial_metric)

    # π^{ij} π_{ij}
    pi_up = Tensor(pi_name, [up(i), up(j)])
    pi_down = Tensor(pi_name, [down(i), down(j)])
    pi_sq = pi_up * pi_down

    # π² = (γ_{ij} π^{ij})² — use trace tensor
    pi_trace = Tensor(K_trace, TIndex[])
    pi_trace_sq = pi_trace * pi_trace

    # R^{(3)} — spatial Ricci scalar (placeholder)
    R3_name = Symbol(:RicScalar_3d_, adm.spatial_metric)
    R3 = Tensor(R3_name, TIndex[])

    # H = π^{ij}π_{ij} - (1/2)π² - R^{(3)}
    pi_sq + tproduct(-1 // 2, TensorExpr[pi_trace_sq]) - R3
end

"""
    momentum_constraint(adm::ADMDecomposition;
                        registry::TensorRegistry=current_registry()) -> TensorExpr

Build the momentum constraint expression:

    H_i = -2 D_j π^j_i

where D is the spatial covariant derivative compatible with γ_{ij}.

The momentum constraint H_i ≈ 0 on the constraint surface.

Ground truth: Arnowitt, Deser, Misner (1962); Wald (1984) Eq 10.2.30.
"""
function momentum_constraint(adm::ADMDecomposition;
                              registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    i = fresh_index(used); push!(used, i)
    j = fresh_index(used)

    pi_name = Symbol(:pi_, adm.spatial_metric)

    # π^j_i (mixed indices)
    pi_mixed = Tensor(pi_name, [up(j), down(i)])

    # -2 D_j π^j_i (the derivative contracts with the up index)
    tproduct(-2 // 1, TensorExpr[TDeriv(down(j), pi_mixed)])
end
