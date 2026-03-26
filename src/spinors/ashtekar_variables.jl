#= Ashtekar-Barbero connection and densitized triad.
#
# The Ashtekar-Barbero connection is the fundamental variable of loop
# quantum gravity:
#   A^i_a = Gamma^i_a + beta * K^i_a
#
# where:
# - Gamma^i_a is the spin connection compatible with the triad
# - K^i_a = K_{ab} e^{bi} is the extrinsic curvature in triad form
# - beta is the Barbero-Immirzi parameter
#
# The conjugate variable is the densitized triad:
#   E^a_i = sqrt(det(q)) * e^a_i
#
# Canonical brackets:
#   {A^i_a(x), E^b_j(y)} = 8*pi*G*beta * delta^i_j * delta^b_a * delta(x,y)
#
# Reference: Ashtekar (1991) Ch 10; Barbero, PRD 51, 5507 (1995).
=#

"""
    define_ashtekar_variables!(reg::TensorRegistry;
                                manifold::Symbol=:Sigma,
                                beta::Symbol=:beta_BI,
                                spatial_metric::Symbol=:gamma)

Register the Ashtekar-Barbero connection, densitized triad, and associated
tensors on a spatial manifold with SU(2) internal structure.

Registers:
- Barbero-Immirzi parameter `beta_BI` (rank-0 scalar)
- Ashtekar connection `A_ash^i_a` (Up SU2 + Down Tangent, no index symmetry)
- Densitized triad `E_ash^a_i` (Up Tangent + Down SU2, no index symmetry)
- Curvature `F_ash_{ab}^i` (Down Tangent, Down Tangent, Up SU2;
  antisymmetric in spatial slots 1,2)
- Spin connection `Gamma_spin^i_a` (Up SU2 + Down Tangent)
- Extrinsic curvature in triad form `K_triad^i_a` (Up SU2 + Down Tangent)

Requires that `manifold` is already registered with a spatial metric and
that the SU(2) VBundle has been defined (via `define_space_spinors!`).

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold Sigma dim=3 metric=gamma
    define_space_spinors!(reg; manifold=:Sigma, metric=:gamma)
    define_ashtekar_variables!(reg; manifold=:Sigma, spatial_metric=:gamma)
end
```

# Reference
Ashtekar (1991) Ch 10; Barbero, PRD 51, 5507 (1995).
"""
function define_ashtekar_variables!(reg::TensorRegistry;
                                     manifold::Symbol=:Sigma,
                                     beta::Symbol=:beta_BI,
                                     spatial_metric::Symbol=:gamma)
    @lock reg.lock begin
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    has_vbundle(reg, :SU2) || error("SU(2) VBundle not registered; call define_space_spinors! first")

    # 1. Register Barbero-Immirzi parameter (scalar, rank 0)
    if !has_tensor(reg, beta)
        register_tensor!(reg, TensorProperties(
            name=beta, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_scalar => true,
                                     :is_barbero_immirzi => true)
        ))
    end

    # Save metric/delta cache (register_tensor! may clobber them)
    saved_metric = get(reg.metric_cache, manifold, nothing)
    saved_delta  = get(reg.delta_cache, manifold, nothing)

    # 2. Register spin connection Gamma_spin^i_a
    #    Index structure: (Up SU2, Down Tangent) -- no index symmetries
    if !has_tensor(reg, :Gamma_spin)
        register_tensor!(reg, TensorProperties(
            name=:Gamma_spin, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_connection => true,
                :index_vbundles => [:SU2, :Tangent])
        ))
    end

    # 3. Register extrinsic curvature in triad form K_triad^i_a
    #    Index structure: (Up SU2, Down Tangent) -- no index symmetries
    if !has_tensor(reg, :K_triad)
        register_tensor!(reg, TensorProperties(
            name=:K_triad, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_extrinsic_triad => true,
                :index_vbundles => [:SU2, :Tangent])
        ))
    end

    # 4. Register Ashtekar connection A_ash^i_a
    #    Index structure: (Up SU2, Down Tangent) -- no index symmetries
    if !has_tensor(reg, :A_ash)
        register_tensor!(reg, TensorProperties(
            name=:A_ash, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_ashtekar_connection => true,
                :barbero_immirzi => beta,
                :index_vbundles => [:SU2, :Tangent])
        ))
    end

    # 5. Register densitized triad E_ash^a_i
    #    Index structure: (Up Tangent, Down SU2) -- no index symmetries
    if !has_tensor(reg, :E_ash)
        register_tensor!(reg, TensorProperties(
            name=:E_ash, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_densitized_triad => true,
                :density_weight => 1,
                :index_vbundles => [:Tangent, :SU2])
        ))
    end

    # 6. Register curvature F_ash_{ab}^i
    #    Index structure: (Down Tangent, Down Tangent, Up SU2)
    #    Antisymmetric in spatial indices (slots 1,2)
    if !has_tensor(reg, :F_ash)
        register_tensor!(reg, TensorProperties(
            name=:F_ash, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[AntiSymmetric(1, 2)],
            options=Dict{Symbol,Any}(
                :is_curvature => true,
                :is_ashtekar_curvature => true,
                :index_vbundles => [:Tangent, :Tangent, :SU2])
        ))
    end

    # Restore metric/delta cache
    if saved_metric !== nothing
        reg.metric_cache[manifold] = saved_metric
    end
    if saved_delta !== nothing
        reg.delta_cache[manifold] = saved_delta
    end

    nothing
    end
end

# -- Expression builders -------------------------------------------------------

"""
    ashtekar_connection_expr(; registry=current_registry()) -> TSum

Return the Ashtekar-Barbero connection expression:

    A^i_a = Gamma_spin^i_a + beta_BI * K_triad^i_a

with fresh SU(2) and spatial indices.
"""
function ashtekar_connection_expr(; registry::TensorRegistry=current_registry())
    has_tensor(registry, :Gamma_spin) || error("Gamma_spin not registered; call define_ashtekar_variables! first")
    has_tensor(registry, :K_triad) || error("K_triad not registered; call define_ashtekar_variables! first")

    # Look up Barbero-Immirzi parameter name from A_ash options
    beta_name = :beta_BI
    if has_tensor(registry, :A_ash)
        props = get_tensor(registry, :A_ash)
        beta_name = get(props.options, :barbero_immirzi, :beta_BI)
    end

    used = Set{Symbol}()
    i = fresh_index(used; vbundle=:SU2)
    push!(used, i)
    a = fresh_index(used; vbundle=:Tangent)

    gamma_term = Tensor(:Gamma_spin, [TIndex(i, Up, :SU2), TIndex(a, Down, :Tangent)])
    k_term = Tensor(:K_triad, [TIndex(i, Up, :SU2), TIndex(a, Down, :Tangent)])
    beta_scalar = Tensor(beta_name, TIndex[])

    # A^i_a = Gamma^i_a + beta * K^i_a
    tsum(TensorExpr[gamma_term,
                     tproduct(1 // 1, TensorExpr[beta_scalar, k_term])])
end

"""
    densitized_triad_expr(; registry=current_registry()) -> Tensor

Return the densitized triad `E_ash^a_i` with fresh indices:
one Up Tangent and one Down SU(2).
"""
function densitized_triad_expr(; registry::TensorRegistry=current_registry())
    has_tensor(registry, :E_ash) || error("E_ash not registered; call define_ashtekar_variables! first")
    used = Set{Symbol}()
    a = fresh_index(used; vbundle=:Tangent)
    push!(used, a)
    i = fresh_index(used; vbundle=:SU2)
    Tensor(:E_ash, [TIndex(a, Up, :Tangent), TIndex(i, Down, :SU2)])
end

"""
    ashtekar_curvature_expr(; registry=current_registry()) -> Tensor

Return the curvature of the Ashtekar connection `F_ash_{ab}^i` with fresh
indices: two Down Tangent (antisymmetric) and one Up SU(2).

The curvature is defined by:
    F^i_{ab} = partial_a A^i_b - partial_b A^i_a + epsilon^i_{jk} A^j_a A^k_b

# Reference
Ashtekar (1991) Ch 10, Eq 10.1.8.
"""
function ashtekar_curvature_expr(; registry::TensorRegistry=current_registry())
    has_tensor(registry, :F_ash) || error("F_ash not registered; call define_ashtekar_variables! first")
    used = Set{Symbol}()
    a = fresh_index(used; vbundle=:Tangent)
    push!(used, a)
    b = fresh_index(used; vbundle=:Tangent)
    push!(used, b)
    i = fresh_index(used; vbundle=:SU2)
    Tensor(:F_ash, [TIndex(a, Down, :Tangent),
                     TIndex(b, Down, :Tangent),
                     TIndex(i, Up, :SU2)])
end

"""
    gauss_constraint_expr(; registry=current_registry()) -> TDeriv

Return the Gauss constraint expression:

    D_a E^a_i = partial_a E^a_i (as abstract derivative)

The full Gauss constraint is G_i = D_a E^a_i = 0, where D is the
gauge-covariant derivative. This returns the partial-derivative term
as a `TDeriv` acting on the densitized triad.

In LQG, the Gauss constraint generates SU(2) gauge transformations.
Its vanishing is equivalent to the torsion-free condition on the triad.

# Reference
Ashtekar (1991) Ch 10; Thiemann (2007) Ch 1.
"""
function gauss_constraint_expr(; registry::TensorRegistry=current_registry())
    has_tensor(registry, :E_ash) || error("E_ash not registered; call define_ashtekar_variables! first")

    used = Set{Symbol}()
    a = fresh_index(used; vbundle=:Tangent)
    push!(used, a)
    i = fresh_index(used; vbundle=:SU2)

    E = Tensor(:E_ash, [TIndex(a, Up, :Tangent), TIndex(i, Down, :SU2)])
    TDeriv(TIndex(a, Down, :Tangent), E)
end
