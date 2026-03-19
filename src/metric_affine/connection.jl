#= General affine connection for metric-affine gravity.
#
# An affine connection Γ^a_{bc} with NO assumed symmetry in lower indices
# (unlike Levi-Civita which has Γ^a_{(bc)}). Decomposes as:
#
#   Γ^a_{bc} = {a; bc} + N^a_{bc}
#
# where {a; bc} is the Levi-Civita connection and N^a_{bc} is the
# distortion tensor. The distortion further splits into:
#
#   N^a_{bc} = K^a_{bc} + L^a_{bc}
#
# where K is contortion (from torsion) and L is disformation (from
# non-metricity).
#
# Ground truth: Hehl, McCrea, Mielke & Ne'eman, Phys. Rep. 258 (1995) 1.
=#

"""
    AffineConnection

Container for a general affine connection and its decomposition.

# Fields
- `name::Symbol`          -- connection name
- `manifold::Symbol`      -- manifold
- `metric::Symbol`        -- compatible metric (for LC part)
- `lc_name::Symbol`       -- Levi-Civita part
- `distortion_name::Symbol` -- distortion tensor N = Γ - LC
- `torsion_name::Symbol`  -- torsion tensor T^a_{bc}
- `nonmetricity_name::Symbol` -- non-metricity Q_{abc}
"""
struct AffineConnection
    name::Symbol
    manifold::Symbol
    metric::Symbol
    lc_name::Symbol
    distortion_name::Symbol
    torsion_name::Symbol
    nonmetricity_name::Symbol
end

function Base.show(io::IO, ac::AffineConnection)
    print(io, "AffineConnection(:$(ac.name), metric=:$(ac.metric))")
end

"""
    define_affine_connection!(reg::TensorRegistry, name::Symbol;
                               manifold::Symbol=:M4, metric::Symbol=:g)
        -> AffineConnection

Register a general affine connection with torsion and non-metricity.

Creates:
- Connection Γ^a_{bc} (rank (1,2), NO lower-index symmetry)
- Levi-Civita part {a; bc} (rank (1,2), symmetric in lower indices)
- Distortion N^a_{bc} = Γ^a_{bc} - {a; bc}
- Torsion T^a_{bc} = Γ^a_{bc} - Γ^a_{cb} (antisymmetric in bc)
- Non-metricity Q_{abc} = -∇_a g_{bc} (symmetric in bc)

The covariant derivative defined by Γ is:
    ∇_a V^b = ∂_a V^b + Γ^b_{ac} V^c

Ground truth: Hehl et al (1995) Sec 2.
"""
function define_affine_connection!(reg::TensorRegistry, name::Symbol;
                                    manifold::Symbol=:M4, metric::Symbol=:g)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    lc_name = Symbol(:LC_, name)
    distortion_name = Symbol(:N_, name)
    torsion_name = Symbol(:T_, name)
    nonmet_name = Symbol(:Q_, name)

    # General connection Γ^a_{bc} — NO symmetry in lower indices
    if !has_tensor(reg, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_connection => true,
                :is_general_affine => true,
                :metric => metric)))
    end

    # Levi-Civita part {a; bc} — symmetric in lower indices
    if !has_tensor(reg, lc_name)
        register_tensor!(reg, TensorProperties(
            name=lc_name, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[Symmetric(2, 3)],
            options=Dict{Symbol,Any}(
                :is_christoffel => true,
                :is_levi_civita => true,
                :metric => metric)))
    end

    # Distortion N^a_{bc} = Γ - LC (no symmetry)
    if !has_tensor(reg, distortion_name)
        register_tensor!(reg, TensorProperties(
            name=distortion_name, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_distortion => true,
                :connection => name)))
    end

    # Torsion T^a_{bc} (antisymmetric in lower indices)
    if !has_tensor(reg, torsion_name)
        register_tensor!(reg, TensorProperties(
            name=torsion_name, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[AntiSymmetric(2, 3)],
            options=Dict{Symbol,Any}(
                :is_torsion => true,
                :connection => name)))
    end

    # Non-metricity Q_{abc} = -∇_a g_{bc} (symmetric in last two indices)
    if !has_tensor(reg, nonmet_name)
        register_tensor!(reg, TensorProperties(
            name=nonmet_name, manifold=manifold, rank=(0, 3),
            symmetries=SymmetrySpec[Symmetric(2, 3)],
            options=Dict{Symbol,Any}(
                :is_nonmetricity => true,
                :connection => name,
                :metric => metric)))
    end

    AffineConnection(name, manifold, metric, lc_name, distortion_name,
                     torsion_name, nonmet_name)
end

"""
    is_metric_compatible(ac::AffineConnection, reg::TensorRegistry) -> Bool

Check if the non-metricity tensor is set to vanish (metric compatibility).
"""
function is_metric_compatible(ac::AffineConnection, reg::TensorRegistry)
    has_tensor(reg, ac.nonmetricity_name) || return false
    tp = get_tensor(reg, ac.nonmetricity_name)
    get(tp.options, :vanishing, false) || tp.vanishing
end

"""
    is_torsion_free(ac::AffineConnection, reg::TensorRegistry) -> Bool

Check if the torsion tensor is set to vanish (symmetric connection).
"""
function is_torsion_free(ac::AffineConnection, reg::TensorRegistry)
    has_tensor(reg, ac.torsion_name) || return false
    tp = get_tensor(reg, ac.torsion_name)
    get(tp.options, :vanishing, false) || tp.vanishing
end

"""
    set_metric_compatible!(reg::TensorRegistry, ac::AffineConnection)

Set the connection to be metric-compatible: Q_{abc} = 0.
"""
function set_metric_compatible!(reg::TensorRegistry, ac::AffineConnection)
    set_vanishing!(reg, ac.nonmetricity_name)
end

"""
    set_torsion_free!(reg::TensorRegistry, ac::AffineConnection)

Set the connection to be torsion-free: T^a_{bc} = 0.
A metric-compatible, torsion-free connection is the Levi-Civita connection.
"""
function set_torsion_free!(reg::TensorRegistry, ac::AffineConnection)
    set_vanishing!(reg, ac.torsion_name)
end
