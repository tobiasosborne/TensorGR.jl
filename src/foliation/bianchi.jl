#= Bianchi I cosmological background.
#
# ds² = -dt² + a₁(t)² dx² + a₂(t)² dy² + a₃(t)² dz²
#
# Three independent scale factors, zero spatial curvature.
# Simplest anisotropic cosmology. Reduces to FLRW when a₁ = a₂ = a₃.
#
# Ground truth: Pitrou, Pereira & Uzan, JCAP 04 (2013) 004, Sec 3.1.
=#

"""
    BianchiIBackground

Bianchi I cosmological background with three independent scale factors.

# Fields
- `name::Symbol`          -- background name
- `manifold::Symbol`      -- ambient manifold
- `scale_factors::NTuple{3,Symbol}` -- scale factor names (a₁, a₂, a₃)
- `foliation::Symbol`     -- associated foliation name
"""
struct BianchiIBackground
    name::Symbol
    manifold::Symbol
    scale_factors::NTuple{3,Symbol}
    foliation::Symbol
end

function Base.show(io::IO, b::BianchiIBackground)
    a1, a2, a3 = b.scale_factors
    print(io, "BianchiI(:$(b.name), a=($(a1),$(a2),$(a3)))")
end

"""
    define_bianchi_I!(reg::TensorRegistry, name::Symbol;
                      manifold::Symbol=:M4,
                      scale_factors::NTuple{3,Symbol}=(:a1, :a2, :a3))
        -> BianchiIBackground

Register a Bianchi I cosmological background.

Creates:
- A 3+1 foliation for the background
- Three scale factor tensors (abstract scalars, functions of time)
- Spatial metric γᵢⱼ = diag(a₁², a₂², a₃²) (abstract)
- Hubble rates Hᵢ = ȧᵢ/aᵢ for each direction

The Bianchi I metric is:
    ds² = -dt² + a₁(t)² dx² + a₂(t)² dy² + a₃(t)² dz²

with zero structure constants (spatially flat, no spatial curvature).

Ground truth: Pitrou et al (2013) Sec 3.1.
"""
function define_bianchi_I!(reg::TensorRegistry, name::Symbol;
                           manifold::Symbol=:M4,
                           scale_factors::NTuple{3,Symbol}=(:a1, :a2, :a3))
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    fol_name = Symbol(name, :_fol)

    # Create foliation if not present
    if !has_foliation(reg, fol_name)
        define_foliation!(reg, fol_name; manifold=manifold,
                          temporal=0, spatial=Int[1,2,3])
    end

    # Register scale factor tensors
    for sf in scale_factors
        if !has_tensor(reg, sf)
            register_tensor!(reg, TensorProperties(
                name=sf, manifold=manifold, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:is_scale_factor => true,
                                         :bianchi_background => name)))
        end
    end

    # Register Hubble rates H_i = da_i/dt / a_i
    for (i, sf) in enumerate(scale_factors)
        H_name = Symbol(:H_, sf)
        if !has_tensor(reg, H_name)
            register_tensor!(reg, TensorProperties(
                name=H_name, manifold=manifold, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:is_hubble_rate => true,
                                         :scale_factor => sf,
                                         :direction => i,
                                         :bianchi_background => name)))
        end
    end

    BianchiIBackground(name, manifold, scale_factors, fol_name)
end

"""
    is_isotropic(b::BianchiIBackground) -> Bool

Check if the Bianchi I background is isotropic (all scale factors equal).
Always returns false at the abstract level — isotropy is a dynamical
condition a₁(t) = a₂(t) = a₃(t), not a structural one.
"""
is_isotropic(::BianchiIBackground) = false

"""
    mean_hubble(b::BianchiIBackground) -> Symbol

Return the symbol for the mean Hubble rate: H = (H₁ + H₂ + H₃)/3.
"""
function mean_hubble(b::BianchiIBackground)
    Symbol(:H_mean_, b.name)
end

"""
    shear_tensor_name(b::BianchiIBackground) -> Symbol

Return the symbol for the shear tensor σᵢⱼ of the background.
The shear measures anisotropic expansion: σᵢⱼ = Hᵢ δᵢⱼ - (H/3) δᵢⱼ.
"""
function shear_tensor_name(b::BianchiIBackground)
    Symbol(:sigma_, b.name)
end
