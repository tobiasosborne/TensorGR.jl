#= Bimetric gravity: dual metric registration.
#
# Hassan-Rosen bimetric theory has two dynamical metrics g_{ab} and f_{ab}
# on the same manifold, with independent curvature tensors.
#
# ds²_g = g_{ab} dx^a dx^b    (physical metric)
# ds²_f = f_{ab} dx^a dx^b    (second metric)
#
# Each metric has its own:
# - Christoffel symbols: Γ^a_{bc}[g], Γ^a_{bc}[f]
# - Riemann tensor: R^a_{bcd}[g], R^a_{bcd}[f]
# - Ricci tensor/scalar: R_{ab}[g], R[g], R_{ab}[f], R[f]
# - Einstein tensor: G_{ab}[g], G_{ab}[f]
#
# The interaction potential involves the matrix square root S = sqrt(g^{-1}f).
#
# Ground truth: Hassan & Rosen, JHEP 02 (2012) 126, arXiv:1109.3515.
=#

"""
    BimetricSetup

Container for a bimetric gravity setup with two metrics on the same manifold.

# Fields
- `manifold::Symbol`  -- the shared manifold
- `metric_g::Symbol`  -- first metric name (physical)
- `metric_f::Symbol`  -- second metric name
- `curvature_g::Dict{Symbol,Symbol}` -- curvature tensor names for g
- `curvature_f::Dict{Symbol,Symbol}` -- curvature tensor names for f
"""
struct BimetricSetup
    manifold::Symbol
    metric_g::Symbol
    metric_f::Symbol
    curvature_g::Dict{Symbol,Symbol}
    curvature_f::Dict{Symbol,Symbol}
end

function Base.show(io::IO, bs::BimetricSetup)
    print(io, "BimetricSetup(:$(bs.manifold), g=:$(bs.metric_g), f=:$(bs.metric_f))")
end

"""
    define_bimetric!(reg::TensorRegistry, metric_g::Symbol, metric_f::Symbol;
                     manifold::Symbol=:M4) -> BimetricSetup

Register two independent metrics on the same manifold, each with its own
complete set of curvature tensors.

For metric `g`, registers: Riem_g, Ric_g, RicScalar_g, Ein_g, Weyl_g, Chris_g
For metric `f`, registers: Riem_f, Ric_f, RicScalar_f, Ein_f, Weyl_f, Chris_f

The naming convention uses metric name as suffix: `:Riem_g`, `:Ric_f`, etc.

Does NOT register the interaction potential (see define_hr_potential! for that).

Ground truth: Hassan & Rosen (2012) Sec 2.
"""
function define_bimetric!(reg::TensorRegistry, metric_g::Symbol, metric_f::Symbol;
                           manifold::Symbol=:M4)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    mp = get_manifold(reg, manifold)

    curvature_g = _register_metric_curvature(reg, metric_g, manifold, mp)
    curvature_f = _register_metric_curvature(reg, metric_f, manifold, mp)

    # Register the matrix square root tensor S^a_b = (g^{-1}f)^{1/2}
    S_name = Symbol(:S_, metric_g, :_, metric_f)
    if !has_tensor(reg, S_name)
        register_tensor!(reg, TensorProperties(
            name=S_name, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_sqrt_matrix => true,
                :bimetric => true,
                :metric_g => metric_g,
                :metric_f => metric_f)))
    end

    BimetricSetup(manifold, metric_g, metric_f, curvature_g, curvature_f)
end

"""Register a metric and its full curvature tensor set."""
function _register_metric_curvature(reg::TensorRegistry, metric::Symbol,
                                     manifold::Symbol, mp::ManifoldProperties)
    dim = mp.dim
    names = Dict{Symbol,Symbol}()

    # Metric (symmetric rank-2)
    if !has_tensor(reg, metric)
        register_tensor!(reg, TensorProperties(
            name=metric, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
    end

    # Christoffel symbol
    chris = Symbol(:Chris_, metric)
    if !has_tensor(reg, chris)
        register_tensor!(reg, TensorProperties(
            name=chris, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[Symmetric(2, 3)],
            options=Dict{Symbol,Any}(:is_christoffel => true, :metric => metric)))
    end
    names[:christoffel] = chris

    # Riemann tensor
    riem = Symbol(:Riem_, metric)
    if !has_tensor(reg, riem)
        register_tensor!(reg, TensorProperties(
            name=riem, manifold=manifold, rank=(0, 4),
            symmetries=SymmetrySpec[
                AntiSymmetric(1, 2), AntiSymmetric(3, 4), Symmetric(1, 3)],
            options=Dict{Symbol,Any}(:is_riemann => true, :metric => metric)))
    end
    names[:riemann] = riem

    # Ricci tensor
    ric = Symbol(:Ric_, metric)
    if !has_tensor(reg, ric)
        register_tensor!(reg, TensorProperties(
            name=ric, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_ricci => true, :metric => metric)))
    end
    names[:ricci] = ric

    # Ricci scalar
    rscalar = Symbol(:RicScalar_, metric)
    if !has_tensor(reg, rscalar)
        register_tensor!(reg, TensorProperties(
            name=rscalar, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_ricci_scalar => true, :metric => metric)))
    end
    names[:ricci_scalar] = rscalar

    # Einstein tensor
    ein = Symbol(:Ein_, metric)
    if !has_tensor(reg, ein)
        register_tensor!(reg, TensorProperties(
            name=ein, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_einstein => true, :metric => metric)))
    end
    names[:einstein] = ein

    # Weyl tensor
    weyl = Symbol(:Weyl_, metric)
    if !has_tensor(reg, weyl)
        register_tensor!(reg, TensorProperties(
            name=weyl, manifold=manifold, rank=(0, 4),
            symmetries=SymmetrySpec[
                AntiSymmetric(1, 2), AntiSymmetric(3, 4), Symmetric(1, 3)],
            options=Dict{Symbol,Any}(:is_weyl => true, :metric => metric)))
    end
    names[:weyl] = weyl

    names
end

"""
    bimetric_field_equations(bs::BimetricSetup;
                             registry::TensorRegistry=current_registry())
        -> Tuple{TensorExpr, TensorExpr}

Return the vacuum bimetric field equations (without interaction terms):

    G_{ab}[g] = 0    (Einstein for g)
    G_{ab}[f] = 0    (Einstein for f)

The interaction terms from the Hassan-Rosen potential are added separately.
Returns (G_g_ab, G_f_ab) as a tuple.
"""
function bimetric_field_equations(bs::BimetricSetup;
                                  registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    G_g = Tensor(bs.curvature_g[:einstein], [down(a), down(b)])
    G_f = Tensor(bs.curvature_f[:einstein], [down(a), down(b)])

    (G_g, G_f)
end
