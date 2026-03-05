#= SVT (Scalar-Vector-Tensor) decomposition of metric perturbations.

On a flat background, the metric perturbation h_{μν} decomposes as:

  h_{00} = 2ϕ
  h_{0i} = ∂_i B + S_i           (S_i transverse: ∂^i S_i = 0)
  h_{ij} = 2ψ δ_{ij} + 2∂_i∂_j E + ∂_i F_j + ∂_j F_i + h^TT_{ij}

The scalar sector contains: ϕ, B, ψ, E
The vector sector contains: S_i, F_i  (both transverse)
The tensor sector contains: h^TT_{ij} (transverse-traceless)

We represent the SVT fields as named tensors and provide substitution rules.
=#

"""
    SVTFields

Container for the SVT decomposition field names.
"""
struct SVTFields
    ϕ::Symbol       # scalar: h_{00} = 2ϕ
    B::Symbol       # scalar: longitudinal part of h_{0i}
    ψ::Symbol       # scalar: trace of spatial part
    E::Symbol       # scalar: longitudinal-longitudinal of h_{ij}
    S::Symbol       # vector: transverse part of h_{0i}
    F::Symbol       # vector: transverse part of h_{ij}
    hTT::Symbol     # tensor: transverse-traceless part
end

SVTFields(; ϕ=:ϕ, B=:B, ψ=:ψ, E=:E, S=:S, F=:F, hTT=:hTT) =
    SVTFields(ϕ, B, ψ, E, S, F, hTT)

const DEFAULT_SVT = SVTFields()

"""
    svt_substitute(expr, h_name; fields=DEFAULT_SVT) -> TensorExpr

Substitute the SVT decomposition into a metric perturbation h_{ab}.
This replaces h_{ab} with the appropriate SVT components based on
the index structure.

Note: This is a symbolic substitution that tags the fields for later
sector projection. Full decomposition requires projector application.
"""
function svt_substitute(expr::TensorExpr, h_name::Symbol;
                        fields::SVTFields=DEFAULT_SVT)
    walk(expr) do node
        if node isa Tensor && node.name == h_name
            return _svt_replace(node, fields)
        end
        node
    end
end

function _svt_replace(t::Tensor, fields::SVTFields)
    # For now, return a labeled version that downstream code can project
    # The actual decomposition depends on whether indices are spatial/temporal
    # which requires the foliation information.
    # Simple approach: tag with SVT marker for later processing
    Tensor(Symbol(:svt_, t.name), t.indices)
end
