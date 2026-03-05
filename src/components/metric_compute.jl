#= MetricCompute: compute Christoffel symbols and curvature from metric components.

Given a metric g_{ij} as a CTensor, compute:
  Γ^a_{bc} = (1/2) g^{ad} (∂_b g_{cd} + ∂_c g_{bd} - ∂_d g_{bc})
  R^a_{bcd} = ∂_c Γ^a_{bd} - ∂_d Γ^a_{bc} + Γ^a_{ce} Γ^e_{bd} - Γ^a_{de} Γ^e_{bc}
  R_{bd} = R^a_{bad}
  R = g^{bd} R_{bd}
=#

"""
    metric_christoffel(g::CTensor, ginv::CTensor, coords::Vector{Symbol};
                       deriv=nothing) -> Array{Any, 3}

Compute Christoffel symbols Γ^a_{bc} from metric components.
If `deriv` is nothing, uses symbolic differentiation via the coordinate symbols.
Otherwise, `deriv` should be a function `deriv(expr, coord) -> expr`.
"""
function metric_christoffel(g_data::Matrix, ginv_data::Matrix,
                            coords::Vector{Symbol};
                            deriv_fn=nothing)
    dim = length(coords)
    @assert size(g_data) == (dim, dim)
    @assert size(ginv_data) == (dim, dim)

    # Compute partial derivatives of metric
    # dg[i,j,k] = ∂_k g_{ij}
    if deriv_fn !== nothing
        dg = Array{Any}(undef, dim, dim, dim)
        for i in 1:dim, j in 1:dim, k in 1:dim
            dg[i, j, k] = deriv_fn(g_data[i, j], coords[k])
        end
    else
        error("Automatic differentiation requires a deriv_fn function")
    end

    # Christoffel: Γ^a_{bc} = (1/2) g^{ad} (∂_b g_{cd} + ∂_c g_{bd} - ∂_d g_{bc})
    Gamma = Array{Any}(undef, dim, dim, dim)
    for a in 1:dim, b in 1:dim, c in 1:dim
        s = 0
        for d in 1:dim
            s += ginv_data[a, d] * (dg[c, d, b] + dg[b, d, c] - dg[b, c, d])
        end
        Gamma[a, b, c] = s / 2
    end

    Gamma
end

"""
    metric_riemann(Gamma::Array, coords::Vector{Symbol};
                   deriv_fn=nothing) -> Array{Any, 4}

Compute Riemann tensor R^a_{bcd} from Christoffel symbols.
R^a_{bcd} = ∂_c Γ^a_{db} - ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} - Γ^a_{de} Γ^e_{cb}

Note: uses the convention R^a_{bcd} where the first lower index is the
"derivative" direction (xAct convention).
"""
function metric_riemann(Gamma::Array, dim::Int;
                        coords::Vector{Symbol}=Symbol[],
                        deriv_fn=nothing)
    Riem = Array{Any}(undef, dim, dim, dim, dim)

    if deriv_fn !== nothing && !isempty(coords)
        dGamma = Array{Any}(undef, dim, dim, dim, dim)
        for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
            dGamma[a, b, c, d] = deriv_fn(Gamma[a, b, c], coords[d])
        end
    else
        # Use finite differences or require derivs pre-computed
        error("deriv_fn required for Riemann computation")
    end

    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        val = dGamma[a, d, b, c] - dGamma[a, c, b, d]
        for e in 1:dim
            val += Gamma[a, c, e] * Gamma[e, d, b] - Gamma[a, d, e] * Gamma[e, c, b]
        end
        Riem[a, b, c, d] = val
    end

    Riem
end

"""
    metric_ricci(Riem::Array, dim::Int) -> Matrix

Compute Ricci tensor R_{bd} = R^a_{bad} by contraction.
"""
function metric_ricci(Riem::Array, dim::Int)
    Ric = Matrix{Any}(undef, dim, dim)
    for b in 1:dim, d in 1:dim
        s = 0
        for a in 1:dim
            s += Riem[a, b, a, d]
        end
        Ric[b, d] = s
    end
    Ric
end

"""
    metric_ricci_scalar(Ric::Matrix, ginv::Matrix, dim::Int) -> Any

Compute Ricci scalar R = g^{bd} R_{bd}.
"""
function metric_ricci_scalar(Ric::Matrix, ginv::Matrix, dim::Int)
    s = 0
    for b in 1:dim, d in 1:dim
        s += ginv[b, d] * Ric[b, d]
    end
    s
end
