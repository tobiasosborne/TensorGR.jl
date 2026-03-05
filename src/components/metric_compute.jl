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

"""
    metric_einstein(Ric::Matrix, R, g::Matrix, dim::Int) -> Matrix

Compute Einstein tensor G_{ab} = R_{ab} - (1/2) g_{ab} R.
"""
function metric_einstein(Ric::Matrix, R, g::Matrix, dim::Int)
    G = Matrix{Any}(undef, dim, dim)
    for a in 1:dim, b in 1:dim
        G[a, b] = Ric[a, b] - g[a, b] * R / 2
    end
    G
end

"""
    metric_weyl(Riem::Array, Ric::Matrix, R, g::Matrix, ginv::Matrix, dim::Int) -> Array

Compute Weyl tensor C_{abcd} from Riemann, Ricci, scalar curvature and metric.
C_{abcd} = R_{abcd} - 2/(d-2)(g_{a[c}R_{d]b} - g_{b[c}R_{d]a})
           + 2/((d-1)(d-2)) R g_{a[c}g_{d]b}

Uses the all-lowered Riemann: R_{abcd} = g_{ae} R^e_{bcd}.
"""
function metric_weyl(Riem::Array, Ric::Matrix, R, g::Matrix, ginv::Matrix, dim::Int)
    dim <= 2 && error("Weyl tensor requires dim > 2")

    # Lower the first index of Riemann: R_{abcd} = g_{ae} R^e_{bcd}
    Riem_down = Array{Any}(undef, dim, dim, dim, dim)
    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        s = 0
        for e in 1:dim
            s += g[a, e] * Riem[e, b, c, d]
        end
        Riem_down[a, b, c, d] = s
    end

    Weyl = Array{Any}(undef, dim, dim, dim, dim)
    c1 = 1 / (dim - 2)
    c2 = 1 / ((dim - 1) * (dim - 2))

    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        # Antisymmetrized Ricci terms
        ricci_part = c1 * (g[a, c] * Ric[d, b] - g[a, d] * Ric[c, b] -
                           g[b, c] * Ric[d, a] + g[b, d] * Ric[c, a])
        # Antisymmetrized metric-scalar terms
        scalar_part = c2 * R * (g[a, c] * g[d, b] - g[a, d] * g[c, b])

        Weyl[a, b, c, d] = Riem_down[a, b, c, d] - ricci_part + scalar_part
    end

    Weyl
end

"""
    metric_kretschmann(Riem::Array, g::Matrix, ginv::Matrix, dim::Int) -> Any

Compute Kretschmann scalar K = R_{abcd} R^{abcd}.
"""
function metric_kretschmann(Riem::Array, g::Matrix, ginv::Matrix, dim::Int)
    # R_{abcd} = g_{ae} R^e_{bcd}
    # R^{abcd} = g^{bf} g^{cg} g^{dh} R^a_{fgh}
    K = 0
    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        R_down = 0
        for e in 1:dim
            R_down += g[a, e] * Riem[e, b, c, d]
        end
        R_up = 0
        for f in 1:dim, h in 1:dim, j in 1:dim
            R_up += ginv[a, f] * ginv[b, h] * ginv[c, j] * Riem[f, h, j, d]  # partial raise
        end
        # Actually need full raise: R^{abcd} needs 4 ginv contractions on R_{abcd}
        # Let me do this properly
    end

    # Proper computation: K = R_{abcd} R^{abcd}
    # R^{abcd} = ginv^{ae} ginv^{bf} ginv^{cg} ginv^{dh} R_{efgh}
    K = 0
    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        R_low = 0
        for e in 1:dim
            R_low += g[a, e] * Riem[e, b, c, d]
        end
        R_up = 0
        for e in 1:dim, f in 1:dim, h in 1:dim, j in 1:dim
            R_efhj = 0
            for k in 1:dim
                R_efhj += g[e, k] * Riem[k, f, h, j]
            end
            R_up += ginv[a, e] * ginv[b, f] * ginv[c, h] * ginv[d, j] * R_efhj
        end
        K += R_low * R_up
    end
    K
end
