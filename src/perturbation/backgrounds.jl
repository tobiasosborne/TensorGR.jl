#= Background geometry rules for perturbation theory.

Provide convenience functions to register curvature rules for
common background geometries: maximally symmetric, vacuum, etc.
=#

"""
    maximally_symmetric_background!(reg, manifold; metric=:g, cosmological_constant=:Λ)

Register background curvature rules for a maximally-symmetric spacetime
(de Sitter / anti-de Sitter / Minkowski):

    R_{abcd} = (2Λ/(d-1)) (g_{ac} g_{bd} - g_{ad} g_{bc})
    R_{ab} = Λ g_{ab}
    R = d · Λ

where Λ is the cosmological constant parameter.
"""
function maximally_symmetric_background!(reg::TensorRegistry, manifold::Symbol;
                                          metric::Symbol=:g,
                                          cosmological_constant::Symbol=:Λ)
    mp = get_manifold(reg, manifold)
    d = mp.dim

    # Register Λ as a scalar tensor
    if !has_tensor(reg, cosmological_constant)
        register_tensor!(reg, TensorProperties(
            name=cosmological_constant, manifold=manifold, rank=(0, 0),
            symmetries=Any[],
            options=Dict{Symbol,Any}(:is_constant => true)))
    end

    Λ = Tensor(cosmological_constant, TIndex[])

    # R_{ab} = Λ g_{ab}
    register_rule!(reg, RewriteRule(
        function(expr)
            expr isa Tensor || return false
            expr.name == :Ric && length(expr.indices) == 2 &&
                expr.indices[1].position == Down && expr.indices[2].position == Down
        end,
        function(expr)
            Tensor(cosmological_constant, TIndex[]) * Tensor(metric, expr.indices)
        end
    ))

    # R = d · Λ
    register_rule!(reg, RewriteRule(
        expr -> expr isa Tensor && expr.name == :RicScalar,
        _ -> tproduct(d // 1, TensorExpr[Tensor(cosmological_constant, TIndex[])])
    ))

    # R_{abcd} = (2Λ/(d-1)) (g_{ac} g_{bd} - g_{ad} g_{bc})
    register_rule!(reg, RewriteRule(
        function(expr)
            expr isa Tensor || return false
            expr.name == :Riem && length(expr.indices) == 4 &&
                all(idx -> idx.position == Down, expr.indices)
        end,
        function(expr)
            a, b, c, d_idx = expr.indices
            coeff = 2 // (d - 1)
            g_ac = Tensor(metric, [a, c])
            g_bd = Tensor(metric, [b, d_idx])
            g_ad = Tensor(metric, [a, d_idx])
            g_bc = Tensor(metric, [b, c])
            tproduct(coeff, TensorExpr[Tensor(cosmological_constant, TIndex[]),
                                        g_ac * g_bd - g_ad * g_bc])
        end
    ))

    nothing
end

"""
    cosmological_background!(reg, manifold; kwargs...)

Alias for `maximally_symmetric_background!`.
"""
cosmological_background!(reg::TensorRegistry, manifold::Symbol; kwargs...) =
    maximally_symmetric_background!(reg, manifold; kwargs...)

"""
    vacuum_background!(reg, manifold; metric=:g)

Register vacuum (Ricci-flat) background rules: `R_{ab} = 0`, `R = 0`.
Riemann tensor is NOT set to zero (Schwarzschild, Kerr have Riem ≠ 0).
"""
function vacuum_background!(reg::TensorRegistry, manifold::Symbol;
                              metric::Symbol=:g)
    # Ric_{ab} = 0
    register_rule!(reg, RewriteRule(
        function(expr)
            expr isa Tensor || return false
            expr.name == :Ric && length(expr.indices) == 2
        end,
        _ -> ZERO
    ))

    # R = 0
    register_rule!(reg, RewriteRule(
        expr -> expr isa Tensor && expr.name == :RicScalar,
        _ -> ZERO
    ))

    nothing
end
