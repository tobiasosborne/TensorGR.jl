#= Sen connection on spatial SU(2) spinors.
#
# The Sen connection D_i is the unique torsion-free connection on
# SU(2) spatial spinors on a spacelike hypersurface that satisfies:
#   D_i epsilon_{PQ} = 0   (preserves spatial spin metric)
#   D_i tau^j_{PQ}  = 0   (compatible with soldering form)
#
# The curvature of D_i encodes both intrinsic 3D geometry and
# extrinsic curvature of the embedding:
#   [D_i, D_j] psi_P = F_{ij P}^Q psi_Q
#
# where F_{ij P}^Q involves both the 3-Riemann and K_{ij}.
#
# Reference: Sen, J. Math. Phys. 22, 1781 (1981), Eq 3.5;
#            Ashtekar (1991) Ch 2.
=#

"""
    define_sen_connection!(reg::TensorRegistry;
                            manifold::Symbol=:Sigma,
                            spatial_metric::Symbol=:gamma,
                            name::Symbol=:D_sen)

Register the Sen connection on the spatial hypersurface.

The Sen connection is the unique torsion-free connection on
spatial SU(2) spinors that preserves both the spatial spin metric
epsilon_{PQ} and the soldering form tau^i_{PQ}:
  D_i epsilon_{PQ} = 0
  D_i tau^j_{PQ}  = 0

Its curvature encodes both the 3D intrinsic curvature and
the extrinsic curvature of the embedding.

Registers:
- Connection coefficient tensor `Gamma_sen^P_{Qi}` (no index symmetries)
- Curvature tensor `F_sen_{ij P}^Q` (antisymmetric in i,j)
- Metricity rules: D_i eps_{PQ} = 0, D_i tau^j_{PQ} = 0

Requires that `define_space_spinors!` has been called first to register
the SU(2) VBundle, eps_space, and tau.

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold Sigma dim=3 metric=gamma
    define_space_spinors!(reg; manifold=:Sigma, metric=:gamma)
    define_sen_connection!(reg; manifold=:Sigma, spatial_metric=:gamma)
end
```

# Reference
Sen, J. Math. Phys. 22, 1781 (1981), Eq 3.5.
"""
function define_sen_connection!(reg::TensorRegistry;
                                 manifold::Symbol=:Sigma,
                                 spatial_metric::Symbol=:gamma,
                                 name::Symbol=:D_sen)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    has_vbundle(reg, :SU2) || error("SU(2) VBundle not registered; call define_space_spinors! first")
    has_tensor(reg, :eps_space) || error("eps_space not registered; call define_space_spinors! first")
    has_tensor(reg, :tau) || error("tau not registered; call define_space_spinors! first")

    # 1. Register connection coefficient tensor Gamma_sen^P_{Qi}
    #    Index structure: (Up SU2, Down SU2, Down Tangent)
    #    No index symmetries (mixed spatial + spinor indices).
    if !has_tensor(reg, :Gamma_sen)
        register_tensor!(reg, TensorProperties(
            name=:Gamma_sen, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_connection => true,
                :connection_name => name,
                :index_vbundles => [:SU2, :SU2, :Tangent])
        ))
    end

    # 2. Register curvature tensor F_sen_{ij P}^Q
    #    Index structure: (Down Tangent, Down Tangent, Down SU2, Up SU2)
    #    Antisymmetric in spatial indices (slots 1,2).
    if !has_tensor(reg, :F_sen)
        register_tensor!(reg, TensorProperties(
            name=:F_sen, manifold=manifold, rank=(1, 3),
            symmetries=SymmetrySpec[AntiSymmetric(1, 2)],
            options=Dict{Symbol,Any}(
                :is_curvature => true,
                :connection_name => name,
                :index_vbundles => [:Tangent, :Tangent, :SU2, :SU2])
        ))
    end

    # 3. Metricity rule: D_i eps_{PQ} = 0
    #    Any TDeriv (tagged D_sen or :partial) acting on eps_space vanishes.
    local _name = name
    register_rule!(reg, RewriteRule(
        function(expr)
            expr isa TDeriv || return false
            (expr.covd == _name || expr.covd == :partial) || return false
            inner = expr.arg
            inner isa Tensor || return false
            inner.name == :eps_space
        end,
        _ -> ZERO
    ))

    # 4. Compatibility rule: D_i tau^j_{PQ} = 0
    register_rule!(reg, RewriteRule(
        function(expr)
            expr isa TDeriv || return false
            (expr.covd == _name || expr.covd == :partial) || return false
            inner = expr.arg
            inner isa Tensor || return false
            inner.name == :tau
        end,
        _ -> ZERO
    ))

    nothing
end

# ── Expression builders ────────────────────────────────────────────────

"""
    sen_connection_expr(; registry=current_registry()) -> Tensor

Return the Sen connection coefficient `Gamma_sen^P_{Qi}` with fresh indices:
one Up SU(2), one Down SU(2), one Down Tangent.
"""
function sen_connection_expr(; registry::TensorRegistry=current_registry())
    has_tensor(registry, :Gamma_sen) || error("Gamma_sen not registered; call define_sen_connection! first")
    used = Set{Symbol}()
    p = fresh_index(used; vbundle=:SU2)
    push!(used, p)
    q = fresh_index(used; vbundle=:SU2)
    push!(used, q)
    i = fresh_index(used; vbundle=:Tangent)
    Tensor(:Gamma_sen, [TIndex(p, Up, :SU2),
                         TIndex(q, Down, :SU2),
                         TIndex(i, Down, :Tangent)])
end

"""
    sen_covd(expr::TensorExpr, idx_name::Symbol;
             name::Symbol=:D_sen,
             registry::TensorRegistry=current_registry()) -> TensorExpr

Apply the Sen covariant derivative D_i to a spatial spinor expression.

Produces the structure: D_i(expr) represented as a TDeriv with a
Tangent-bundle index and CovD tag `:D_sen`.

The Sen derivative acts on SU(2) spinor-valued fields. For a spinor
psi_P, D_i psi_P is the spatial covariant derivative that preserves
both eps_{PQ} and tau^j_{PQ}.

# Arguments
- `expr`: the spinor expression to differentiate
- `idx_name`: name for the spatial derivative index (e.g., :i)
- `name`: the Sen connection CovD tag (default `:D_sen`)

# Returns
A `TDeriv` with the spatial index and CovD tag.

# Reference
Sen, J. Math. Phys. 22, 1781 (1981), Eq 3.5.
"""
function sen_covd(expr::TensorExpr, idx_name::Symbol;
                  name::Symbol=:D_sen,
                  registry::TensorRegistry=current_registry())
    TDeriv(TIndex(idx_name, Down, :Tangent), expr, name)
end

"""
    sen_covd_expr(expr::TensorExpr;
                  name::Symbol=:D_sen,
                  registry::TensorRegistry=current_registry()) -> TensorExpr

Apply the Sen covariant derivative with an automatically generated fresh
spatial index. Returns `D_i(expr)` as a `TDeriv`.
"""
function sen_covd_expr(expr::TensorExpr;
                       name::Symbol=:D_sen,
                       registry::TensorRegistry=current_registry())
    all_idxs = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idxs)
    i = fresh_index(used; vbundle=:Tangent)
    sen_covd(expr, i; name=name, registry=registry)
end

"""
    sen_curvature_expr(; registry=current_registry()) -> Tensor

Return the Sen curvature tensor `F_sen_{ij P}^Q` with fresh indices:
two Down Tangent (antisymmetric), one Down SU(2), one Up SU(2).

The curvature is defined by
    [D_i, D_j] psi_P = F_{ij P}^Q psi_Q
and encodes both the 3-Riemann tensor and the extrinsic curvature.

# Reference
Sen, J. Math. Phys. 22, 1781 (1981), Eq 3.5.
"""
function sen_curvature_expr(; registry::TensorRegistry=current_registry())
    has_tensor(registry, :F_sen) || error("F_sen not registered; call define_sen_connection! first")
    used = Set{Symbol}()
    i = fresh_index(used; vbundle=:Tangent)
    push!(used, i)
    j = fresh_index(used; vbundle=:Tangent)
    push!(used, j)
    p = fresh_index(used; vbundle=:SU2)
    push!(used, p)
    q = fresh_index(used; vbundle=:SU2)
    Tensor(:F_sen, [TIndex(i, Down, :Tangent),
                     TIndex(j, Down, :Tangent),
                     TIndex(p, Down, :SU2),
                     TIndex(q, Up, :SU2)])
end
