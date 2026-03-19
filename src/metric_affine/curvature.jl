# Metric-affine Riemann tensor for general affine connection.
#
# R^a_{bcd}(Γ) = ∂_c Γ^a_{bd} - ∂_d Γ^a_{bc} + Γ^a_{ce}Γ^e_{bd} - Γ^a_{de}Γ^e_{bc}
#
# Decomposition into Riemannian + distortion parts:
#   R^a_{bcd}(Γ) = R^a_{bcd}(LC) + ∇_c N^a_{bd} - ∇_d N^a_{bc}
#                  + N^a_{ce}N^e_{bd} - N^a_{de}N^e_{bc}
#
# Key differences from Riemannian case:
#   - R_{abcd} ≠ R_{cdab} in general (pair symmetry only with metric compatibility)
#   - R_{[abc]d} ≠ 0 in general (first Bianchi with torsion)
#   - R_{abcd} = -R_{abdc} still holds (antisymmetry in last two indices)
#
# Ground truth: Hehl et al, Phys. Rep. 258 (1995), Sec 3.

"""
    MAFieldStrength

Container for the metric-affine curvature tensors.

# Fields
- `riemann_name::Symbol`      -- R^a_{bcd}(Γ) full curvature
- `ricci_name::Symbol`        -- R_{bd} = R^a_{bad}
- `scalar_name::Symbol`       -- R = g^{bd} R_{bd}
- `connection::Symbol`        -- parent connection
"""
struct MAFieldStrength
    riemann_name::Symbol
    ricci_name::Symbol
    scalar_name::Symbol
    connection::Symbol
end

function Base.show(io::IO, fs::MAFieldStrength)
    print(io, "MAFieldStrength(Riem=:$(fs.riemann_name), Ric=:$(fs.ricci_name))")
end

"""
    define_ma_curvature!(reg::TensorRegistry, ac::AffineConnection;
                          manifold::Symbol=:M4) -> MAFieldStrength

Register the curvature tensors for a general affine connection.

Creates:
- Full Riemann R^a_{bcd}(Γ) — antisymmetric in c,d only (NOT pair-symmetric)
- Ricci tensor R_{bd} = R^a_{bad} — NOT necessarily symmetric (asymmetric Ricci)
- Ricci scalar R = g^{bd} R_{bd}

Note: for a general affine connection, the Ricci tensor has an antisymmetric
part: R_{[ab]} ∝ (curvature of the trace of the connection), which
relates to the electromagnetic-like U(1) piece (Weyl vector).

Ground truth: Hehl et al (1995) Sec 3.
"""
function define_ma_curvature!(reg::TensorRegistry, ac::AffineConnection;
                               manifold::Symbol=ac.manifold)
    riem_name = Symbol(:Riem_, ac.name)
    ric_name = Symbol(:Ric_, ac.name)
    scalar_name = Symbol(:RicScalar_, ac.name)

    # Full Riemann: antisymmetric in last two indices ONLY
    if !has_tensor(reg, riem_name)
        register_tensor!(reg, TensorProperties(
            name=riem_name, manifold=manifold, rank=(1, 3),
            symmetries=SymmetrySpec[AntiSymmetric(3, 4)],
            options=Dict{Symbol,Any}(
                :is_curvature => true,
                :is_metric_affine => true,
                :connection => ac.name,
                :no_pair_symmetry => true)))
    end

    # Ricci: NO symmetry (asymmetric in general)
    if !has_tensor(reg, ric_name)
        register_tensor!(reg, TensorProperties(
            name=ric_name, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_ricci => true,
                :is_metric_affine => true,
                :connection => ac.name,
                :asymmetric => true)))
    end

    # Ricci scalar
    if !has_tensor(reg, scalar_name)
        register_tensor!(reg, TensorProperties(
            name=scalar_name, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_ricci_scalar => true,
                :is_metric_affine => true,
                :connection => ac.name)))
    end

    MAFieldStrength(riem_name, ric_name, scalar_name, ac.name)
end

"""
    ma_riemann_decomposition(ac::AffineConnection;
                              registry::TensorRegistry=current_registry()) -> TensorExpr

Return the decomposition of the metric-affine Riemann tensor into
Riemannian part + distortion contributions:

    R^a_{bcd}(Γ) = R^a_{bcd}(LC) + ∇_c N^a_{bd} - ∇_d N^a_{bc}
                   + N^a_{ce} N^e_{bd} - N^a_{de} N^e_{bc}

Returns a TSum with the Riemannian term + 4 distortion terms.
"""
function ma_riemann_decomposition(ac::AffineConnection;
                                   registry::TensorRegistry=current_registry())
    N = ac.distortion_name

    # R^a_{bcd}(LC) — standard Riemannian curvature
    R_LC = Tensor(:Riem, [up(:a), down(:b), down(:c), down(:d)])

    # ∇_c N^a_{bd}
    covd_N1 = TDeriv(down(:c), Tensor(N, [up(:a), down(:b), down(:d)]), :D)

    # -∇_d N^a_{bc}
    covd_N2 = TDeriv(down(:d), Tensor(N, [up(:a), down(:b), down(:c)]), :D)

    # N^a_{ce} N^e_{bd}
    NN1 = tproduct(1 // 1, TensorExpr[
        Tensor(N, [up(:a), down(:c), down(:e)]),
        Tensor(N, [up(:e), down(:b), down(:d)])
    ])

    # -N^a_{de} N^e_{bc}
    NN2 = tproduct(1 // 1, TensorExpr[
        Tensor(N, [up(:a), down(:d), down(:e)]),
        Tensor(N, [up(:e), down(:b), down(:c)])
    ])

    tsum(TensorExpr[
        R_LC,
        covd_N1,
        tproduct(-1 // 1, TensorExpr[covd_N2]),
        NN1,
        tproduct(-1 // 1, TensorExpr[NN2])
    ])
end
