# Distortion tensor decomposition: N = K + L.
#
# The distortion tensor N^a_{bc} = Γ^a_{bc} - {a;bc} decomposes as:
#   N^a_{bc} = K^a_{bc} + L^a_{bc}
#
# Contortion (from torsion):
#   K^a_{bc} = (1/2)(T^a_{bc} + T_b^a_c + T_c^a_b)
#   K_{abc} = -K_{bac}  (antisymmetric in first two indices when lowered)
#
# Disformation (from non-metricity):
#   L^a_{bc} = (1/2)(Q^a_{bc} - Q_b^a_c - Q_c^a_b)
#   L_{abc} = L_{acb}  (symmetric in last two indices when lowered)
#
# Key identity: N = K + L (complete, unique decomposition)
#
# Ground truth: Hehl et al, Phys. Rep. 258 (1995), Eq 2.8;
#               Schouten, Ricci-Calculus (1954).

"""
    DistortionDecomposition

Decomposition of the distortion tensor N = K + L.

# Fields
- `contortion_name::Symbol`   -- K^a_{bc} (from torsion)
- `disformation_name::Symbol` -- L^a_{bc} (from non-metricity)
- `connection::Symbol`        -- parent affine connection
"""
struct DistortionDecomposition
    contortion_name::Symbol
    disformation_name::Symbol
    connection::Symbol
end

function Base.show(io::IO, dd::DistortionDecomposition)
    print(io, "DistortionDecomp(K=:$(dd.contortion_name), L=:$(dd.disformation_name))")
end

"""
    decompose_distortion!(reg::TensorRegistry, ac::AffineConnection;
                           manifold::Symbol=:M4) -> DistortionDecomposition

Register the contortion and disformation tensors.

Contortion K^a_{bc}: derives from torsion T^a_{bc}, antisymmetric in
first two lowered indices. 24 independent components in d=4.

Disformation L^a_{bc}: derives from non-metricity Q_{abc}, symmetric in
last two lowered indices. 40 independent components in d=4.

Together: N^a_{bc} = K^a_{bc} + L^a_{bc} (distortion = contortion + disformation).

Ground truth: Hehl et al (1995) Eq 2.8.
"""
function decompose_distortion!(reg::TensorRegistry, ac::AffineConnection;
                                manifold::Symbol=ac.manifold)
    K_name = Symbol(:K_, ac.name)
    L_name = Symbol(:L_, ac.name)

    # Contortion K^a_{bc} — antisymmetric in first two lowered indices
    # K_{abc} = -K_{bac}, which means K^a_{bc} has no simple slot symmetry
    # in the mixed-position form
    if !has_tensor(reg, K_name)
        register_tensor!(reg, TensorProperties(
            name=K_name, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_contortion => true,
                :connection => ac.name,
                :torsion => ac.torsion_name,
                :definition => "K^a_{bc} = (1/2)(T^a_{bc} + T_b^a_c + T_c^a_b)")))
    end

    # Disformation L^a_{bc} — symmetric in last two lowered indices
    # L_{abc} = L_{acb}, which in mixed form means L^a_{bc} = L^a_{cb}
    if !has_tensor(reg, L_name)
        register_tensor!(reg, TensorProperties(
            name=L_name, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[Symmetric(2, 3)],
            options=Dict{Symbol,Any}(
                :is_disformation => true,
                :connection => ac.name,
                :nonmetricity => ac.nonmetricity_name,
                :definition => "L^a_{bc} = (1/2)(Q^a_{bc} - Q_b^a_c - Q_c^a_b)")))
    end

    DistortionDecomposition(K_name, L_name, ac.name)
end

"""
    contortion_from_torsion(T_name::Symbol; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the contortion tensor as an expression in terms of torsion:

    K^a_{bc} = (1/2)(T^a_{bc} + T_b^a_c + T_c^a_b)

The result has free indices a (Up), b (Down), c (Down).
"""
function contortion_from_torsion(T_name::Symbol;
                                 registry::TensorRegistry=current_registry())
    # T^a_{bc}
    T1 = Tensor(T_name, [up(:a), down(:b), down(:c)])
    # T_b^a_c = g_{bd} g^{ae} T^d_{ec} — raise/lower to get mixed form
    # For abstract algebra, just express with index positions:
    # T_{ba}^{.}_c means T with indices b(Down), a(Up), c(Down)
    # But T is defined with (Up, Down, Down), so T_b^a_c needs metric contractions.
    # For the abstract expression, use the metric explicitly:
    T2 = tproduct(1 // 1, TensorExpr[
        Tensor(:g, [down(:b), down(:d)]),
        Tensor(:g, [up(:a), up(:e)]),
        Tensor(T_name, [up(:d), down(:e), down(:c)])
    ])
    T3 = tproduct(1 // 1, TensorExpr[
        Tensor(:g, [down(:c), down(:d)]),
        Tensor(:g, [up(:a), up(:e)]),
        Tensor(T_name, [up(:d), down(:e), down(:b)])
    ])

    tsum(TensorExpr[
        tproduct(1 // 2, TensorExpr[T1]),
        tproduct(1 // 2, TensorExpr[T2]),
        tproduct(1 // 2, TensorExpr[T3])
    ])
end

"""
    disformation_from_nonmetricity(Q_name::Symbol;
                                    registry::TensorRegistry=current_registry()) -> TensorExpr

Return the disformation tensor as an expression in terms of non-metricity:

    L^a_{bc} = (1/2)(Q^a_{bc} - Q_b^a_c - Q_c^a_b)

where Q^a_{bc} = g^{ad} Q_{dbc}, etc.
"""
function disformation_from_nonmetricity(Q_name::Symbol;
                                         registry::TensorRegistry=current_registry())
    # Q^a_{bc} = g^{ad} Q_{dbc}
    Q1 = tproduct(1 // 1, TensorExpr[
        Tensor(:g, [up(:a), up(:d)]),
        Tensor(Q_name, [down(:d), down(:b), down(:c)])
    ])
    # Q_b^a_c = g^{ae} Q_{bec}
    Q2 = tproduct(1 // 1, TensorExpr[
        Tensor(:g, [up(:a), up(:e)]),
        Tensor(Q_name, [down(:b), down(:e), down(:c)])
    ])
    # Q_c^a_b = g^{ae} Q_{ceb}
    Q3 = tproduct(1 // 1, TensorExpr[
        Tensor(:g, [up(:a), up(:e)]),
        Tensor(Q_name, [down(:c), down(:e), down(:b)])
    ])

    tsum(TensorExpr[
        tproduct(1 // 2, TensorExpr[Q1]),
        tproduct(-1 // 2, TensorExpr[Q2]),
        tproduct(-1 // 2, TensorExpr[Q3])
    ])
end
