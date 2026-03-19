# Non-metricity tensor irreducible decomposition.
#
# Q_{abc} = -nabla_a g_{bc} (symmetric in b,c).
#
# In d dimensions, Q_{abc} decomposes into 4 irreducible parts:
#   Q_a = g^{bc} Q_{abc}              (Weyl vector, trace on b,c)
#   tilde_Q_a = Q^b_{ab}              (second trace, on a,b)
#   Q_{abc} = (1/d) g_{bc} Q_a        (Weyl part)
#           + (2/(d+2)) g_{a(b} W_{c)} - (2/d(d+2)) g_{bc} W_a  (mixed)
#           + Omega_{abc}              (totally traceless)
#
# where W_a = tilde_Q_a - Q_a/d and Omega is the traceless remainder.
#
# Ground truth: Hehl, McCrea, Mielke & Ne'eman, Phys. Rep. 258 (1995), Sec 2.4.

"""
    weyl_vector_expr(Q_name::Symbol; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the Weyl vector Q_a = g^{bc} Q_{abc} (trace of non-metricity on last two indices).
"""
function weyl_vector_expr(Q_name::Symbol;
                          registry::TensorRegistry=current_registry())
    g_up = Tensor(:g, [up(:b), up(:c)])
    Q = Tensor(Q_name, [down(:a), down(:b), down(:c)])
    tproduct(1 // 1, TensorExpr[g_up, Q])
end

"""
    second_trace_expr(Q_name::Symbol; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the second trace tilde_Q_c = g^{ab} Q_{abc} = Q^b_{bc}
(trace of non-metricity on first and second indices).
"""
function second_trace_expr(Q_name::Symbol;
                           registry::TensorRegistry=current_registry())
    g_up = Tensor(:g, [up(:a), up(:b)])
    Q = Tensor(Q_name, [down(:a), down(:b), down(:c)])
    tproduct(1 // 1, TensorExpr[g_up, Q])
end

"""
    NonmetricityDecomposition

Irreducible decomposition of the non-metricity tensor Q_{abc}.

# Fields
- `Q_name::Symbol`          -- non-metricity tensor name
- `weyl_name::Symbol`       -- Weyl vector Q_a = g^{bc}Q_{abc}
- `second_trace_name::Symbol` -- second trace tilde_Q_a = Q^b_{ab}
- `traceless_name::Symbol`  -- traceless tensor part Omega_{abc}
"""
struct NonmetricityDecomposition
    Q_name::Symbol
    weyl_name::Symbol
    second_trace_name::Symbol
    traceless_name::Symbol
end

function Base.show(io::IO, nd::NonmetricityDecomposition)
    print(io, "NonmetricityDecomposition(:$(nd.Q_name), weyl=:$(nd.weyl_name))")
end

"""
    decompose_nonmetricity!(reg::TensorRegistry, Q_name::Symbol;
                             manifold::Symbol=:M4, dim::Int=4)
        -> NonmetricityDecomposition

Register the irreducible parts of the non-metricity tensor Q_{abc}:
- Weyl vector Q_a (rank (0,1))
- Second trace tilde_Q_a (rank (0,1))
- Traceless part Omega_{abc} (rank (0,3), symmetric in b,c, all traces zero)

Ground truth: Hehl et al (1995) Sec 2.4.
"""
function decompose_nonmetricity!(reg::TensorRegistry, Q_name::Symbol;
                                  manifold::Symbol=:M4, dim::Int=4)
    has_tensor(reg, Q_name) || error("Non-metricity tensor $Q_name not registered")

    weyl_name = Symbol(:Qvec_, Q_name)
    second_name = Symbol(:Qtilde_, Q_name)
    traceless_name = Symbol(:Omega_, Q_name)

    # Weyl vector Q_a = g^{bc} Q_{abc}
    if !has_tensor(reg, weyl_name)
        register_tensor!(reg, TensorProperties(
            name=weyl_name, manifold=manifold, rank=(0, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_weyl_vector => true,
                :nonmetricity => Q_name,
                :definition => "g^{bc} Q_{abc}")))
    end

    # Second trace tilde_Q_a = g^{bc} Q_{bac} = Q^b_{ab}
    if !has_tensor(reg, second_name)
        register_tensor!(reg, TensorProperties(
            name=second_name, manifold=manifold, rank=(0, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_second_trace => true,
                :nonmetricity => Q_name,
                :definition => "Q^b_{ab}")))
    end

    # Traceless part Omega_{abc} (symmetric in b,c, all traces vanish)
    if !has_tensor(reg, traceless_name)
        register_tensor!(reg, TensorProperties(
            name=traceless_name, manifold=manifold, rank=(0, 3),
            symmetries=SymmetrySpec[Symmetric(2, 3)],
            options=Dict{Symbol,Any}(
                :is_nonmetricity_traceless => true,
                :nonmetricity => Q_name,
                :dim => dim,
                :traces_vanish => true)))
    end

    NonmetricityDecomposition(Q_name, weyl_name, second_name, traceless_name)
end
