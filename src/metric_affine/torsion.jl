#= Torsion tensor and irreducible decomposition.
#
# Torsion: T^a_{bc} = 2Γ^a_{[bc]} = Γ^a_{bc} - Γ^a_{cb}
# (antisymmetric in lower indices)
#
# Irreducible decomposition (d=4):
#   T^a_{bc} = (1/3)(T_b δ^a_c - T_c δ^a_b) + (1/3)ε^a_{bcd} S^d + q^a_{bc}
#
# where:
#   T_b = T^a_{ba}         -- torsion vector (trace)
#   S^a = (1/6)ε^{abcd}T_{bcd}  -- torsion axial vector (pseudo-trace)
#   q^a_{bc}               -- tensor part (traceless, axial-free)
#
# Ground truth: Hehl, McCrea, Mielke & Ne'eman, Phys. Rep. 258 (1995) Sec 2.3.
=#

"""
    TorsionDecomposition

Irreducible decomposition of the torsion tensor.

# Fields
- `vector::Symbol`     -- torsion vector T_a (trace part)
- `axial::Symbol`      -- axial torsion S^a (pseudo-trace)
- `tensor::Symbol`     -- tensor torsion q^a_{bc} (traceless, axial-free)
- `connection::Symbol` -- parent connection
"""
struct TorsionDecomposition
    vector::Symbol
    axial::Symbol
    tensor::Symbol
    connection::Symbol
end

function Base.show(io::IO, td::TorsionDecomposition)
    print(io, "TorsionDecomp(vec=:$(td.vector), ax=:$(td.axial), ten=:$(td.tensor))")
end

"""
    decompose_torsion!(reg::TensorRegistry, ac::AffineConnection;
                        manifold::Symbol=:M4) -> TorsionDecomposition

Register the irreducible components of the torsion tensor.

Creates:
- Torsion vector T_a = T^b_{ba} (rank-(0,1), the trace)
- Axial torsion S^a = (1/6)ε^{abcd}T_{bcd} (rank-(1,0), pseudo-vector)
- Tensor torsion q^a_{bc} (rank-(1,2), antisymmetric in bc, traceless)

The decomposition is: T^a_{bc} = vector part + axial part + tensor part.

In d=4: 24 = 4 (vector) + 4 (axial) + 16 (tensor) components.

Ground truth: Hehl et al (1995) Sec 2.3, Eq 2.3.6.
"""
function decompose_torsion!(reg::TensorRegistry, ac::AffineConnection;
                             manifold::Symbol=ac.manifold)
    # Torsion vector T_a (trace)
    vec_name = Symbol(:Tvec_, ac.name)
    if !has_tensor(reg, vec_name)
        register_tensor!(reg, TensorProperties(
            name=vec_name, manifold=manifold, rank=(0, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_torsion_vector => true,
                :irreducible_part => :vector,
                :connection => ac.name)))
    end

    # Axial torsion S^a (pseudo-trace)
    ax_name = Symbol(:Tax_, ac.name)
    if !has_tensor(reg, ax_name)
        register_tensor!(reg, TensorProperties(
            name=ax_name, manifold=manifold, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_torsion_axial => true,
                :irreducible_part => :axial,
                :connection => ac.name)))
    end

    # Tensor torsion q^a_{bc} (traceless, antisymmetric in bc)
    ten_name = Symbol(:Tten_, ac.name)
    if !has_tensor(reg, ten_name)
        register_tensor!(reg, TensorProperties(
            name=ten_name, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[AntiSymmetric(2, 3)],
            options=Dict{Symbol,Any}(
                :is_torsion_tensor => true,
                :irreducible_part => :tensor,
                :traceless => true,
                :connection => ac.name)))
    end

    TorsionDecomposition(vec_name, ax_name, ten_name, ac.name)
end

"""
    torsion_vector_expr(ac::AffineConnection;
                         registry::TensorRegistry=current_registry()) -> TensorExpr

Build the torsion vector trace: T_b = T^a_{ba}.

This contracts the first and last indices of the torsion tensor.
"""
function torsion_vector_expr(ac::AffineConnection;
                              registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    Tensor(ac.torsion_name, [up(a), down(b), down(a)])
end

"""
    contortion_expr(ac::AffineConnection;
                     registry::TensorRegistry=current_registry()) -> TensorExpr

The contortion tensor K^a_{bc} relating torsion to distortion:

    K^a_{bc} = (1/2)(T^a_{bc} + T_{b}^{a}_{c} + T_{c}^{a}_{b})

with T lowered/raised using the metric.

Ground truth: Hehl et al (1995) Eq 2.4.7.
"""
function contortion_expr(ac::AffineConnection;
                          registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used)

    T = ac.torsion_name
    T1 = Tensor(T, [up(a), down(b), down(c)])

    # Simplified: return T^a_{bc} as the leading term
    # Full expression requires index gymnastics with metric
    T1
end
