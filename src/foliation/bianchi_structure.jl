#= Bianchi class A structure constants.
#
# Class A models have trace-free structure constants: C^i_{ij} = 0.
# Parametrized by diagonal matrix n^{ij} = diag(n₁, n₂, n₃):
#   C^i_{jk} = ε_{jkl} n^{li}
#
# Bianchi types (class A):
#   I:     n = (0,0,0)   — flat, trivial
#   II:    n = (1,0,0)   — one non-zero
#   VI₀:  n = (1,-1,0)  — two non-zero, opposite sign
#   VII₀: n = (1,1,0)   — two non-zero, same sign
#   VIII:  n = (1,1,-1)  — all non-zero, one different sign
#   IX:    n = (1,1,1)   — all non-zero, same sign (S³ topology)
#
# Ground truth: Ellis & MacCallum, CMP 12, 108 (1969); Pitrou et al Sec 3.2.
=#

"""
    BianchiStructureConstants

Structure constants for a Bianchi class A model, parametrized by the
diagonal matrix n^{ij} = diag(n₁, n₂, n₃).

The structure constants are: C^i_{jk} = ε_{jkl} n^{li}

# Fields
- `type::Symbol`       -- Bianchi type (:I, :II, :VI0, :VII0, :VIII, :IX)
- `n::NTuple{3,Int}` -- diagonal entries (n₁, n₂, n₃)
"""
struct BianchiStructureConstants
    type::Symbol
    n::NTuple{3,Int}
end

function Base.show(io::IO, bsc::BianchiStructureConstants)
    print(io, "Bianchi $(bsc.type), n=", bsc.n)
end

"""
    bianchi_type(n::NTuple{3,Int}) -> Symbol

Classify the Bianchi type from the diagonal n-parameters.

Ground truth: Ellis & MacCallum (1969) Table 1.
"""
function bianchi_type(n::NTuple{3,Int})
    sorted = sort(collect(n))
    if sorted == [0, 0, 0]
        return :I
    elseif count(!=(0), n) == 1
        return :II
    elseif sorted == [-1, 0, 1]
        return :VI0
    elseif sorted == [0, 1, 1]
        return :VII0
    elseif sorted == [-1, 1, 1]
        return :VIII
    elseif sorted == [1, 1, 1]
        return :IX
    else
        error("Not a standard Bianchi class A model: n=$n")
    end
end

# Named constructors for each Bianchi type
bianchi_I()     = BianchiStructureConstants(:I,     (0, 0, 0))
bianchi_II()    = BianchiStructureConstants(:II,    (1, 0, 0))
bianchi_VI0()   = BianchiStructureConstants(:VI0,   (1, -1, 0))
bianchi_VII0()  = BianchiStructureConstants(:VII0,  (1, 1, 0))
bianchi_VIII()  = BianchiStructureConstants(:VIII,  (1, 1, -1))
bianchi_IX()    = BianchiStructureConstants(:IX,    (1, 1, 1))

"""
    structure_constant(bsc::BianchiStructureConstants, i::Int, j::Int, k::Int) -> Int

Compute C^i_{jk} = ε_{jkl} n^{li} for the given Bianchi model.

Uses the 3D Levi-Civita symbol: ε_{123} = +1, antisymmetric.
"""
function structure_constant(bsc::BianchiStructureConstants, i::Int, j::Int, k::Int)
    (1 <= i <= 3 && 1 <= j <= 3 && 1 <= k <= 3) ||
        error("Indices must be in 1..3")

    # ε_{jkl} n^{li} = ε_{jkl} n_l δ^{li} (diagonal n)
    # Sum over l:
    result = 0
    for l in 1:3
        eps_jkl = _levi_civita_3d(j, k, l)
        eps_jkl == 0 && continue
        # n^{li} = n_l * δ_{li} (diagonal)
        l == i || continue
        result += eps_jkl * bsc.n[l]
    end
    result
end

"""3D Levi-Civita symbol: ε_{ijk}."""
function _levi_civita_3d(i::Int, j::Int, k::Int)
    (i == j || j == k || i == k) && return 0
    # Even permutation of (1,2,3) → +1, odd → -1
    perm = (i, j, k)
    if perm == (1,2,3) || perm == (2,3,1) || perm == (3,1,2)
        return 1
    else
        return -1
    end
end

"""
    verify_jacobi(bsc::BianchiStructureConstants) -> Bool

Verify the Jacobi identity: C^i_{[jk} C^j_{l]m} = 0.

This is automatically satisfied for class A models (trace-free structure
constants), but serves as a consistency check.

Ground truth: Ellis & MacCallum (1969) Sec 2.
"""
function verify_jacobi(bsc::BianchiStructureConstants)
    # Jacobi: C^l_{im} C^m_{jk} + C^l_{jm} C^m_{ki} + C^l_{km} C^m_{ij} = 0
    # (cyclic sum over i,j,k for all l)
    for l in 1:3, i in 1:3, j in 1:3, k in 1:3
        total = 0
        for m in 1:3
            total += structure_constant(bsc, l, i, m) * structure_constant(bsc, m, j, k)
            total += structure_constant(bsc, l, j, m) * structure_constant(bsc, m, k, i)
            total += structure_constant(bsc, l, k, m) * structure_constant(bsc, m, i, j)
        end
        total == 0 || return false
    end
    true
end

"""
    is_class_A(bsc::BianchiStructureConstants) -> Bool

Check if the structure constants are class A (trace-free): C^i_{ij} = 0.
"""
function is_class_A(bsc::BianchiStructureConstants)
    for j in 1:3
        trace = sum(structure_constant(bsc, i, i, j) for i in 1:3)
        trace == 0 || return false
    end
    true
end
