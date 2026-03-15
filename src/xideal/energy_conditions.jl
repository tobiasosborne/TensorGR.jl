#= Energy condition checker for component metrics.

Given the Ricci tensor, Ricci scalar, and metric (or the mixed stress-energy
tensor T^a_b directly), check the four standard pointwise energy conditions:
NEC, WEC, SEC, DEC.

References:
  - Wald, "General Relativity" (1984), Sec 9.2
  - Hawking & Ellis, "Large Scale Structure of Space-Time" (1973), Sec 4.3
=#

using LinearAlgebra: eigen, eigvals, norm

"""
    EnergyConditionResult

Result of energy condition evaluation at a spacetime point.

Fields:
- `NEC::Bool`: Null energy condition (rho + p_i >= 0 for all i)
- `WEC::Bool`: Weak energy condition (rho >= 0 and NEC)
- `SEC::Bool`: Strong energy condition (rho + sum(p_i) >= 0 and NEC)
- `DEC::Bool`: Dominant energy condition (rho >= |p_i| for all i)
- `rho::Any`: Energy density (timelike eigenvalue of -T^a_b)
- `pressures::Vector`: Principal pressures (spacelike eigenvalues of T^a_b)
"""
struct EnergyConditionResult
    NEC::Bool
    WEC::Bool
    SEC::Bool
    DEC::Bool
    rho::Any
    pressures::Vector
end

function Base.show(io::IO, ec::EnergyConditionResult)
    print(io, "EnergyConditions(NEC=", ec.NEC, ", WEC=", ec.WEC,
          ", SEC=", ec.SEC, ", DEC=", ec.DEC, ")")
end

"""
    check_energy_conditions(Ric::Matrix, R, g::Matrix, ginv::Matrix;
                            dim::Int=size(g,1), atol=1e-10) -> EnergyConditionResult

Check all four energy conditions from the Ricci tensor, Ricci scalar, and metric.

Computes the Einstein tensor G_{ab} = R_{ab} - (1/2) R g_{ab}, then sets
T_{ab} = G_{ab} (i.e., 8piG = 1), raises one index to get T^a_b = g^{ac} T_{cb},
and diagonalizes to find the energy density and principal pressures.
"""
function check_energy_conditions(Ric::Matrix, R, g::Matrix, ginv::Matrix;
                                 dim::Int=size(g, 1), atol=1e-10)
    # Einstein tensor (covariant): G_{ab} = R_{ab} - (1/2) R g_{ab}
    G_down = Matrix{Float64}(undef, dim, dim)
    for a in 1:dim, b in 1:dim
        G_down[a, b] = Float64(Ric[a, b]) - 0.5 * Float64(R) * Float64(g[a, b])
    end

    # T_{ab} = G_{ab} (with 8piG = 1)
    # Mixed tensor: T^a_b = g^{ac} T_{cb}
    T_mixed = Matrix{Float64}(undef, dim, dim)
    for a in 1:dim, b in 1:dim
        s = 0.0
        for c in 1:dim
            s += Float64(ginv[a, c]) * G_down[c, b]
        end
        T_mixed[a, b] = s
    end

    return check_energy_conditions(T_mixed, g; dim=dim, atol=atol)
end

"""
    check_energy_conditions(T_mixed::Matrix, g::Matrix;
                            dim::Int=size(g,1), atol=1e-10) -> EnergyConditionResult

Check all four energy conditions from the pre-computed mixed energy-momentum
tensor T^a_b and the metric g_{ab}.
"""
function check_energy_conditions(T_mixed::Matrix, g::Matrix;
                                 dim::Int=size(g, 1), atol=1e-10)
    # Vacuum check: if T^a_b is essentially zero, all conditions trivially hold
    if all(abs(T_mixed[a, b]) < atol for a in 1:dim, b in 1:dim)
        return EnergyConditionResult(true, true, true, true, 0.0, zeros(dim - 1))
    end

    rho, pressures = _eigendecompose_stress_energy(T_mixed, g; dim=dim, atol=atol)

    # NEC: rho + p_i >= 0 for all principal pressures
    nec = all(rho + p >= -atol for p in pressures)

    # WEC: rho >= 0 and NEC
    wec = (rho >= -atol) && nec

    # SEC: rho + sum(p_i) >= 0 and rho + p_i >= 0 for all i
    sec = (rho + sum(pressures) >= -atol) && nec

    # DEC: rho >= |p_i| for all i (implies WEC)
    dec = all(rho >= abs(p) - atol for p in pressures)

    return EnergyConditionResult(nec, wec, sec, dec, rho, pressures)
end

"""
    _eigendecompose_stress_energy(T_mixed::Matrix, g::Matrix;
                                  dim::Int=size(g,1), atol=1e-10)
        -> (rho, pressures)

Diagonalize T^a_b and identify the timelike eigenvalue (energy density rho)
and spacelike eigenvalues (principal pressures p_i).

The timelike eigenvector satisfies g_{ab} v^a v^b < 0. Its eigenvalue is -rho.
The spacelike eigenvalues are the pressures p_i.

For a perfect fluid T_{ab} = (rho+p) u_a u_b + p g_{ab}, the mixed tensor
T^a_b has eigenvalue -rho for the timelike eigenvector u^a and eigenvalue p
for each spacelike eigenvector.
"""
function _eigendecompose_stress_energy(T_mixed::Matrix, g::Matrix;
                                       dim::Int=size(g, 1), atol=1e-10)
    T_float = Float64.(T_mixed)
    g_float = Float64.(g)

    F = eigen(T_float)
    eigenvalues = real.(F.values)
    eigenvectors = real.(F.vectors)

    # Classify each eigenvector as timelike or spacelike via g_{ab} v^a v^b
    timelike_idx = -1
    timelike_norm = Inf
    for i in 1:dim
        v = eigenvectors[:, i]
        gnorm = 0.0
        for a in 1:dim, b in 1:dim
            gnorm += g_float[a, b] * v[a] * v[b]
        end
        # Most negative g-norm = most timelike
        if gnorm < timelike_norm
            timelike_norm = gnorm
            timelike_idx = i
        end
    end

    # Energy density: the eigenvalue of the timelike eigenvector is -rho
    rho = -eigenvalues[timelike_idx]

    # Pressures: all other eigenvalues
    pressures = Float64[]
    for i in 1:dim
        if i != timelike_idx
            push!(pressures, eigenvalues[i])
        end
    end

    return (rho, pressures)
end
