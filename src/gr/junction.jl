#= Israel junction conditions for thin shells.

Given a hypersurface Sigma separating two spacetime regions, the Israel
junction conditions relate the jump in extrinsic curvature across Sigma to
the surface stress-energy tensor S_{ab} on the shell:

    [K_{ab}] - gamma_{ab} [K] = -8 pi S_{ab}

where [X] = X^+ - X^- denotes the jump across Sigma, gamma_{ab} is the
induced metric, and [K] = gamma^{ab} [K_{ab}] is the trace of the jump.
=#

"""
    JunctionData

Stores the geometric data for Israel junction conditions on a thin shell.

Fields:
- `hypersurface`: name of the hypersurface (must be registered via `define_hypersurface!`)
- `K_plus`: name of the extrinsic curvature tensor on the outside (+ side)
- `K_minus`: name of the extrinsic curvature tensor on the inside (- side)
- `S`: name of the surface stress-energy tensor
- `induced_metric`: name of the induced metric on the shell
"""
struct JunctionData
    hypersurface::Symbol
    K_plus::Symbol
    K_minus::Symbol
    S::Symbol
    induced_metric::Symbol
end

"""
    define_junction!(reg, hypersurface;
                     K_plus=:Kp, K_minus=:Km, S=:S,
                     induced_metric=:gamma_jc) -> JunctionData

Register the tensors needed for Israel junction conditions on a
codimension-1 hypersurface and return a `JunctionData` struct.

The hypersurface must already be registered via `define_hypersurface!`.

Registers (if not already present):
- `K_plus` (rank-(0,2) symmetric): extrinsic curvature from the + side
- `K_minus` (rank-(0,2) symmetric): extrinsic curvature from the - side
- `S` (rank-(0,2) symmetric): surface stress-energy tensor
- `induced_metric` (rank-(0,2) symmetric): induced metric on the shell

# Example
```julia
reg = TensorRegistry()
@manifold M4 dim=4 metric=g
define_hypersurface!(reg, :Sigma; ambient=:M4, metric=:g, signature=-1)
jd = define_junction!(reg, :Sigma)
```
"""
function define_junction!(reg::TensorRegistry, hypersurface::Symbol;
                          K_plus::Symbol=:Kp,
                          K_minus::Symbol=:Km,
                          S::Symbol=:S,
                          induced_metric::Symbol=:gamma_jc)
    key = Symbol(:hypersurface_, hypersurface)
    haskey(reg.foliations, key) ||
        error("No hypersurface ':$hypersurface' found in registry. " *
              "Call define_hypersurface! first.")
    sp = reg.foliations[key]
    sp.codimension == 1 ||
        error("Israel junction conditions require codimension-1 " *
              "hypersurface (got codimension $(sp.codimension)).")

    ambient = sp.ambient
    sym2 = SymmetrySpec[Symmetric(1, 2)]

    for tname in (K_plus, K_minus, S, induced_metric)
        if !has_tensor(reg, tname)
            register_tensor!(reg, TensorProperties(
                name=tname, manifold=ambient, rank=(0, 2),
                symmetries=sym2))
        end
    end

    JunctionData(hypersurface, K_plus, K_minus, S, induced_metric)
end

"""
    israel_equation(K_plus, K_minus, induced_metric;
                    dim=4, idx_a=down(:a), idx_b=down(:b)) -> TensorExpr

Return the Israel junction condition as a tensor equation (LHS = 0):

    [K_{ab}] - gamma_{ab} [K] + 8 pi S_{ab} = 0

where `[K_{ab}] = K^+_{ab} - K^-_{ab}` is the jump in extrinsic curvature,
`gamma_{ab}` is the induced metric on the shell, and `[K] = gamma^{cd}[K_{cd}]`
is the trace of the jump.

The returned expression equals the left-hand side; setting it to zero
gives the junction condition.

Arguments:
- `K_plus`, `K_minus`: names of the extrinsic curvature tensors (+ and - sides)
- `induced_metric`: name of the induced metric on the shell
- `dim`: ambient spacetime dimension (default 4)
- `idx_a`, `idx_b`: free indices for the equation (default `down(:a)`, `down(:b)`)

# Example
```julia
eq = israel_equation(:Kp, :Km, :gamma_jc)
```
"""
function israel_equation(K_plus::Symbol, K_minus::Symbol,
                         induced_metric::Symbol;
                         dim::Int=4,
                         idx_a::TIndex=down(:a), idx_b::TIndex=down(:b))
    used = Set{Symbol}([idx_a.name, idx_b.name])
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    gamma_ab = Tensor(induced_metric, [idx_a, idx_b])
    gamma_up = Tensor(induced_metric, [up(c), up(d)])
    S_ab = Tensor(:S, [idx_a, idx_b])

    # Build expanded form:
    #   K+_{ab} - K-_{ab}
    #   - gamma_{ab} gamma^{cd} K+_{cd}
    #   + gamma_{ab} gamma^{cd} K-_{cd}
    #   + 8 pi S_{ab}
    tsum(TensorExpr[
        Tensor(K_plus, [idx_a, idx_b]),
        tproduct(-1 // 1, TensorExpr[Tensor(K_minus, [idx_a, idx_b])]),
        tproduct(-1 // 1, TensorExpr[gamma_ab, gamma_up,
                                      Tensor(K_plus, [down(c), down(d)])]),
        tproduct(1 // 1, TensorExpr[gamma_ab, gamma_up,
                                     Tensor(K_minus, [down(c), down(d)])]),
        tproduct(8 // 1, TensorExpr[TScalar(:pi), S_ab])
    ])
end

"""
    junction_stress_energy(K_plus, K_minus, induced_metric;
                           dim=4, idx_a=down(:a), idx_b=down(:b)) -> TensorExpr

Solve the Israel junction condition for the surface stress-energy tensor:

    S_{ab} = -(1/8pi) ([K_{ab}] - gamma_{ab} [K])

where `[K_{ab}] = K^+_{ab} - K^-_{ab}` and `[K] = gamma^{cd}[K_{cd}]`.

Arguments:
- `K_plus`, `K_minus`: names of the extrinsic curvature tensors (+ and - sides)
- `induced_metric`: name of the induced metric on the shell
- `dim`: ambient spacetime dimension (default 4)
- `idx_a`, `idx_b`: free indices (default `down(:a)`, `down(:b)`)

# Example
```julia
S_expr = junction_stress_energy(:Kp, :Km, :gamma_jc)
```
"""
function junction_stress_energy(K_plus::Symbol, K_minus::Symbol,
                                induced_metric::Symbol;
                                dim::Int=4,
                                idx_a::TIndex=down(:a), idx_b::TIndex=down(:b))
    used = Set{Symbol}([idx_a.name, idx_b.name])
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used)

    gamma_ab = Tensor(induced_metric, [idx_a, idx_b])
    gamma_up = Tensor(induced_metric, [up(c), up(d)])

    # S_{ab} = -(1/8pi) ([K_{ab}] - gamma_{ab} [K])
    # Expanded:
    #   -(1/8pi) K+_{ab} + (1/8pi) K-_{ab}
    #   + (1/8pi) gamma_{ab} gamma^{cd} K+_{cd}
    #   - (1/8pi) gamma_{ab} gamma^{cd} K-_{cd}
    #
    # We represent 1/pi as the scalar symbol :inv_pi so the expression
    # stays rational in coefficients.
    tsum(TensorExpr[
        tproduct(-1 // 8, TensorExpr[TScalar(:inv_pi),
                                      Tensor(K_plus, [idx_a, idx_b])]),
        tproduct(1 // 8, TensorExpr[TScalar(:inv_pi),
                                     Tensor(K_minus, [idx_a, idx_b])]),
        tproduct(1 // 8, TensorExpr[TScalar(:inv_pi), gamma_ab, gamma_up,
                                     Tensor(K_plus, [down(c), down(d)])]),
        tproduct(-1 // 8, TensorExpr[TScalar(:inv_pi), gamma_ab, gamma_up,
                                      Tensor(K_minus, [down(c), down(d)])])
    ])
end
