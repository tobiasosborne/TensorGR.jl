# Tutorial

This tutorial walks through the core features of TensorGR.jl with worked examples.
Each section corresponds to an example script in the `examples/` directory.

## 1. Getting Started

Every TensorGR computation begins by creating a registry and defining a manifold.

```julia
using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    # Define a 4-dimensional manifold with metric g
    @manifold M4 dim=4 metric=g

    # This automatically registers:
    #   - ManifoldProperties for M4 (dim=4)
    #   - Metric tensor g_{ab} (symmetric, rank (0,2))
    #   - Kronecker delta delta^a_b
    #   - Tangent vector bundle (dim=4)
end
```

### Building Tensor Expressions

Tensors are constructed with a name and a vector of indices:

```julia
R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
g_inv = Tensor(:g, [up(:a), up(:b)])
V = Tensor(:V, [up(:a)])
```

The convenience functions `up(:a)` and `down(:a)` create `TIndex` values with the specified position.

### Metric Contraction

The simplification engine automatically contracts metrics:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g

    # g^{ab} g_{bc} = delta^a_c
    product = Tensor(:g, [up(:a), up(:b)]) * Tensor(:g, [down(:b), down(:c)])
    result = simplify(product)
    # result == Tensor(:delta, [up(:a), down(:c)])

    # delta^a_a = dim = 4
    trace = simplify(Tensor(:delta, [up(:a), down(:a)]))
    # trace == TScalar(4//1)
end
```

### Symmetry Verification

Register curvature tensors and verify their symmetries:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)

    R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
    R2 = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])

    # Antisymmetry: R_{abcd} + R_{bacd} = 0
    simplify(R1 + R2)  # => TScalar(0//1)

    # Pair symmetry: R_{abcd} = R_{cdab}
    R3 = Tensor(:Riem, [down(:c), down(:d), down(:a), down(:b)])
    simplify(R1 - R3)  # => TScalar(0//1)
end
```

### Output Formats

```julia
expr = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
to_latex(expr)    # "Riem_{a b c d}"
to_unicode(expr)  # "Riem_a_b_c_d"
```

> **See also:** [`examples/01_getting_started.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/01_getting_started.jl)

---

## 2. Covariant Derivatives

Define a covariant derivative and expand into Christoffel symbols.

### Defining a CovD

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    @define_tensor V on=M4 rank=(1,0)
    @covd D on=M4 metric=g

    # This registers:
    #   - CovD "D" with metric compatibility
    #   - Christoffel symbol GammaD^a_{bc} (symmetric in lower indices)
end
```

### Expanding Covariant Derivatives

```julia
# nabla_a V^b = partial_a V^b + Gamma^b_{ac} V^c
nabla_V = TDeriv(down(:a), Tensor(:V, [up(:b)]))
expanded = covd_to_christoffel(nabla_V, :D)
# Result: d_a(V^b) + GammaD^b_{ac} V^c

# nabla_a W_b = partial_a W_b - Gamma^c_{ab} W_c
nabla_W = TDeriv(down(:a), Tensor(:W, [down(:b)]))
expanded_W = covd_to_christoffel(nabla_W, :D)
```

### Christoffel in Terms of Metric Gradients

```julia
christoffel_to_grad_metric(:g, up(:a), down(:b), down(:c))
# (1/2) g^{ad} (d_b g_{cd} + d_c g_{bd} - d_d g_{bc})
```

### Commuting Derivatives

Commuting covariant derivatives produces Riemann curvature:

```julia
double_deriv = TDeriv(down(:b), TDeriv(down(:a), Tensor(:V, [up(:c)])))
sorted = commute_covds(double_deriv, :D)
# Result: d_a(d_b(V^c)) + Riem^c_{dba} V^d
```

### Bianchi Identity

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    for r in bianchi_rules(); register_rule!(reg, r); end

    # Contracted Bianchi: nabla^a G_{ab} = 0
    bianchi = simplify(TDeriv(up(:a), Tensor(:Ein, [down(:a), down(:b)])))
    # bianchi == TScalar(0//1)

    # nabla^a R_{ab} = (1/2) nabla_b R
    ricci_bianchi = simplify(TDeriv(up(:a), Tensor(:Ric, [down(:a), down(:b)])))
    # (1//2) d_b(RicScalar)
end
```

> **See also:** [`examples/02_covariant_derivatives.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/02_covariant_derivatives.jl)

---

## 3. Curvature Decomposition

### Riemann-Weyl Decomposition

In *d* dimensions, the Riemann tensor decomposes into Weyl (traceless) and Ricci parts:

```julia
a, b, c, d = down(:a), down(:b), down(:c), down(:d)

# R_{abcd} = C_{abcd} + (Ricci terms) + (scalar terms)
decomp = riemann_to_weyl(a, b, c, d, :g; dim=4)

# Inverse: express Weyl in terms of Riemann
inv_decomp = weyl_to_riemann(a, b, c, d, :g; dim=4)
```

### Other Curvature Tensors

```julia
# Schouten tensor: P_{ab} = 1/(d-2)(R_{ab} - R g_{ab}/(2(d-1)))
schouten = schouten_to_ricci(a, b, :g; dim=4)

# Trace-free Ricci: S_{ab} = R_{ab} - (1/d) g_{ab} R
tfric = tfricci_expr(a, b, :g; dim=4)

# Einstein tensor: G_{ab} = R_{ab} - (1/2) g_{ab} R
einstein = einstein_to_ricci(a, b, :g)
```

### Curvature Conversions

Convert all curvature tensors to a common basis:

```julia
# Replace Weyl, Schouten, Einstein, TFRicci with Riemann + Ricci + metric
to_riemann(expr; metric=:g, dim=4)

# Replace everything with Ricci + scalar + metric (where possible)
to_ricci(expr; metric=:g, dim=4)
```

### Kretschmann Scalar

```julia
# K = R_{abcd} R^{abcd} as an abstract expression
kretsch = kretschmann_expr(:g; dim=4)
```

> **See also:** [`examples/03_curvature_decomposition.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/03_curvature_decomposition.jl)

---

## 4. Perturbation Theory

TensorGR implements xPert-style metric perturbation at arbitrary order.

### Setting Up a Perturbation

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)

    # Define g -> g + eps*h
    mp = define_metric_perturbation!(reg, :g, :h)
end
```

### Inverse Metric Perturbation

```julia
# delta(g^{ab}) = -g^{ac} g^{bd} h_{cd}
delta_ginv = deltainverse_metric(mp, up(:a), up(:b), 1)

# Second order: delta^2(g^{ab})
delta2_ginv = deltainverse_metric(mp, up(:a), up(:b), 2)
```

### Christoffel and Curvature Perturbations

```julia
# delta(Gamma^a_{bc}) at first order
delta_gamma = deltachristoffel(mp, up(:a), down(:b), down(:c), 1)
# = (1/2) g^{ad} (d_b h_{cd} + d_c h_{bd} - d_d h_{bc})

# delta(R_{ab}) at first order
delta_ricci = deltaricci(mp, down(:a), down(:b), 1)

# delta(R) at first order
delta_R = deltaricci_scalar(mp, 1)
```

### Background Field Equations

Set background curvature to zero for vacuum spacetimes:

```julia
background_solution!(reg, [:Ric, :RicScalar, :Ein])
# Now simplify(Tensor(:Ric, [down(:a), down(:b)])) => 0
```

### Expanding General Expressions

```julia
# Perturb any tensor expression at a given order
expand_perturbation(expr, mp, order)
```

> **See also:** [`examples/04_perturbation_theory.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/04_perturbation_theory.jl)

---

## 5. Component Calculations (Schwarzschild)

TensorGR can compute curvature quantities from explicit metric components.

### Setting Up Coordinates

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_chart!(reg, :Schw; manifold=:M4, coords=[:t, :r, :theta, :phi])
end
```

### Computing Christoffel Symbols

```julia
# Provide metric components and a derivative function
Gamma = metric_christoffel(g_data, ginv_data, [:t, :r, :theta, :phi];
                           deriv_fn=my_diff)
```

### Computing Curvature

```julia
Riem  = metric_riemann(Gamma, dim; coords=coords, deriv_fn=my_diff)
Ric   = metric_ricci(Riem, dim)
R     = metric_ricci_scalar(Ric, ginv_data, dim)
G     = metric_einstein(Ric, R, g_data, dim)
Weyl  = metric_weyl(Riem, Ric, R, g_data, ginv_data, dim)
K     = metric_kretschmann(Riem, g_data, ginv_data, dim)
```

For the Schwarzschild solution, the Ricci tensor vanishes (vacuum) and the Kretschmann scalar is:

```
K = 48 M^2 / r^6
```

### Riemann in Terms of Christoffel Symbols

```julia
riemann_to_christoffel(up(:a), down(:b), down(:c), down(:d), :Gamma)
# d_c(Gamma^a_{db}) - d_d(Gamma^a_{cb}) + Gamma^a_{ce} Gamma^e_{db} - Gamma^a_{de} Gamma^e_{cb}
```

> **See also:** [`examples/05_schwarzschild.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/05_schwarzschild.jl)

---

## 6. Exterior Calculus

### Defining Forms

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_form!(reg, :A; manifold=:M4, degree=1)  # 1-form
    define_form!(reg, :F; manifold=:M4, degree=2)  # 2-form

    # Forms are automatically antisymmetric:
    simplify(Tensor(:F, [down(:a), down(:b)]) + Tensor(:F, [down(:b), down(:a)]))
    # => 0
end
```

### Wedge Product

```julia
A1 = Tensor(:A, [down(:a)])
A2 = Tensor(:A, [down(:b)])
w = wedge(A1, A2, 1, 1)
# Coefficient (p+q)!/(p!q!) = 2
```

### Exterior Derivative

```julia
dA = exterior_d(A1, 1, down(:b))
# d_b(A_a)
```

### Interior Product

```julia
v = Tensor(:V, [up(:a)])
alpha = Tensor(:F, [down(:a), down(:b)])
iv_alpha = interior_product(v, alpha)
# V^a F_{ab}
```

### Cartan's Magic Formula

The Lie derivative of a form equals `d(iota_v omega) + iota_v(d omega)`:

```julia
cartan_lie_d(v, omega, degree, deriv_idx)
```

### Connection and Curvature Forms

```julia
# Connection 1-form: omega^a_b = Gamma^a_{cb} dx^c
omega = connection_form(:Gamma, up(:a), down(:b), down(:c))

# Curvature 2-form via second structure equation:
# Omega^a_b = d(omega^a_b) + omega^a_c ^ omega^c_b
Omega = curvature_form(:Gamma, up(:a), down(:b), down(:c), down(:d))
```

> **See also:** [`examples/06_exterior_calculus.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/06_exterior_calculus.jl)

---

## 7. Vector Bundles and Gauge Theory

TensorGR supports multiple vector bundles, enabling gauge theory computations where indices belong to different fiber spaces.

### Defining a Vector Bundle

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g

    # Define an SU(2) gauge bundle with 3-dimensional fibers
    define_vbundle!(reg, :SU2; manifold=:M4, dim=3,
                    indices=[:A, :B, :C, :D, :E])
end
```

### Mixed-Bundle Tensors

Indices carry their bundle identity:

```julia
# Field strength: gauge index A (SU2) + spacetime indices mu,nu (Tangent)
F = Tensor(:F, [up(:A, :SU2), down(:mu), down(:nu)])
# F.indices[1].vbundle == :SU2
# F.indices[2].vbundle == :Tangent
```

### Cross-Bundle Protection

Indices with the same name but different bundles are never contracted:

```julia
T = Tensor(:T, [up(:a, :Tangent)])
S = Tensor(:S, [down(:a, :SU2)])
product = T * S

dummy_pairs(product)  # empty -- no contraction
free_indices(product)  # both 'a' indices are free
```

### Same-Bundle Contraction

Indices on the same bundle contract normally:

```julia
T1 = Tensor(:T, [up(:A, :SU2), down(:mu)])
T2 = Tensor(:S, [down(:A, :SU2), up(:nu)])
dp = dummy_pairs(T1 * T2)
# One pair: A^SU2 contracted
```

### Yang-Mills Example

```julia
F_up = Tensor(:F, [up(:A, :SU2), down(:mu), down(:nu)])
F_dn = Tensor(:F, [down(:A, :SU2), up(:mu), up(:nu)])
lagrangian = F_up * F_dn
# Three dummy pairs: A (SU2), mu (Tangent), nu (Tangent)
```

> **See also:** [`examples/07_gauge_theory.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/07_gauge_theory.jl)

---

## Simplification Pipeline

The `simplify` function is the main workhorse. It runs a fixed-point loop of:

1. **`expand_products`** -- distribute multiplication over addition
2. **`contract_metrics`** -- eliminate `g` and `delta` contractions (raising/lowering)
3. **`canonicalize`** -- sort index slots using xperm.c symmetry generators
4. **`collect_terms`** -- combine terms that differ only by scalar coefficient
5. **`apply_rules`** -- apply registered rewrite rules (Bianchi, vanishing, etc.)

```julia
result = simplify(expr)                        # default registry
result = simplify(expr; registry=reg)          # explicit registry
result = simplify(expr; maxiter=50)            # increase iteration limit
```

## Rewrite Rules

Define custom rewrite rules with pattern matching:

```julia
# Set a tensor to zero
set_vanishing!(reg, :Torsion)

# Custom rule: replace Einstein with Ricci decomposition
register_rule!(reg, RewriteRule(
    expr -> expr isa Tensor && expr.name == :Ein,
    expr -> einstein_to_ricci(expr.indices[1], expr.indices[2], :g)
))
```

Or use the `make_rule` function for pattern-based rules with automatic symmetry handling:

```julia
lhs = Tensor(:Ein, [down(:a), down(:b)])
rhs = einstein_to_ricci(down(:a), down(:b), :g)
rule = make_rule(lhs, rhs)
```

---

## 8. Equation Solver (`solve_tensors`)

The `solve_tensors` function solves linear tensor equations for unknown tensors. Given an equation of the form `expr = 0`, it decomposes each term, identifies unknowns, and returns rewrite rules for the solution.

### Solving Einstein's Equation for the Stress-Energy Tensor

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor T on=M4 rank=(0,2) symmetries=[Symmetric(1,2)]

    # Einstein's equation: G_{ab} - 8pi T_{ab} = 0
    G = Tensor(:Ein, [down(:a), down(:b)])
    T = Tensor(:T, [down(:a), down(:b)])
    equation = G - tproduct(8 // 1, TensorExpr[TScalar(:pi), T])

    # Solve for T_{ab}
    rules = solve_tensors(equation, [:T])
    # Returns a RewriteRule: T_{ab} => (1/8pi) G_{ab}
end
```

### Roundtrip Verification

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor T on=M4 rank=(0,2) symmetries=[Symmetric(1,2)]

    # Solve G_{ab} = 8pi T_{ab} for T
    G = Tensor(:Ein, [down(:a), down(:b)])
    T = Tensor(:T, [down(:a), down(:b)])
    equation = G - tproduct(8 // 1, TensorExpr[TScalar(:pi), T])
    rules = solve_tensors(equation, [:T])

    # Register the solution as a rule and verify roundtrip
    for r in rules
        register_rule!(reg, r)
    end

    # Substituting the solution back into the equation should give zero
    result = simplify(equation)
    # result == TScalar(0//1)
end
```

The solver also supports systems of equations via `solve_tensors(equations::Vector, unknowns)` and optional trace-taking with `take_traces=true`.

---

## 9. Metric Engine

The `define_metric!` function is a one-liner that registers a metric together with its inverse, Kronecker delta, epsilon tensor, Levi-Civita covariant derivative with Christoffel symbols, all curvature tensors, and Bianchi rules.

### Full Metric Setup

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g

    # Full DefMetric setup with explicit signature
    define_metric!(reg, :g; manifold=:M4, signature=lorentzian(4))

    # This registers: g_{ab}, delta^a_b, epsilon_{abcd}, curvature tensors,
    # Levi-Civita CovD (named nabla_g), Christoffel symbols, and Bianchi rules.
end
```

### Flat Metrics

Mark a metric as flat to automatically set all curvature tensors and Christoffel symbols to zero:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=eta
    define_metric!(reg, :eta; manifold=:M4, signature=lorentzian(4))

    set_flat!(reg, :eta)
    # Now: Riem = 0, Ric = 0, RicScalar = 0, Weyl = 0, Ein = 0, Christoffel = 0

    result = simplify(Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)]))
    # result == TScalar(0//1)
end
```

### Freezing Metrics

Freeze a metric to prevent it from participating in index contraction:

```julia
freeze_metric!(reg, :g)
# g^{ab} g_{bc} is no longer simplified to delta^a_c

unfreeze_metric!(reg, :g)
# Contraction resumes
```

### Conformal Metrics

Declare a conformal relationship between two metrics:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)
    define_metric!(reg, :g_tilde; manifold=:M4)

    # g_tilde = e^{2f} g
    set_conformal_to!(reg, :g_tilde, :g, :f)
end
```

### Metric Determinant and Volume Element

```julia
# Symbolic determinant det(g)
det_g = metric_det_expr(:g)

# Volume element sqrt(-det(g)) for Lorentzian signature
vol = sqrt_det_expr(:g; neg=true)
```

### Signatures

```julia
lorentzian(4)   # MetricSignature(-,+,+,+)
euclidean(3)    # MetricSignature(+,+,+)
sign_det(lorentzian(4))  # -1
```

---

## 10. Topological Invariants

TensorGR provides constructors for topological densities in 4D, useful in modified gravity and anomaly analysis.

### Pontryagin Density

The Pontryagin (Chern-Pontryagin) density is the pseudoscalar `*(R wedge R)`:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # P = epsilon^{abcd} R_{ab}^{ef} R_{cdef}
    P = pontryagin_density(:g)
end
```

### Euler (Gauss-Bonnet) Density

The Euler density in 4D is `E_4 = R^2 - 4 R_{ab} R^{ab} + R_{abcd} R^{abcd}`:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    E4 = euler_density(:g; dim=4)
    # Kretschmann - 4 Ricci^2 + RicciScalar^2
end
```

### Chern-Simons Gravitational Coupling

Couple an axion/dilaton scalar field to the Pontryagin density:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)
    @define_tensor theta on=M4 rank=(0,0)

    # S_CS = theta * epsilon^{abcd} R_{ab}^{ef} R_{cdef}
    S_CS = chern_simons_action(Tensor(:theta, TIndex[]), :g)
end
```

> **See also:** [`examples/08_postquantum_gravity.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/08_postquantum_gravity.jl) for topological terms in higher-derivative gravity actions.

---

## 11. Hypersurface & ADM

Define codimension-1 hypersurface embeddings for boundary terms, junction conditions, and ADM decompositions.

### Defining a Hypersurface

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # Define a spacelike hypersurface with timelike normal (signature=-1)
    hs = define_hypersurface!(reg, :Sigma;
        ambient=:M4, metric=:g,
        normal_name=:n, extrinsic_name=:K, induced_name=:gamma,
        signature=-1)

    # This registers:
    #   - Unit normal n_a (rank-1, with n_a n^a = -1 rule)
    #   - Induced metric gamma_{ab} (symmetric)
    #   - Extrinsic curvature K_{ab} (symmetric)
    #   - Projector P^a_b onto the hypersurface
end
```

### Induced Metric

The induced metric on the hypersurface is `gamma_{ab} = g_{ab} - sigma * n_a n_b`:

```julia
# For timelike normal (signature = -1):
# gamma_{ab} = g_{ab} + n_a n_b
gamma = induced_metric_expr(down(:a), down(:b), :g, :n; signature=-1)
```

### Extrinsic Curvature

The extrinsic curvature (second fundamental form) is `K_{ab} = -nabla_a n_b`:

```julia
K = extrinsic_curvature_expr(down(:a), down(:b), :n, :g)
# Returns -d_a(n_b); expand with covd_to_christoffel for connection terms
```

### Projector

The projection tensor onto the hypersurface:

```julia
# P^a_b = delta^a_b + n^a n_b  (for timelike normal)
P = projector_expr(up(:a), down(:b), :n; signature=-1)
```

### Normal Normalization

The `define_hypersurface!` function automatically registers a rule so that `n_a n^a` simplifies to the signature value:

```julia
with_registry(reg) do
    # ... after define_hypersurface! ...
    nn = Tensor(:n, [down(:a)]) * Tensor(:n, [up(:a)])
    simplify(nn)  # => TScalar(-1//1) for timelike normal
end
```

---

## 12. Perturbation Theory (Advanced)

Building on Section 4, TensorGR provides specialized background geometries, Isaacson averaging for gravitational wave stress-energy, and variational derivatives.

### Maximally Symmetric Backgrounds

For de Sitter, anti-de Sitter, or Minkowski backgrounds, register curvature rules in terms of a cosmological constant:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # Register: R_{abcd} = (2Lambda/(d-1))(g_{ac}g_{bd} - g_{ad}g_{bc})
    #           R_{ab} = Lambda * g_{ab}
    #           R = d * Lambda
    maximally_symmetric_background!(reg, :M4; metric=:g, cosmological_constant=:Lambda)

    # Now curvature simplifies using these rules
    Ric = Tensor(:Ric, [down(:a), down(:b)])
    result = simplify(Ric)
    # result = Lambda * g_{ab}
end
```

### Vacuum Backgrounds

For Ricci-flat backgrounds (Schwarzschild, Kerr), set only Ricci to zero while keeping Riemann nonzero:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    vacuum_background!(reg, :M4; metric=:g)
    # Now: Ric_{ab} = 0, RicScalar = 0
    # But: Riem_{abcd} is NOT set to zero
end
```

### Isaacson Averaging

Compute the effective stress-energy tensor of gravitational waves via short-wavelength averaging. The `isaacson_average` function keeps only terms bilinear in the perturbation and discards linear and higher-order terms:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)
    mp = define_metric_perturbation!(reg, :g, :h)

    # Expand Einstein tensor to second order
    delta2_G = expand_perturbation(
        Tensor(:Ein, [down(:a), down(:b)]), mp, 2)

    # Isaacson average: keep only h*h terms (bilinear)
    T_eff = isaacson_average(delta2_G, :h)
    # Terms linear in h are discarded (average to zero)
    # Terms with 0 or >2 factors of h are also discarded
end
```

### Variational Derivatives

Compute the Euler-Lagrange equations by varying a Lagrangian with respect to a field:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    @define_tensor phi on=M4 rank=(0,0)

    # Scalar field Lagrangian: L = (1/2) g^{ab} d_a(phi) d_b(phi)
    phi_field = Tensor(:phi, TIndex[])
    L = (1 // 2) * grad_squared(phi_field, :g)

    # delta L / delta phi = -box(phi)
    eom = variational_derivative(L, :phi)
end
```

For metric variations, use `metric_variation` which implements the identities `delta(g^{ab})/delta(g^{cd})` and `delta(g_{ab})/delta(g^{cd})` via the Leibniz rule:

```julia
# Vary an expression with respect to g^{cd}
delta_expr = metric_variation(expr, :g, down(:c), down(:d))
```

> **See also:** [`examples/11_6deriv_gravity_dS.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/11_6deriv_gravity_dS.jl) for perturbation theory on a de Sitter background.

---

## 13. Quadratic Action & Spin Projectors

For analyzing the particle content of gravitational theories, TensorGR provides quadratic form analysis and Barnes-Rivers spin-projection operators.

### Quadratic Forms

A quadratic Lagrangian `L = Phi_i M_{ij}(k) Phi_j` defines a kinetic matrix whose inverse is the propagator:

```julia
# Build a quadratic form from field pairs
fields = [:phi, :psi]
entries = Dict(
    (:phi, :phi) => :(k^2),
    (:phi, :psi) => 0,
    (:psi, :psi) => :(k^2 - m^2)
)
qf = quadratic_form(entries, fields)

# Compute the propagator (matrix inverse)
prop = propagator(qf)

# Compute the determinant
det = determinant(qf)
```

### Barnes-Rivers Spin Projectors

In momentum space, symmetric rank-2 fields decompose into spin sectors using transverse and longitudinal projectors:

```julia
mu, nu, rho, sigma = down(:mu), down(:nu), down(:rho), down(:sigma)

# Transverse projector: theta_{mu nu} = eta_{mu nu} - k_mu k_nu / k^2
theta = theta_projector(mu, nu; metric=:eta, k_name=:k, k_sq=:k2)

# Longitudinal projector: omega_{mu nu} = k_mu k_nu / k^2
omega = omega_projector(mu, nu; k_name=:k, k_sq=:k2)
```

### Spin-2, Spin-1, and Spin-0 Projectors

The six Barnes-Rivers operators decompose a symmetric tensor into irreducible spin sectors:

```julia
# Spin-2 (transverse-traceless graviton)
P2 = spin2_projector(mu, nu, rho, sigma; dim=4, metric=:eta, k_name=:k, k_sq=:k2)

# Spin-1 (vector)
P1 = spin1_projector(mu, nu, rho, sigma; metric=:eta, k_name=:k, k_sq=:k2)

# Spin-0 scalar (transverse trace)
P0s = spin0s_projector(mu, nu, rho, sigma; dim=4, metric=:eta, k_name=:k, k_sq=:k2)

# Spin-0 w (longitudinal)
P0w = spin0w_projector(mu, nu, rho, sigma; k_name=:k, k_sq=:k2)

# Transfer operators between spin-0 sectors
Tsw = transfer_sw(mu, nu, rho, sigma; dim=4, metric=:eta, k_name=:k, k_sq=:k2)
Tws = transfer_ws(mu, nu, rho, sigma; dim=4, metric=:eta, k_name=:k, k_sq=:k2)
```

These projectors satisfy the completeness relation `P2 + P1 + P0s + P0w = I` (symmetrized identity on rank-2 symmetric tensors) and are idempotent: `P_i * P_j = delta_{ij} P_i`.

> **See also:** [`examples/08_postquantum_gravity.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/08_postquantum_gravity.jl) for spin projectors applied to higher-derivative gravity propagators.

---

## 14. 3+1 Foliation & SVT Decomposition

TensorGR supports the 3+1 decomposition of spacetime tensors and their scalar-vector-tensor (SVT) decomposition, widely used in cosmological perturbation theory.

### Defining a Foliation

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # Define a standard 3+1 foliation: temporal=0, spatial=[1,2,3]
    fol = define_foliation!(reg, :flat31;
        manifold=:M4, temporal=0, spatial=[1, 2, 3])
end
```

### Splitting Spacetime Indices

Replace an abstract spacetime index with a sum over temporal and spatial components:

```julia
# Split a single index
expr = Tensor(:V, [up(:a)])
split_expr = split_spacetime(expr, :a, fol)
# V^0 + V^1 + V^2 + V^3

# Split all free indices in an expression
full_split = split_all_spacetime(expr, fol)
```

### SVT Decomposition of Metric Perturbations

The standard SVT decomposition of a symmetric rank-2 perturbation `h_{ab}` is:
- `h_{00} = 2 Phi` (scalar)
- `h_{0i} = d_i B + S_i` (scalar + transverse vector)
- `h_{ij} = 2 psi delta_{ij} + 2 d_i d_j E + d_i F_j + d_j F_i + hTT_{ij}` (scalar + vector + TT tensor)

In Bardeen gauge (`B = E = F = 0`), this simplifies to:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)
    fol = define_foliation!(reg, :flat31; manifold=:M4, temporal=0, spatial=[1,2,3])

    # SVT fields with default names: phi, B, psi, E, S, F, hTT
    fields = SVTFields()  # or SVTFields(; phi=:Phi, psi=:Psi, ...)
end
```

### End-to-End Pipeline

The `foliate_and_decompose` function chains all steps (split, substitute, constrain, collect):

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)
    fol = define_foliation!(reg, :flat31; manifold=:M4, temporal=0, spatial=[1,2,3])

    # Some expression involving h_{ab}
    mp = define_metric_perturbation!(reg, :g, :h)
    expr = expand_perturbation(Tensor(:Ein, [down(:a), down(:b)]), mp, 1)

    # Decompose into SVT sectors
    sectors = foliate_and_decompose(expr, :h; foliation=fol, gauge=:bardeen)
    # Returns Dict{Symbol, TensorExpr}:
    #   :scalar      => terms with only Phi, psi
    #   :vector      => terms with only S
    #   :tensor      => terms with only hTT
    #   :mixed       => cross-sector terms (should vanish)
    #   :pure_scalar => terms with no SVT fields
end
```

### Component Classification

Inspect how indices were classified after splitting:

```julia
# Check if an index is temporal or spatial
is_temporal_component(idx, fol)
is_spatial_component(idx, fol)

# Get the component pattern for a tensor
t = Tensor(:h, [down(:_0), down(:_1)])
pattern = component_pattern(t, fol)
# [:temporal, :spatial]
```

> **See also:** [`examples/11_6deriv_gravity_dS.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/11_6deriv_gravity_dS.jl) for a full worked example with perturbation theory and cosmological backgrounds.
