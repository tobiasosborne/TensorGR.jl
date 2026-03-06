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
