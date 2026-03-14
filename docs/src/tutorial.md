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

## 3. Curvature Algebra

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

### Kretschner Scalar

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
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)

    # Define g -> g + eps*h
    mp = define_metric_perturbation!(reg, :g, :h)
end
```

### Inverse Metric Perturbation

The exported function uses the unicode delta character:

```julia
# First-order perturbation of the inverse metric:
# delta(g^{ab}) = -g^{ac} g^{bd} h_{cd}
delta_ginv = δinverse_metric(mp, up(:a), up(:b), 1)

# Second order: delta^2(g^{ab})
delta2_ginv = δinverse_metric(mp, up(:a), up(:b), 2)
```

### Christoffel and Curvature Perturbations

```julia
# delta(Gamma^a_{bc}) at first order
dGamma = δchristoffel(mp, up(:a), down(:b), down(:c), 1)
# = (1/2) g^{ad} (d_b h_{cd} + d_c h_{bd} - d_d h_{bc})

# delta(R_{ab}) at first order
dRic = δricci(mp, down(:a), down(:b), 1)

# delta(R) at first order
dR = δricci_scalar(mp, 1)
```

### Background Field Equations

Set background curvature to zero for vacuum spacetimes:

```julia
background_solution!(reg, [:Ric, :RicScalar, :Ein])
# Now simplify(Tensor(:Ric, [down(:a), down(:b)])) => 0
```

### Gauge Transformations

Under an infinitesimal diffeomorphism generated by a vector field xi,
the metric perturbation transforms as h -> h + Lie_xi(g):

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    mp = define_metric_perturbation!(reg, :g, :h)

    # The gauge vector
    xi = Tensor(:xi, [up(:a)])

    # Gauge-transformed perturbation at first order
    h_ab = Tensor(:h, [down(:a), down(:b)])
    h_new = gauge_transformation(h_ab, xi, :g; order=1)
    # h_new = h_{ab} + nabla_a xi_b + nabla_b xi_a
end
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

## 7. Particle Spectrum Analysis

TensorGR provides tools for analyzing the particle content of gravitational
theories by decomposing the quadratic action into spin sectors using the
Barnes-Rivers projection operators.

### Barnes-Rivers Spin Projectors

In momentum space, a symmetric rank-2 field h decomposes into irreducible spin
components via the transverse and longitudinal projectors:

```julia
mu, nu, rho, sigma = down(:mu), down(:nu), down(:rho), down(:sigma)

# Transverse projector: theta_{mu nu} = eta_{mu nu} - k_mu k_nu / k^2
theta = theta_projector(mu, nu; metric=:g, k_name=:k, k_sq=:k2)

# Longitudinal projector: omega_{mu nu} = k_mu k_nu / k^2
omega = omega_projector(mu, nu; k_name=:k, k_sq=:k2)
```

The six Barnes-Rivers operators decompose h into spin-2, spin-1, and two
spin-0 sectors. They are idempotent and sum to the symmetrized identity:

```julia
# Spin-2 (transverse-traceless graviton)
P2 = spin2_projector(mu, nu, rho, sigma; dim=4, metric=:g, k_name=:k, k_sq=:k2)

# Spin-1 (vector)
P1 = spin1_projector(mu, nu, rho, sigma; metric=:g, k_name=:k, k_sq=:k2)

# Spin-0 scalar (transverse trace)
P0s = spin0s_projector(mu, nu, rho, sigma; dim=4, metric=:g, k_name=:k, k_sq=:k2)

# Spin-0 w (longitudinal)
P0w = spin0w_projector(mu, nu, rho, sigma; k_name=:k, k_sq=:k2)
```

### Position-Space Kernel Extraction

Given a quadratic action `S^(2) = integral h(x) K(partial) h(x) dx`, the kinetic
kernel K determines the propagator and the particle content. The function
`extract_kernel_direct` extracts K directly from a position-space bilinear
expression, correctly handling the two-momentum physics where the two field
copies carry opposite momenta (+k and -k):

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    mp = define_metric_perturbation!(reg, :g, :h)

    # Build the Einstein-Hilbert bilinear: h^{ab} delta^(1) G_{ab}
    d1R_ab = simplify(δricci(mp, down(:a), down(:b), 1))
    d1R    = simplify(δricci_scalar(mp, 1))
    h_up   = Tensor(:h, [up(:a), up(:b)])
    trh    = Tensor(:h, [up(:c), down(:c)])

    EH_bilinear = h_up * d1R_ab - (1//2) * trh * d1R

    # Extract the kinetic kernel (converts derivatives to momenta)
    K = extract_kernel_direct(EH_bilinear, :h; registry=reg)
    # K is a KineticKernel with bilinear terms
end
```

### Spin Projection

Project the extracted kernel onto each spin sector to get the form factors
(functions of k-squared):

```julia
with_registry(reg) do
    f2  = spin_project(K, :spin2;  dim=4, metric=:g, k_name=:k, k_sq=:k2, registry=reg)
    f0s = spin_project(K, :spin0s; dim=4, metric=:g, k_name=:k, k_sq=:k2, registry=reg)
    f1  = spin_project(K, :spin1;  dim=4, metric=:g, k_name=:k, k_sq=:k2, registry=reg)
    f0w = spin_project(K, :spin0w; dim=4, metric=:g, k_name=:k, k_sq=:k2, registry=reg)

    # Evaluate at a specific k^2 value
    val_2  = _eval_spin_scalar(f2, 1.0)   # spin-2 form factor at k^2=1
    val_0s = _eval_spin_scalar(f0s, 1.0)  # spin-0s form factor at k^2=1
    # For the Fierz-Pauli (EH) kernel: val_2 = 2.5, val_0s = -1.0
end
```

### Pre-Built Momentum-Space Kernels

TensorGR provides pre-built Fourier-space kernels for common quadratic
Lagrangians, useful for cross-checking:

```julia
with_registry(reg) do
    # Fierz-Pauli (Einstein-Hilbert) kernel
    K_FP = build_FP_momentum_kernel(reg)

    # R^2 kernel: (delta R)^2
    K_R2 = build_R2_momentum_kernel(reg)

    # Ric^2 kernel: (delta R_{ab})^2
    K_Ric2 = build_Ric2_momentum_kernel(reg)

    # Combined 4-derivative kernel: kappa*K_FP - 2*alpha1*K_R2 - 2*alpha2*K_Ric2
    K_4d = build_6deriv_flat_kernel(reg; kappa=1//1, alpha1=1//10, alpha2=1//5)
end
```

### De Sitter Spectrum (Bueno-Cano Analysis)

For higher-derivative gravity on a de Sitter background, TensorGR implements
the Bueno-Cano parameterization to compute the physical spectrum analytically:

```julia
# Compute spectrum of R + alpha1*R^2 + alpha2*Ric^2 on de Sitter
result = dS_spectrum_6deriv(kappa=1.0, alpha1=0.01, alpha2=0.02, Lambda=0.1)

result.kappa_eff_inv   # inverse effective Newton constant
result.m2_graviton    # massive spin-2 mass squared
result.m2_scalar      # spin-0 mass squared
result.flat_f2        # flat-space spin-2 form factor coefficients (c1, c2)
result.flat_f0        # flat-space spin-0 form factor coefficients (c1, c2)
```

The Bueno-Cano parameters for individual Lagrangian terms can also be
computed separately:

```julia
p_EH  = bc_EH(1.0, 0.1)          # Einstein-Hilbert: S = kappa*R
p_R2  = bc_R2(0.01, 0.1)         # R^2 term
p_Ric = bc_RicSq(0.02, 0.1)      # Ric_{ab} Ric^{ab} term
p_total = p_EH + p_R2 + p_Ric    # additive composition

# Predict spin-projected form factors from BC parameters
ff = bc_to_form_factors(p_total, 1.0, 0.1)
ff.f_spin2    # spin-2 form factor at k^2=1
ff.f_spin0s   # spin-0s form factor at k^2=1
```

---

## 8. Geodesic Integration

TensorGR provides the `GeodesicEquation` type and an ODE right-hand side for
numerically integrating geodesics on a given spacetime. The system is designed
for use with DifferentialEquations.jl.

### Setting Up the Geodesic Equation

Provide a metric function that returns the metric and its inverse at a spacetime
point. The Christoffel symbols are computed automatically via central finite
differences, or you can supply an analytic function:

```julia
using TensorGR, LinearAlgebra

# Schwarzschild metric in (t, r, theta, phi) coordinates
function schwarzschild_metric(x; M=1.0)
    r, theta = x[2], x[3]
    f = 1.0 - 2.0 * M / r
    sth2 = sin(theta)^2
    g    = diagm([-f, 1.0/f, r^2, r^2 * sth2])
    ginv = diagm([-1.0/f, f, 1.0/r^2, 1.0/(r^2 * sth2)])
    (g, ginv)
end

# Build the geodesic equation (Christoffels computed numerically)
geq = setup_geodesic(schwarzschild_metric; dim=4, is_timelike=true)
```

### Evaluating the ODE Right-Hand Side

The state vector `u` has length `2*dim`: the first `dim` entries are the
position x^mu, and the next `dim` are the 4-velocity dx^mu/d(tau).
The function `geodesic_rhs!` computes the geodesic acceleration
dv^mu/d(tau) = -Gamma^mu_{alpha beta} v^alpha v^beta:

```julia
# ISCO orbit at r = 6M in Schwarzschild
M_bh = 1.0
r_isco = 6.0 * M_bh
f_isco = 1.0 - 2.0 * M_bh / r_isco   # 2/3

E = sqrt(8.0 / 9.0)            # energy per unit mass
L = 2.0 * sqrt(3.0)            # angular momentum per unit mass
vt = E / f_isco
vphi = L / r_isco^2

x0 = [0.0, r_isco, pi/2, 0.0]
v0 = [vt, 0.0, 0.0, vphi]

# Evaluate the RHS at the initial condition
u = vcat(x0, v0)
du = similar(u)
geodesic_rhs!(du, u, geq, 0.0)
# du[1:4] = velocities, du[5:8] = accelerations (zero for circular orbit)
```

### Full Integration with DifferentialEquations.jl

When DifferentialEquations.jl is available, use `integrate_geodesic` for a
complete solution:

```julia
using DifferentialEquations

# Integrate for one orbit in proper time
tau_orbit = 2.0 * pi / vphi
sol = integrate_geodesic(geq, x0, v0, (0.0, tau_orbit);
                         abstol=1e-12, reltol=1e-12)

sol.retcode   # :Success
sol.t         # proper time values
sol.x         # positions at each time step
sol.v         # velocities at each time step

# The radius should remain at r=6M (circular orbit)
max_r_err = maximum(abs(pt[2] - r_isco) for pt in sol.x)
# max_r_err < 1e-6
```

The integrator preserves the norm g_{mu nu} v^mu v^nu = -1 for timelike
geodesics and = 0 for null geodesics.

---

## 9. Equation of State and TOV

TensorGR provides equation-of-state types and a Tolman-Oppenheimer-Volkoff
(TOV) integrator for computing the structure of static, spherically symmetric
stars in general relativity.

### Equation of State Types

Three EOS types are available, all subtypes of `EquationOfState`:

```julia
# Barotropic: p = w * rho (constant equation-of-state parameter)
dust      = BarotropicEOS(0)       # w = 0 (pressureless dust)
radiation = BarotropicEOS(1//3)    # w = 1/3 (radiation)
dark_e    = BarotropicEOS(-1)      # w = -1 (cosmological constant)

# Polytropic: p = K * rho^gamma
neutron_star_eos = PolytropicEOS(1//10, 2//1)

# Tabular: linear interpolation from data
tab_eos = TabularEOS([0.0, 1.0, 2.0, 4.0],   # rho values
                     [0.0, 0.5, 1.5, 3.5])    # p values
```

Evaluate pressure and sound speed:

```julia
pressure(radiation, 3.0)      # => 1.0  (= 1/3 * 3)
sound_speed(radiation, 1.0)   # => 1//3 (= dp/drho = w)

pressure(neutron_star_eos, 8.0)           # K * rho^gamma
sound_speed(neutron_star_eos, 1.0)        # K * gamma * rho^(gamma-1)

pressure(tab_eos, 0.5)                    # => 0.25 (linear interpolation)
```

### Setting Up the TOV System

The TOV equations describe hydrostatic equilibrium for a spherically symmetric
star. The state vector is `[m(r), p(r)]` (enclosed mass and pressure as functions
of the areal radius r):

```
dm/dr = 4 pi r^2 rho
dp/dr = -(rho + p)(m + 4 pi r^3 p) / (r(r - 2m))
```

```julia
# Polytropic neutron star with central density rho_c = 1.0
eos = PolytropicEOS(1//10, 2//1)
tov = setup_tov(eos, 1.0)

tov.rho_c    # central density
tov.p_c      # central pressure (from EOS)
tov.u0       # initial state [m(r0), p(r0)] from series expansion
tov.r0       # starting radius (default 1e-4, avoids r=0 singularity)
```

### Evaluating the TOV Right-Hand Side

The function `tov_rhs!` is an in-place ODE right-hand side compatible with
DifferentialEquations.jl:

```julia
du = zeros(2)
tov_rhs!(du, tov.u0, tov, tov.r0)
# du[1] = dm/dr at r = r0
# du[2] = dp/dr at r = r0
```

### Integration with DifferentialEquations.jl

To find the stellar structure, integrate outward from the center until the
pressure drops to zero (the stellar surface):

```julia
using DifferentialEquations

# Define a callback to stop integration when p <= 0 (surface)
surface_cb = ContinuousCallback(
    (u, r, integrator) -> u[2],   # trigger when p = 0
    (integrator) -> terminate!(integrator))

prob = ODEProblem(tov_rhs!, tov.u0, (tov.r0, 20.0), tov)
sol = solve(prob, Tsit5(); callback=surface_cb, abstol=1e-10, reltol=1e-10)

R_star = sol.t[end]      # stellar radius
M_star = sol[1, end]      # total mass
```

---

## 10. Hypersurfaces and Boundaries

TensorGR supports codimension-1 hypersurface embeddings for boundary terms,
the ADM decomposition, Israel junction conditions, and thin-shell formalism.

### Defining a Hypersurface

```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # Define a spacelike hypersurface with timelike normal (signature=-1)
    hs = define_hypersurface!(reg, :Sigma;
        ambient=:M4, metric=:g,
        normal_name=:n, extrinsic_name=:K, induced_name=:gamma,
        signature=-1)

    # This registers:
    #   - Unit normal n_a (with n_a n^a = -1 rule)
    #   - Induced metric gamma_{ab} (symmetric)
    #   - Extrinsic curvature K_{ab} (symmetric)
    #   - Projector P^a_b onto the hypersurface
end
```

### Gauss-Codazzi Relations

The Gauss equation relates the intrinsic Riemann tensor of the hypersurface
to the ambient Riemann tensor and the extrinsic curvature:

```julia
# Gauss equation: {}^(3)R_{abcd} = R_{abcd} + sigma*(K_{ac}K_{bd} - K_{ad}K_{bc})
gauss = gauss_equation(down(:a), down(:b), down(:c), down(:d);
                       Riem=:Riem, K=:K, signature=-1)
```

The Codazzi equation relates the covariant derivative of the extrinsic curvature
to the normal projection of the ambient Riemann tensor:

```julia
# Codazzi: D_a K_{bc} - D_b K_{ac} = sigma * R_{dabc} n^d
codazzi = codazzi_equation(down(:a), down(:b), down(:c);
                           Riem=:Riem, K=:K, normal=:n, signature=-1)
```

Both are also available as rewrite rules:

```julia
rules = gauss_codazzi_rules(; Riem=:Riem, K=:K, normal=:n,
                              signature=-1, intrinsic_Riem=:Riem3)
for r in rules
    register_rule!(reg, r)
end
```

### GHY Boundary Term

The Gibbons-Hawking-York boundary term ensures a well-posed Dirichlet
variational problem for the Einstein-Hilbert action. The GHY term is
`S_GHY = 2 K`, where K is the trace of the extrinsic curvature:

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)
    define_hypersurface!(reg, :Sigma; ambient=:M4, metric=:g, signature=-1)

    S_ghy = ghy_boundary_term(reg, :Sigma)
    # Returns 2 * g^{ab} K_{ab}
end
```

### Integration by Parts with Boundary Terms

When integrating by parts in the bulk, the boundary contribution is often
needed. The function `ibp_with_boundary` returns both the bulk and boundary
parts:

```julia
with_registry(reg) do
    phi = Tensor(:phi, TIndex[])
    T   = Tensor(:T, [up(:a)])
    expr = tproduct(1//1, TensorExpr[TDeriv(down(:a), phi), T])

    bulk, boundary = ibp_with_boundary(expr, :phi)
    # bulk     = -phi * d_a(T^a)  (IBP-transferred expression)
    # boundary = expr - bulk       (total-derivative surface contribution)
end
```

### Israel Junction Conditions

For a thin shell separating two spacetime regions, the Israel junction
conditions relate the jump in extrinsic curvature across the shell to the
surface stress-energy tensor:

```
[K_{ab}] - gamma_{ab} [K] = -8 pi S_{ab}
```

```julia
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)
    define_hypersurface!(reg, :Sigma; ambient=:M4, metric=:g, signature=-1)

    # Register junction condition tensors
    jd = define_junction!(reg, :Sigma;
                          K_plus=:Kp, K_minus=:Km, S=:S,
                          induced_metric=:gamma_jc)
    # jd is a JunctionData struct
end
```

Build the Israel equation as a tensor expression (LHS = 0):

```julia
eq = israel_equation(:Kp, :Km, :gamma_jc)
# Returns [K_{ab}] - gamma_{ab} [K] + 8*pi*S_{ab}
```

Alternatively, solve for the surface stress-energy directly:

```julia
S_expr = junction_stress_energy(:Kp, :Km, :gamma_jc)
# Returns S_{ab} = -(1/8pi) ([K_{ab}] - gamma_{ab} [K])
```

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

## 8a. Equation Solver (`solve_tensors`)

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

## Metric Engine

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

## Topological Invariants

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

## Perturbation Theory (Advanced)

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

> **See also:** [`examples/11_6deriv_gravity_dS.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/11_6deriv_gravity_dS.jl) for perturbation theory on a de Sitter background.

---

## 3+1 Foliation & SVT Decomposition

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

> **See also:** [`examples/11_6deriv_gravity_dS.jl`](https://github.com/tobiasosborne/TensorGR.jl/blob/master/examples/11_6deriv_gravity_dS.jl) for a full worked example with perturbation theory and cosmological backgrounds.
