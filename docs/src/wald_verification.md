# Wald's *General Relativity* — Verification Guide

This page walks through key identities from Robert Wald's *General Relativity*
(University of Chicago Press, 1984) and shows how to verify them in TensorGR.jl
using the tensor REPL mode.

All examples can be typed directly at the `tensor>` prompt (press `\` to enter
tensor mode) or used programmatically via `parse_tex` and `simplify`.

## Setup

```julia
using TensorGR
@manifold M4 dim=4 metric=g
init_repl_mode!()    # press \ to enter tensor mode
```

## Chapter 3: Curvature

### Metric properties

The metric trace equals the manifold dimension (Eq 3.1.14):

```
tensor> g^{ab} g_{ab}
  g^a^b g_a_b
tensor> simplify %
  4
```

The metric inverse contracts to the Kronecker delta (Eq 3.1.12):

```
tensor> g^{ac} g_{cb}
  g^a^c g_c_b
tensor> simplify %
  δ^a_b
```

### Riemann tensor symmetries

The Riemann tensor has three algebraic symmetries. All are verified by
showing that the antisymmetric/symmetric combination simplifies to zero.

**First pair antisymmetry** (Eq 3.2.14): ``R_{abcd} = -R_{bacd}``

```julia
R_abcd = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
R_bacd = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])
simplify(R_abcd + R_bacd)   # → 0
```

**Second pair antisymmetry** (Eq 3.2.14): ``R_{abcd} = -R_{abdc}``

```julia
R_abcd = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
R_abdc = Tensor(:Riem, [down(:a), down(:b), down(:d), down(:c)])
simplify(R_abcd + R_abdc)   # → 0
```

**Pair exchange symmetry** (Eq 3.2.15): ``R_{abcd} = R_{cdab}``

```julia
R_abcd = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
R_cdab = Tensor(:Riem, [down(:c), down(:d), down(:a), down(:b)])
simplify(R_abcd - R_cdab)   # → 0
```

**First Bianchi identity** (Eq 3.2.16): ``R_{a[bcd]} = 0``

The cyclic sum ``R_{abcd} + R_{acdb} + R_{adbc} = 0`` requires the
Level 2 (Bianchi) simplification from the Invar module, which goes beyond
the basic `simplify` pipeline:

```julia
using TensorGR: simplify_level2
expr = R_abcd + R_acdb + R_adbc
simplify_level2(expr)   # → 0  (uses Bianchi rule)
```

### Ricci tensor

The Ricci tensor is the trace of the Riemann tensor (Eq 3.2.25),
and is symmetric:

```julia
R_ab = Tensor(:Ric, [down(:a), down(:b)])
R_ba = Tensor(:Ric, [down(:b), down(:a)])
simplify(R_ab - R_ba)   # → 0  (symmetry)
```

The Ricci scalar is the trace of the Ricci tensor (Eq 3.2.26):

```
tensor> g^{ab} R_{ab}
  g^a^b Ric_a_b
tensor> simplify %
  RicScalar
```

### Riemann contractions (Appendix C)

Single contraction gives the Ricci tensor:

```julia
Riem = Tensor(:Riem, [up(:a), down(:b), down(:a), down(:c)])
simplify(Riem)   # → Ric_b_c
```

Double contraction gives the Ricci scalar:

```julia
Riem = Tensor(:Riem, [up(:a), up(:b), down(:a), down(:b)])
simplify(Riem)   # → RicScalar
```

### Einstein tensor (Eq 3.2.27–3.2.30)

The Einstein tensor ``G_{ab} = R_{ab} - \frac{1}{2}g_{ab}R``:

```
tensor> R_{ab} - \frac{1}{2} g_{ab} R
  Ric_a_b + (-1//2) g_a_b RicScalar
```

It is symmetric (Eq 3.2.28):

```julia
G_ab = Tensor(:Ein, [down(:a), down(:b)])
G_ba = Tensor(:Ein, [down(:b), down(:a)])
simplify(G_ab - G_ba)   # → 0
```

### Weyl tensor (Section 3.4)

The Weyl tensor shares all symmetries of the Riemann tensor:

```julia
C_abcd = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
C_bacd = Tensor(:Weyl, [down(:b), down(:a), down(:c), down(:d)])
simplify(C_abcd + C_bacd)   # → 0  (antisymmetry)

C_cdab = Tensor(:Weyl, [down(:c), down(:d), down(:a), down(:b)])
simplify(C_abcd - C_cdab)   # → 0  (pair exchange)
```

## Chapter 6: Schwarzschild Solution

### Kretschner scalar (Eq 6.1.7)

Using symbolic components (requires `Symbolics.jl`):

```julia
using Symbolics
@variables r M
coords = [:t, :r, :theta, :phi]
f = 1 - 2M/r
diag = [-f, 1/f, r^2, r^2*sin(theta)^2]
sm = symbolic_diagonal_metric(coords, diag)
K = symbolic_kretschmann(sm)
simplify(K)   # → 48M²/r⁶
```

The abstract Schwarzschild potentials can be verified via the
Regge-Wheeler/Zerilli infrastructure:

```julia
V_RW = schwarzschild_rw_potential(2)
V_RW(6, 1)   # → 5/54  (at r=6M, M=1)
V_RW(2, 1)   # → 0     (vanishes at horizon)
```

## Chapter 7: Linearized Gravity

### First-order perturbation (Eq 7.5.5–7.5.7)

```julia
mp = define_metric_perturbation!(reg, :g, :h)
δ1Ric = δricci(mp, down(:a), down(:b), 1)    # δ¹R_{ab}
δ1R   = δricci_scalar(mp, 1)                   # δ¹R
```

Both are nonzero and have the correct free indices:

```julia
free_indices(δ1Ric)   # → [a↓, b↓]
```

## Chapter 10: Hypersurfaces

### Induced metric (Eq 10.2.13)

For a timelike hypersurface with unit normal ``n^a``:

```julia
γ = induced_metric_expr(down(:a), down(:b), :g, :n; signature=-1)
# → g_a_b + n_a n_b   (timelike: σ = -1, so -σ = +1)
```

### Gibbons-Hawking-York boundary term (Eq E.1.23)

```julia
define_hypersurface!(reg, :Sigma; ambient=:M4, metric=:g, signature=-1)
S_ghy = ghy_boundary_term(reg, :Sigma)
# → 2 g^a^b K_a_b   (trace of extrinsic curvature)
```

## Summary: What Simplifies Automatically

| Identity | Wald Eq | `simplify` | Notes |
|----------|---------|------------|-------|
| ``g^{ab}g_{ab} = 4`` | 3.1.14 | **Yes** | Metric trace |
| ``g^{ac}g_{cb} = \delta^a_b`` | 3.1.12 | **Yes** | Metric inverse |
| ``R_{abcd} = -R_{bacd}`` | 3.2.14 | **Yes** | xperm canonicalization |
| ``R_{abcd} = R_{cdab}`` | 3.2.15 | **Yes** | xperm canonicalization |
| ``R_{a[bcd]} = 0`` | 3.2.16 | Level 2 | Needs `simplify_level2` |
| ``R_{ab} = R_{ba}`` | 3.2.25 | **Yes** | xperm canonicalization |
| ``g^{ab}R_{ab} = R`` | 3.2.26 | **Yes** | Metric contraction |
| ``G_{ab} = G_{ba}`` | 3.2.28 | **Yes** | xperm canonicalization |
| ``G^a_a = -R`` | 3.2.30 | **No** | Needs Einstein trace rule |
| ``g^{ac}C_{abcd} = 0`` | 3.4 | **No** | Needs Weyl trace-free rule |
| ``\nabla^a G_{ab} = 0`` | 3.2.17 | **Yes** | Via `commute_covds` |
