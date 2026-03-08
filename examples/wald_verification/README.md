# Wald Verification Scripts

Standalone verification scripts that test fundamental GR identities using
TensorGR.jl's abstract tensor algebra. Each script sets up a manifold, metric,
and curvature tensors, then verifies standard identities by simplifying
expressions to zero.

## Running

Each script is standalone and can be run from the project root:

```bash
julia --project examples/wald_verification/01_covariant_derivative_identities.jl
julia --project examples/wald_verification/02_lie_derivatives.jl
julia --project examples/wald_verification/03_linearised_gravity.jl
julia --project examples/wald_verification/07_curvature_decomposition.jl
julia --project examples/wald_verification/08_exterior_calculus.jl
```

Or run all at once:

```bash
for f in examples/wald_verification/0*.jl; do
    echo "=== Running $f ==="
    julia --project "$f" || echo "FAILED: $f"
done
```

## Identity Reference Table

| Script | # | Identity | Wald Reference |
|--------|---|----------|----------------|
| 01 | 1 | Metric compatibility: nabla_a g_{bc} = 0 | Eq. 3.1.29 |
| 01 | 2 | Riemann antisymmetry: R_{abcd} = -R_{bacd} | Eq. 3.2.14 |
| 01 | 3 | Riemann pair symmetry: R_{abcd} = R_{cdab} | Eq. 3.2.15 |
| 01 | 4 | Riemann antisymmetry: R_{abcd} = -R_{abdc} | Eq. 3.2.14 |
| 01 | 5 | First Bianchi: R_{a[bcd]} = 0 | Eq. 3.2.16 |
| 01 | 6 | Ricci from Riemann: R^b_{abc} = R_{ac} | Eq. 3.2.25 |
| 01 | 7 | Contracted Bianchi: nabla^a G_{ab} = 0 | Eq. 3.2.30 |
| 01 | 8 | Ricci Bianchi: nabla^a R_{ab} = (1/2) nabla_b R | Eq. 3.2.29 |
| 01 | 9 | Ricci symmetry: R_{ab} = R_{ba} | Sec. 3.2 |
| 01 | 10 | Einstein symmetry: G_{ab} = G_{ba} | Sec. 3.2 |
| 02 | 1 | Lie derivative of metric: L_v g = nabla v + nabla v | Eq. C.3.6 |
| 02 | 2 | Lie bracket structure: [v,w]^a | Eq. C.1.1 |
| 02 | 3 | Lie bracket antisymmetry: [v,w] + [w,v] = 0 | Sec. C.1 |
| 02 | 4 | Lie derivative of scalar: L_v f = v^a d_a f | Eq. C.1.2 |
| 02 | 5 | Lie derivative of vector = Lie bracket | Eq. C.1.4 |
| 03 | 1 | Metric perturbation: delta(g_{ab}) = h_{ab} | Eq. 7.5.4 |
| 03 | 2 | Inverse metric perturbation | Eq. 7.5.8 |
| 03 | 3 | Linearised Riemann tensor | Eq. 7.5.14 |
| 03 | 4 | Linearised Christoffel symbol | Eq. 7.5.12 |
| 03 | 5 | Linearised Ricci tensor | Eq. 7.5.16 |
| 03 | 6 | Linearised Ricci scalar | Sec. 7.5 |
| 03 | 7 | Symmetry of linearised Ricci | Sec. 7.5 |
| 03 | 8 | Linearised Einstein tensor | Sec. 7.5 |
| 07 | 1 | Weyl decomposition roundtrip | Sec. 3.2 |
| 07 | 2 | Weyl trace-freeness: C^a_{bac} = 0 | Sec. 3.2 |
| 07 | 3 | Einstein-Ricci roundtrip | Eq. 3.2.28 |
| 07 | 4 | Einstein tensor definition | Eq. 3.2.28 |
| 07 | 5 | Decomposition structure check | Sec. 3.2 |
| 08 | 1 | d^2 = 0 for scalars | Appendix B |
| 08 | 2 | d^2 = 0 for 1-forms | Appendix B |
| 08 | 3 | Maxwell-Bianchi: dF = 0 | Appendix B |

## Lecture Mapping

These verifications align with a standard GR lecture sequence:

- **Lecture 1-2** (Manifolds, tensors): Script 01 (sections 1-4)
- **Lecture 3** (Curvature): Script 01 (sections 5-10), Script 07
- **Lecture 4** (Lie derivatives): Script 02
- **Lecture 5** (Differential forms): Script 08
- **Lecture 10+** (Linearised gravity): Script 03
