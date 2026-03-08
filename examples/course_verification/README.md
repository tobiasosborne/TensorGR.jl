# Course Verification: Introduction to General Relativity

Systematic computational verification of the GR lecture course using TensorGR.jl.
Each script corresponds to a lecture (or group of lectures) and verifies every
nontrivial calculation using Julia — no `println` cheating.

## Lecture Map

| Script | Lectures | Topic | Method |
|--------|----------|-------|--------|
| `lec07_derivative_operators.jl` | 7-8 | Covariant derivatives, parallel transport, Christoffel symbols | Abstract algebra |
| `lec09_10_curvature.jl` | 9-11 | Riemann tensor, Ricci, Einstein, Bianchi identities | Abstract algebra |
| `lec12_lie_derivatives.jl` | 12 | Lie derivatives, Killing vectors, Newtonian limit | Abstract algebra |
| `lec13_einstein_equations.jl` | 13 | Einstein field equations, trace, dust, conservation | Abstract algebra |
| `lec14_gravitational_radiation.jl` | 14 | Linearised gravity, gauge transformations, wave equation | Perturbation theory |
| `lec17_friedmann.jl` | 15-17 | FLRW metric, Christoffel, Ricci, Friedmann equations | Symbolic components |
| `lec19_schwarzschild.jl` | 18-19 | Schwarzschild metric, vacuum solution, Kretschmann | Symbolic components |
| `lec21_22_geodesics.jl` | 20-22 | Killing vectors, effective potential, ISCO, light bending | Symbolic algebra |

## Lectures Not Covered Computationally

| Lectures | Topic | Reason |
|----------|-------|--------|
| 1-2 | Prerelativity gravitation, equivalence principle | Conceptual (no tensor calculations) |
| 3-6 | Manifolds, tangent spaces, tensors, flows | Definitions and proofs (no verifiable calculations) |
| 15-16 | Spaces of constant curvature, cosmological dynamics | Mostly qualitative; calculations covered in lec17 |
| 20 | Interior solutions, TOV equation | Requires ODE integration (not tensor algebra) |
| 23 | Retrospective and outlook | Conceptual review |

## Running

```bash
# Run a single verification
julia --project examples/course_verification/lec09_10_curvature.jl

# Run all (from project root)
for f in examples/course_verification/lec*.jl; do
    echo "=== $(basename $f) ==="
    julia --project "$f"
    echo
done
```

## Methods Used

- **Abstract algebra**: TensorGR's symbolic tensor engine (simplify, canonicalize, contract)
- **Perturbation theory**: expand_perturbation, δRiemann, δRicci, gauge_transformation
- **Symbolic components**: SymbolicMetric + Symbolics.jl for Christoffel/Riemann/Ricci from explicit metrics
- **Symbolic algebra**: Symbolics.jl for effective potential, ISCO conditions, etc.
