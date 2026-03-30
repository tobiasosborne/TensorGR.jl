# REPL Tensor Mode

TensorGR provides an interactive REPL mode for direct manipulation of tensor expressions using LaTeX-style input. The mode is activated by pressing `\` (backslash) at the Julia REPL.

## Activation

```julia
using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
end

init_repl_mode!(reg)
# Press \ to enter tensor mode, backspace on empty line to exit
```

Or auto-activate via environment variable:

```julia
ENV["TENSORGR_REPL"] = "1"
using TensorGR
```

## Commands

All commands apply to the last result (`%`) by default, or accept an explicit expression:

| Command | Description |
|---------|-------------|
| `simplify` | Full simplification pipeline |
| `canon` | Canonicalize only (xperm, no collection) |
| `expand` | Expand products |
| `contract` | Contract metrics |
| `covd` | Expand covariant derivatives to Christoffel symbols |
| `perturb` | Linearize expression (first-order metric perturbation) |
| `latex` | Print LaTeX form |
| `indices` | Show free indices |
| `terms` | Count terms |
| `level2` | Simplify with Bianchi + cyclic identities |
| `to_riemann` | Convert to Riemann basis |
| `to_ricci` | Convert to Ricci basis |

## Name Resolution

Standard LaTeX names are automatically resolved to registered tensors:

| LaTeX Input | Resolved Name | Condition |
|-------------|---------------|-----------|
| `R_{abcd}` | `Riem` | 4 indices |
| `R_{ab}` | `Ric` | 2 indices |
| `R` | `RicScalar` | 0 indices |
| `G_{ab}` | `Ein` | 2 indices |
| `C_{abcd}` | `Weyl` | 4 indices |
| `\Gamma^a_{bc}` | `Christoffel` | 3 indices |

## Variables

Store and recall named expressions:

```
tensor> expr = R_{abcd}
tensor> result = simplify expr
tensor> result
```

## Pipe Syntax

Chain commands with `|`:

```
tensor> g^{ab} R_{ab} | contract | simplify
tensor> x = R_{abcd} R^{abcd} | simplify
```

## Numbered History

Results are numbered. Use `%N` to recall:

```
tensor> R_{abcd}         [1] Riem_{a b c d}
tensor> g_{ab}           [2] g_{a b}
tensor> simplify %1      [3] ...
```

## Workspace Commands

| Command | Description |
|---------|-------------|
| `vars` | List all stored variables |
| `info` | Inspect expression (indices, terms, tensors, symmetries) |
| `registry` | Show registered manifolds and tensors |
| `define T_{ab}` | Register a new tensor in the active registry |
| `sub pattern -> replacement` | Apply substitution to last result |

## Tab Completion

Press Tab to complete:
- Command names (`sim` -> `simplify`)
- Variable names
- Registered tensor names
- LaTeX names (`\alp` -> `\alpha`)

## Derivative Shorthands

- `\partial_a T^{bc}` produces a partial derivative
- `\nabla_a T^{bc}` resolves to the active covariant derivative

## API Reference

```@docs
init_repl_mode!
set_tensor_registry!
```
