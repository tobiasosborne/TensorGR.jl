using Documenter
using TensorGR

makedocs(
    sitename = "TensorGR.jl",
    modules = [TensorGR],
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API Reference" => [
            "Types & Registry" => "api/types.md",
            "Algebra" => "api/algebra.md",
            "GR Objects" => "api/gr.md",
            "Spinor Formalism" => "api/spinors.md",
            "Curvature Invariants" => "api/invariants.md",
            "Matter Fields" => "api/matter.md",
            "Scalar-Tensor Theories" => "api/scalar_tensor.md",
            "Perturbation Theory" => "api/perturbation.md",
            "Bimetric Gravity" => "api/bimetric.md",
            "Metric-Affine Geometry" => "api/metric_affine.md",
            "Component Calculations" => "api/components.md",
            "Spherical Harmonics" => "api/harmonics.md",
            "Feynman Rules & EFT" => "api/feynman.md",
            "PPN Formalism" => "api/ppn.md",
            "Geodesics" => "api/geodesics.md",
            "Exterior Calculus" => "api/exterior.md",
            "3+1 Foliation" => "api/foliation.md",
            "SVT Decomposition" => "api/svt.md",
            "Quadratic Action" => "api/action.md",
            "Hamiltonian / ADM" => "api/hamiltonian.md",
            "Covariant Phase Space" => "api/phase_space.md",
            "Fermions & Clifford" => "api/fermions.md",
            "Frame Bundle / Tetrads" => "api/tetrads.md",
            "Metric Ansatz" => "api/ansatz.md",
            "Advanced Features" => "api/advanced.md",
        ],
        "xperm.c Internals" => "xperm_internals.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
)
