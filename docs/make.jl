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
            "Matter Fields" => "api/matter.md",
            "Perturbation Theory" => "api/perturbation.md",
            "Component Calculations" => "api/components.md",
            "Geodesics" => "api/geodesics.md",
            "Exterior Calculus" => "api/exterior.md",
            "3+1 Foliation" => "api/foliation.md",
            "SVT Decomposition" => "api/svt.md",
            "Quadratic Action" => "api/action.md",
            "Metric Ansatz" => "api/ansatz.md",
            "Advanced Features" => "api/advanced.md",
        ],
        "xperm.c Internals" => "xperm_internals.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
)
