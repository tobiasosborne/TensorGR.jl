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
            "Perturbation Theory" => "api/perturbation.md",
            "Component Calculations" => "api/components.md",
            "Exterior Calculus" => "api/exterior.md",
        ],
        "xperm.c Internals" => "xperm_internals.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
)
