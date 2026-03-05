using Test
using TensorGR

@testset "TensorGR.jl" begin
    include("test_types.jl")
    include("test_registry.jl")
    include("test_show.jl")
    include("test_display.jl")
    include("test_xperm.jl")
    include("test_indices.jl")
    include("test_walk.jl")
    include("test_escape_hatch.jl")
    include("test_arithmetic.jl")
    include("test_macros.jl")
    include("test_contraction.jl")
    include("test_canonicalize.jl")
    include("test_derivatives.jl")
    include("test_gr_objects.jl")
    include("test_linearize.jl")
    include("test_svt.jl")
    include("test_quadratic_action.jl")
    include("test_integration.jl")
    # Phase 0: Infrastructure
    include("test_rules.jl")
    include("test_scalar_algebra.jl")
    include("test_simplify.jl")
    # Phase 1: Full Canonicalization
    include("test_phase1.jl")
    # Phase 2: Covariant Derivatives
    include("test_phase2.jl")
    # Phase 3: Perturbation Theory
    include("test_phase3.jl")
    # Phase 4: Component Calculations
    include("test_phase4.jl")
    # Phase 5+6: Curvature Algebra + Exterior Calculus
    include("test_phase56.jl")
    # Curvature algebra conversions
    include("test_conversions.jl")
    # Phase 7: Advanced Features
    include("test_phase7.jl")
    # Phase 8: Hardening & Extensions
    include("test_phase8.jl")
    # Integration V2: Full pipeline
    include("test_integration_v2.jl")
    # Symmetrize / Antisymmetrize
    include("test_symmetrize.jl")
    # New features (streams A-K)
    include("test_new_features.jl")
end
