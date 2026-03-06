# ============================================================================
# Benchmark 11: Superfield Integrals — High-Rank Canonicalization
#
# Paper: Green, Peeters, Stahn (arXiv:hep-th/0506161)
# Reproduces: High-rank tensor contractions in 10D supergravity
# Exercises: xperm.c canonicalization at scale, high-dimensional manifolds
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))

@testset "Bench 11: Superfield — High-Rank Canonicalization" begin
    reg = TensorRegistry()
    with_registry(reg) do
        # 10D manifold — use register_manifold! directly for >6 indices
        idx10 = [:A,:B,:C,:D,:E,:F,:H,:I,:J,:K,:L,:M,:N,:P,:Q,:R,:S,:T,:U,:V]
        register_manifold!(reg, ManifoldProperties(:M10, 10, :G, :d, idx10))
        register_tensor!(reg, TensorProperties(
            name=:G, manifold=:M10, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        define_curvature_tensors!(reg, :M10, :G)

        @testset "10D Riemann contraction" begin
            Riem_down = Tensor(:Riem, [down(:A), down(:B), down(:C), down(:D)])
            Riem_up = Tensor(:Riem, [up(:A), up(:B), up(:C), up(:D)])
            kretschner = Riem_down * Riem_up
            @test kretschner isa TProduct
            result = simplify(kretschner)
            @test result != TScalar(0 // 1)
            println("  R_{ABCD}R^{ABCD} in 10D: $(count_terms(result)) terms")
        end

        @testset "Triple Riemann contraction" begin
            R1 = Tensor(:Riem, [down(:A), down(:B), down(:C), down(:D)])
            R2 = Tensor(:Riem, [up(:A), up(:B), down(:E), down(:F)])
            R3 = Tensor(:Riem, [up(:C), up(:D), up(:E), up(:F)])
            triple = R1 * R2 * R3

            tc = timed_compute() do
                simplify(triple)
            end
            @test tc.result != TScalar(0 // 1)
            println("  R R R (triple, 10D): $(count_terms(tc.result)) terms ($(round(tc.time, digits=3))s)")
        end
    end
end
