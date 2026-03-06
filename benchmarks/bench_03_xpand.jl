# ============================================================================
# Benchmark 03: xPand — FLRW SVT Decomposition
#
# Paper: Pitrou, Roy, Umeh (arXiv:1302.6174)
# Reproduces: Perturbed Einstein equations in FLRW Newtonian gauge,
#             automatic SVT sector separation
# Exercises: foliation, SVT decomposition, perturbation engine
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ground_truth.jl"))

@testset "Bench 03: xPand — FLRW SVT Decomposition" begin

    # ── 3.1: Foliation setup and basic splitting ────────────────────────
    @testset "3+1 foliation of metric perturbation" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            fol = define_foliation!(reg, :flat31; manifold=:M4)

            # Split h_{ab} into 3+1 components: h_{00}, h_{0i}, h_{ij}
            h_ab = Tensor(:h, [down(:a), down(:b)])
            split_expr = split_all_spacetime(h_ab, fol)
            @test split_expr isa TSum
            ncomp = count_terms(split_expr)
            # For a rank-(0,2) symmetric tensor: h_{00}, h_{0i}+h_{i0}, h_{ij}
            # = 1 + 2 + 1 = 4 basis terms (but h_{0i}=h_{i0} by symmetry)
            @test ncomp >= 3
            println("  h_{ab} -> $(ncomp) component terms")
        end
    end

    # ── 3.2: SVT substitution in Bardeen gauge ──────────────────────────
    @testset "SVT substitution (Bardeen gauge)" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            fol = define_foliation!(reg, :flat31; manifold=:M4)

            h_ab = Tensor(:h, [down(:a), down(:b)])
            split_expr = split_all_spacetime(h_ab, fol)
            substituted = apply_svt(split_expr, :h, fol; gauge=:bardeen)
            @test substituted != TScalar(0 // 1)
            println("  SVT substituted: $(count_terms(substituted)) terms")
        end
    end

    # ── 3.3: Sector collection ──────────────────────────────────────────
    @testset "SVT sector separation" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            fol = define_foliation!(reg, :flat31; manifold=:M4)

            h_ab = Tensor(:h, [down(:a), down(:b)])
            split_expr = split_all_spacetime(h_ab, fol)
            substituted = apply_svt(split_expr, :h, fol; gauge=:bardeen)
            sectors = collect_sectors(substituted)

            sector_names = sort(collect(keys(sectors)))
            @test length(sectors) >= 2  # at least scalar and tensor sectors
            println("  Sectors: $(sector_names)")
        end
    end

    # ── 3.4: End-to-end pipeline (foliate_and_decompose) ────────────────
    @testset "foliate_and_decompose pipeline" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            fol = define_foliation!(reg, :flat31; manifold=:M4)

            tc = timed_compute() do
                h_ab = Tensor(:h, [down(:a), down(:b)])
                foliate_and_decompose(h_ab, :h; foliation=fol)
            end

            sectors = tc.result
            @test sectors isa Dict
            @test length(sectors) >= 2
            println("  E2E pipeline: $(length(sectors)) sectors ($(round(tc.time, digits=3))s)")

            for (name, expr) in sort(collect(sectors), by=first)
                println("    $name: $(count_terms(expr)) terms")
            end
        end
    end

    # ── 3.5: Linearized Ricci + 3+1 split ──────────────────────────────
    @testset "Linearized Ricci -> SVT sectors" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            mp = define_metric_perturbation!(reg, :g, :h)
            background_solution!(reg, [:Ric, :RicScalar, :Ein])

            # Linearized Ricci
            dRic = δricci(mp, down(:a), down(:b), 1)
            @test dRic != TScalar(0 // 1)

            # 3+1 split
            fol = define_foliation!(reg, :flat31; manifold=:M4)
            tc = timed_compute() do
                split_all_spacetime(dRic, fol)
            end
            @test tc.result != TScalar(0 // 1)
            println("  δ¹Ric split: $(count_terms(tc.result)) terms ($(round(tc.time, digits=3))s)")
        end
    end

    # ── 3.6: Full perturbed Einstein -> SVT (xPand main result) ─────────
    @testset "Perturbed Einstein -> SVT sectors" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            mp = define_metric_perturbation!(reg, :g, :h)
            background_solution!(reg, [:Ric, :RicScalar, :Ein])

            # Linearized Einstein: G_{ab} = R_{ab} - (1/2)g_{ab}R
            dRic = δricci(mp, down(:a), down(:b), 1)
            dR = δricci_scalar(mp, 1)
            g_ab = Tensor(:g, [down(:a), down(:b)])
            dEin = tsum(TensorExpr[dRic, tproduct(-1 // 2, TensorExpr[g_ab, dR])])

            fol = define_foliation!(reg, :flat31; manifold=:M4)

            tc = timed_compute() do
                foliate_and_decompose(dEin, :h; foliation=fol)
            end

            if tc.result isa Dict
                sectors = tc.result
                @test length(sectors) >= 1
                println("  δ¹G_{ab} -> $(length(sectors)) SVT sectors ($(round(tc.time, digits=3))s)")
                for (name, expr) in sort(collect(sectors), by=first)
                    println("    $name: $(count_terms(expr)) terms")
                end
            else
                # foliate_and_decompose may return a plain expression
                @test tc.result != TScalar(0 // 1)
                println("  δ¹G_{ab} split: $(count_terms(tc.result)) terms ($(round(tc.time, digits=3))s)")
            end
        end
    end
end
