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

            h_ab = Tensor(:h, [down(:a), down(:b)])
            split_expr = split_all_spacetime(h_ab, fol)
            @test split_expr isa TSum
            @test count_terms(split_expr) == XPAND_SPLIT_HAB_TERMS
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
            @test count_terms(substituted) == XPAND_SVT_SUBSTITUTED_TERMS
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

            @test Set(keys(sectors)) == XPAND_SECTOR_NAMES
        end
    end

    # ── 3.4: End-to-end pipeline (foliate_and_decompose) ────────────────
    @testset "foliate_and_decompose pipeline" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            fol = define_foliation!(reg, :flat31; manifold=:M4)

            h_ab = Tensor(:h, [down(:a), down(:b)])
            sectors = foliate_and_decompose(h_ab, :h; foliation=fol)

            @test sectors isa Dict
            @test Set(keys(sectors)) == XPAND_E2E_SECTOR_NAMES
        end
    end

    # ── 3.5: Linearized Ricci + 3+1 split ──────────────────────────────
    @testset "Linearized Ricci -> 3+1 split" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            mp = define_metric_perturbation!(reg, :g, :h)
            background_solution!(reg, [:Ric, :RicScalar, :Ein])

            dRic = δricci(mp, down(:a), down(:b), 1)
            @test dRic != TScalar(0 // 1)

            fol = define_foliation!(reg, :flat31; manifold=:M4)
            result = split_all_spacetime(dRic, fol)
            @test result != TScalar(0 // 1)
            @test count_terms(result) == XPAND_DRIC_SPLIT_TERMS
        end
    end

    # ── 3.6: Full perturbed Einstein -> SVT ──────────────────────────────
    @testset "Perturbed Einstein -> SVT sectors" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            mp = define_metric_perturbation!(reg, :g, :h)
            background_solution!(reg, [:Ric, :RicScalar, :Ein])

            dRic = δricci(mp, down(:a), down(:b), 1)
            dR = δricci_scalar(mp, 1)
            g_ab = Tensor(:g, [down(:a), down(:b)])
            dEin = tsum(TensorExpr[dRic, tproduct(-1 // 2, TensorExpr[g_ab, dR])])

            fol = define_foliation!(reg, :flat31; manifold=:M4)
            result = foliate_and_decompose(dEin, :h; foliation=fol)

            if result isa Dict
                # Should produce sectors
                @test length(result) >= 3
                # Each sector must be non-trivial
                for (name, expr) in result
                    if name != :pure_scalar  # pure_scalar may have 0 or many terms
                        @test count_terms(expr) > 0
                    end
                end
            else
                @test result != TScalar(0 // 1)
            end
        end
    end
end
