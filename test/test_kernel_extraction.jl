#= Tests for position-space kernel extraction and spin projection.
#
# Validates extract_kernel_direct (two-momentum-correct) against:
# - FP position-space Lagrangian → must match Fourier-space FP kernel
# - Analytic R² and Ric² kernels → spin projections at k²=1
# - Full 4-derivative flat spectrum → Buoninfante form factors
# - Phase correction for asymmetric derivative distributions
=#

using Test
using TensorGR

function _setup_flat_registry()
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)
        @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    end
    reg
end

@testset "Kernel Extraction" begin

    # ── Phase correction ─────────────────────────────────────────────

    @testset "two-momentum phase" begin
        reg = _setup_flat_registry()
        with_registry(reg) do
            h1 = Tensor(:h, [down(:a), down(:b)])
            h2 = Tensor(:h, [down(:e), down(:f)])

            # n_L=1, n_R=1 → phase=+1 (symmetric)
            sym = TProduct(1 // 1, TensorExpr[TDeriv(down(:c), h1), TDeriv(down(:d), h2)])
            K = extract_kernel_direct(sym, :h; registry=reg)
            @test length(K.terms) == 1
            @test K.terms[1].coeff isa TProduct && K.terms[1].coeff.scalar > 0

            # n_L=2, n_R=0 → phase=-1 (asymmetric)
            asym_L = TProduct(1 // 1, TensorExpr[
                TDeriv(down(:d), TDeriv(down(:c), h1)), h2])
            K2 = extract_kernel_direct(asym_L, :h; registry=reg)
            @test length(K2.terms) == 1
            @test K2.terms[1].coeff isa TProduct && K2.terms[1].coeff.scalar < 0

            # n_L=0, n_R=2 → phase=-1 (asymmetric)
            asym_R = TProduct(1 // 1, TensorExpr[
                h1, TDeriv(down(:d), TDeriv(down(:c), h2))])
            K3 = extract_kernel_direct(asym_R, :h; registry=reg)
            @test length(K3.terms) == 1
            @test K3.terms[1].coeff isa TProduct && K3.terms[1].coeff.scalar < 0

            # n=1 (odd) → dropped
            odd = TProduct(1 // 1, TensorExpr[TDeriv(down(:c), h1), h2])
            K4 = extract_kernel_direct(odd, :h; registry=reg)
            @test length(K4.terms) == 0

            # total derivative ∂(bilinear) → dropped
            bilin = TProduct(1 // 1, TensorExpr[TDeriv(down(:c), h1), TDeriv(down(:d), h2)])
            td = TDeriv(down(:e), bilin)
            K5 = extract_kernel_direct(td, :h; registry=reg)
            @test length(K5.terms) == 0
        end
    end

    # ── Fierz-Pauli position-space ───────────────────────────────────

    @testset "FP position-space matches Fourier-space" begin
        reg = _setup_flat_registry()
        with_registry(reg) do
            # FP Lagrangian: 4 terms, all n_L=n_R=1
            t1 = (1 // 2) * TDeriv(down(:c), Tensor(:h, [down(:a), down(:b)])) *
                             TDeriv(up(:c), Tensor(:h, [up(:a), up(:b)]))
            t2 = (-1 // 1) * TDeriv(down(:b), Tensor(:h, [up(:a), up(:b)])) *
                              TDeriv(down(:c), Tensor(:h, [up(:c), down(:a)]))
            t3 = TDeriv(down(:a), Tensor(:h, [up(:a), up(:b)])) *
                 TDeriv(down(:b), Tensor(:h, [up(:c), down(:c)]))
            t4 = (-1 // 2) * TDeriv(down(:a), Tensor(:h, [up(:b), down(:b)])) *
                              TDeriv(up(:a), Tensor(:h, [up(:c), down(:c)]))

            K_pos = extract_kernel_direct(t1 + t2 + t3 + t4, :h; registry=reg)
            K_FP  = build_FP_momentum_kernel(reg)
            kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

            for k2 in [0.5, 1.0, 2.0]
                for s in [:spin2, :spin1, :spin0s, :spin0w]
                    pos = _eval_spin_scalar(spin_project(K_pos, s; kw...), k2)
                    fp  = _eval_spin_scalar(spin_project(K_FP,  s; kw...), k2)
                    @test abs(pos - fp) < 1e-8
                end
            end
        end
    end

    # ── FP absolute values ───────────────────────────────────────────

    @testset "FP spin projections" begin
        reg = _setup_flat_registry()
        with_registry(reg) do
            K = build_FP_momentum_kernel(reg)
            kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)
            @test abs(_eval_spin_scalar(spin_project(K, :spin2;  kw...), 1.0) - 2.5) < 1e-10
            @test abs(_eval_spin_scalar(spin_project(K, :spin1;  kw...), 1.0))       < 1e-10
            @test abs(_eval_spin_scalar(spin_project(K, :spin0s; kw...), 1.0) + 1.0) < 1e-10
            @test abs(_eval_spin_scalar(spin_project(K, :spin0w; kw...), 1.0))       < 1e-10
        end
    end

    # ── Linearized Einstein tensor = FP ──────────────────────────────

    @testset "h^{ab} delta1G_{ab} = FP kernel" begin
        reg = _setup_flat_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)
            set_vanishing!(reg, :Ric); set_vanishing!(reg, :RicScalar); set_vanishing!(reg, :Riem)
            kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

            d1R_ab = simplify(δricci(mp, down(:a), down(:b), 1); registry=reg)
            d1R    = simplify(δricci_scalar(mp, 1); registry=reg)
            h_up   = Tensor(:h, [up(:a), up(:b)])
            trh    = Tensor(:g, [up(:p), up(:q)]) * Tensor(:h, [down(:p), down(:q)])

            K = extract_kernel_direct(h_up * d1R_ab - (1 // 2) * trh * d1R, :h; registry=reg)
            K_FP = build_FP_momentum_kernel(reg)

            for s in [:spin2, :spin1, :spin0s, :spin0w]
                v  = _eval_spin_scalar(spin_project(K,    s; kw...), 1.0)
                fp = _eval_spin_scalar(spin_project(K_FP, s; kw...), 1.0)
                @test abs(v - fp) < 1e-8
            end
        end
    end

    # ── (δR)² and (δRic)² ────────────────────────────────────────────

    @testset "(delta R)^2 spin projections" begin
        reg = _setup_flat_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)
            set_vanishing!(reg, :Ric); set_vanishing!(reg, :RicScalar); set_vanishing!(reg, :Riem)
            kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

            d1R = simplify(δricci_scalar(mp, 1); registry=reg)
            K = extract_kernel_direct(d1R * d1R, :h; registry=reg)
            @test abs(_eval_spin_scalar(spin_project(K, :spin2;  kw...), 1.0))       < 1e-8
            @test abs(_eval_spin_scalar(spin_project(K, :spin0s; kw...), 1.0) - 3.0) < 1e-8
            @test abs(_eval_spin_scalar(spin_project(K, :spin1;  kw...), 1.0))       < 1e-8
            @test abs(_eval_spin_scalar(spin_project(K, :spin0w; kw...), 1.0))       < 1e-8
        end
    end

    @testset "(delta Ric)^2 spin projections" begin
        reg = _setup_flat_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)
            set_vanishing!(reg, :Ric); set_vanishing!(reg, :RicScalar); set_vanishing!(reg, :Riem)
            kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

            d1Ric_ab = simplify(δricci(mp, down(:a), down(:b), 1); registry=reg)
            d1Ric_cd = simplify(δricci(mp, down(:c), down(:d), 1); registry=reg)
            dRic_sq = d1Ric_ab * d1Ric_cd *
                      Tensor(:g, [up(:a), up(:c)]) * Tensor(:g, [up(:b), up(:d)])
            K = extract_kernel_direct(dRic_sq, :h; registry=reg)
            @test abs(_eval_spin_scalar(spin_project(K, :spin2;  kw...), 1.0) - 1.25) < 1e-8
            @test abs(_eval_spin_scalar(spin_project(K, :spin0s; kw...), 1.0) - 1.0)  < 1e-8
            @test abs(_eval_spin_scalar(spin_project(K, :spin1;  kw...), 1.0))        < 1e-8
            @test abs(_eval_spin_scalar(spin_project(K, :spin0w; kw...), 1.0))        < 1e-8
        end
    end

    # ── Full 4-derivative spectrum cross-check ───────────────────────
    # K = κ K_EH - 2α₁ K_{R²} - 2α₂ K_{Ric²}
    # Verified against analytic Buoninfante form factors (2012.11829 Eq.2.13)

    @testset "4-derivative flat spectrum" begin
        reg = _setup_flat_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)
            set_vanishing!(reg, :Ric); set_vanishing!(reg, :RicScalar); set_vanishing!(reg, :Riem)
            kw = (dim=4, metric=:g, k_name=:k, k_sq=:k², registry=reg)

            d1R_ab = simplify(δricci(mp, down(:a), down(:b), 1); registry=reg)
            d1R    = simplify(δricci_scalar(mp, 1); registry=reg)
            d1Ric_cd = simplify(δricci(mp, down(:c), down(:d), 1); registry=reg)
            h_up = Tensor(:h, [up(:a), up(:b)])
            trh  = Tensor(:g, [up(:p), up(:q)]) * Tensor(:h, [down(:p), down(:q)])

            K_EH = extract_kernel_direct(
                h_up * d1R_ab - (1 // 2) * trh * d1R, :h; registry=reg)
            K_Rsq = extract_kernel_direct(d1R * d1R, :h; registry=reg)
            K_Ric2 = extract_kernel_direct(
                d1R_ab * d1Ric_cd *
                Tensor(:g, [up(:a), up(:c)]) * Tensor(:g, [up(:b), up(:d)]),
                :h; registry=reg)

            for (α₁, α₂) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 3)]
                kernels = KineticKernel[K_EH]
                α₁ != 0 && push!(kernels, scale_kernel(K_Rsq,  Rational{Int}(-2α₁)))
                α₂ != 0 && push!(kernels, scale_kernel(K_Ric2, Rational{Int}(-2α₂)))
                K = combine_kernels(kernels)

                ref = flat_6deriv_spin_projections(reg;
                    κ=1 // 1, α₁=Rational{Int}(α₁), α₂=Rational{Int}(α₂))
                for s in [:spin2, :spin1, :spin0s, :spin0w]
                    v = _eval_spin_scalar(spin_project(K, s; kw...), 1.0)
                    r = _eval_spin_scalar(getfield(ref, s), 1.0)
                    @test abs(v - r) < 1e-6
                end
            end
        end
    end

    # ── Edge cases ───────────────────────────────────────────────────

    @testset "edge cases" begin
        reg = _setup_flat_registry()
        with_registry(reg) do
            @test extract_kernel_direct(TScalar(0 // 1), :h; registry=reg).terms |> isempty
            @test extract_kernel_direct(TScalar(42 // 1), :h; registry=reg).terms |> isempty
            @test extract_kernel_direct(Tensor(:h, [down(:a), down(:b)]), :h; registry=reg).terms |> isempty
        end
    end

end  # main testset
