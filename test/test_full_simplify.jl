using Test
using TensorGR
using TensorGR: free_indices

@testset "FullSimplification (TGR-ulo.1)" begin

    function fs_registry(; dim::Int=4)
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M dim=dim metric=g registry=reg
            define_curvature_tensors!(reg, :M, :g)
        end
        reg
    end

    # ── Gauss-Bonnet → 0 in d=4 via DDI ────────────────────────────
    @testset "E_4 → 0 in d=4 (DDI auto-applied)" begin
        reg = fs_registry(dim=4)
        with_registry(reg) do
            E4 = euler_density(:g; dim=4, registry=reg)
            result = full_simplify(E4; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── Scalar identity: R stays R ──────────────────────────────────
    @testset "RicScalar → RicScalar" begin
        reg = fs_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            result = full_simplify(R; registry=reg)
            @test result isa Tensor
            @test result.name == :RicScalar
        end
    end

    # ── Without DDIs: E_4 has 3 terms ──────────────────────────────
    @testset "E_4 without DDIs: 3 terms" begin
        reg = fs_registry(dim=4)
        with_registry(reg) do
            E4 = euler_density(:g; dim=4, registry=reg)
            result = full_simplify(E4; registry=reg, use_ddis=false)
            @test result isa TSum
            @test length(result.terms) == 3
        end
    end

    # ── Basis :riemann explicitly ───────────────────────────────────
    @testset "basis=:riemann" begin
        reg = fs_registry(dim=4)
        with_registry(reg) do
            used = Set{Symbol}()
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used); push!(used, b)
            Ric_sq = Tensor(:Ric, [down(a), down(b)]) * Tensor(:Ric, [up(a), up(b)])
            result = full_simplify(Ric_sq; registry=reg, basis=:riemann)
            @test isempty(free_indices(result))
        end
    end

    # ── Basis :weyl explicitly ──────────────────────────────────────
    @testset "basis=:weyl" begin
        reg = fs_registry(dim=4)
        with_registry(reg) do
            used = Set{Symbol}()
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used); push!(used, b)
            c = fresh_index(used); push!(used, c)
            d = fresh_index(used); push!(used, d)
            Riem_sq = Tensor(:Riem, [down(a), down(b), down(c), down(d)]) *
                      Tensor(:Riem, [up(a), up(b), up(c), up(d)])
            result = full_simplify(Riem_sq; registry=reg, basis=:weyl)
            @test isempty(free_indices(result))
            # Weyl basis: should contain Weyl tensors
            str = string(result)
            @test occursin("Weyl", str) || occursin("Ric", str)
        end
    end

    # ── basis=:auto picks shorter result ────────────────────────────
    @testset "basis=:auto" begin
        reg = fs_registry(dim=4)
        with_registry(reg) do
            used = Set{Symbol}()
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used); push!(used, b)
            Ric_sq = Tensor(:Ric, [down(a), down(b)]) * Tensor(:Ric, [up(a), up(b)])
            result = full_simplify(Ric_sq; registry=reg, basis=:auto)
            @test isempty(free_indices(result))
        end
    end

    # ── Invalid basis throws ────────────────────────────────────────
    @testset "invalid basis throws" begin
        reg = fs_registry()
        R = Tensor(:RicScalar, TIndex[])
        @test_throws ArgumentError full_simplify(R; registry=reg, basis=:invalid)
    end

    # ── Weyl tensor input (to_riemann normalizes first) ─────────────
    @testset "Weyl input normalized to Riemann" begin
        reg = fs_registry(dim=4)
        with_registry(reg) do
            used = Set{Symbol}()
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used); push!(used, b)
            c = fresh_index(used); push!(used, c)
            d = fresh_index(used); push!(used, d)
            W = Tensor(:Weyl, [down(a), down(b), down(c), down(d)]) *
                Tensor(:Weyl, [up(a), up(b), up(c), up(d)])
            result = full_simplify(W; registry=reg, basis=:riemann)
            @test isempty(free_indices(result))
            # In Riemann basis, should not contain Weyl
            str = string(result)
            @test !occursin("Weyl", str)
        end
    end

    # ── Dimension auto-inference ────────────────────────────────────
    @testset "dim auto-inferred from registry" begin
        reg = fs_registry(dim=4)
        with_registry(reg) do
            E4 = euler_density(:g; dim=4, registry=reg)
            # Don't pass dim explicitly — should auto-infer d=4
            result = full_simplify(E4; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── Zero expression stays zero ──────────────────────────────────
    @testset "zero stays zero" begin
        reg = fs_registry()
        with_registry(reg) do
            result = full_simplify(TScalar(0 // 1); registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

end
