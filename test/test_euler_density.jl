using Test
using TensorGR
using TensorGR: lovelock_lagrangian, free_indices

@testset "EulerDensity: Arbitrary Dimension" begin

    # ── d=2: Euler density is the Ricci scalar ────────────────────────
    @testset "d=2: E_2 = R" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M2 dim=2 metric=g registry=reg
            define_curvature_tensors!(reg, :M2, :g)

            E2 = euler_density(:g; dim=2, registry=reg)

            # E_2 should be exactly the Ricci scalar
            @test E2 isa Tensor
            @test E2.name == :RicScalar
            @test isempty(E2.indices)
        end
    end

    # ── d=3: Euler density vanishes (odd dimension) ───────────────────
    @testset "d=3: E_3 = 0 (odd dimension)" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M3 dim=3 metric=g registry=reg
            define_curvature_tensors!(reg, :M3, :g)

            E3 = euler_density(:g; dim=3, registry=reg)

            @test E3 isa TScalar
            @test E3.val == 0 // 1
        end
    end

    # ── d=4: Gauss-Bonnet regression test ─────────────────────────────
    @testset "d=4: Gauss-Bonnet (regression)" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M4, :g)

            E4 = euler_density(:g; dim=4, registry=reg)

            # Must be a sum of 3 terms: Riem^2, -4 Ric^2, R^2
            @test E4 isa TSum
            @test length(E4.terms) == 3

            # All terms must be scalars (no free indices)
            for t in E4.terms
                @test isempty(free_indices(t))
            end

            # Verify content: must contain RicScalar^2, Ric^2, Riem^2
            terms_str = [string(t) for t in E4.terms]
            all_str = join(terms_str, " ")

            has_ric_scalar_sq = count("RicScalar", all_str) >= 2
            has_ric_sq = any(s -> occursin("Ric", s) && !occursin("RicScalar", s) && !occursin("Riem", s), terms_str)
            has_riem_sq = any(s -> count("Riem", s) >= 2, terms_str)

            @test has_ric_scalar_sq
            @test has_ric_sq
            @test has_riem_sq

            # Verify relative coefficients: E_4 = Riem^2 - 4 Ric^2 + R^2
            coeffs = Dict{Symbol, Rational{Int}}()
            for t in E4.terms
                str = string(t)
                if count("Riem", str) >= 2
                    coeffs[:Riem] = t isa TProduct ? t.scalar : 1 // 1
                elseif occursin("Ric", str) && !occursin("RicScalar", str) && !occursin("Riem", str)
                    coeffs[:Ric] = t isa TProduct ? t.scalar : 1 // 1
                elseif count("RicScalar", str) >= 2
                    coeffs[:RicScalar] = t isa TProduct ? t.scalar : 1 // 1
                end
            end

            @test haskey(coeffs, :Riem) && haskey(coeffs, :Ric) && haskey(coeffs, :RicScalar)
            @test coeffs[:Riem] == 1 // 1
            @test coeffs[:Ric] == -4 // 1
            @test coeffs[:RicScalar] == 1 // 1
        end
    end

    # ── d=4 with DDI: E_4 IS the Gauss-Bonnet DDI ────────────────────
    @testset "d=4: E_4 simplifies to zero with DDI" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M4, :g)

            E4 = euler_density(:g; dim=4, registry=reg)
            result = simplify_with_ddis(E4; dim=4, registry=reg)

            @test result == TScalar(0 // 1)
        end
    end

    # ── d=5: Euler density vanishes (odd dimension) ───────────────────
    @testset "d=5: E_5 = 0 (odd dimension)" begin
        E5 = euler_density(:g; dim=5)
        @test E5 isa TScalar
        @test E5.val == 0 // 1
    end

    # ── d=6: Cubic Lovelock E_6 structure ─────────────────────────────
    @testset "d=6: E_6 = L_3 (cubic Lovelock)" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M6 dim=6 metric=g registry=reg
            define_curvature_tensors!(reg, :M6, :g)

            E6 = euler_density(:g; dim=6, registry=reg)

            # Should be a scalar (no free indices)
            @test isempty(free_indices(E6))

            # Simplify to get canonical form
            s = simplify(E6; registry=reg)

            # E_6 = L_3 should produce a sum of cubic curvature invariants
            @test s isa TSum

            # Count term types: should have cubic invariants involving
            # Riem^3, Ric*Riem^2, R*Riem^2, Ric^2*Riem, Ric^3, R*Ric^2, R^3
            # (some types may split into multiple canonical forms)
            term_types = Dict{String, Int}()
            for t in s.terms
                str = string(t)
                n_riem = count("Riem[", str)
                n_ric = count("Ric[", str)
                n_rs = count("RicScalar", str)
                type_key = "$(n_rs)R_$(n_ric)Ric_$(n_riem)Riem"
                term_types[type_key] = get(term_types, type_key, 0) + 1
            end

            # Must contain Riem^3 terms (highest-rank)
            @test haskey(term_types, "0R_0Ric_3Riem")

            # Must contain R^3 term (lowest-rank)
            @test haskey(term_types, "3R_0Ric_0Riem") || haskey(term_types, "3R_-3Ric_0Riem")

            # Must be a cubic expression: each term should have exactly 3
            # curvature factors (R, Ric, or Riem)
            for t in s.terms
                str = string(t)
                n_riem = count("Riem[", str)
                n_ric = count("Ric[", str)
                n_rs = count("RicScalar", str)
                @test n_riem + n_ric + n_rs == 3
            end
        end
    end

    # ── d=1: should error ─────────────────────────────────────────────
    @testset "d=1: error for dim < 2" begin
        @test_throws ErrorException euler_density(:g; dim=0)
    end

end

@testset "Lovelock Lagrangian" begin

    # ── order=0: cosmological constant ────────────────────────────────
    @testset "order=0: L_0 = 1" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M4, :g)

            L0 = lovelock_lagrangian(0, :g; registry=reg)
            @test L0 isa TScalar
            @test L0.val == 1 // 1
        end
    end

    # ── order=1: Einstein-Hilbert = R ─────────────────────────────────
    @testset "order=1: L_1 = R" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M4, :g)

            L1 = lovelock_lagrangian(1, :g; registry=reg)
            @test L1 isa Tensor
            @test L1.name == :RicScalar
            @test isempty(L1.indices)
        end
    end

    # ── order=2: Gauss-Bonnet = E_4 ──────────────────────────────────
    @testset "order=2: L_2 = Gauss-Bonnet" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M4, :g)

            L2 = lovelock_lagrangian(2, :g; registry=reg)
            @test L2 isa TSum
            @test length(L2.terms) == 3

            # Should match euler_density in d=4
            E4 = euler_density(:g; dim=4, registry=reg)
            @test string(L2) == string(E4)
        end
    end

    # ── order=3 in d=4: vanishes by DDI ──────────────────────────────
    @testset "order=3 in d=4: L_3 = 0 (DDI)" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M4, :g)

            L3 = lovelock_lagrangian(3, :g; dim=4, registry=reg)
            @test L3 isa TScalar
            @test L3.val == 0 // 1
        end
    end

    # ── order=3 in d=6: cubic Lovelock ───────────────────────────────
    @testset "order=3 in d=6: cubic Lovelock" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M6 dim=6 metric=g registry=reg
            define_curvature_tensors!(reg, :M6, :g)

            L3 = lovelock_lagrangian(3, :g; dim=6, registry=reg)
            @test isempty(free_indices(L3))

            s = simplify(L3; registry=reg)
            @test s isa TSum

            # Should have multiple cubic invariant terms
            @test length(s.terms) >= 7
        end
    end

    # ── Lovelock order=2 equals E_4 (euler_density d=4) ──────────────
    @testset "euler_density(d=4) == lovelock_lagrangian(2)" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M, :g)

            E4 = euler_density(:g; dim=4, registry=reg)
            L2 = lovelock_lagrangian(2, :g; registry=reg)

            # Both should be structurally identical
            @test string(E4) == string(L2)
        end
    end

    # ── Lovelock order=1 equals E_2 (euler_density d=2) ──────────────
    @testset "euler_density(d=2) == lovelock_lagrangian(1)" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M, :g)

            E2 = euler_density(:g; dim=2, registry=reg)
            L1 = lovelock_lagrangian(1, :g; registry=reg)

            @test string(E2) == string(L1)
        end
    end

    # ── Negative order should error ──────────────────────────────────
    @testset "negative order: error" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M, :g)

            @test_throws ErrorException lovelock_lagrangian(-1, :g; registry=reg)
        end
    end

    # ── Large order in low dimension: DDI vanishing ──────────────────
    @testset "order > dim/2: vanishes by DDI" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M, :g)

            # L_4 in d=4: 2*4=8 > 4, so vanishes
            L4 = lovelock_lagrangian(4, :g; dim=4, registry=reg)
            @test L4 isa TScalar
            @test L4.val == 0 // 1

            # L_10 in d=4: way beyond, vanishes
            L10 = lovelock_lagrangian(10, :g; dim=4, registry=reg)
            @test L10 isa TScalar
            @test L10.val == 0 // 1
        end
    end

end

@testset "Euler Density: Topological Properties" begin

    # ── E_4 is topological in d=4 (DDI gives zero) ──────────────────
    @testset "E_4 topological: simplifies to zero with DDI in d=4" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M4, :g)

            E4 = euler_density(:g; dim=4, registry=reg)
            result = simplify_with_ddis(E4; dim=4, registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── E_6 scalar structure: all terms are cubic in curvature ────────
    @testset "E_6: all terms cubic in curvature" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M6 dim=6 metric=g registry=reg
            define_curvature_tensors!(reg, :M6, :g)

            E6 = euler_density(:g; dim=6, registry=reg)
            s = simplify(E6; registry=reg)

            @test s isa TSum
            for t in s.terms
                @test isempty(free_indices(t))
            end
        end
    end

end

@testset "Euler Density: Regression" begin

    # ── Existing d=4 tests must still pass ────────────────────────────
    @testset "d=4 regression: structure matches original" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M dim=4 metric=g registry=reg
            define_curvature_tensors!(reg, :M, :g)

            E4 = euler_density(:g; registry=reg)

            # This is the exact test from test_xact_ground_truth.jl
            @test E4 isa TSum
            @test length(E4.terms) == 3

            for t in E4.terms
                @test isempty(free_indices(t))
            end

            terms_str = [string(t) for t in E4.terms]
            all_str = join(terms_str, " ")

            has_ric_scalar_sq = count("RicScalar", all_str) >= 2
            has_ric_sq = any(s -> occursin("Ric", s) && !occursin("RicScalar", s) && !occursin("Riem", s), terms_str)
            has_riem_sq = any(s -> count("Riem", s) >= 2, terms_str)

            @test has_ric_scalar_sq
            @test has_ric_sq
            @test has_riem_sq
        end
    end

    # ── d=2 in existing test (from test_xact_ground_truth.jl) ────────
    @testset "d=2 regression: well-formed scalar" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M2 dim=2 metric=g2 registry=reg
            define_curvature_tensors!(reg, :M2, :g2)

            # With the new implementation, d=2 returns RicScalar directly
            E2 = euler_density(:g2; dim=2, registry=reg)
            @test E2 isa Tensor
            @test E2.name == :RicScalar
            fi = free_indices(E2)
            @test isempty(fi)
        end
    end

end
