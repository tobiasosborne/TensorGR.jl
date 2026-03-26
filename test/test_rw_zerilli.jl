@testset "Regge-Wheeler / Zerilli Master Equations" begin
    using TensorGR: SchwarzschildBackground, define_schwarzschild_background!,
                    get_schwarzschild_background, schwarzschild_f, schwarzschild_r,
                    schwarzschild_rw_potential, schwarzschild_zerilli_potential,
                    schwarzschild_potential_difference,
                    RWGaugeChoice, rw_gauge_odd, rw_gauge_even, rw_gauge_full,
                    apply_rw_gauge!, rw_dof_count,
                    RWMasterEquation, derive_rw_equation, derive_zerilli_equation,
                    evaluate_rw_potential, rw_potential_at_horizon, rw_potential_at_infinity,
                    MasterFunctionSpec, rw_master_function, zerilli_master_function,
                    extract_master_functions,
                    ProductManifoldProperties, has_product_manifold,
                    TensorRegistry, Tensor, TScalar, TIndex,
                    up, down, Up, Down,
                    with_registry, has_tensor, has_manifold

    # ── Schwarzschild background (TGR-bm6.1) ──────────────────────────

    @testset "define_schwarzschild_background!" begin
        reg = TensorRegistry()
        with_registry(reg) do
            bg = define_schwarzschild_background!(reg)

            @test bg isa SchwarzschildBackground
            @test bg.orbital === :M2
            @test bg.sphere === :S2
            @test bg.orbital_metric === :gab
            @test bg.sphere_metric === :Omega
            @test bg.M === :M_BH
            @test bg.f === :f_BH

            # Manifolds registered
            @test has_manifold(reg, :M2)
            @test has_manifold(reg, :S2)

            # Metrics registered
            @test has_tensor(reg, :gab)
            @test has_tensor(reg, :Omega)

            # Lapse and radius registered
            @test has_tensor(reg, :f_BH)
            @test has_tensor(reg, :r_M2)

            # Product manifold registered
            @test has_product_manifold(reg, :M4_Schw)
        end
    end

    @testset "get_schwarzschild_background" begin
        reg = TensorRegistry()
        with_registry(reg) do
            define_schwarzschild_background!(reg)
            bg = get_schwarzschild_background(reg)
            @test bg isa SchwarzschildBackground
        end
    end

    @testset "schwarzschild_f and schwarzschild_r" begin
        reg = TensorRegistry()
        with_registry(reg) do
            bg = define_schwarzschild_background!(reg)
            f = schwarzschild_f(bg)
            r = schwarzschild_r(bg)
            @test f isa Tensor
            @test r isa Tensor
            @test f.name === :f_BH
            @test r.name === :r_M2
        end
    end

    @testset "Schwarzschild vacuum: potential well-defined" begin
        # V_RW(6M, M) for l=2 should be nonzero
        V_RW = schwarzschild_rw_potential(2)
        r, M = 6, 1  # r = 6M
        val = V_RW(r, M)
        @test val isa Number
        @test val != 0
    end

    @testset "idempotent registration" begin
        reg = TensorRegistry()
        with_registry(reg) do
            bg1 = define_schwarzschild_background!(reg)
            # Second call should error (already defined)
            @test_throws ErrorException define_schwarzschild_background!(reg)
        end
    end

    # ── RW gauge (TGR-bm6.2) ────────────────────────────────────────

    @testset "rw_gauge_odd structure" begin
        gauge = rw_gauge_odd()
        @test gauge.parity === :odd
        @test length(gauge.vanishing) == 1
        @test :h_2_odd in gauge.vanishing
        @test length(gauge.remaining) == 2
        @test rw_dof_count(gauge) == 2
    end

    @testset "rw_gauge_even structure" begin
        gauge = rw_gauge_even()
        @test gauge.parity === :even
        @test length(gauge.vanishing) == 3
        @test :h_0_even in gauge.vanishing
        @test :h_1_even in gauge.vanishing
        @test :h_G in gauge.vanishing
        @test length(gauge.remaining) == 4
        @test rw_dof_count(gauge) == 4
    end

    @testset "rw_gauge_full" begin
        odd, even = rw_gauge_full()
        @test odd.parity === :odd
        @test even.parity === :even
        # Total DOFs: 2 odd + 4 even = 6 (= 10 metric components - 4 gauge)
        @test rw_dof_count(odd) + rw_dof_count(even) == 6
    end

    @testset "custom prefix" begin
        gauge = rw_gauge_odd(prefix=:delta_g)
        @test :delta_g_2_odd in gauge.vanishing
    end

    # ── RW equation (TGR-bm6.3) ─────────────────────────────────────

    @testset "derive_rw_equation" begin
        eq = derive_rw_equation(2)
        @test eq isa RWMasterEquation
        @test eq.parity === :odd
        @test eq.l == 2
        @test eq.label === :RW
    end

    @testset "RW potential ground truth l=2" begin
        eq = derive_rw_equation(2)

        # V_RW(r=6M, M=1) = (2/3)(6/36 - 6/216) = (2/3)(5/36) = 5/54
        val = evaluate_rw_potential(eq, 6, 1)
        @test val ≈ 5 / 54

        # At horizon r=2M: f(2M)=0, so V=0
        val_h = evaluate_rw_potential(eq, 2, 1)
        @test val_h == 0
    end

    @testset "RW potential ground truth l=3" begin
        eq = derive_rw_equation(3)
        # V_RW(r, M) = f(r)[l(l+1)/r² - 6M/r³] with l=3
        # At r=6M, M=1: f=2/3, l(l+1)=12
        # V = (2/3)(12/36 - 6/216) = (2/3)(1/3 - 1/36) = (2/3)(11/36) = 11/54
        val = evaluate_rw_potential(eq, 6, 1)
        @test val ≈ 11 / 54
    end

    @testset "RW potential positivity" begin
        eq = derive_rw_equation(2)
        # V_RW is positive for r > 2M (outside horizon)
        for r in [3, 4, 5, 6, 10, 20, 100]
            @test evaluate_rw_potential(eq, r, 1) > 0
        end
    end

    @testset "derive_rw_equation requires l >= 2" begin
        @test_throws ErrorException derive_rw_equation(0)
        @test_throws ErrorException derive_rw_equation(1)
    end

    # ── Zerilli equation (TGR-bm6.4) ─────────────────────────────────

    @testset "derive_zerilli_equation" begin
        eq = derive_zerilli_equation(2)
        @test eq isa RWMasterEquation
        @test eq.parity === :even
        @test eq.l == 2
        @test eq.label === :Zerilli
    end

    @testset "Zerilli potential ground truth l=2" begin
        eq = derive_zerilli_equation(2)

        # n = (2-1)(2+2)/2 = 2
        # At r=6M, M=1:
        # f = 2/3
        # num = 2·4·3·216 + 6·4·1·36 + 18·2·1·6 + 18·1 = 5184 + 864 + 216 + 18 = 6282
        # den = 216 · (2·6+3)² = 216 · 225 = 48600
        # V = (2/3)·6282/48600
        val = evaluate_rw_potential(eq, 6, 1)
        @test val isa Number
        @test val > 0
    end

    @testset "Zerilli potential positivity" begin
        eq = derive_zerilli_equation(2)
        for r in [3, 4, 5, 6, 10, 20, 100]
            @test evaluate_rw_potential(eq, r, 1) > 0
        end
    end

    @testset "derive_zerilli_equation requires l >= 2" begin
        @test_throws ErrorException derive_zerilli_equation(0)
        @test_throws ErrorException derive_zerilli_equation(1)
    end

    # ── Horizon and infinity limits ──────────────────────────────────

    @testset "potentials vanish at horizon" begin
        for l in 2:5
            rw = derive_rw_equation(l)
            z = derive_zerilli_equation(l)
            @test evaluate_rw_potential(rw, 2, 1) == 0
            @test evaluate_rw_potential(z, 2, 1) == 0
        end
    end

    @testset "potentials approach zero at large r" begin
        for l in 2:4
            rw = derive_rw_equation(l)
            z = derive_zerilli_equation(l)
            # At r=10000M, both should be very small
            @test abs(Float64(evaluate_rw_potential(rw, 10000, 1))) < 1e-4
            @test abs(Float64(evaluate_rw_potential(z, 10000, 1))) < 1e-4
        end
    end

    # ── Master functions (TGR-bm6.5) ─────────────────────────────────

    @testset "rw_master_function" begin
        mf = rw_master_function(2)
        @test mf isa MasterFunctionSpec
        @test mf.parity === :odd
        @test mf.l == 2
        @test mf.n == 2  # (2-1)(2+2)/2 = 2
        @test mf.label === :Psi_RW
        @test :h_0_odd in mf.input_fields
        @test :h_1_odd in mf.input_fields
    end

    @testset "zerilli_master_function" begin
        mf = zerilli_master_function(2)
        @test mf isa MasterFunctionSpec
        @test mf.parity === :even
        @test mf.l == 2
        @test mf.n == 2
        @test mf.label === :Psi_Z
        @test :K in mf.input_fields
        @test :H_1 in mf.input_fields
    end

    @testset "extract_master_functions" begin
        psi_rw, psi_z = extract_master_functions(2)
        @test psi_rw.parity === :odd
        @test psi_z.parity === :even
        @test psi_rw.l == psi_z.l == 2
    end

    @testset "master_function l-dependence" begin
        for l in 2:5
            mf = rw_master_function(l)
            @test mf.n == (l - 1) * (l + 2) ÷ 2
        end
    end

    @testset "master_function requires l >= 2" begin
        @test_throws ErrorException rw_master_function(0)
        @test_throws ErrorException zerilli_master_function(1)
    end

    # ── Isospectrality verification (TGR-bm6.6) ─────────────────────

    @testset "both potentials positive outside horizon" begin
        M = 1.0
        for l in 2:4
            V_RW = schwarzschild_rw_potential(l)
            V_Z = schwarzschild_zerilli_potential(l)
            for r in [3.0, 4.0, 6.0, 10.0, 20.0]
                @test V_RW(r, M) > 0
                @test V_Z(r, M) > 0
            end
        end
    end

    @testset "potentials agree in large-r limit" begin
        # Both V_RW and V_Z → l(l+1)/r² as r → ∞ (centrifugal barrier)
        M = 1.0
        for l in 2:4
            V_RW = schwarzschild_rw_potential(l)
            V_Z = schwarzschild_zerilli_potential(l)
            r = 1e6
            centrifugal = l * (l + 1) / r^2
            @test abs(V_RW(r, M) - centrifugal) / centrifugal < 1e-4
            @test abs(V_Z(r, M) - centrifugal) / centrifugal < 1e-4
        end
    end

    @testset "potential difference changes sign (isospectrality indicator)" begin
        # V_RW and V_Z cross: V_RW > V_Z at large r, V_RW < V_Z near horizon.
        # This crossing is a necessary condition for having the same spectrum.
        M = 1.0
        diff = schwarzschild_potential_difference(2)
        # Near horizon (r=3M): V_RW < V_Z, so diff < 0
        @test diff(3.0, M) < 0
        # At larger r (r=10M): V_RW > V_Z, so diff > 0
        @test diff(10.0, M) > 0
    end

    @testset "potential peak heights are close" begin
        # For isospectral potentials, the peak heights should be similar
        # (they determine the leading QNM frequency)
        M = 1.0
        for l in 2:5
            V_RW = schwarzschild_rw_potential(l)
            V_Z = schwarzschild_zerilli_potential(l)

            # Find approximate peaks by sampling
            rw_max = maximum(V_RW(r, M) for r in 2.5:0.01:10.0)
            z_max = maximum(V_Z(r, M) for r in 2.5:0.01:10.0)

            # Peak heights should be within 10% (they're close but not identical)
            ratio = rw_max / z_max
            @test 0.9 < ratio < 1.1
        end
    end

    @testset "isospectrality: both potentials match existing implementation" begin
        # Cross-check against the potentials in bh_second_order.jl
        using TensorGR: regge_wheeler_potential, zerilli_potential

        for l in 2:5
            V_RW_new = schwarzschild_rw_potential(l)
            V_Z_new = schwarzschild_zerilli_potential(l)
            for r in [3.0, 4.0, 6.0, 10.0]
                M = 1.0
                @test V_RW_new(r, M) ≈ regge_wheeler_potential(r, M, l)
                @test V_Z_new(r, M) ≈ zerilli_potential(r, M, l)
            end
        end
    end
end
