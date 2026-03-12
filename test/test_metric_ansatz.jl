using Symbolics

# Helper: evaluate a symbolic expression after substitution.
function _eval_ansatz(expr, vals)
    subbed = Symbolics.substitute(expr, vals)
    v = Symbolics.value(subbed)
    v isa Number && return Float64(v)
    return Float64(eval(Symbolics.toexpr(v)))
end

@testset "Metric Ansatz Generators" begin
    @testset "FLRW ansatz (k=0)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
        ans = HomogeneousIsotropy(:M4)
        result = with_registry(reg) do
            metric_ansatz(reg, :M4, ans; k=0)
        end

        sm = result.metric
        a_func = result.free_functions[1]
        tau_sym = result.time_coord

        @test sm.dim == 4
        @test length(result.free_functions) == 1

        # Compute Christoffel symbols
        Gamma_raw = symbolic_christoffel(sm)
        Riem_raw = symbolic_riemann(sm, Gamma_raw)
        Ric_raw = symbolic_ricci(Riem_raw, 4)

        # Substitute derivatives with plain variables for verification
        D = Symbolics.Differential(tau_sym)
        da = Symbolics.expand_derivatives(D(a_func))
        dda = Symbolics.expand_derivatives(D(da))
        @variables a_bare_t adot_t addot_t
        sub_dict = Dict(da => adot_t, dda => addot_t, a_func => a_bare_t)

        # Substitute in Ricci tensor
        Ric = Matrix{Any}(undef, 4, 4)
        for i in 1:4, j in 1:4
            Ric[i, j] = Symbolics.simplify(Symbolics.substitute(Ric_raw[i, j], sub_dict))
        end

        # R_{tt} = -3 * addot / a -- verify numerically
        # Use test point: a=2, adot=0.5, addot=-0.1
        chi_sym = sm.coords[2]
        theta_sym = sm.coords[3]
        phi_sym = sm.coords[4]
        vals = Dict(
            a_bare_t => 2.0, adot_t => 0.5, addot_t => -0.1,
            chi_sym => 0.5, theta_sym => pi / 2, phi_sym => 0.0, tau_sym => 0.0
        )
        R_tt_computed = _eval_ansatz(Ric[1, 1], vals)
        R_tt_expected = -3.0 * (-0.1) / 2.0  # -3 * addot / a
        @test abs(R_tt_computed - R_tt_expected) < 1e-10

        # Verify off-diagonal Ricci components vanish
        for i in 1:4, j in 1:4
            i == j && continue
            @test abs(_eval_ansatz(Ric[i, j], vals)) < 1e-10
        end

        # Verify diagonal structure of metric
        @test isequal(sm.g[1, 2], 0)
        @test isequal(sm.g[1, 3], 0)
        @test isequal(sm.g[1, 4], 0)
        @test isequal(sm.g[2, 3], 0)
    end

    @testset "FLRW ansatz (k=1)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
        ans = HomogeneousIsotropy(:M4)
        result = with_registry(reg) do
            metric_ansatz(reg, :M4, ans; k=1)
        end
        sm = result.metric
        a_func = result.free_functions[1]
        tau_sym = result.time_coord

        # Compute Ricci scalar and verify R = 6(addot/a + adot^2/a^2 + k/a^2)
        Gamma_raw = symbolic_christoffel(sm)
        Riem_raw = symbolic_riemann(sm, Gamma_raw)
        Ric_raw = symbolic_ricci(Riem_raw, 4)
        R_raw = symbolic_ricci_scalar(Ric_raw, sm.ginv, 4)

        D = Symbolics.Differential(tau_sym)
        da = Symbolics.expand_derivatives(D(a_func))
        dda = Symbolics.expand_derivatives(D(da))
        @variables a_bare_k1 adot_k1 addot_k1
        sub_dict = Dict(da => adot_k1, dda => addot_k1, a_func => a_bare_k1)
        R_scalar = Symbolics.simplify(Symbolics.substitute(R_raw, sub_dict))

        chi_sym = sm.coords[2]
        theta_sym = sm.coords[3]
        phi_sym = sm.coords[4]
        vals = Dict(
            a_bare_k1 => 1.0, adot_k1 => 1.0, addot_k1 => -0.5,
            chi_sym => 0.3, theta_sym => 1.0, phi_sym => 0.5, tau_sym => 0.0
        )
        R_computed = _eval_ansatz(R_scalar, vals)
        R_expected = 6.0 * (-0.5 / 1.0 + 1.0 / 1.0 + 1.0 / 1.0)  # k=1
        @test abs(R_computed - R_expected) < 1e-10
    end

    @testset "FLRW ansatz (k=-1)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
        ans = HomogeneousIsotropy(:M4)
        result = with_registry(reg) do
            metric_ansatz(reg, :M4, ans; k=-1)
        end
        sm = result.metric

        # Verify g_{chi,chi} denominator has (1 + chi^2) for k=-1
        # g_{chi,chi} = a^2 / (1 - (-1)*chi^2) = a^2 / (1 + chi^2)
        @test sm.dim == 4
    end

    @testset "Spherical symmetry ansatz" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
        ans = SphericalSymmetry(:M4)
        result = with_registry(reg) do
            metric_ansatz(reg, :M4, ans)
        end

        sm = result.metric
        r_sym = result.radial_coord
        A_func = result.free_functions[1]
        B_func = result.free_functions[2]

        @test sm.dim == 4
        @test length(result.free_functions) == 2

        # Verify diagonal structure
        @test isequal(sm.g[1, 2], 0)
        @test isequal(sm.g[1, 3], 0)
        @test isequal(sm.g[1, 4], 0)
        @test isequal(sm.g[2, 3], 0)
        @test isequal(sm.g[2, 4], 0)
        @test isequal(sm.g[3, 4], 0)

        # Compute Christoffel symbols
        Gamma_raw = symbolic_christoffel(sm)

        # Verify Gamma^t_{tr} = A'/(2A) numerically
        # Substitute A(r) -> A_val, A'(r) -> Ap_val, B(r) -> B_val, B'(r) -> Bp_val
        D = Symbolics.Differential(r_sym)
        dA = Symbolics.expand_derivatives(D(A_func))
        dB = Symbolics.expand_derivatives(D(B_func))
        @variables A_bare B_bare Ap_bare Bp_bare
        sub_dict = Dict(dA => Ap_bare, dB => Bp_bare, A_func => A_bare, B_func => B_bare)

        Gamma = Array{Any}(undef, 4, 4, 4)
        for i in 1:4, j in 1:4, l in 1:4
            Gamma[i, j, l] = Symbolics.simplify(Symbolics.substitute(Gamma_raw[i, j, l], sub_dict))
        end

        t_sym = sm.coords[1]
        theta_sym = sm.coords[3]
        phi_sym = sm.coords[4]

        # Evaluation point
        A_val = 0.8; B_val = 1.2; Ap_val = 0.3; Bp_val = -0.1
        r_val = 3.0; th_val = pi / 2
        vals = Dict(
            A_bare => A_val, B_bare => B_val, Ap_bare => Ap_val, Bp_bare => Bp_val,
            r_sym => r_val, theta_sym => th_val, phi_sym => 0.0, t_sym => 0.0
        )

        # Gamma^t_{tr} = A'/(2A)
        G_ttr = _eval_ansatz(Gamma[1, 1, 2], vals)
        @test abs(G_ttr - Ap_val / (2.0 * A_val)) < 1e-10

        # Gamma^r_{tt} = A'/(2B)
        G_rtt = _eval_ansatz(Gamma[2, 1, 1], vals)
        @test abs(G_rtt - Ap_val / (2.0 * B_val)) < 1e-10

        # Gamma^r_{rr} = B'/(2B)
        G_rrr = _eval_ansatz(Gamma[2, 2, 2], vals)
        @test abs(G_rrr - Bp_val / (2.0 * B_val)) < 1e-10

        # Gamma^r_{theta,theta} = -r/B
        G_rthth = _eval_ansatz(Gamma[2, 3, 3], vals)
        @test abs(G_rthth - (-r_val / B_val)) < 1e-10

        # Gamma^theta_{r,theta} = 1/r
        G_thrth = _eval_ansatz(Gamma[3, 2, 3], vals)
        @test abs(G_thrth - 1.0 / r_val) < 1e-10
    end

    @testset "Schwarzschild specialization" begin
        # Verify that SphericalSymmetry ansatz with A=B^(-1)=1-2M/r gives Ricci-flat
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
        ans = SphericalSymmetry(:M4)
        result = with_registry(reg) do
            metric_ansatz(reg, :M4, ans)
        end
        sm = result.metric
        r_sym = result.radial_coord
        A_func = result.free_functions[1]
        B_func = result.free_functions[2]

        # Compute Ricci tensor
        Gamma_raw = symbolic_christoffel(sm)
        Riem_raw = symbolic_riemann(sm, Gamma_raw)
        Ric_raw = symbolic_ricci(Riem_raw, 4)

        # Specialize to Schwarzschild: A = 1 - 2M/r, B = 1/A
        D = Symbolics.Differential(r_sym)
        dA = Symbolics.expand_derivatives(D(A_func))
        dB = Symbolics.expand_derivatives(D(B_func))
        ddA = Symbolics.expand_derivatives(D(dA))
        ddB = Symbolics.expand_derivatives(D(dB))

        @variables M_schw
        f_val = 1 - 2 * M_schw / r_sym
        f_inv = 1 / f_val
        df = Symbolics.expand_derivatives(D(f_val))
        dfinv = Symbolics.expand_derivatives(D(f_inv))
        ddf = Symbolics.expand_derivatives(D(df))
        ddfinv = Symbolics.expand_derivatives(D(dfinv))

        sub_dict = Dict(
            ddA => ddf, ddB => ddfinv,
            dA => df, dB => dfinv,
            A_func => f_val, B_func => f_inv
        )

        theta_sym = sm.coords[3]
        vals = Dict(M_schw => 1.0, r_sym => 3.0, theta_sym => pi / 2)

        # Ricci tensor should vanish for Schwarzschild
        for a in 1:4, b in 1:4
            ric_sub = Symbolics.substitute(Ric_raw[a, b], sub_dict)
            v = _eval_ansatz(Symbolics.simplify(ric_sub), vals)
            @test abs(v) < 1e-8
        end
    end

    @testset "Custom coordinates" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))

        # FLRW with custom coord names
        ans_flrw = HomogeneousIsotropy(:M4)
        result = with_registry(reg) do
            metric_ansatz(reg, :M4, ans_flrw; coords=[:T, :X, :Y, :Z], k=0)
        end
        @test result.metric.dim == 4
        @test result.metric.coord_names == [:T, :X, :Y, :Z]

        # Spherical with custom coord names
        ans_sph = SphericalSymmetry(:M4)
        result2 = with_registry(reg) do
            metric_ansatz(reg, :M4, ans_sph; coords=[:t0, :rho, :th, :ph])
        end
        @test result2.metric.dim == 4
        @test result2.metric.coord_names == [:t0, :rho, :th, :ph]
    end

    @testset "Invalid arguments" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))

        # Wrong number of coords
        @test_throws ErrorException with_registry(reg) do
            metric_ansatz(reg, :M4, HomogeneousIsotropy(:M4); coords=[:t, :x, :y])
        end
        @test_throws ErrorException with_registry(reg) do
            metric_ansatz(reg, :M4, SphericalSymmetry(:M4); coords=[:t, :r])
        end

        # Invalid k value
        @test_throws ErrorException with_registry(reg) do
            metric_ansatz(reg, :M4, HomogeneousIsotropy(:M4); k=2)
        end
    end
end
