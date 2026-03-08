using Symbolics

# Helper: evaluate a symbolic expression after substitution.
# Handles the case where trig functions remain symbolic after substitution.
function _eval_sym(expr, vals)
    subbed = Symbolics.substitute(expr, vals)
    v = Symbolics.value(subbed)
    v isa Number && return Float64(v)
    return Float64(eval(Symbolics.toexpr(v)))
end

@testset "Symbolic Components" begin
    @testset "sym_deriv basics" begin
        @variables x_sd y_sd
        @test isequal(sym_deriv(x_sd^2, x_sd), 2x_sd)
        @test isequal(sym_deriv(x_sd * y_sd, x_sd), y_sd)
        @test isequal(sym_deriv(sin(x_sd), x_sd), cos(x_sd))
        @test isequal(sym_deriv(x_sd^2, y_sd), 0)
    end

    @testset "SymbolicMetric construction" begin
        @variables t_c x_c y_c z_c
        sm = symbolic_diagonal_metric(
            [t_c, x_c, y_c, z_c],
            [Symbolics.Num(-1), Symbolics.Num(1), Symbolics.Num(1), Symbolics.Num(1)])
        @test sm.dim == 4
        @test length(sm.coords) == 4
        @test sm.g[1, 1] == -1
        @test sm.g[2, 2] == 1
        @test sm.g[1, 2] == 0
        @test sm.ginv[1, 1] == -1
        @test sm.ginv[2, 2] == 1
    end

    @testset "Flat Minkowski" begin
        @variables t_m x_m y_m z_m
        sm = symbolic_diagonal_metric(
            [t_m, x_m, y_m, z_m],
            [Symbolics.Num(-1), Symbolics.Num(1), Symbolics.Num(1), Symbolics.Num(1)])
        result = symbolic_curvature_from_metric(sm)
        # All Christoffels should be 0
        for a in 1:4, b in 1:4, c in 1:4
            @test isequal(result.Gamma[a, b, c], 0)
        end
        # All Riemann should be 0
        for a in 1:4, b in 1:4, c in 1:4, d in 1:4
            @test isequal(result.Riem[a, b, c, d], 0)
        end
        # Ricci scalar = 0
        @test isequal(result.R, 0)
        # Kretschmann = 0
        @test isequal(result.K, 0)
    end

    @testset "2D Polar coordinates" begin
        @variables r_p θ_p
        sm = symbolic_diagonal_metric([r_p, θ_p], [Symbolics.Num(1), r_p^2])
        Gamma = symbolic_christoffel(sm)
        # Γ^r_{θθ} = -r
        @test isequal(Symbolics.simplify(Gamma[1, 2, 2] + r_p), 0)
        # Γ^θ_{rθ} = 1/r
        @test isequal(Symbolics.simplify(Gamma[2, 1, 2] - 1 / r_p), 0)
        # Γ^θ_{θr} = 1/r (symmetric in lower indices)
        @test isequal(Symbolics.simplify(Gamma[2, 2, 1] - 1 / r_p), 0)
        # Flat space: all Riemann = 0
        Riem = symbolic_riemann(sm, Gamma)
        for a in 1:2, b in 1:2, c in 1:2, d in 1:2
            @test isequal(Symbolics.simplify(Riem[a, b, c, d]), 0)
        end
    end

    @testset "2D Sphere (unit radius)" begin
        @variables θ_s φ_s
        sm = symbolic_diagonal_metric([θ_s, φ_s], [Symbolics.Num(1), sin(θ_s)^2])
        result = symbolic_curvature_from_metric(sm)
        # Ricci scalar R = 2 for unit 2-sphere (numerical; trig not auto-simplified)
        for θ_val in [0.5, 1.0, 1.5]
            @test abs(_eval_sym(result.R, Dict(θ_s => θ_val)) - 2.0) < 1e-12
        end
        # Christoffel spot-checks at θ = π/4
        local vals = Dict(θ_s => π / 4)
        # Γ^θ_{φφ} = -sin(θ)cos(θ)
        @test abs(_eval_sym(result.Gamma[1, 2, 2], vals) -
                  (-sin(π / 4) * cos(π / 4))) < 1e-12
        # Γ^φ_{θφ} = cos(θ)/sin(θ)
        @test abs(_eval_sym(result.Gamma[2, 1, 2], vals) -
                  cos(π / 4) / sin(π / 4)) < 1e-12
        # Kretschmann K = 4 for unit 2-sphere
        for θ_val in [0.5, 1.0, 1.5]
            @test abs(_eval_sym(result.K, Dict(θ_s => θ_val)) - 4.0) < 1e-10
        end
    end

    @testset "Schwarzschild Ricci-flat" begin
        @variables t_ss r_ss θ_ss φ_ss M_ss
        local f = 1 - 2 * M_ss / r_ss
        sm = symbolic_diagonal_metric(
            [t_ss, r_ss, θ_ss, φ_ss],
            [-f, 1 / f, r_ss^2, r_ss^2 * sin(θ_ss)^2])
        Gamma = symbolic_christoffel(sm)
        Riem = symbolic_riemann(sm, Gamma)
        Ric = symbolic_ricci(Riem, 4)
        # Schwarzschild is Ricci-flat
        local vals = Dict(M_ss => 1.0, r_ss => 3.0, θ_ss => π / 2)
        for a in 1:4, b in 1:4
            @test abs(_eval_sym(Ric[a, b], vals)) < 1e-10
        end
        # Ricci scalar should also be 0
        R = symbolic_ricci_scalar(Ric, sm.ginv, 4)
        @test abs(_eval_sym(R, vals)) < 1e-10
    end

    @testset "Schwarzschild Kretschmann" begin
        @variables t_sk r_sk θ_sk φ_sk M_sk
        local f_sk = 1 - 2 * M_sk / r_sk
        sm = symbolic_diagonal_metric(
            [t_sk, r_sk, θ_sk, φ_sk],
            [-f_sk, 1 / f_sk, r_sk^2, r_sk^2 * sin(θ_sk)^2])
        result = symbolic_curvature_from_metric(sm)
        # K = 48 M^2 / r^6 for Schwarzschild
        for (M_val, r_val) in [(1.0, 3.0), (2.0, 5.0), (0.5, 4.0)]
            local vals = Dict(M_sk => M_val, r_sk => r_val, θ_sk => π / 2)
            local K_val = _eval_sym(result.K, vals)
            local K_exact = 48.0 * M_val^2 / r_val^6
            @test abs(K_val - K_exact) < 1e-10
        end
    end

    @testset "Einstein tensor trace" begin
        # For any metric in d dimensions: G^a_a = (1 - d/2) R
        # Use polar coords (non-trig, exact simplification)
        @variables r_et θ_et
        sm = symbolic_diagonal_metric([r_et, θ_et], [Symbolics.Num(1), r_et^2])
        result = symbolic_curvature_from_metric(sm)
        local dim = 2
        local trace_G = sum(sm.ginv[a, b] * result.G[b, a] for a in 1:dim, b in 1:dim)
        local expected = (1 - dim / 2) * result.R  # = 0 for dim=2
        @test isequal(Symbolics.simplify(trace_G - expected), 0)
    end

    @testset "symbolic_metric (non-diagonal)" begin
        @variables u_nd v_nd
        # 2D metric with off-diagonal terms
        g_mat = Symbolics.Num[1 1; 1 2]
        sm = symbolic_metric([u_nd, v_nd], g_mat)
        @test sm.dim == 2
        # Check inverse: g * ginv = I
        for i in 1:2, j in 1:2
            local prod_ij = sum(sm.g[i, k] * sm.ginv[k, j] for k in 1:2)
            local expected = (i == j) ? 1 : 0
            @test isequal(Symbolics.simplify(prod_ij - expected), 0)
        end
        # Flat metric: all curvature should be 0 (constant metric)
        local result = symbolic_curvature_from_metric(sm)
        for a in 1:2, b in 1:2, c in 1:2
            @test isequal(result.Gamma[a, b, c], 0)
        end
    end
end
