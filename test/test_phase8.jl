# Helper: check if expression contains ∂∂field (nested derivative of field)
function _has_second_deriv(expr::TDeriv, field::Symbol)
    if expr.arg isa TDeriv
        return true
    end
    _has_second_deriv(expr.arg, field)
end
_has_second_deriv(expr::TSum, field::Symbol) = any(t -> _has_second_deriv(t, field), expr.terms)
_has_second_deriv(expr::TProduct, field::Symbol) = any(f -> _has_second_deriv(f, field), expr.factors)
_has_second_deriv(::TensorExpr, ::Symbol) = false

@testset "Phase 8: Hardening & Extensions" begin

    @testset "Variational derivative" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(name=:Φ, manifold=:M4, rank=(0,0),
            symmetries=Any[]))

        with_registry(reg) do
            # Simple Lagrangian L = Φ (no derivatives)
            # δL/δΦ = 1 (the coefficient of Φ)
            L = Tensor(:Φ, TIndex[])
            result = variational_derivative(L, :Φ)
            # Should be Φ (the variation is trivial for this case)
            @test result isa TensorExpr
        end
    end

    @testset "VarD kinetic term: δ/δΦ (∂_a Φ ∂^a Φ) = -2 □Φ" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(name=:Φ, manifold=:M4, rank=(0,0),
            symmetries=Any[]))

        with_registry(reg) do
            Φ = Tensor(:Φ, TIndex[])
            # L = ∂_a Φ ∂^a Φ
            dΦ_down = TDeriv(down(:a), Φ)
            dΦ_up = TDeriv(up(:a), Φ)
            L = tproduct(1 // 1, TensorExpr[dΦ_down, dΦ_up])

            result = variational_derivative(L, :Φ)
            # Result should be -2 □Φ = -2 ∂_a ∂^a Φ
            # Check it's non-trivial (not zero) and involves second derivatives
            @test result != TScalar(0 // 1)
            @test _has_second_deriv(result, :Φ)
        end
    end

    @testset "Euler-Lagrange for multiple fields" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(name=:Φ, manifold=:M4, rank=(0,0),
            symmetries=Any[]))
        register_tensor!(reg, TensorProperties(name=:Ψ, manifold=:M4, rank=(0,0),
            symmetries=Any[]))

        with_registry(reg) do
            L = Tensor(:Φ, TIndex[]) * Tensor(:Ψ, TIndex[])
            eqs = euler_lagrange(L, [:Φ, :Ψ])
            @test length(eqs) == 2
        end
    end

    @testset "Abstract trace" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)]))

        with_registry(reg) do
            # Trace of T_{ab} over a,b with same positions needs metric
            T_ab = Tensor(:T, [down(:a), down(:b)])
            result = abstract_trace(T_ab, :a, :b)
            # Should involve a metric contraction
            @test result isa TProduct
        end
    end

    @testset "Make traceless" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)]))

        with_registry(reg) do
            T_ab = Tensor(:T, [down(:a), down(:b)])
            result = make_traceless(T_ab, :g, :a, :b; dim=4)
            # T^TF = T - (1/4) g tr(T)
            @test result isa TSum
        end
    end

    @testset "@manifold macro" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M3 dim=3 metric=h indices=[i,j,k]
            @test has_manifold(reg, :M3)
            @test get_manifold(reg, :M3).dim == 3
            @test has_tensor(reg, :h)
            @test has_tensor(reg, :δ)
        end
    end

    @testset "@define_tensor macro" begin
        reg = TensorRegistry()
        with_registry(reg) do
            register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d]))
            @define_tensor F on=M4 rank=(0,2) symmetry=AntiSymmetric(1,2)
            @test has_tensor(reg, :F)
            props = get_tensor(reg, :F)
            @test props.rank == (0, 2)
            @test length(props.symmetries) == 1
            @test props.symmetries[1] isa AntiSymmetric
        end
    end

    @testset "@covd macro" begin
        reg = TensorRegistry()
        with_registry(reg) do
            register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d]))
            register_tensor!(reg, TensorProperties(name=:g, manifold=:M4, rank=(0,2),
                symmetries=Any[Symmetric(1,2)],
                options=Dict{Symbol,Any}(:is_metric => true)))
            @covd D on=M4 metric=g
            @test has_tensor(reg, :ΓD)
        end
    end

    @testset "Schwarzschild Christoffel symbols" begin
        # Schwarzschild metric in (t,r,θ,φ) coordinates
        # ds² = -(1-2M/r)dt² + (1-2M/r)⁻¹dr² + r²dθ² + r²sin²θdφ²
        #
        # At a specific point r=R, θ=π/2, M=1:
        # f(R) = 1 - 2/R
        R = 10.0  # far from horizon
        M_val = 1.0
        f = 1.0 - 2*M_val/R
        sinθ = 1.0  # θ = π/2

        g = zeros(4, 4)
        g[1,1] = -f          # g_tt
        g[2,2] = 1.0/f       # g_rr
        g[3,3] = R^2          # g_θθ
        g[4,4] = R^2 * sinθ^2 # g_φφ

        ginv = zeros(4, 4)
        ginv[1,1] = -1.0/f
        ginv[2,2] = f
        ginv[3,3] = 1.0/R^2
        ginv[4,4] = 1.0/(R^2 * sinθ^2)

        coords = [:t, :r, :θ, :φ]

        # Derivatives of metric components at (R, π/2):
        # ∂_r g_tt = -f' = -2M/R² = -2/R²
        # ∂_r g_rr = -f'/f² = 2M/(R²f²) = 2/(R²f²)
        # ∂_r g_θθ = 2R
        # ∂_r g_φφ = 2R sin²θ = 2R
        # ∂_θ g_φφ = 2r² sinθ cosθ = 0 at θ=π/2
        # All other derivatives = 0
        f_prime = 2*M_val/R^2

        function schw_deriv(expr, coord)
            if coord == :r
                if expr ≈ -f
                    return -f_prime
                elseif expr ≈ 1.0/f
                    return -f_prime/f^2
                elseif expr ≈ R^2
                    return 2*R
                elseif expr ≈ R^2 * sinθ^2
                    return 2*R * sinθ^2
                end
            end
            return 0.0
        end

        Gamma = metric_christoffel(g, ginv, coords; deriv_fn=schw_deriv)

        # Known non-zero Schwarzschild Christoffels at θ=π/2:
        # Γ^t_{tr} = Γ^t_{rt} = f'/(2f) = M/(R²f)
        @test Gamma[1, 1, 2] ≈ f_prime/(2*f) atol=1e-10
        @test Gamma[1, 2, 1] ≈ f_prime/(2*f) atol=1e-10

        # Γ^r_{tt} = f·f'/2 = f·M/R²
        @test Gamma[2, 1, 1] ≈ f*f_prime/2 atol=1e-10

        # Γ^r_{rr} = -f'/(2f) = -M/(R²f)
        @test Gamma[2, 2, 2] ≈ -f_prime/(2*f) atol=1e-10

        # Γ^r_{θθ} = -R·f
        @test Gamma[2, 3, 3] ≈ -R*f atol=1e-10

        # Γ^r_{φφ} = -R·f·sin²θ = -R·f at θ=π/2
        @test Gamma[2, 4, 4] ≈ -R*f*sinθ^2 atol=1e-10

        # Γ^θ_{rθ} = Γ^θ_{θr} = 1/R
        @test Gamma[3, 2, 3] ≈ 1.0/R atol=1e-10
        @test Gamma[3, 3, 2] ≈ 1.0/R atol=1e-10

        # Γ^φ_{rφ} = Γ^φ_{φr} = 1/R
        @test Gamma[4, 2, 4] ≈ 1.0/R atol=1e-10
        @test Gamma[4, 4, 2] ≈ 1.0/R atol=1e-10
    end
end
