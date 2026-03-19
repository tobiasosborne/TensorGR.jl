# Ground truth: Higuchi, Nucl. Phys. B 282, 397 (1987);
#              Hassan, Schmidt-May & von Strauss, PLB 715, 335 (2012).
#
# The Higuchi bound states that for a massive spin-2 field on de Sitter
# with cosmological constant Lambda > 0:
#
#   m^2 >= 2*Lambda/3
#
# Below this bound the helicity-0 mode becomes a ghost.
# At the bound m^2 = 2*Lambda/3, the theory is partially massless.

@testset "Bimetric Higuchi bound" begin

    # ─── Basic higuchi_bound tests ───────────────────────────────────────

    @testset "higuchi_bound returns 2*Lambda/3 for numeric Lambda" begin
        @test TensorGR.higuchi_bound(3) == 2
        @test TensorGR.higuchi_bound(6) == 4
        @test TensorGR.higuchi_bound(3 // 2) == 1 // 1
        @test TensorGR.higuchi_bound(0) == 0
        @test TensorGR.higuchi_bound(9) == 6
        @test TensorGR.higuchi_bound(1) == 2 // 3
    end

    @testset "higuchi_bound for symbolic Lambda returns Expr" begin
        result = TensorGR.higuchi_bound(:Lambda)
        @test result isa Expr
    end

    # ─── higuchi_coefficient tests ───────────────────────────────────────

    @testset "higuchi_coefficient at the bound is exactly zero" begin
        # m^2 = 2*Lambda/3 => coefficient = 0
        for Lambda in [3, 6, 9, 3 // 2]
            bound = TensorGR.higuchi_bound(Lambda)
            @test TensorGR.higuchi_coefficient(bound, Lambda) == 0
        end
    end

    @testset "higuchi_coefficient above the bound is positive" begin
        # m^2 > 2*Lambda/3 => healthy, positive coefficient
        Lambda = 3
        bound = TensorGR.higuchi_bound(Lambda)  # = 2
        @test TensorGR.higuchi_coefficient(bound + 1, Lambda) > 0
        @test TensorGR.higuchi_coefficient(bound + 1 // 10, Lambda) > 0
        @test TensorGR.higuchi_coefficient(100, Lambda) > 0
    end

    @testset "higuchi_coefficient below the bound is negative" begin
        # m^2 < 2*Lambda/3 => ghost, negative coefficient
        Lambda = 3
        bound = TensorGR.higuchi_bound(Lambda)  # = 2
        @test TensorGR.higuchi_coefficient(bound - 1, Lambda) < 0
        @test TensorGR.higuchi_coefficient(bound - 1 // 10, Lambda) < 0
        @test TensorGR.higuchi_coefficient(0, Lambda) < 0
    end

    @testset "higuchi_coefficient with symbolic args returns Expr" begin
        result = TensorGR.higuchi_coefficient(:m2, :Lambda)
        @test result isa Expr
    end

    # ─── is_higuchi_healthy tests ────────────────────────────────────────

    @testset "is_higuchi_healthy returns true above or at the bound" begin
        Lambda = 3
        bound = TensorGR.higuchi_bound(Lambda)  # = 2
        @test TensorGR.is_higuchi_healthy(bound, Lambda) == true      # at bound
        @test TensorGR.is_higuchi_healthy(bound + 1, Lambda) == true  # above bound
        @test TensorGR.is_higuchi_healthy(100, Lambda) == true        # well above
    end

    @testset "is_higuchi_healthy returns false below the bound" begin
        Lambda = 3
        bound = TensorGR.higuchi_bound(Lambda)  # = 2
        @test TensorGR.is_higuchi_healthy(bound - 1, Lambda) == false
        @test TensorGR.is_higuchi_healthy(0, Lambda) == false
        @test TensorGR.is_higuchi_healthy(-1, Lambda) == false
    end

    @testset "is_higuchi_healthy errors on symbolic arguments" begin
        @test_throws ErrorException TensorGR.is_higuchi_healthy(:m2, 3)
        @test_throws ErrorException TensorGR.is_higuchi_healthy(1, :Lambda)
        @test_throws ErrorException TensorGR.is_higuchi_healthy(:m2, :Lambda)
    end

    # ─── Integration with bimetric setup ─────────────────────────────────

    @testset "Integration: bimetric FP mass vs Higuchi bound" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        bs = define_bimetric!(reg, :g, :f; manifold=:M4)

        # Choose beta parameters and c=1 so that m^2_FP is known.
        # beta1=0, beta2=1, beta3=0, c=1: m^2_FP = m^2 * (0 + 2*1 + 0) / (1+1) = m^2
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=0, beta2=1, beta3=0, beta4=0)
        m2_FP = fierz_pauli_mass_squared(params, 1)
        @test m2_FP == 1

        # For Lambda=1: bound = 2/3; m^2_FP=1 > 2/3 => healthy
        @test TensorGR.is_higuchi_healthy(m2_FP, 1) == true
        @test TensorGR.higuchi_coefficient(m2_FP, 1) == 1 // 3

        # For Lambda=3: bound = 2; m^2_FP=1 < 2 => ghost
        @test TensorGR.is_higuchi_healthy(m2_FP, 3) == false
        @test TensorGR.higuchi_coefficient(m2_FP, 3) == -1
    end

    @testset "Integration: larger m_sq satisfies stronger bound" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        bs = define_bimetric!(reg, :g, :f; manifold=:M4)

        # m_sq=10, beta2=1, c=1 => m^2_FP = 10
        params = HassanRosenParams(m_sq=10, beta0=0, beta1=0, beta2=1, beta3=0, beta4=0)
        m2_FP = fierz_pauli_mass_squared(params, 1)
        @test m2_FP == 10

        # Lambda=15: bound = 10 => at bound (partially massless)
        @test TensorGR.is_higuchi_healthy(m2_FP, 15) == true
        @test TensorGR.higuchi_coefficient(m2_FP, 15) == 0

        # Lambda=14: bound = 28/3 < 10 => healthy
        @test TensorGR.is_higuchi_healthy(m2_FP, 14) == true
        @test TensorGR.higuchi_coefficient(m2_FP, 14) > 0

        # Lambda=16: bound = 32/3 > 10 => ghost
        @test TensorGR.is_higuchi_healthy(m2_FP, 16) == false
        @test TensorGR.higuchi_coefficient(m2_FP, 16) < 0
    end

    # ─── Partially massless point ────────────────────────────────────────

    @testset "Partially massless point: m^2_FP = 2*Lambda/3" begin
        # Choose beta, c so that m^2_FP = 2*Lambda/3 exactly.
        # beta2=1, c=1, m_sq=1 => m^2_FP = 1
        # Lambda = 3/2 => bound = 2*(3/2)/3 = 1 => at bound
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=0, beta2=1, beta3=0, beta4=0)
        m2_FP = fierz_pauli_mass_squared(params, 1)  # = 1
        Lambda = 3 // 2

        @test m2_FP == TensorGR.higuchi_bound(Lambda)
        @test TensorGR.higuchi_coefficient(m2_FP, Lambda) == 0
        @test TensorGR.is_higuchi_healthy(m2_FP, Lambda) == true  # at bound is healthy (>=)
    end

    @testset "Partially massless with non-unit parameters" begin
        # beta1=1, beta2=1, beta3=1, c=1, m_sq=1 => m^2_FP = (1+2+1)/2 = 2
        # Lambda = 3 => bound = 2 => at bound
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=1, beta2=1, beta3=1, beta4=0)
        m2_FP = fierz_pauli_mass_squared(params, 1)  # = 2
        @test m2_FP == 2

        Lambda = 3
        @test TensorGR.higuchi_bound(Lambda) == 2
        @test TensorGR.higuchi_coefficient(m2_FP, Lambda) == 0
    end

    # ─── Flat space: Lambda = 0 ──────────────────────────────────────────

    @testset "Flat space (Lambda=0): bound is 0, any m^2 >= 0 is healthy" begin
        @test TensorGR.higuchi_bound(0) == 0
        @test TensorGR.higuchi_coefficient(0, 0) == 0
        @test TensorGR.higuchi_coefficient(1, 0) == 1
        @test TensorGR.is_higuchi_healthy(0, 0) == true
        @test TensorGR.is_higuchi_healthy(1, 0) == true
        @test TensorGR.is_higuchi_healthy(100, 0) == true

        # Negative mass squared in flat space is unhealthy
        # (tachyonic, but this is m^2 < 0 < bound=0)
        @test TensorGR.is_higuchi_healthy(-1, 0) == false
    end

    # ─── AdS: Lambda < 0 ────────────────────────────────────────────────

    @testset "AdS (Lambda<0): bound is negative, m^2 >= 0 always healthy" begin
        # Lambda = -3: bound = 2*(-3)/3 = -2
        @test TensorGR.higuchi_bound(-3) == -2
        @test TensorGR.is_higuchi_healthy(0, -3) == true    # 0 >= -2
        @test TensorGR.is_higuchi_healthy(1, -3) == true    # 1 >= -2
        @test TensorGR.is_higuchi_healthy(-1, -3) == true   # -1 >= -2

        # Even at the bound
        @test TensorGR.is_higuchi_healthy(-2, -3) == true
        @test TensorGR.higuchi_coefficient(-2, -3) == 0

        # Below the bound in AdS (pathological, m^2 very negative)
        @test TensorGR.is_higuchi_healthy(-3, -3) == false
    end

    @testset "AdS coefficient is positive for physical masses" begin
        Lambda = -6
        # bound = -4, any m^2 > -4 has positive coefficient
        @test TensorGR.higuchi_coefficient(0, Lambda) == 4
        @test TensorGR.higuchi_coefficient(1, Lambda) == 5
        @test TensorGR.higuchi_coefficient(-3, Lambda) == 1
    end

    # ─── Rational arithmetic precision ───────────────────────────────────

    @testset "Rational arithmetic: exact results" begin
        # Lambda = 1: bound = 2/3
        @test TensorGR.higuchi_bound(1) == 2 // 3
        @test TensorGR.higuchi_bound(1) isa Rational

        # Lambda = 7: bound = 14/3
        @test TensorGR.higuchi_bound(7) == 14 // 3

        # Coefficient with rationals
        @test TensorGR.higuchi_coefficient(1, 1) == 1 // 3
        @test TensorGR.higuchi_coefficient(1 // 2, 3 // 4) == 1 // 2 - 1 // 2
        @test TensorGR.higuchi_coefficient(1 // 2, 3 // 4) == 0
    end
end
