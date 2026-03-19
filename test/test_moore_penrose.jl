# Ground truth: Barker (2024) arXiv:2406.09500, PSALTer Section 5.
# Also: standard linearized gravity propagator — spin-2 and spin-0s
# sectors are physical, spin-1 and spin-0w are gauge zero modes.
#
# For Fierz-Pauli (linearized GR):
#   K_FP = 2.5k² P₂ - k² P₀ₛ (spin-1 and spin-0w vanish)
#   K_FP⁺ = (1/(2.5k²)) P₂ + (1/(-k²)) P₀ₛ

@testset "Moore-Penrose propagator" begin

    @testset "SpinSectorDecomposition construction" begin
        d = SpinSectorDecomposition(5//2, 0, -1, 0)
        @test d.spin2 == 5//2
        @test d.spin1 == 0
        @test d.spin0s == -1
        @test d.spin0w == 0
    end

    @testset "Fierz-Pauli kernel: spin-2=5k²/2, spin-0s=-k², others=0" begin
        # At k²=1: a₂ = 5/2, a₁ = 0, a₀ₛ = -1, a₀w = 0
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)

        prop = moore_penrose_propagator(K_FP)

        # Spin-2: 1/(5/2) = 2/5
        @test prop.spin2 == 2//5

        # Spin-1: gauge zero mode -> 0 in propagator
        @test prop.spin1 == 0

        # Spin-0s: 1/(-1) = -1
        @test prop.spin0s == -1//1

        # Spin-0w: gauge zero mode -> 0 in propagator
        @test prop.spin0w == 0
    end

    @testset "R² kernel: only spin-0s sector" begin
        # K_R² = 3k⁴ P₀ₛ (all other sectors vanish)
        K_R2 = SpinSectorDecomposition(0, 0, 3, 0)

        prop = moore_penrose_propagator(K_R2)

        @test prop.spin2 == 0   # no spin-2 mode
        @test prop.spin1 == 0
        @test prop.spin0s == 1//3  # only propagating sector
        @test prop.spin0w == 0
    end

    @testset "(δRic)² kernel: spin-2 + spin-0s" begin
        # K_Ric² = 5k⁴/4 P₂ + k⁴ P₀ₛ
        K_Ric2 = SpinSectorDecomposition(5//4, 0, 1, 0)

        prop = moore_penrose_propagator(K_Ric2)

        @test prop.spin2 == 4//5
        @test prop.spin1 == 0
        @test prop.spin0s == 1//1
        @test prop.spin0w == 0
    end

    @testset "All sectors nonzero" begin
        d = SpinSectorDecomposition(2, 3, 4, 5)
        prop = moore_penrose_propagator(d)
        @test prop.spin2 == 1//2
        @test prop.spin1 == 1//3
        @test prop.spin0s == 1//4
        @test prop.spin0w == 1//5
    end

    @testset "All sectors zero -> zero propagator" begin
        d = SpinSectorDecomposition(0, 0, 0, 0)
        prop = moore_penrose_propagator(d)
        @test prop.spin2 == 0
        @test prop.spin1 == 0
        @test prop.spin0s == 0
        @test prop.spin0w == 0
    end

    @testset "no_ghost: Fierz-Pauli has ghost in spin-0s" begin
        # FP: spin2=+5/2, spin0s=-1 -> opposite signs -> ghost!
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)
        @test !no_ghost(K_FP)  # the scalar mode is a ghost in FP

        # Pure spin-2: no ghost
        K_pure = SpinSectorDecomposition(1, 0, 0, 0)
        @test no_ghost(K_pure)

        # All positive: no ghost
        K_pos = SpinSectorDecomposition(1, 2, 3, 4)
        @test no_ghost(K_pos)

        # All negative: consistent signs -> no ghost (just wrong overall sign convention)
        K_neg = SpinSectorDecomposition(-1, -2, 0, 0)
        @test no_ghost(K_neg)
    end

    @testset "no_tachyon" begin
        # Positive mass-squared: no tachyon
        m2_ok = SpinSectorDecomposition(1, 0, 2, 0)
        @test no_tachyon(m2_ok)

        # Negative mass-squared: tachyon
        m2_bad = SpinSectorDecomposition(1, 0, -1, 0)
        @test !no_tachyon(m2_bad)

        # Zero mass: ok (massless)
        m2_zero = SpinSectorDecomposition(0, 0, 0, 0)
        @test no_tachyon(m2_zero)
    end

    @testset "Symbolic form factors" begin
        # Symbolic k² as an expression
        k2 = :k²
        d = SpinSectorDecomposition(:($k2 * 5 // 2), 0, :(-$k2), 0)
        prop = moore_penrose_propagator(d)
        # Spin-2: 1/(k²*5//2) as Expr
        @test prop.spin2 isa Expr
        # Spin-1: 0
        @test prop.spin1 == 0
        # Spin-0s: 1/(-k²) as Expr
        @test prop.spin0s isa Expr
    end

    @testset "Display" begin
        d = SpinSectorDecomposition(1, 0, -1, 0)
        s = sprint(show, d)
        @test occursin("SpinSectorDecomposition", s)
        @test occursin("spin2", s)
    end

    @testset "Float tolerance" begin
        d = SpinSectorDecomposition(1.0, 1e-15, -1.0, 1e-16)
        # With default atol=0: tiny values treated as nonzero
        prop1 = moore_penrose_propagator(d)
        @test prop1.spin1 != 0  # 1e-15 is not exactly zero

        # With atol=1e-10: tiny values treated as zero (gauge)
        prop2 = moore_penrose_propagator(d; atol=1e-10)
        @test prop2.spin1 == 0
        @test prop2.spin0w == 0
    end
end
