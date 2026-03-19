# Ground truth: Peskin & Schroeder (1995) Eqs 3.76-3.79;
#               Nishi, Am. J. Phys. 73, 1160 (2005), Table I.

@testset "Fierz identities" begin

    @testset "Clifford basis" begin
        @test Int(CB_S) == 0
        @test Int(CB_V) == 1
        @test Int(CB_T) == 2
        @test Int(CB_A) == 3
        @test Int(CB_P) == 4
    end

    @testset "Fierz matrix dimensions" begin
        F = fierz_matrix()
        @test size(F) == (5, 5)
        @test eltype(F) == Rational{Int}
    end

    @testset "Fierz matrix: scalar-scalar entry (P&S Eq 3.78)" begin
        # (ψ̄₁ ψ₂)(ψ̄₃ ψ₄) Fierz rearrangement
        # F_{SS} = -1/4
        @test fierz_coefficient(CB_S, CB_S) == -1//4
    end

    @testset "Fierz matrix: known entries from Nishi Table I" begin
        F = fierz_matrix()

        # Row S: [-1/4, -1/4, -1/8, 1/4, -1/4]
        @test F[1, 1] == -1//4   # SS
        @test F[1, 2] == -1//4   # SV
        @test F[1, 3] == -1//8   # ST
        @test F[1, 4] == 1//4    # SA
        @test F[1, 5] == -1//4   # SP

        # Row V: [-1, 1/2, 0, -1/2, 1]
        @test F[2, 1] == -1//1   # VS
        @test F[2, 2] == 1//2    # VV
        @test F[2, 3] == 0//1    # VT
        @test F[2, 4] == -1//2   # VA
        @test F[2, 5] == 1//1    # VP

        # Row P: [-1/4, 1/4, -1/8, -1/4, -1/4]
        @test F[5, 1] == -1//4   # PS
        @test F[5, 2] == 1//4    # PV
        @test F[5, 3] == -1//8   # PT
        @test F[5, 4] == -1//4   # PA
        @test F[5, 5] == -1//4   # PP
    end

    @testset "Fierz matrix symmetry: F_{SP} = F_{PS}" begin
        @test fierz_coefficient(CB_S, CB_P) == fierz_coefficient(CB_P, CB_S)
    end

    @testset "Fierz completeness" begin
        @test fierz_identity_check()
    end

    @testset "Clifford basis dimensions sum to 16" begin
        dims = [CLIFFORD_DIM[b] for b in instances(CliffordBasis)]
        @test sum(dims) == 16  # 4² = 16 (Dirac space dimension)
    end

    @testset "All basis elements named" begin
        for b in instances(CliffordBasis)
            @test haskey(CLIFFORD_NAMES, b)
            @test !isempty(CLIFFORD_NAMES[b])
        end
    end
end
