using Test
using TensorGR
using LinearAlgebra

@testset "Segre classification" begin

    @testset "Vacuum (R_ab = 0): Segre {(1,111)}" begin
        # Schwarzschild or any vacuum spacetime: all Ricci eigenvalues = 0
        dim = 4
        Ric = zeros(dim, dim)
        ginv = diagm([-1.0, 1.0, 1.0, 1.0])
        st = segre_classify(Ric, ginv)
        @test st.notation == "{(1111)}"
        @test length(st.eigenvalues) == 1
        @test abs(st.eigenvalues[1]) < 1e-10
        @test st.multiplicities == [4]
        @test st.is_degenerate == false
    end

    @testset "de Sitter (R_ab = Lambda g_ab): Segre {(1111)}" begin
        # de Sitter: R_ab = Lambda * g_ab, so R^a_b = Lambda * delta^a_b
        # All eigenvalues = Lambda (fully degenerate)
        dim = 4
        Lambda = 3.0
        g = diagm([-1.0, 1.0, 1.0, 1.0])
        Ric = Lambda .* g
        ginv = diagm([-1.0, 1.0, 1.0, 1.0])
        st = segre_classify(Ric, ginv)
        @test st.notation == "{(1111)}"
        @test length(st.eigenvalues) == 1
        @test abs(st.eigenvalues[1] - Lambda) < 1e-10
        @test st.multiplicities == [4]
        @test st.is_degenerate == false
    end

    @testset "Perfect fluid: Segre {1,(111)}" begin
        # Perfect fluid: T^a_b = diag(-rho, p, p, p)
        # Via Einstein equations, R^a_b has one timelike eigenvalue and
        # three degenerate spacelike eigenvalues.
        dim = 4
        rho = 5.0
        p = 2.0
        # R^a_b for perfect fluid (mixed tensor, eigenvalues directly)
        R_mixed = diagm([-rho, p, p, p])
        st = segre_classify(R_mixed; atol=1e-10)
        @test st.notation == "{1,(111)}"
        @test length(st.eigenvalues) == 2
        @test st.multiplicities == [1, 3] || st.multiplicities == [3, 1]
        @test st.is_degenerate == false
    end

    @testset "Generic 4 distinct eigenvalues: Segre {1,111}" begin
        # R^a_b with 4 distinct real eigenvalues
        dim = 4
        R_mixed = diagm([1.0, 2.0, 3.0, 4.0])
        st = segre_classify(R_mixed; atol=1e-10)
        @test st.notation == "{1,111}"
        @test length(st.eigenvalues) == 4
        @test all(m == 1 for m in st.multiplicities)
        @test st.is_degenerate == false
    end

    @testset "Two pairs of degenerate eigenvalues: Segre {(11),(11)}" begin
        # Two eigenvalues, each with multiplicity 2
        dim = 4
        R_mixed = diagm([1.0, 1.0, 3.0, 3.0])
        st = segre_classify(R_mixed; atol=1e-10)
        @test st.notation == "{(11),(11)}"
        @test length(st.eigenvalues) == 2
        @test st.multiplicities == [2, 2]
        @test st.is_degenerate == false
    end

    @testset "One eigenvalue mult 3, one mult 1: Segre {1,(111)} or {(111),1}" begin
        # One eigenvalue with multiplicity 3, one with multiplicity 1
        dim = 4
        R_mixed = diagm([5.0, 2.0, 2.0, 2.0])
        st = segre_classify(R_mixed; atol=1e-10)
        @test st.notation == "{1,(111)}"
        @test length(st.eigenvalues) == 2
    end

    @testset "Jordan block: Segre type with degenerate flag" begin
        # A 2x2 Jordan block for eigenvalue 1, plus eigenvalues 3 and 4
        dim = 4
        R_mixed = [1.0 1.0 0.0 0.0;
                   0.0 1.0 0.0 0.0;
                   0.0 0.0 3.0 0.0;
                   0.0 0.0 0.0 4.0]
        st = segre_classify(R_mixed; atol=1e-10)
        @test st.is_degenerate == true
        @test length(st.eigenvalues) == 3
        # Eigenvalue 1 has multiplicity 2 with a Jordan block of size 2
        idx = findfirst(ev -> abs(ev - 1.0) < 1e-10, st.eigenvalues)
        @test idx !== nothing
        @test st.multiplicities[idx] == 2
        @test st.jordan_sizes[idx] == [2]
    end

    @testset "Mixed Ricci from Ric and ginv" begin
        # Verify that passing (Ric, ginv) gives same result as passing R_mixed
        dim = 4
        ginv = diagm([-1.0, 1.0, 1.0, 1.0])
        g = diagm([-1.0, 1.0, 1.0, 1.0])  # Minkowski metric (self-inverse)
        Ric = diagm([3.0, 1.0, 1.0, 1.0])
        R_mixed = ginv * Ric
        st1 = segre_classify(Ric, ginv)
        st2 = segre_classify(R_mixed)
        @test st1.notation == st2.notation
        @test st1.multiplicities == st2.multiplicities
    end

    @testset "3D case: all eigenvalues distinct" begin
        dim = 3
        R_mixed = diagm([1.0, 2.0, 3.0])
        st = segre_classify(R_mixed; atol=1e-10)
        @test length(st.eigenvalues) == 3
        @test all(m == 1 for m in st.multiplicities)
        @test st.is_degenerate == false
    end

    @testset "Non-diagonal mixed Ricci" begin
        # A non-diagonal matrix with known eigenvalues (similar to diag(1,2,3,4))
        dim = 4
        D = diagm([1.0, 2.0, 3.0, 4.0])
        # Construct a similarity transform
        P = [1.0 0.5 0.0 0.0;
             0.0 1.0 0.3 0.0;
             0.0 0.0 1.0 0.2;
             0.0 0.0 0.0 1.0]
        R_mixed = P * D * inv(P)
        st = segre_classify(R_mixed; atol=1e-8)
        @test st.notation == "{1,111}"
        @test length(st.eigenvalues) == 4
        @test st.is_degenerate == false
    end

    @testset "SegreType display" begin
        R_mixed = zeros(4, 4)
        st = segre_classify(R_mixed)
        @test sprint(show, st) == "SegreType({(1111)})"
    end

end
