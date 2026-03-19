# Ground truth: GHP (1973), Eq 3.2 (6 commutator equations).

@testset "GHP commutator relations" begin

    @testset "Table structure" begin
        table = ghp_commutator_table()
        @test length(table) == 6

        # All 6 commutator pairs
        pairs = Set([(r.op1, r.op2) for r in table])
        @test (:thorn, :thorn_prime) in pairs
        @test (:thorn, :edth) in pairs
        @test (:thorn, :edth_prime) in pairs
        @test (:thorn_prime, :edth) in pairs
        @test (:thorn_prime, :edth_prime) in pairs
        @test (:edth, :edth_prime) in pairs
    end

    @testset "Descriptions contain curvature terms" begin
        table = ghp_commutator_table()
        for r in table
            @test !isempty(r.description)
            @test r.description isa String
        end

        # [þ, þ'] should mention Psi_2 and Phi_11
        comm1 = first(r for r in table if r.op1 == :thorn && r.op2 == :thorn_prime)
        @test occursin("Ψ₂", comm1.description) || occursin("Psi", comm1.description)

        # [ð, ð'] should mention curvature
        comm6 = first(r for r in table if r.op1 == :edth && r.op2 == :edth_prime)
        @test occursin("Ψ₂", comm6.description) || occursin("Psi", comm6.description)
    end

    @testset "Weight consistency of commutators" begin
        ops = [:thorn, :thorn_prime, :edth, :edth_prime]
        for i in 1:length(ops)
            for j in (i+1):length(ops)
                w = ghp_commutator_weight_consistency(ops[i], ops[j])
                @test w isa GHPWeight
                # Commutator total shift = sum of individual shifts
                expected = ghp_weight_shift(ops[i]) + ghp_weight_shift(ops[j])
                @test w == expected
            end
        end
    end

    @testset "[þ, þ'] weight shift = {0, 0}" begin
        w = ghp_commutator_weight_consistency(:thorn, :thorn_prime)
        @test w == GHPWeight(0, 0)
    end

    @testset "[ð, ð'] weight shift = {0, 0}" begin
        w = ghp_commutator_weight_consistency(:edth, :edth_prime)
        @test w == GHPWeight(0, 0)
    end

    @testset "[þ, ð] weight shift = {2, 0}" begin
        w = ghp_commutator_weight_consistency(:thorn, :edth)
        @test w == GHPWeight(2, 0)
    end

    @testset "Display" begin
        table = ghp_commutator_table()
        s = sprint(show, table[1])
        @test occursin("thorn", s)
    end
end
