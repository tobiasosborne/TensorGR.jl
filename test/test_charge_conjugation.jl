# Ground truth: Wald, GR, Appendix B;
#               Freedman & Van Proeyen, Supergravity (2012), Ch 3.

@testset "Charge conjugation matrix" begin

    @testset "ChargeConjugation type" begin
        C = ChargeConjugation()
        @test C isa TensorExpr
        @test C == ChargeConjugation()
        @test isempty(indices(C))
        @test isempty(free_indices(C))
        @test isempty(children(C))
    end

    @testset "AST integration" begin
        C = ChargeConjugation()
        @test derivative_order(C) == 0
        @test is_well_formed(C)
        @test walk(identity, C) == C
        @test rename_dummy(C, :a, :b) == C
    end

    @testset "Display" begin
        C = ChargeConjugation()
        @test sprint(show, C) == "C"
        @test to_latex(C) == "C"
        @test to_unicode(C) == "C"
    end

    @testset "Properties" begin
        props = charge_conjugation_properties()

        # C^T = -C
        @test props.antisymmetric

        # C^{-1} = C^†
        @test props.unitary

        # C γ^a C^{-1} = -(γ^a)^T
        @test props.gamma_conjugation == -1

        # C σ^{ab} C^{-1} = -(σ^{ab})^T
        @test props.sigma_conjugation == -1

        # C γ^5 C^{-1} = (γ^5)^T
        @test props.gamma5_conjugation == 1
    end

    @testset "Majorana condition" begin
        mc = majorana_condition()
        @test mc isa String
        @test occursin("self-conjugate", mc)
    end

    @testset "Hermiticity" begin
        C = ChargeConjugation()
        @test dagger(C) == C
    end
end
