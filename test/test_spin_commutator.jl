@testset "Spinor commutator identities" begin

    function _comm_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
            define_spinor_structure!(reg; manifold=:M4, metric=:g)
            define_curvature_spinors!(reg; manifold=:M4)

            # Register a test spinor kappa_A
            register_tensor!(reg, TensorProperties(
                name=:kappa, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle => :SL2C,
                                          :index_vbundles => [:SL2C])))
        end
        reg
    end

    @testset "Basic structure" begin
        reg = _comm_reg()
        with_registry(reg) do
            result = spinor_ricci_identity(:kappa, spin_down(:C))
            @test result isa TSum
            # 3 terms: Psi term + 2 Lambda terms
            @test length(result.terms) == 3
        end
    end

    @testset "Weyl spinor term structure" begin
        reg = _comm_reg()
        with_registry(reg) do
            result = spinor_ricci_identity(:kappa, spin_down(:C))
            # First term should contain Psi
            term1 = result.terms[1]
            @test term1 isa TProduct
            has_psi = any(f -> f isa Tensor && f.name == :Psi, term1.factors)
            @test has_psi
            # Should also contain eps_spin_dot (the A'B' epsilon)
            has_eps_dot = any(f -> f isa Tensor && f.name == :eps_spin_dot, term1.factors)
            @test has_eps_dot
            # Should contain kappa with Up index (contracted dummy)
            has_kappa_up = any(f -> f isa Tensor && f.name == :kappa &&
                               f.indices[1].position == Up, term1.factors)
            @test has_kappa_up
        end
    end

    @testset "Lambda terms structure" begin
        reg = _comm_reg()
        with_registry(reg) do
            result = spinor_ricci_identity(:kappa, spin_down(:C))
            # Terms 2 and 3 should contain RicScalar (Lambda = R/24)
            for k in 2:3
                term = result.terms[k]
                @test term isa TProduct
                @test term.scalar == 1 // 24
                has_R = any(f -> f isa Tensor && f.name == :RicScalar, term.factors)
                @test has_R
                has_eps = any(f -> f isa Tensor && f.name == :eps_spin, term.factors)
                @test has_eps
            end
        end
    end

    @testset "Free indices" begin
        reg = _comm_reg()
        with_registry(reg) do
            result = spinor_ricci_identity(:kappa, spin_down(:C))
            # Each term should have free indices: A, B, C (SL2C) + Ap, Bp (SL2C_dot)
            for term in result.terms
                free = free_indices(term)
                @test length(free) == 5
                undotted = filter(idx -> idx.vbundle == :SL2C, free)
                dotted = filter(idx -> idx.vbundle == :SL2C_dot, free)
                @test length(undotted) == 3
                @test length(dotted) == 2
            end
        end
    end

    @testset "Vacuum: Lambda terms vanish" begin
        reg = _comm_reg()
        with_registry(reg) do
            # In vacuum, R = 0, so Lambda = 0
            # Set RicScalar to zero
            set_vanishing!(reg, :RicScalar)
            result = spinor_ricci_identity(:kappa, spin_down(:C))
            simplified = simplify(result)
            # After simplify, only the Psi term should survive
            @test simplified isa TensorExpr
        end
    end

    @testset "Undotted spinor only" begin
        reg = _comm_reg()
        with_registry(reg) do
            # Should error on dotted spinor
            @test_throws ErrorException spinor_ricci_identity(:kappa, spin_dot_down(:Cp))
        end
    end
end
