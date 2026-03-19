# Ground truth: Hassan & Rosen, JHEP 02 (2012) 126, Sec 3;
#              Torsello et al, CQG 37, 025013 (2020), Sec 4.

@testset "Bimetric mass eigenstates" begin

    function _bim_setup(; c=1, beta1=0, beta2=1, beta3=0, m_sq=1)
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        bs = define_bimetric!(reg, :g, :f; manifold=:M4)
        params = HassanRosenParams(m_sq=m_sq, beta0=0, beta1=beta1,
                                    beta2=beta2, beta3=beta3, beta4=0)
        bp = define_bimetric_perturbation!(reg, bs, params; background_ratio=c)
        reg, bs, bp
    end

    @testset "Eigenstate structure for c=1, beta2=1" begin
        reg, bs, bp = _bim_setup(c=1, beta2=1)
        result = TensorGR.bimetric_mass_eigenstates(bp)

        # Returns correct NamedTuple
        @test haskey(result, :massless)
        @test haskey(result, :massive)
        @test haskey(result, :m2_FP)

        # Both eigenstates are TSum (linear combinations)
        @test result.massless isa TSum
        @test result.massive isa TSum

        # Each has exactly 2 terms (delta_g and delta_f contributions)
        @test length(result.massless.terms) == 2
        @test length(result.massive.terms) == 2
    end

    @testset "Normalization coefficients for c=1" begin
        # c=1: norm = 1/(1+1) = 1/2
        # Massless: (1/2)(delta_g + delta_f) -> coefficients 1/2 and 1/2
        # Massive:  (1/2)(delta_g - delta_f) -> coefficients 1/2 and -1/2
        reg, bs, bp = _bim_setup(c=1, beta2=1)
        result = TensorGR.bimetric_mass_eigenstates(bp)

        function get_coeff(term)
            term isa TProduct ? term.scalar : 1 // 1
        end

        # For c=1: massless = (1/2)delta_g + (1/2)delta_f
        m_coeffs = sort([get_coeff(t) for t in result.massless.terms])
        @test m_coeffs == [1 // 2, 1 // 2]

        # For c=1: massive = (1/2)delta_g - (1/2)delta_f
        v_coeffs = sort([get_coeff(t) for t in result.massive.terms])
        @test v_coeffs == [-1 // 2, 1 // 2]
    end

    @testset "Normalization coefficients for c=2" begin
        # c=2: norm = 1/(1+4) = 1/5, c^2 = 4
        # Massless: (1/5)(delta_g + 4 delta_f) -> coefficients 1/5 and 4/5
        # Massive:  (1/5)(delta_g - delta_f) -> coefficients 1/5 and -1/5
        reg, bs, bp = _bim_setup(c=2, beta2=1)
        result = TensorGR.bimetric_mass_eigenstates(bp)

        function get_coeff(term)
            term isa TProduct ? term.scalar : 1 // 1
        end

        m_coeffs = sort([get_coeff(t) for t in result.massless.terms])
        @test m_coeffs == [1 // 5, 4 // 5]

        v_coeffs = sort([get_coeff(t) for t in result.massive.terms])
        @test v_coeffs == [-1 // 5, 1 // 5]
    end

    @testset "Decoupling: mass matrix null space is (1,1)" begin
        # The mass matrix M has null vector (1,1) meaning delta_g + delta_f is massless.
        # The other eigenvector is (c^2, -1).
        # These are the raw mass matrix eigenvectors; the physical eigenstates in
        # bimetric_mass_eigenstates include kinetic normalization factors.
        for c in [1, 2, 3]
            params = HassanRosenParams(m_sq=1, beta0=0, beta1=1, beta2=1, beta3=1, beta4=0)
            M = bimetric_mass_matrix(params, c)

            # Null vector: (1, 1)
            v0 = [1, 1]
            Mv0 = M * v0
            @test Mv0[1] == 0
            @test Mv0[2] == 0

            # Other eigenvector: (c^2, -1)
            v1 = [c^2, -1]
            Mv1 = M * v1
            # Mv1 should be proportional to (c^2, -1)
            # Check: Mv1[1] * (-1) == Mv1[2] * c^2
            @test Mv1[1] * (-1) == Mv1[2] * c^2
            @test Mv1[1] != 0  # non-trivial eigenvalue
        end
    end

    @testset "Decoupling: mass matrix for c=1 applied to eigenstates" begin
        reg, bs, bp = _bim_setup(c=1, beta2=1)
        M = bimetric_mass_matrix(bp.params, 1)

        # For c=1, massless eigenvector is (1,1), massive is (1,-1)
        # Both conventions agree at c=1.
        v_massless = [1, 1]
        Mv = M * v_massless
        @test Mv[1] == 0
        @test Mv[2] == 0

        v_massive = [1, -1]
        Mv2 = M * v_massive
        @test Mv2[1] != 0
        @test Mv2[1] == -Mv2[2]  # proportional to (1,-1)
    end

    @testset "Inverse transform: round-trip for c=1" begin
        reg, bs, bp = _bim_setup(c=1, beta2=1)
        inv = TensorGR.bimetric_inverse_transform(bp)

        @test inv.delta_g isa TSum
        @test inv.delta_f isa TSum
        @test length(inv.delta_g.terms) == 2
        @test length(inv.delta_f.terms) == 2

        function get_coeff(term)
            term isa TProduct ? term.scalar : 1 // 1
        end

        # For c=1: delta_g = gamma + chi (both coefficient 1)
        dg_coeffs = sort([get_coeff(t) for t in inv.delta_g.terms])
        @test dg_coeffs == [1 // 1, 1 // 1]

        # delta_f = gamma - chi (coefficients 1, -1)
        df_coeffs = sort([get_coeff(t) for t in inv.delta_f.terms])
        @test df_coeffs == [-1 // 1, 1 // 1]
    end

    @testset "Inverse transform for c=2" begin
        reg, bs, bp = _bim_setup(c=2, beta2=1)
        inv = TensorGR.bimetric_inverse_transform(bp)

        function get_coeff(term)
            term isa TProduct ? term.scalar : 1 // 1
        end

        # delta_g = gamma + 4*chi
        dg_coeffs = sort([get_coeff(t) for t in inv.delta_g.terms])
        @test dg_coeffs == [1 // 1, 4 // 1]

        # delta_f = gamma - chi
        df_coeffs = sort([get_coeff(t) for t in inv.delta_f.terms])
        @test df_coeffs == [-1 // 1, 1 // 1]
    end

    @testset "m2_FP matches fierz_pauli_mass_squared" begin
        for (c, b1, b2, b3) in [(1, 0, 1, 0), (2, 1, 0, 1), (1, 1, 1, 1), (3, 0, 2, 0)]
            reg, bs, bp = _bim_setup(c=c, beta1=b1, beta2=b2, beta3=b3)
            result = TensorGR.bimetric_mass_eigenstates(bp)
            @test result.m2_FP == fierz_pauli_mass_squared(bp.params, c)
        end
    end

    @testset "Algebraic round-trip: forward then inverse recovers identity for c=1" begin
        # For c=1: norm=1/2
        # Forward: [gamma; chi] = (1/2)[1 1; 1 -1] [delta_g; delta_f]
        # Inverse: [delta_g; delta_f] = [1 1; 1 -1] [gamma; chi]
        # Product: [1 1; 1 -1] * (1/2)[1 1; 1 -1] = (1/2)[2 0; 0 2] = I
        T = Rational{Int}[1//2 1//2; 1//2 -1//2]
        S = Rational{Int}[1 1; 1 -1]
        @test S * T == Rational{Int}[1 0; 0 1]
    end

    @testset "Algebraic round-trip for general c" begin
        # Forward: gamma = (1/(1+c^2))(delta_g + c^2 delta_f)
        #          chi   = (1/(1+c^2))(delta_g - delta_f)
        # T = (1/(1+c^2)) [1 c^2; 1 -1]
        # Inverse: delta_g = gamma + c^2 chi
        #          delta_f = gamma - chi
        # S = [1 c^2; 1 -1]
        # S*T = (1/(1+c^2)) [1+c^2  c^2-c^2; 1-1  c^2+1]
        #     = (1/(1+c^2)) [1+c^2  0; 0  1+c^2] = I
        # Round-trip works for ALL c.
        for c in [1, 2, 3, 5]
            norm = 1 // (1 + c^2)
            c2 = c^2 // 1
            T = [norm norm*c2; norm -norm]
            S = Rational{Int}[1 c2; 1 -1]
            @test S * T == Rational{Int}[1 0; 0 1]
        end
    end

    @testset "Eigenstate tensor names are correct" begin
        reg, bs, bp = _bim_setup(c=1, beta2=1)
        result = TensorGR.bimetric_mass_eigenstates(bp)

        function tensor_names(expr::TSum)
            names = Symbol[]
            for term in expr.terms
                if term isa TProduct
                    for f in term.factors
                        f isa Tensor && push!(names, f.name)
                    end
                elseif term isa Tensor
                    push!(names, term.name)
                end
            end
            sort(names)
        end

        @test tensor_names(result.massless) == sort([bp.delta_g, bp.delta_f])
        @test tensor_names(result.massive) == sort([bp.delta_g, bp.delta_f])
    end

    @testset "Inverse transform tensor names are correct" begin
        reg, bs, bp = _bim_setup(c=1, beta2=1)
        inv = TensorGR.bimetric_inverse_transform(bp)

        function tensor_names(expr::TSum)
            names = Symbol[]
            for term in expr.terms
                if term isa TProduct
                    for f in term.factors
                        f isa Tensor && push!(names, f.name)
                    end
                elseif term isa Tensor
                    push!(names, term.name)
                end
            end
            sort(names)
        end

        @test tensor_names(inv.delta_g) == sort([bp.massless_mode, bp.massive_mode])
        @test tensor_names(inv.delta_f) == sort([bp.massless_mode, bp.massive_mode])
    end

    @testset "Zero mass parameters: both modes become massless" begin
        reg, bs, bp = _bim_setup(c=1, beta1=0, beta2=0, beta3=0)
        result = TensorGR.bimetric_mass_eigenstates(bp)
        @test result.m2_FP == 0
    end

    @testset "Eigenstates have correct index structure" begin
        # Both eigenstates should carry two down indices
        reg, bs, bp = _bim_setup(c=1, beta2=1)
        result = TensorGR.bimetric_mass_eigenstates(bp)

        for term in result.massless.terms
            if term isa TProduct
                for f in term.factors
                    if f isa Tensor
                        @test length(f.indices) == 2
                        @test all(idx -> idx.position == Down, f.indices)
                    end
                end
            end
        end
    end
end
