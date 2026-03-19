@testset "Irreducible spinor decomposition" begin

    function _irred_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_spinor_structure!(reg; manifold=:M4, metric=:g)
        end
        reg
    end

    @testset "Rank-2 undotted decomposition" begin
        reg = _irred_reg()
        # Register a generic rank-2 spinor phi_{AB} (no symmetry)
        register_tensor!(reg, TensorProperties(
            name=:phi2, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:vbundle => :SL2C,
                                      :index_vbundles => [:SL2C, :SL2C])))

        with_registry(reg) do
            phi = Tensor(:phi2, [spin_down(:A), spin_down(:B)])
            result = irreducible_decompose(phi)

            # Should be a TSum of symmetric part + trace part
            @test result isa TSum
            @test length(result.terms) >= 2
        end
    end

    @testset "Rank-2 dotted decomposition" begin
        reg = _irred_reg()
        register_tensor!(reg, TensorProperties(
            name=:phi2_dot, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:vbundle => :SL2C_dot,
                                      :index_vbundles => [:SL2C_dot, :SL2C_dot])))

        with_registry(reg) do
            phi = Tensor(:phi2_dot, [spin_dot_down(:Ap), spin_dot_down(:Bp)])
            result = irreducible_decompose(phi)
            @test result isa TSum
        end
    end

    @testset "Higher rank returns unchanged" begin
        reg = _irred_reg()
        define_weyl_spinor!(reg; manifold=:M4)

        with_registry(reg) do
            # Psi is rank-4 -- irreducible_decompose returns unchanged for now
            psi = Tensor(:Psi, [spin_down(:A), spin_down(:B), spin_down(:C), spin_down(:D)])
            result = irreducible_decompose(psi)
            @test result isa Tensor  # returned unchanged
        end
    end

    @testset "Scalar input unchanged" begin
        reg = _irred_reg()
        with_registry(reg) do
            s = TScalar(42)
            @test irreducible_decompose(s) == s
        end
    end

    @testset "Non-spinor tensor unchanged" begin
        reg = _irred_reg()
        with_registry(reg) do
            T = Tensor(:g, [down(:a), down(:b)])
            result = irreducible_decompose(T)
            @test result isa Tensor
            @test result.name == :g
        end
    end

    @testset "Single spinor index unchanged" begin
        reg = _irred_reg()
        register_tensor!(reg, TensorProperties(
            name=:psi1, manifold=:M4, rank=(0, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:vbundle => :SL2C,
                                      :index_vbundles => [:SL2C])))

        with_registry(reg) do
            psi = Tensor(:psi1, [spin_down(:A)])
            result = irreducible_decompose(psi)
            @test result isa Tensor  # unchanged
        end
    end
end
