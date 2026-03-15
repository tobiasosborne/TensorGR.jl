@testset "Multi-field Horndeski scalar-tensor theory" begin

    # -- Helper: set up a fresh registry with manifold + metric + curvature --
    function _make_multi_horn_registry()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d,
            [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q,:r,:s,:t,:u,:v,:w]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true,
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1),
            symmetries=SymmetrySpec[],
            is_delta=true,
            options=Dict{Symbol,Any}(:is_delta => true)))
        define_curvature_tensors!(reg, :M4, :g)
        reg
    end

    # -- MultiScalarTensorFunction naming --

    @testset "MultiScalarTensorFunction naming" begin
        mg4 = MultiScalarTensorFunction(:MG4, 0, 0)
        @test multi_g_tensor_name(mg4) == :MG4

        mg4x = differentiate_MG(mg4, :X)
        @test mg4x.field_derivs == 0
        @test mg4x.X_derivs == 1
        @test multi_g_tensor_name(mg4x) == :MG4_X

        mg4xx = differentiate_MG(mg4x, :X)
        @test mg4xx.X_derivs == 2
        @test multi_g_tensor_name(mg4xx) == :MG4_XX

        mg4phi = differentiate_MG(mg4, :phi)
        @test mg4phi.field_derivs == 1
        @test mg4phi.X_derivs == 0
        @test multi_g_tensor_name(mg4phi) == :MG4_phi
    end

    # -- define_multi_horndeski! registration --

    @testset "define_multi_horndeski! registers tensors" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=2, manifold=:M4, metric=:g)

            @test mht isa MultiHorndeskiTheory
            @test mht.n_fields == 2
            @test mht.manifold == :M4
            @test mht.metric == :g
            @test mht.field_names == [:phi1, :phi2]
            @test mht.field_metric == :Gfield

            # Scalar fields registered
            @test has_tensor(reg, :phi1)
            @test has_tensor(reg, :phi2)

            # Field-space metric registered
            @test has_tensor(reg, :Gfield)

            # G-functions registered
            @test has_tensor(reg, :MG2)
            @test has_tensor(reg, :MG4)
            @test has_tensor(reg, :MG4_X)
            @test has_tensor(reg, :MG5)

            # Per-field G3 functions
            @test has_tensor(reg, Symbol("MG3_1"))
            @test has_tensor(reg, Symbol("MG3_2"))

            # Field-space VBundle
            @test has_vbundle(reg, :FieldSpace)
        end
    end

    @testset "define_multi_horndeski! custom field names" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=3, manifold=:M4, metric=:g,
                                           field_names=[:chi, :psi, :sigma])
            @test mht.field_names == [:chi, :psi, :sigma]
            @test has_tensor(reg, :chi)
            @test has_tensor(reg, :psi)
            @test has_tensor(reg, :sigma)
        end
    end

    # -- Kinetic matrix symmetry: X^{IJ} = X^{JI} --

    @testset "Kinetic matrix X^{IJ} symmetry" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=2, manifold=:M4, metric=:g)

            X12 = kinetic_matrix(mht, 1, 2; registry=reg)
            X21 = kinetic_matrix(mht, 2, 1; registry=reg)

            # Both should be scalar (no free spacetime indices)
            @test isempty(free_indices(X12))
            @test isempty(free_indices(X21))

            # X12 and X21 should have same structure: products with scalar -1//2
            @test X12 isa TProduct
            @test X21 isa TProduct
            @test X12.scalar == -1 // 2
            @test X21.scalar == -1 // 2

            # Both contain exactly one g tensor and two TDeriv factors
            function count_derivs(expr::TProduct)
                count(f -> f isa TDeriv, expr.factors)
            end
            @test count_derivs(X12) == 2
            @test count_derivs(X21) == 2
        end
    end

    @testset "Kinetic matrix diagonal X^{II}" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=2, manifold=:M4, metric=:g)

            # X^{11} = -(1/2) g^{ab} d_a phi1 d_b phi1
            X11 = kinetic_matrix(mht, 1, 1; registry=reg)
            @test X11 isa TProduct
            @test X11.scalar == -1 // 2
            @test isempty(free_indices(X11))

            # All TDeriv arguments should reference phi1
            for f in X11.factors
                if f isa TDeriv
                    @test f.arg isa Tensor
                    @test f.arg.name == :phi1
                end
            end
        end
    end

    @testset "Kinetic matrix full NxN" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=2, manifold=:M4, metric=:g)

            Xmat = kinetic_matrix_full(mht; registry=reg)
            @test size(Xmat) == (2, 2)

            # All entries scalar
            for I in 1:2, J in 1:2
                @test isempty(free_indices(Xmat[I, J]))
            end
        end
    end

    # -- L2 for multi-field theory --

    @testset "Multi-field L2 = MG2 (scalar)" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=2, manifold=:M4, metric=:g)

            L2 = multi_horndeski_L2(mht; registry=reg)
            @test L2 isa Tensor
            @test L2.name == :MG2
            @test isempty(L2.indices)
        end
    end

    # -- L3 for multi-field theory --

    @testset "Multi-field L3 sums over fields" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=2, manifold=:M4, metric=:g)

            L3 = multi_horndeski_L3(mht; registry=reg)
            # For N=2, L3 should be a TSum of 2 terms
            @test L3 isa TSum
            @test length(L3.terms) == 2
            @test isempty(free_indices(L3))
        end
    end

    @testset "Multi-field L3 N=1 has single term" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=1, manifold=:M4, metric=:g)

            L3 = multi_horndeski_L3(mht; registry=reg)
            # For N=1, L3 should be a single TProduct (not TSum)
            @test L3 isa TProduct
            @test L3.scalar == -1 // 1
            @test isempty(free_indices(L3))
        end
    end

    # -- L4 for multi-field theory --

    @testset "Multi-field L4 contains R coupling" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=2, manifold=:M4, metric=:g)

            L4 = multi_horndeski_L4(mht; registry=reg)
            @test L4 isa TSum
            @test isempty(free_indices(L4))
        end
    end

    @testset "Multi-field L4 N=1 same structure as single-field" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=1, manifold=:M4, metric=:g)

            L4 = multi_horndeski_L4(mht; registry=reg)
            # Should be a TSum of 2 terms: MG4*R and MG4_X*[bracket]
            @test L4 isa TSum
            @test length(L4.terms) == 2
            @test isempty(free_indices(L4))
        end
    end

    # -- N=1 reduction to standard Horndeski --

    @testset "N=1 reduction: to_single_field" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=1, manifold=:M4, metric=:g,
                                           field_names=[:phi])

            ht = to_single_field(mht; registry=reg)
            @test ht isa HorndeskiTheory
            @test ht.manifold == :M4
            @test ht.metric == :g
            @test ht.scalar_field == :phi

            # Single-field Horndeski functions should be registered
            @test has_tensor(reg, :G2)
            @test has_tensor(reg, :G3)
            @test has_tensor(reg, :G4)
            @test has_tensor(reg, :G5)
        end
    end

    @testset "N=1 reduction: L2 matches single-field" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            # Multi-field with N=1
            mht = define_multi_horndeski!(reg; n_fields=1, manifold=:M4, metric=:g,
                                           field_names=[:phi])
            L2_multi = multi_horndeski_L2(mht; registry=reg)

            # Single-field Horndeski
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            L2_single = horndeski_L2(ht; registry=reg)

            # Both are rank-0 tensors
            @test isempty(free_indices(L2_multi))
            @test isempty(free_indices(L2_single))
            @test L2_multi isa Tensor
            @test L2_single isa Tensor
        end
    end

    @testset "N=1 kinetic matrix matches kinetic_X" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=1, manifold=:M4, metric=:g,
                                           field_names=[:phi])

            X11 = kinetic_matrix(mht, 1, 1; registry=reg)

            # Compare with single-field kinetic_X
            X_single = kinetic_X(:phi, :g; registry=reg)

            # Both should be -(1/2) g^{ab} d_a phi d_b phi
            @test X11 isa TProduct
            @test X_single isa TProduct
            @test X11.scalar == X_single.scalar  # both -1//2
            @test isempty(free_indices(X11))
            @test isempty(free_indices(X_single))
        end
    end

    # -- Field space metric --

    @testset "Field space metric G_{IJ} properties" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=3, manifold=:M4, metric=:g)

            # Field metric is registered with Symmetric symmetry
            props = get_tensor(reg, :Gfield)
            @test props.rank == (0, 2)
            @test length(props.symmetries) == 1
            @test props.symmetries[1] isa Symmetric
            @test props.options[:is_field_metric] == true
            @test props.options[:n_fields] == 3
        end
    end

    # -- Edge cases --

    @testset "Error for n_fields=0" begin
        reg = _make_multi_horn_registry()
        @test_throws ErrorException define_multi_horndeski!(reg;
            n_fields=0, manifold=:M4, metric=:g)
    end

    @testset "Error for mismatched field_names" begin
        reg = _make_multi_horn_registry()
        @test_throws ErrorException define_multi_horndeski!(reg;
            n_fields=2, manifold=:M4, metric=:g,
            field_names=[:phi1])
    end

    @testset "Error for missing manifold" begin
        reg = TensorRegistry()
        @test_throws ErrorException define_multi_horndeski!(reg;
            n_fields=2, manifold=:M4, metric=:g)
    end

    @testset "Kinetic matrix out-of-range field index" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=2, manifold=:M4, metric=:g)
            @test_throws ErrorException kinetic_matrix(mht, 0, 1; registry=reg)
            @test_throws ErrorException kinetic_matrix(mht, 1, 3; registry=reg)
        end
    end

    @testset "to_single_field rejects N>1" begin
        reg = _make_multi_horn_registry()
        with_registry(reg) do
            mht = define_multi_horndeski!(reg; n_fields=2, manifold=:M4, metric=:g)
            @test_throws ErrorException to_single_field(mht; registry=reg)
        end
    end

end
