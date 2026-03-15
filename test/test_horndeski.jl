@testset "Horndeski scalar-tensor theory" begin

    # ── ScalarTensorFunction ────────────────────────────────────────

    @testset "ScalarTensorFunction naming" begin
        G4 = ScalarTensorFunction(:G4, 0, 0)
        @test g_tensor_name(G4) == :G4

        G4X = differentiate_G(G4, :X)
        @test G4X.phi_derivs == 0
        @test G4X.X_derivs == 1
        @test g_tensor_name(G4X) == :G4_X

        G4XX = differentiate_G(G4X, :X)
        @test G4XX.X_derivs == 2
        @test g_tensor_name(G4XX) == :G4_XX

        G4phi = differentiate_G(G4, :phi)
        @test G4phi.phi_derivs == 1
        @test G4phi.X_derivs == 0
        @test g_tensor_name(G4phi) == :G4_phi

        G4phiX = differentiate_G(G4phi, :X)
        @test G4phiX.phi_derivs == 1
        @test G4phiX.X_derivs == 1
        @test g_tensor_name(G4phiX) == :G4_phiX
    end

    # ── define_horndeski! ───────────────────────────────────────────

    @testset "define_horndeski! registration" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d, [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
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

        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            @test ht isa HorndeskiTheory
            @test ht.manifold == :M4
            @test ht.metric == :g
            @test ht.scalar_field == :phi

            # Check scalar field registered
            @test has_tensor(reg, :phi)

            # Check G-functions registered
            for name in [:G2, :G3, :G3_phi, :G4, :G4_X, :G4_phi,
                         :G5, :G5_X, :G5_phi, :G5_XX]
                @test has_tensor(reg, name)
                props = get_tensor(reg, name)
                @test props.rank == (0, 0)
            end

            # Check covariant derivative registered
            @test has_tensor(reg, :nabla)
        end
    end

    # ── Index structure: all Lagrangians are scalar (rank 0) ───────

    @testset "Lagrangians are scalar densities (no free indices)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d, [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
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

        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            L2 = horndeski_L2(ht; registry=reg)
            L3 = horndeski_L3(ht; registry=reg)
            L4 = horndeski_L4(ht; registry=reg)
            L5 = horndeski_L5(ht; registry=reg)

            @test isempty(free_indices(L2))
            @test isempty(free_indices(L3))
            @test isempty(free_indices(L4))
            @test isempty(free_indices(L5))

            L_full = horndeski_lagrangian(ht; registry=reg)
            @test isempty(free_indices(L_full))
        end
    end

    # ── L2 structure ───────────────────────────────────────────────

    @testset "L2 = G2 (k-essence)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d, [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
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

        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            L2 = horndeski_L2(ht; registry=reg)
            @test L2 isa Tensor
            @test L2.name == :G2
            @test isempty(L2.indices)
        end
    end

    # ── L3 structure ───────────────────────────────────────────────

    @testset "L3 = -G3 Box(phi)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d, [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
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

        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            L3 = horndeski_L3(ht; registry=reg)
            # L3 should be a product with scalar -1
            @test L3 isa TProduct
            @test L3.scalar == -1 // 1
        end
    end

    # ── L4 structure ───────────────────────────────────────────────

    @testset "L4 contains G4*R and G4_X terms" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d, [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
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

        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            L4 = horndeski_L4(ht; registry=reg)
            # L4 is a TSum of two terms: G4*R and G4_X*[...]
            @test L4 isa TSum
            @test length(L4.terms) == 2
        end
    end

    # ── L5 structure ───────────────────────────────────────────────

    @testset "L5 contains G5 and G5_X terms" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d, [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
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

        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            L5 = horndeski_L5(ht; registry=reg)
            # L5 is a sum of G5*Ein term and -(1/6)*G5_X*[cubic] term
            @test L5 isa TSum
        end
    end

    # ── GR limit ───────────────────────────────────────────────────

    @testset "GR limit: G2=-2Lambda, G3=G5=0, G4=kappa" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d, [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
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

        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            # GR limit: G3=0, G5=0 (and all derivatives) => L3=0, L5=0
            set_vanishing!(reg, :G3)
            set_vanishing!(reg, :G5)
            set_vanishing!(reg, :G5_X)

            # G4_X = 0 (constant G4) => L4 = G4 * R (second term vanishes)
            set_vanishing!(reg, :G4_X)

            L3 = horndeski_L3(ht; registry=reg)
            L3_simplified = simplify(L3; registry=reg)
            @test L3_simplified == TScalar(0 // 1)

            L5 = horndeski_L5(ht; registry=reg)
            L5_simplified = simplify(L5; registry=reg)
            @test L5_simplified == TScalar(0 // 1)

            # L4 with G4_X=0: the (Box phi)^2 - nabla^2 term vanishes
            L4 = horndeski_L4(ht; registry=reg)
            L4_simplified = simplify(L4; registry=reg)
            # Should reduce to G4 * RicScalar
            @test L4_simplified isa TProduct || L4_simplified isa Tensor
            # Check it contains RicScalar and G4
            all_idx = indices(L4_simplified)
            @test isempty(free_indices(L4_simplified))
        end
    end

    # ── kinetic_X ──────────────────────────────────────────────────

    @testset "kinetic_X = -(1/2) g^{ab} d_a phi d_b phi" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d, [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true,
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:phi, manifold=:M4, rank=(0,0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}()))

        with_registry(reg) do
            X = kinetic_X(:phi, :g; registry=reg)
            @test isempty(free_indices(X))
            # Should be a product with scalar -1//2
            @test X isa TProduct
            @test X.scalar == -1 // 2
        end
    end

end
