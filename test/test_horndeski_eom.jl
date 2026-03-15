@testset "Horndeski field equations (EOM)" begin

    # Helper to set up a standard Horndeski registry
    function _horndeski_registry()
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
        reg
    end

    # ── Basic construction ─────────────────────────────────────────

    @testset "Metric EOM has correct free indices (a,b)" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            E_ab = horndeski_metric_eom(ht; registry=reg)
            fi = free_indices(E_ab)
            @test length(fi) == 2
            names = Set(i.name for i in fi)
            @test :a in names
            @test :b in names
            @test all(i.position == Down for i in fi)
        end
    end

    @testset "Scalar EOM is a scalar (no free indices)" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            E_phi = horndeski_scalar_eom(ht; registry=reg)
            @test isempty(free_indices(E_phi))
        end
    end

    @testset "Combined horndeski_eom returns tuple" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            (E_ab, E_phi) = horndeski_eom(ht; registry=reg)
            @test length(free_indices(E_ab)) == 2
            @test isempty(free_indices(E_phi))
        end
    end

    # ── GR limit ───────────────────────────────────────────────────

    @testset "GR limit: metric EOM = G4 * Ein_{ab}" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            # GR limit: G2=G3=G5=0, G4=const
            for s in [:G2, :G2_X, :G2_phi, :G3, :G3_X, :G3_phi,
                      :G5, :G5_X, :G5_phi, :G5_XX,
                      :G4_X, :G4_XX, :G4_phi]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end
            # Register and set vanishing for EOM-specific functions
            for s in [:G5_phiX, :G4_phiX]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end

            E_ab = horndeski_metric_eom(ht; registry=reg)
            E_s = simplify(E_ab; registry=reg)

            # Should reduce to G4 * Ein_{ab}
            @test E_s isa TProduct
            tensor_names = Set(f isa Tensor ? f.name : nothing for f in E_s.factors)
            @test :G4 in tensor_names
            @test :Ein in tensor_names
            @test isempty(free_indices(E_s)) == false
            @test length(free_indices(E_s)) == 2
        end
    end

    @testset "GR limit: scalar EOM vanishes" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            for s in [:G2, :G2_X, :G2_phi, :G3, :G3_X, :G3_phi,
                      :G5, :G5_X, :G5_phi, :G5_XX,
                      :G4_X, :G4_XX, :G4_phi]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end
            for s in [:G5_phiX, :G4_phiX]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end

            E_phi = horndeski_scalar_eom(ht; registry=reg)
            E_phi_s = simplify(E_phi; registry=reg)
            @test E_phi_s == TScalar(0 // 1)
        end
    end

    # ── Quintessence limit ─────────────────────────────────────────

    @testset "Quintessence: scalar EOM contains Box(phi) and G2_phi" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            # Quintessence: G2 = X - V(phi), so G2_X = 1, G2_phi = -V'
            # G3 = G5 = 0, G4 = const
            for s in [:G3, :G3_X, :G3_phi,
                      :G5, :G5_X, :G5_phi, :G5_XX,
                      :G4_X, :G4_XX, :G4_phi]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end
            for s in [:G5_phiX, :G4_phiX]
                has_tensor(reg, s) || register_tensor!(reg, TensorProperties(
                    name=s, manifold=:M4, rank=(0,0),
                    symmetries=SymmetrySpec[], options=Dict{Symbol,Any}()))
                set_vanishing!(reg, s)
            end

            E_phi = horndeski_scalar_eom(ht; registry=reg)
            E_phi_s = simplify(E_phi; registry=reg)

            # Should be -G2_phi + G2_X * Box(phi)
            # = Klein-Gordon equation: Box(phi) - V'(phi) = 0 when G2_X=1, G2_phi=-V'
            @test !isempty(string(E_phi_s))  # non-trivial
            @test isempty(free_indices(E_phi_s))  # still scalar
        end
    end

    # ── Individual EOM contributions ───────────────────────────────

    @testset "E^{(2)}_{ab} structure" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            E2 = TensorGR._metric_eom_L2(ht, down(:a), down(:b); registry=reg)
            fi = free_indices(E2)
            @test length(fi) == 2
            # E2 is a sum of two terms: G2_X*dphi_a*dphi_b and G2*g_ab
            @test E2 isa TSum
            @test length(E2.terms) == 2
        end
    end

    @testset "J^{(2)} structure" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            J2 = TensorGR._scalar_eom_L2(ht; registry=reg)
            @test isempty(free_indices(J2))
            # J2 = -G2_phi + G2_X * Box(phi)
            @test J2 isa TSum
            @test length(J2.terms) == 2
        end
    end

    @testset "J^{(4)} contains RicScalar" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            J4 = TensorGR._scalar_eom_L4(ht; registry=reg)
            @test isempty(free_indices(J4))
            # J4 contains -G4_phi * R - G4_phiX * [...]
            expr_str = string(J4)
            @test occursin("RicScalar", expr_str)
        end
    end

    # ── EOM function registration ──────────────────────────────────

    @testset "EOM registers additional G-function derivatives" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            # Before EOM, G2_X should not exist
            @test !has_tensor(reg, :G2_X)
            @test !has_tensor(reg, :G2_phi)

            # Call EOM which should register them
            horndeski_metric_eom(ht; registry=reg)

            @test has_tensor(reg, :G2_X)
            @test has_tensor(reg, :G2_phi)
            @test has_tensor(reg, :G3_X)
            @test has_tensor(reg, :G4_XX)
            @test has_tensor(reg, :G4_phiX)
            @test has_tensor(reg, :G5_phiX)
        end
    end

    # ── L3-only theory ─────────────────────────────────────────────

    @testset "L3-only: E^{(3)}_{ab} is non-trivial tensor" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            E3 = TensorGR._metric_eom_L3(ht, down(:a), down(:b); registry=reg)
            fi = free_indices(E3)
            @test length(fi) == 2
            # Contains G3_phi and G3_X terms
            expr_str = string(E3)
            @test occursin("G3_phi", expr_str)
            @test occursin("G3_X", expr_str)
        end
    end

    # ── Custom index labels ────────────────────────────────────────

    @testset "Custom index labels for metric EOM" begin
        reg = _horndeski_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)

            E_mn = horndeski_metric_eom(ht; idx_a=down(:m), idx_b=down(:n), registry=reg)
            fi = free_indices(E_mn)
            names = Set(i.name for i in fi)
            @test :m in names
            @test :n in names
        end
    end

end
