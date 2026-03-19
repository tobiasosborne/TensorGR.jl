@testset "PPN-to-component bridge" begin

    function _ppn_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l]))
        define_ppn_potentials!(reg; manifold=:M4)
        reg
    end

    @testset "ppn_foliation!" begin
        reg = _ppn_reg()
        with_registry(reg) do
            fol = ppn_foliation!(reg)
            @test fol isa FoliationProperties
            @test fol.name == :ppn
            @test fol.manifold == :M4
            @test fol.temporal_component == 0
            @test fol.spatial_components == [1, 2, 3]
            @test fol.spatial_dim == 3

            # Auxiliary tensors registered
            @test has_tensor(reg, :delta_s)
            @test has_tensor(reg, :gamma_s)
            @test has_tensor(reg, :N_ppn)
            @test has_tensor(reg, :N_ppn_i)
        end
    end

    @testset "ppn_foliation! idempotent" begin
        reg = _ppn_reg()
        with_registry(reg) do
            fol1 = ppn_foliation!(reg)
            fol2 = ppn_foliation!(reg)
            @test fol1 === fol2
        end
    end

    @testset "ppn_decompose from dict" begin
        reg = _ppn_reg()
        with_registry(reg) do
            metric = ppn_metric_ansatz(ppn_gr(), reg; order=1)
            mc = ppn_decompose(metric)

            @test mc isa PPNMetricComponents
            @test mc.g00 isa TensorExpr
            @test mc.g0i isa TensorExpr
            @test mc.gij isa TensorExpr
        end
    end

    @testset "ppn_decompose convenience" begin
        reg = _ppn_reg()
        with_registry(reg) do
            mc = ppn_decompose(ppn_gr(), reg; order=1)
            @test mc isa PPNMetricComponents
        end
    end

    @testset "ppn_decompose from dict: missing key" begin
        bad = Dict{Tuple{Symbol,Symbol}, TensorExpr}()
        @test_throws ErrorException ppn_decompose(bad)
    end

    @testset "ppn_compose roundtrip" begin
        reg = _ppn_reg()
        with_registry(reg) do
            metric = ppn_metric_ansatz(ppn_gr(), reg; order=2)
            mc = ppn_decompose(metric)
            recomposed = ppn_compose(mc)

            @test recomposed[(:time, :time)] == metric[(:time, :time)]
            @test recomposed[(:time, :space)] == metric[(:time, :space)]
            @test recomposed[(:space, :space)] == metric[(:space, :space)]
        end
    end

    @testset "ppn_compose from components" begin
        g00 = TScalar(-1 // 1)
        g0i = TScalar(0 // 1)
        gij = TScalar(1 // 1)

        result = ppn_compose(g00, g0i, gij)
        @test result isa Dict
        @test result[(:time, :time)] == g00
        @test result[(:time, :space)] == g0i
        @test result[(:space, :space)] == gij
    end

    @testset "ppn_decompose: GR order=1 structure" begin
        reg = _ppn_reg()
        with_registry(reg) do
            mc = ppn_decompose(ppn_gr(), reg; order=1)

            # g00 at 1PN = -1 + 2U (a TSum)
            @test mc.g00 isa TSum
            @test length(mc.g00.terms) == 2  # -1 and +2U

            # g0i at 1PN = 0
            @test mc.g0i == TScalar(0 // 1)

            # gij at 1PN = (1 + 2U)delta_ij (should be a TSum or TProduct)
            @test mc.gij isa TensorExpr
        end
    end

    @testset "ppn_decompose: GR order=2 structure" begin
        reg = _ppn_reg()
        with_registry(reg) do
            mc = ppn_decompose(ppn_gr(), reg; order=2)

            # g00 at 2PN has more terms (U, U², Phi_W, Phi_1..4, A)
            @test mc.g00 isa TSum
            @test length(mc.g00.terms) > 2

            # g0i at 2PN is nonzero for GR (has V and W)
            @test mc.g0i != TScalar(0 // 1)
        end
    end

    @testset "ppn_christoffel_1pn" begin
        reg = _ppn_reg()
        with_registry(reg) do
            ppn_foliation!(reg)
            mc = ppn_decompose(ppn_gr(), reg; order=1)
            gamma = ppn_christoffel_1pn(mc; registry=reg)

            @test gamma isa PPNChristoffelComponents

            # Γ⁰₀₀ = ∂₀U (scalar, represented as TScalar placeholder)
            @test gamma.G000 isa TScalar

            # Γ⁰₀ⱼ = ∂ⱼU (derivative)
            @test gamma.G00j isa TDeriv

            # Γⁱ₀₀ = ∂ⁱU (derivative with up index)
            @test gamma.Gi00 isa TDeriv

            # Γⁱⱼₖ is a sum of 3 terms
            @test gamma.Gijk isa TSum
            @test length(gamma.Gijk.terms) == 3
        end
    end

    @testset "ppn_christoffel convenience" begin
        reg = _ppn_reg()
        with_registry(reg) do
            ppn_foliation!(reg)
            gamma = ppn_christoffel(ppn_gr(), reg; order=1)
            @test gamma isa PPNChristoffelComponents
        end
    end

    @testset "ppn_christoffel order validation" begin
        reg = _ppn_reg()
        with_registry(reg) do
            ppn_foliation!(reg)
            @test_throws ErrorException ppn_christoffel(ppn_gr(), reg; order=2)
        end
    end

    @testset "PPNMetricComponents display" begin
        mc = PPNMetricComponents(TScalar(1), TScalar(0), TScalar(1))
        s = sprint(show, mc)
        @test occursin("PPNMetricComponents", s)
    end

    @testset "PPNChristoffelComponents display" begin
        z = TScalar(0 // 1)
        gc = PPNChristoffelComponents(z, z, z, z, z, z)
        s = sprint(show, gc)
        @test occursin("PPNChristoffelComponents", s)
    end

end
