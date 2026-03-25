@testset "Symmetry-Reduced Metric Ansatz" begin
    using TensorGR: symmetry_reduce, MetricAnsatzResult,
                    independent_components, constrained_components,
                    SphericalSymmetry, AxialSymmetry, StaticSymmetry,
                    HomogeneousIsotropy

    # ── SphericalSymmetry ────────────────────────────────────────────

    @testset "SphericalSymmetry: Schwarzschild-type metric" begin
        sym = SphericalSymmetry(:M4)
        result = symmetry_reduce(sym)

        @test result isa MetricAnsatzResult
        @test result.dim == 4
        @test result.coords == [:t, :r, :theta, :phi]
        @test length(result.components) == 4  # diagonal
        @test result.free_functions == [:A_metric, :B_metric]
        @test result.essential_coords == [:r]  # functions of r only
        @test independent_components(result) == 2
        @test constrained_components(result) == 10 - 4  # 6 off-diagonal = 0
    end

    @testset "SphericalSymmetry: custom coords" begin
        sym = SphericalSymmetry(:M4)
        result = symmetry_reduce(sym; coords=[:tau, :rho, :th, :ph])
        @test result.coords == [:tau, :rho, :th, :ph]
        @test result.essential_coords == [:rho]
    end

    @testset "SphericalSymmetry: wrong number of coords" begin
        sym = SphericalSymmetry(:M4)
        @test_throws ErrorException symmetry_reduce(sym; coords=[:t, :r, :theta])
    end

    # ── StaticSymmetry ───────────────────────────────────────────────

    @testset "StaticSymmetry: time-independent metric" begin
        sym = StaticSymmetry(:M4)
        result = symmetry_reduce(sym)

        @test result.dim == 4
        @test result.coords == [:t, :x, :y, :z]
        # 1 lapse + 6 spatial metric components = 7
        @test length(result.components) == 7
        @test :N_lapse in result.free_functions
        @test length(result.free_functions) == 7
        @test result.essential_coords == [:x, :y, :z]
    end

    @testset "StaticSymmetry: no time cross-terms" begin
        sym = StaticSymmetry(:M4)
        result = symmetry_reduce(sym)
        # g_{ti} = 0 for static metric (no (1,2), (1,3), (1,4))
        for i in 2:4
            @test !haskey(result.components, (1, i))
        end
    end

    # ── HomogeneousIsotropy ──────────────────────────────────────────

    @testset "HomogeneousIsotropy: FLRW metric" begin
        sym = HomogeneousIsotropy(:M4)
        result = symmetry_reduce(sym; k=0)

        @test result.dim == 4
        @test length(result.components) == 4  # diagonal
        @test result.free_functions == [:a_scale]  # just the scale factor
        @test result.essential_coords == [:t]  # a depends on t only
        @test independent_components(result) == 1
    end

    @testset "HomogeneousIsotropy: curvature parameter" begin
        for k in [-1, 0, 1]
            sym = HomogeneousIsotropy(:M4)
            result = symmetry_reduce(sym; k=k)
            @test result.components[(2, 2)] == (:a_squared_over_f, k)
        end
    end

    @testset "HomogeneousIsotropy: g_{tt} = -1 exactly" begin
        sym = HomogeneousIsotropy(:M4)
        result = symmetry_reduce(sym)
        @test result.components[(1, 1)] == -1  # exact, not a function
    end

    @testset "HomogeneousIsotropy: invalid k" begin
        sym = HomogeneousIsotropy(:M4)
        @test_throws ErrorException symmetry_reduce(sym; k=2)
    end

    # ── AxialSymmetry ────────────────────────────────────────────────

    @testset "AxialSymmetry: Lewis-Papapetrou metric" begin
        sym = AxialSymmetry(:M4)
        result = symmetry_reduce(sym)

        @test result.dim == 4
        @test length(result.components) == 5  # 4 diagonal + 1 off-diagonal
        @test haskey(result.components, (1, 4))  # g_{tφ} cross-term (frame dragging)
        @test length(result.free_functions) == 5
        @test result.essential_coords == [:r, :theta]
    end

    @testset "AxialSymmetry: has frame-dragging" begin
        sym = AxialSymmetry(:M4)
        result = symmetry_reduce(sym)
        # g_{tφ} != 0 (frame dragging from rotation)
        @test haskey(result.components, (1, 4))
    end

    # ── Counting ─────────────────────────────────────────────────────

    @testset "component counting: 4D" begin
        # Total components in 4D symmetric metric: 4*5/2 = 10
        total = 10

        # Spherical + static: 2 free (A,B), 4 non-zero, 6 zero
        sph = symmetry_reduce(SphericalSymmetry(:M4))
        @test independent_components(sph) == 2
        @test constrained_components(sph) == 6

        # Static: 7 free (lapse + 6 spatial), 3 zero (cross-terms)
        sta = symmetry_reduce(StaticSymmetry(:M4))
        @test constrained_components(sta) == 3

        # FLRW: 1 free (scale factor), 4 non-zero, 6 zero
        flrw = symmetry_reduce(HomogeneousIsotropy(:M4))
        @test independent_components(flrw) == 1
        @test constrained_components(flrw) == 6

        # Axial: 5 free, 5 non-zero, 5 zero
        axial = symmetry_reduce(AxialSymmetry(:M4))
        @test independent_components(axial) == 5
        @test constrained_components(axial) == 5
    end

    @testset "symmetry hierarchy: more symmetry → fewer components" begin
        n_static = independent_components(symmetry_reduce(StaticSymmetry(:M4)))
        n_axial = independent_components(symmetry_reduce(AxialSymmetry(:M4)))
        n_spherical = independent_components(symmetry_reduce(SphericalSymmetry(:M4)))
        n_flrw = independent_components(symmetry_reduce(HomogeneousIsotropy(:M4)))

        # More symmetry → fewer free functions
        @test n_static >= n_axial  # axial adds rotation symmetry
        @test n_axial >= n_spherical  # spherical adds full SO(3)
        @test n_spherical >= n_flrw  # FLRW adds homogeneity
    end

    # ── Display ──────────────────────────────────────────────────────

    @testset "display" begin
        result = symmetry_reduce(SphericalSymmetry(:M4))
        s = sprint(show, result)
        @test occursin("MetricAnsatz", s)
        @test occursin("free functions", s)
    end
end
