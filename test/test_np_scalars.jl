@testset "NP Weyl and Ricci scalars" begin

    function _np_scalar_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_spinor_structure!(reg; manifold=:M4, metric=:g)
            define_null_tetrad!(reg; manifold=:M4, metric=:g)
        end
        reg
    end

    @testset "Weyl scalar Psi_n definitions" begin
        reg = _np_scalar_reg()
        with_registry(reg) do
            for n in 0:4
                psi_n = weyl_scalar(n)
                @test psi_n isa TProduct

                # Should contain Weyl tensor + 4 tetrad vectors = 5 factors
                @test length(psi_n.factors) == 5

                # First factor is Weyl with 4 down indices
                @test psi_n.factors[1] isa Tensor
                @test psi_n.factors[1].name == :Weyl
                @test length(psi_n.factors[1].indices) == 4
                @test all(idx -> idx.position == Down, psi_n.factors[1].indices)

                # Remaining 4 factors are tetrad vectors with up indices
                for k in 2:5
                    @test psi_n.factors[k] isa Tensor
                    @test psi_n.factors[k].name in (:np_l, :np_n, :np_m, :np_mbar)
                    @test psi_n.factors[k].indices[1].position == Up
                end

                # Result is a scalar (all indices contracted)
                free = free_indices(psi_n)
                @test isempty(free)
            end
        end
    end

    @testset "Weyl scalar specific tetrad vectors" begin
        reg = _np_scalar_reg()
        with_registry(reg) do
            # Psi_0 = C_{abcd} l^a m^b l^c m^d
            psi0 = weyl_scalar(0)
            tetrad_names = [f.name for f in psi0.factors[2:5]]
            @test tetrad_names == [:np_l, :np_m, :np_l, :np_m]

            # Psi_2 = C_{abcd} l^a m^b mbar^c n^d
            psi2 = weyl_scalar(2)
            tetrad_names2 = [f.name for f in psi2.factors[2:5]]
            @test tetrad_names2 == [:np_l, :np_m, :np_mbar, :np_n]

            # Psi_4 = C_{abcd} n^a mbar^b n^c mbar^d
            psi4 = weyl_scalar(4)
            tetrad_names4 = [f.name for f in psi4.factors[2:5]]
            @test tetrad_names4 == [:np_n, :np_mbar, :np_n, :np_mbar]
        end
    end

    @testset "weyl_scalars returns all 5" begin
        reg = _np_scalar_reg()
        with_registry(reg) do
            all_psi = weyl_scalars()
            @test length(all_psi) == 5
            for (n, psi_n) in enumerate(all_psi)
                @test psi_n isa TProduct
                @test length(psi_n.factors) == 5
            end
        end
    end

    @testset "Invalid Weyl scalar index" begin
        reg = _np_scalar_reg()
        with_registry(reg) do
            @test_throws ErrorException weyl_scalar(5)
            @test_throws ErrorException weyl_scalar(-1)
        end
    end

    @testset "Ricci scalar Phi_{ij} definitions" begin
        reg = _np_scalar_reg()
        with_registry(reg) do
            for i in 0:2, j in 0:2
                phi_ij = ricci_scalar_np(i, j)

                if i == 1 && j == 1
                    # Phi_{11} is a sum of two terms
                    @test phi_ij isa TSum
                    @test length(phi_ij.terms) == 2
                else
                    # All others are single products
                    @test phi_ij isa TProduct

                    # Coefficient is -1/2
                    @test phi_ij.scalar == -1 // 2

                    # Contains Ricci + 2 tetrad vectors = 3 factors
                    @test length(phi_ij.factors) == 3
                    @test phi_ij.factors[1].name == :Ric

                    # Result is a scalar
                    free = free_indices(phi_ij)
                    @test isempty(free)
                end
            end
        end
    end

    @testset "Ricci scalar Phi_{00} structure" begin
        reg = _np_scalar_reg()
        with_registry(reg) do
            # Phi_{00} = -(1/2) R_{ab} l^a l^b
            phi00 = ricci_scalar_np(0, 0)
            @test phi00.factors[2].name == :np_l
            @test phi00.factors[3].name == :np_l
        end
    end

    @testset "Ricci scalar Phi_{22} structure" begin
        reg = _np_scalar_reg()
        with_registry(reg) do
            # Phi_{22} = -(1/2) R_{ab} n^a n^b
            phi22 = ricci_scalar_np(2, 2)
            @test phi22.factors[2].name == :np_n
            @test phi22.factors[3].name == :np_n
        end
    end

    @testset "Phi_{11} special structure" begin
        reg = _np_scalar_reg()
        with_registry(reg) do
            # Phi_{11} = -(1/4)(R_{ab} l^a n^b + R_{cd} m^c mbar^d)
            phi11 = ricci_scalar_np(1, 1)
            @test phi11 isa TSum
            @test length(phi11.terms) == 2

            # Each term has coefficient -1/4
            @test phi11.terms[1].scalar == -1 // 4
            @test phi11.terms[2].scalar == -1 // 4
        end
    end

    @testset "NP Lambda = R/24" begin
        reg = _np_scalar_reg()
        with_registry(reg) do
            lam = np_lambda()
            @test lam isa TProduct
            @test lam.scalar == 1 // 24
            @test length(lam.factors) == 1
            @test lam.factors[1].name == :RicScalar
        end
    end

    @testset "Invalid Ricci scalar indices" begin
        reg = _np_scalar_reg()
        with_registry(reg) do
            @test_throws ErrorException ricci_scalar_np(3, 0)
            @test_throws ErrorException ricci_scalar_np(0, 3)
            @test_throws ErrorException ricci_scalar_np(-1, 0)
        end
    end
end
