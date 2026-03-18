@testset "Spinor dummy pair analysis (TGR-e1l)" begin

    # ── Test 1: free_indices with spinor indices ──
    @testset "free_indices spinor" begin
        psi = Tensor(:psi, [spin_down(:A)])
        fi = free_indices(psi)
        @test length(fi) == 1
        @test fi[1] == spin_down(:A)
        @test fi[1].vbundle == :SL2C
    end

    # ── Test 2: dummy pairs in spinor product ──
    @testset "spinor dummy pairs" begin
        # psi^A chi_A — contracted spinor pair
        psi = Tensor(:psi, [spin_up(:A)])
        chi = Tensor(:chi, [spin_down(:A)])
        prod = psi * chi

        fi = free_indices(prod)
        @test isempty(fi)  # A is a dummy

        # The pair should be recognized as a contraction
        all_idxs = indices(prod)
        a_up = count(idx -> idx.name == :A && idx.position == Up && idx.vbundle == :SL2C, all_idxs)
        a_down = count(idx -> idx.name == :A && idx.position == Down && idx.vbundle == :SL2C, all_idxs)
        @test a_up == 1
        @test a_down == 1
    end

    # ── Test 3: no cross-bundle pairing ──
    @testset "no cross-bundle pairing" begin
        # spin_up(:A, :SL2C) and spin_dot_down(:A, :SL2C_dot) should NOT pair
        t1 = Tensor(:psi, [spin_up(:A)])
        t2 = Tensor(:phi, [spin_dot_down(:A)])
        prod = t1 * t2

        fi = free_indices(prod)
        # Both are free because different vbundles
        @test length(fi) == 2
    end

    # ── Test 4: fresh_index with vbundle=:SL2C ──
    @testset "fresh_index SL2C" begin
        used = Set{Symbol}()
        idx = fresh_index(used; vbundle=:SL2C)
        @test idx == :A

        push!(used, :A)
        idx2 = fresh_index(used; vbundle=:SL2C)
        @test idx2 == :B

        # Exhaust A-F
        for s in [:B, :C, :D, :E, :F]
            push!(used, s)
        end
        idx7 = fresh_index(used; vbundle=:SL2C)
        @test idx7 == :A1  # Extended name
    end

    # ── Test 5: fresh_index with vbundle=:SL2C_dot ──
    @testset "fresh_index SL2C_dot" begin
        used = Set{Symbol}()
        idx = fresh_index(used; vbundle=:SL2C_dot)
        @test idx == :Ap

        push!(used, :Ap)
        idx2 = fresh_index(used; vbundle=:SL2C_dot)
        @test idx2 == :Bp
    end

    # ── Test 6: fresh_index default is Tangent (backward compat) ──
    @testset "fresh_index default Tangent" begin
        used = Set{Symbol}()
        idx = fresh_index(used)
        @test idx == :a  # lowercase, same as before

        idx2 = fresh_index(used; vbundle=:Tangent)
        @test idx2 == :a  # explicitly Tangent
    end

    # ── Test 7: mixed expression with Tangent and SL2C indices ──
    @testset "mixed tangent + spinor expression" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))
        define_spinor_bundles!(reg; manifold=:M4)

        with_registry(reg) do
            # sigma^a_{A Ap} — soldering form (tangent + undotted + dotted)
            sigma = Tensor(:sigma, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
            fi = free_indices(sigma)
            @test length(fi) == 3

            # Contract tangent index: sigma^a_{A Ap} V_a
            V = Tensor(:V, [down(:a)])
            prod = sigma * V
            fi2 = free_indices(prod)
            @test length(fi2) == 2  # A and Ap remain free
            # The two free indices should be spinor indices
            @test all(is_spinor_index, fi2)
        end
    end

    # ── Test 8: fresh_index for unknown vbundle falls back to Tangent alphabet ──
    @testset "fresh_index unknown vbundle fallback" begin
        used = Set{Symbol}()
        idx = fresh_index(used; vbundle=:SomeOtherBundle)
        @test idx == :a  # falls back to tangent alphabet
    end
end
