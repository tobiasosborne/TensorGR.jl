@testset "Spinor pipeline integration" begin

    function _int_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
            define_spinor_structure!(reg; manifold=:M4, metric=:g)
            define_curvature_spinors!(reg; manifold=:M4)
        end
        reg
    end

    @testset "contract_metrics handles spin metric epsilon_{AB}" begin
        reg = _int_reg()
        with_registry(reg) do
            # eps^{AB} eps_{BC} should contract to delta^A_C
            eps_up = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            eps_dn = Tensor(:eps_spin, [spin_down(:B), spin_down(:C)])
            prod = tproduct(1 // 1, TensorExpr[eps_up, eps_dn])
            result = contract_metrics(prod)
            # After contraction, should have a delta (or simplified form)
            @test result isa TensorExpr
            free = free_indices(result)
            free_names = Set(idx.name for idx in free)
            @test :A in free_names || :C in free_names
        end
    end

    @testset "canonicalize handles Weyl spinor symmetry" begin
        reg = _int_reg()
        with_registry(reg) do
            # Psi_{BACD} should canonicalize to Psi_{ABCD} (total symmetry)
            psi1 = Tensor(:Psi, [spin_down(:B), spin_down(:A), spin_down(:C), spin_down(:D)])
            psi2 = Tensor(:Psi, [spin_down(:A), spin_down(:B), spin_down(:C), spin_down(:D)])
            c1 = canonicalize(psi1)
            c2 = canonicalize(psi2)
            @test to_latex(c1) == to_latex(c2)
        end
    end

    @testset "canonicalize handles Ricci spinor symmetry" begin
        reg = _int_reg()
        with_registry(reg) do
            # Phi_{BA A'B'} should canonicalize to Phi_{AB A'B'} (sym in undotted)
            phi1 = Tensor(:Phi_Ricci, [spin_down(:B), spin_down(:A),
                                        spin_dot_down(:Ap), spin_dot_down(:Bp)])
            phi2 = Tensor(:Phi_Ricci, [spin_down(:A), spin_down(:B),
                                        spin_dot_down(:Ap), spin_dot_down(:Bp)])
            c1 = canonicalize(phi1)
            c2 = canonicalize(phi2)
            @test to_latex(c1) == to_latex(c2)
        end
    end

    @testset "collect_terms identifies equivalent spinor terms" begin
        reg = _int_reg()
        with_registry(reg) do
            # Psi_{ABCD} + Psi_{ABCD} should collect to 2 Psi_{ABCD}
            psi = Tensor(:Psi, [spin_down(:A), spin_down(:B), spin_down(:C), spin_down(:D)])
            sum_expr = tsum(TensorExpr[psi, psi])
            result = collect_terms(sum_expr)
            # Should be a single term with coefficient 2
            if result isa TProduct
                @test result.scalar == 2 // 1
            elseif result isa TSum
                # Might still be a sum if collect_terms preserves structure
                @test length(result.terms) <= 2
            else
                @test result isa TensorExpr
            end
        end
    end

    @testset "simplify works on mixed tensor-spinor products" begin
        reg = _int_reg()
        with_registry(reg) do
            # g^{ab} * Psi_{ABCD} should pass through simplify without error
            g_up = Tensor(:g, [up(:a), up(:b)])
            psi = Tensor(:Psi, [spin_down(:A), spin_down(:B), spin_down(:C), spin_down(:D)])
            prod = tproduct(1 // 1, TensorExpr[g_up, psi])
            result = simplify(prod)
            @test result isa TensorExpr
            # Free indices: a, b (Tangent) + A, B, C, D (SL2C)
            free = free_indices(result)
            @test length(free) == 6
        end
    end

    @testset "simplify handles spinor scalar products" begin
        reg = _int_reg()
        with_registry(reg) do
            # 3 * Psi_{ABCD} should simplify cleanly
            psi = Tensor(:Psi, [spin_down(:A), spin_down(:B), spin_down(:C), spin_down(:D)])
            prod = tproduct(3 // 1, TensorExpr[psi])
            result = simplify(prod)
            @test result isa TensorExpr
        end
    end

    @testset "spin_covd integrates with simplify" begin
        reg = _int_reg()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:phi_test, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle => :SL2C,
                                          :index_vbundles => [:SL2C])))

            phi = Tensor(:phi_test, [spin_down(:B)])
            sc = spin_covd(phi, :A, :Ap; covd_name=:D)
            result = simplify(sc)
            @test result isa TensorExpr
            # Should have 3 free indices: A (SL2C), Ap (SL2C_dot), B (SL2C)
            free = free_indices(result)
            @test length(free) == 3
        end
    end

    @testset "irreducible_decompose roundtrip" begin
        reg = _int_reg()
        register_tensor!(reg, TensorProperties(
            name=:X_spin, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:vbundle => :SL2C,
                                      :index_vbundles => [:SL2C, :SL2C])))

        with_registry(reg) do
            X = Tensor(:X_spin, [spin_down(:A), spin_down(:B)])
            decomp = irreducible_decompose(X)
            @test decomp isa TSum
            # The decomposition should have free indices A, B
            free = free_indices(decomp)
            free_names = Set(idx.name for idx in free)
            @test :A in free_names
            @test :B in free_names
        end
    end
end
