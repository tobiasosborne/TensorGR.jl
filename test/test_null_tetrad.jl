@testset "Newman-Penrose null tetrad" begin

    function _np_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_spinor_structure!(reg; manifold=:M4, metric=:g)
            define_null_tetrad!(reg; manifold=:M4, metric=:g)
        end
        reg
    end

    @testset "Registration" begin
        reg = _np_reg()
        @test has_tensor(reg, :np_l)
        @test has_tensor(reg, :np_n)
        @test has_tensor(reg, :np_m)
        @test has_tensor(reg, :np_mbar)

        for vname in (:np_l, :np_n, :np_m, :np_mbar)
            props = get_tensor(reg, vname)
            @test props.rank == (1, 0)
            @test props.options[:is_null_tetrad] == true
        end
    end

    @testset "Null conditions: v_a v^a = 0" begin
        reg = _np_reg()
        with_registry(reg) do
            for vname in (:np_l, :np_n, :np_m, :np_mbar)
                v_up = Tensor(vname, [up(:a)])
                v_dn = Tensor(vname, [down(:a)])
                prod = tproduct(1 // 1, TensorExpr[v_up, v_dn])
                result = simplify(prod)
                @test result == TScalar(0) || to_latex(result) == "0"
            end
        end
    end

    @testset "Normalization: l_a n^a = -1" begin
        reg = _np_reg()
        with_registry(reg) do
            l_dn = Tensor(:np_l, [down(:a)])
            n_up = Tensor(:np_n, [up(:a)])
            prod = tproduct(1 // 1, TensorExpr[l_dn, n_up])
            result = simplify(prod)
            @test result == TScalar(-1)
        end
    end

    @testset "Normalization: m_a mbar^a = 1" begin
        reg = _np_reg()
        with_registry(reg) do
            m_dn = Tensor(:np_m, [down(:a)])
            mbar_up = Tensor(:np_mbar, [up(:a)])
            prod = tproduct(1 // 1, TensorExpr[m_dn, mbar_up])
            result = simplify(prod)
            @test result == TScalar(1)
        end
    end

    @testset "Orthogonality: l_a m^a = 0" begin
        reg = _np_reg()
        with_registry(reg) do
            l_dn = Tensor(:np_l, [down(:a)])
            m_up = Tensor(:np_m, [up(:a)])
            prod = tproduct(1 // 1, TensorExpr[l_dn, m_up])
            result = simplify(prod)
            @test result == TScalar(0) || to_latex(result) == "0"
        end
    end

    @testset "Completeness relation structure" begin
        reg = _np_reg()
        with_registry(reg) do
            comp = np_completeness()
            @test comp isa TSum
            @test length(comp.terms) == 4
            # Each term should have two free indices a, b
            for term in comp.terms
                free = free_indices(term)
                @test length(free) == 2
            end
        end
    end

    @testset "Idempotent registration" begin
        reg = _np_reg()
        define_null_tetrad!(reg; manifold=:M4, metric=:g)
        @test has_tensor(reg, :np_l)
    end
end
