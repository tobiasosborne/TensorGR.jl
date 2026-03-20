using Test
using TensorGR

@testset "Bimetric Field Equations" begin

    # Helper: collect all Tensor names from an expression
    function _tensor_names(expr::TensorExpr)
        names = Set{Symbol}()
        _collect_names!(names, expr)
        names
    end
    _collect_names!(s::Set{Symbol}, t::Tensor) = push!(s, t.name)
    _collect_names!(::Set{Symbol}, ::TScalar) = nothing
    function _collect_names!(s::Set{Symbol}, p::TProduct)
        for f in p.factors; _collect_names!(s, f); end
    end
    function _collect_names!(s::Set{Symbol}, ts::TSum)
        for t in ts.terms; _collect_names!(s, t); end
    end
    function _collect_names!(s::Set{Symbol}, d::TDeriv)
        _collect_names!(s, d.arg)
    end

    # Helper: standard bimetric setup
    function _eom_setup()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        define_curvature_tensors!(reg, :M4, :g)
        bs = define_bimetric!(reg, :g, :f; manifold=:M4)
        reg, bs
    end

    # ── Y-tensor tests ───────────────────────────────────────────────

    @testset "Y_0 = identity (delta)" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y0 = TensorGR.y_tensor(0, :S_g_f; registry=reg)
            @test Y0 isa Tensor
            @test Y0.name == :delta || Y0.name == :δ
            @test length(Y0.indices) == 2
            # One up, one down
            @test Y0.indices[1].position == Up
            @test Y0.indices[2].position == Down
        end
    end

    @testset "Y_1 has 2 terms" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y1 = TensorGR.y_tensor(1, :S_g_f; registry=reg)
            @test Y1 isa TSum
            @test length(Y1.terms) == 2

            fi = free_indices(Y1)
            @test length(fi) == 2
            ups = count(idx -> idx.position == Up, fi)
            downs = count(idx -> idx.position == Down, fi)
            @test ups == 1
            @test downs == 1
        end
    end

    @testset "Y_2 has 3 terms" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y2 = TensorGR.y_tensor(2, :S_g_f; registry=reg)
            @test Y2 isa TSum
            @test length(Y2.terms) == 3

            fi = free_indices(Y2)
            @test length(fi) == 2
            ups = count(idx -> idx.position == Up, fi)
            downs = count(idx -> idx.position == Down, fi)
            @test ups == 1
            @test downs == 1
        end
    end

    @testset "Y_3 has 4 terms" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y3 = TensorGR.y_tensor(3, :S_g_f; registry=reg)
            @test Y3 isa TSum
            @test length(Y3.terms) == 4

            fi = free_indices(Y3)
            @test length(fi) == 2
            ups = count(idx -> idx.position == Up, fi)
            downs = count(idx -> idx.position == Down, fi)
            @test ups == 1
            @test downs == 1
        end
    end

    @testset "Y_n out of range throws error" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            @test_throws ErrorException TensorGR.y_tensor(-1, :S_g_f; registry=reg)
            @test_throws ErrorException TensorGR.y_tensor(4, :S_g_f; registry=reg)
        end
    end

    @testset "Y_0 contains delta" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y0 = TensorGR.y_tensor(0, :S_g_f; registry=reg)
            names = _tensor_names(Y0)
            @test :δ in names
        end
    end

    @testset "Y_1 contains S and delta" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y1 = TensorGR.y_tensor(1, :S_g_f; registry=reg)
            names = _tensor_names(Y1)
            @test :S_g_f in names
            @test :δ in names
        end
    end

    @testset "Y_2 contains S and delta" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y2 = TensorGR.y_tensor(2, :S_g_f; registry=reg)
            names = _tensor_names(Y2)
            @test :S_g_f in names
            @test :δ in names
        end
    end

    @testset "Y_3 contains S and delta" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y3 = TensorGR.y_tensor(3, :S_g_f; registry=reg)
            names = _tensor_names(Y3)
            @test :S_g_f in names
            @test :δ in names
        end
    end

    # ── Y-tensor recursion verification ──────────────────────────────
    # Y_{n+1} should equal S * Y_n + (-1)^{n+1} e_{n+1} I structurally.
    # We verify term counts: if Y_n has N terms, then S*Y_n has N terms (each
    # multiplied by S), plus 1 new term from e_{n+1}*I, giving N+1 total.

    @testset "Y-tensor recursion: term count Y_0 -> Y_1" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y0 = TensorGR.y_tensor(0, :S_g_f; registry=reg)
            Y1 = TensorGR.y_tensor(1, :S_g_f; registry=reg)
            # Y_0 = delta (1 term), Y_1 should have 1 + 1 = 2 terms
            @test Y0 isa Tensor  # single term
            @test Y1 isa TSum
            @test length(Y1.terms) == 2
        end
    end

    @testset "Y-tensor recursion: term count Y_1 -> Y_2" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y1 = TensorGR.y_tensor(1, :S_g_f; registry=reg)
            Y2 = TensorGR.y_tensor(2, :S_g_f; registry=reg)
            # Y_1 has 2 terms, Y_2 should have 2 + 1 = 3 terms
            @test length(Y1.terms) == 2
            @test length(Y2.terms) == 3
        end
    end

    @testset "Y-tensor recursion: term count Y_2 -> Y_3" begin
        reg, bs = _eom_setup()
        with_registry(reg) do
            Y2 = TensorGR.y_tensor(2, :S_g_f; registry=reg)
            Y3 = TensorGR.y_tensor(3, :S_g_f; registry=reg)
            # Y_2 has 3 terms, Y_3 should have 3 + 1 = 4 terms
            @test length(Y2.terms) == 3
            @test length(Y3.terms) == 4
        end
    end

    # ── interaction_tensor_g tests ───────────────────────────────────

    @testset "interaction_tensor_g: rank-(0,2)" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            Vg = TensorGR.interaction_tensor_g(bs, params; registry=reg)
            @test Vg isa TensorExpr
            fi = free_indices(Vg)
            @test length(fi) == 2
            # Both indices down
            @test all(idx -> idx.position == Down, fi)
        end
    end

    @testset "interaction_tensor_g: contains metric g and S" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            Vg = TensorGR.interaction_tensor_g(bs, params; registry=reg)
            names = _tensor_names(Vg)
            @test :g in names
            @test :S_g_f in names
        end
    end

    @testset "interaction_tensor_g: zero betas -> zero" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=1, beta0=0, beta1=0, beta2=0, beta3=0, beta4=0)
        with_registry(reg) do
            Vg = TensorGR.interaction_tensor_g(bs, params; registry=reg)
            @test Vg isa TScalar
            @test Vg.val == 0 // 1
        end
    end

    @testset "interaction_tensor_g: cosmological constant (beta0 only)" begin
        reg, bs = _eom_setup()
        # Only beta0 nonzero: V^{(g)}_{ab} = -beta_0 * g_{ac} * Y_0^c_b = -beta_0 * g_{ac} * delta^c_b = -beta_0 * g_{ab}
        params = HassanRosenParams(; m_sq=1, beta0=:B0, beta1=0, beta2=0, beta3=0, beta4=0)
        with_registry(reg) do
            Vg = TensorGR.interaction_tensor_g(bs, params; registry=reg)
            @test Vg isa TProduct
            # Should contain g and delta (from Y_0 = delta)
            names = _tensor_names(Vg)
            @test :g in names
            @test :δ in names
            fi = free_indices(Vg)
            @test length(fi) == 2
            @test all(idx -> idx.position == Down, fi)
        end
    end

    @testset "interaction_tensor_g: cosmological constant (numeric beta0=1)" begin
        reg, bs = _eom_setup()
        # beta0=1 only: V^{(g)} = -g_{ac} delta^c_b
        # Overall sign from formula: -(-1)^0 = -1
        params = HassanRosenParams(; m_sq=1, beta0=1, beta1=0, beta2=0, beta3=0, beta4=0)
        with_registry(reg) do
            Vg = TensorGR.interaction_tensor_g(bs, params; registry=reg)
            @test Vg isa TProduct
            @test Vg.scalar == -1 // 1  # overall sign is -1
        end
    end

    # ── interaction_tensor_f tests ───────────────────────────────────

    @testset "interaction_tensor_f: rank-(0,2)" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            Vf = TensorGR.interaction_tensor_f(bs, params; registry=reg)
            @test Vf isa TensorExpr
            fi = free_indices(Vf)
            @test length(fi) == 2
            # Both indices down
            @test all(idx -> idx.position == Down, fi)
        end
    end

    @testset "interaction_tensor_f: contains metric f and Sinv" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            Vf = TensorGR.interaction_tensor_f(bs, params; registry=reg)
            names = _tensor_names(Vf)
            @test :f in names
            @test :Sinv_g_f in names
        end
    end

    @testset "interaction_tensor_f: registers Sinv tensor" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            TensorGR.interaction_tensor_f(bs, params; registry=reg)
            @test has_tensor(reg, :Sinv_g_f)
            @test get_tensor(reg, :Sinv_g_f).rank == (1, 1)
        end
    end

    @testset "interaction_tensor_f: zero interior betas -> zero" begin
        reg, bs = _eom_setup()
        # beta1=beta2=beta3=beta4=0 means f-equation interaction vanishes
        params = HassanRosenParams(; m_sq=1, beta0=0, beta1=0, beta2=0, beta3=0, beta4=0)
        with_registry(reg) do
            Vf = TensorGR.interaction_tensor_f(bs, params; registry=reg)
            @test Vf isa TScalar
            @test Vf.val == 0 // 1
        end
    end

    @testset "interaction_tensor_f: uses reversed betas (beta4 first)" begin
        reg, bs = _eom_setup()
        # Only beta4 nonzero: f-equation uses beta_{4-n} for n=0, so beta_4
        # This corresponds to the n=0 term: -(-1)^0 * beta_4 * f_{ac} * Y_0(gamma)^c_b
        # = -beta_4 * f_{ac} * delta^c_b = -beta_4 * f_{ab}
        params = HassanRosenParams(; m_sq=1, beta0=0, beta1=0, beta2=0, beta3=0, beta4=:B4)
        with_registry(reg) do
            Vf = TensorGR.interaction_tensor_f(bs, params; registry=reg)
            @test Vf isa TProduct
            names = _tensor_names(Vf)
            @test :f in names
            @test :δ in names
        end
    end

    # ── bimetric_eom_g tests ─────────────────────────────────────────

    @testset "bimetric_eom_g: well-formed expression" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            eom_g = TensorGR.bimetric_eom_g(bs, params; registry=reg)
            @test eom_g isa TensorExpr
        end
    end

    @testset "bimetric_eom_g: contains Einstein tensor" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            eom_g = TensorGR.bimetric_eom_g(bs, params; registry=reg)
            names = _tensor_names(eom_g)
            @test :Ein_g in names
        end
    end

    @testset "bimetric_eom_g: contains interaction terms (S and metric)" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            eom_g = TensorGR.bimetric_eom_g(bs, params; registry=reg)
            names = _tensor_names(eom_g)
            @test :S_g_f in names
            @test :g in names
        end
    end

    @testset "bimetric_eom_g: rank-(0,2) free indices" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            eom_g = TensorGR.bimetric_eom_g(bs, params; registry=reg)
            fi = free_indices(eom_g)
            @test length(fi) == 2
            @test all(idx -> idx.position == Down, fi)
        end
    end

    @testset "bimetric_eom_g: m^2=0 gives pure Einstein" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=0, beta0=1, beta1=1, beta2=1, beta3=1, beta4=1)
        with_registry(reg) do
            eom_g = TensorGR.bimetric_eom_g(bs, params; registry=reg)
            # With m^2=0, the interaction terms vanish, leaving just G_{ab}
            @test eom_g isa Tensor
            @test eom_g.name == :Ein_g
        end
    end

    # ── bimetric_eom_f tests ─────────────────────────────────────────

    @testset "bimetric_eom_f: well-formed expression" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            eom_f = TensorGR.bimetric_eom_f(bs, params; registry=reg)
            @test eom_f isa TensorExpr
        end
    end

    @testset "bimetric_eom_f: contains Einstein tensor for f" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            eom_f = TensorGR.bimetric_eom_f(bs, params; registry=reg)
            names = _tensor_names(eom_f)
            @test :Ein_f in names
        end
    end

    @testset "bimetric_eom_f: contains interaction terms (Sinv and metric f)" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            eom_f = TensorGR.bimetric_eom_f(bs, params; registry=reg)
            names = _tensor_names(eom_f)
            @test :Sinv_g_f in names
            @test :f in names
        end
    end

    @testset "bimetric_eom_f: rank-(0,2) free indices" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            eom_f = TensorGR.bimetric_eom_f(bs, params; registry=reg)
            fi = free_indices(eom_f)
            @test length(fi) == 2
            @test all(idx -> idx.position == Down, fi)
        end
    end

    @testset "bimetric_eom_f: m^2=0 gives pure Einstein" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=0, beta0=1, beta1=1, beta2=1, beta3=1, beta4=1)
        with_registry(reg) do
            eom_f = TensorGR.bimetric_eom_f(bs, params; registry=reg)
            # With m^2=0, the interaction terms vanish, leaving just G_{ab}[f]
            @test eom_f isa Tensor
            @test eom_f.name == :Ein_f
        end
    end

    # ── Cross-consistency tests ──────────────────────────────────────

    @testset "g and f equations have same free index structure" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            eom_g = TensorGR.bimetric_eom_g(bs, params; registry=reg)
            eom_f = TensorGR.bimetric_eom_f(bs, params; registry=reg)
            fi_g = free_indices(eom_g)
            fi_f = free_indices(eom_f)
            @test length(fi_g) == length(fi_f)
            @test all(idx -> idx.position == Down, fi_g)
            @test all(idx -> idx.position == Down, fi_f)
        end
    end

    @testset "g-equation uses S, f-equation uses S-inverse" begin
        reg, bs = _eom_setup()
        params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2, beta3=:b3, beta4=1)
        with_registry(reg) do
            eom_g = TensorGR.bimetric_eom_g(bs, params; registry=reg)
            eom_f = TensorGR.bimetric_eom_f(bs, params; registry=reg)
            names_g = _tensor_names(eom_g)
            names_f = _tensor_names(eom_f)
            # g-equation uses S (forward), f-equation uses Sinv (inverse)
            @test :S_g_f in names_g
            @test :Sinv_g_f in names_f
        end
    end

    @testset "Cosmological constant limit: beta0 only, V_g proportional to g" begin
        reg, bs = _eom_setup()
        # beta0=1, all others zero: V^{(g)} = -g_{ac} delta^c_b = -g_{ab}
        # So the g-equation is G_{ab} - m^2 g_{ab} = 0 (cosmological constant)
        params = HassanRosenParams(; m_sq=1, beta0=1, beta1=0, beta2=0, beta3=0, beta4=0)
        with_registry(reg) do
            Vg = TensorGR.interaction_tensor_g(bs, params; registry=reg)
            @test Vg isa TProduct
            # V_g should be -1 * g_{ac} * delta^c_b
            @test Vg.scalar == -1 // 1
            # Contains g and delta, NO S tensors
            names = _tensor_names(Vg)
            @test :g in names
            @test :δ in names
            @test !(:S_g_f in names)
        end
    end

end
