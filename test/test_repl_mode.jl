@testset "REPL Tensor Mode" begin
    using TensorGR: _process_tensor_input, _init_commands!, _parse_and_resolve,
                    _display_tensor_result, _tensor_registry, set_tensor_registry!,
                    TensorREPL,
                    TensorRegistry, Tensor, TProduct, TSum, TDeriv, TScalar,
                    up, down, with_registry, current_registry,
                    simplify, to_unicode, to_latex

    function _setup_repl_test()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        set_tensor_registry!(reg)
        _init_commands!()
        reg
    end

    # ── Name resolution ──────────────────────────────────────────────

    @testset "name resolution" begin
        reg = _setup_repl_test()

        @testset "R with 4 indices → Riem" begin
            expr = _parse_and_resolve("R_{abcd}")
            @test expr isa Tensor
            @test expr.name === :Riem
            @test length(expr.indices) == 4
        end

        @testset "R with 2 indices → Ric" begin
            expr = _parse_and_resolve("R_{ab}")
            @test expr isa Tensor
            @test expr.name === :Ric
        end

        @testset "R with 0 indices → RicScalar" begin
            expr = _parse_and_resolve("R")
            @test expr isa Tensor
            @test expr.name === :RicScalar
        end

        @testset "G with 2 indices → Ein" begin
            expr = _parse_and_resolve("G_{ab}")
            @test expr isa Tensor
            @test expr.name === :Ein
        end

        @testset "g stays g (registered metric)" begin
            expr = _parse_and_resolve("g_{ab}")
            @test expr isa Tensor
            @test expr.name === :g
        end

        @testset "resolves inside products" begin
            expr = _parse_and_resolve("g^{ab} R_{ab}")
            @test expr isa TProduct
            names = [f.name for f in expr.factors if f isa Tensor]
            @test :Ric in names
            @test :g in names
        end

        @testset "resolves inside sums" begin
            expr = _parse_and_resolve("R_{ab} + G_{ab}")
            @test expr isa TSum
            term_names = [t.name for t in expr.terms if t isa Tensor]
            @test :Ric in term_names
            @test :Ein in term_names
        end
    end

    # ── Simplification via tensor mode ───────────────────────────────

    @testset "simplify through process_tensor_input" begin
        reg = _setup_repl_test()

        @testset "metric trace: g^{ab}g_{ab} → 4" begin
            _process_tensor_input("g^{ab} g_{ab}")
            result = _process_tensor_input("simplify %")
            @test result == TScalar(4 // 1)
        end

        @testset "Ricci trace: g^{ab}R_{ab} → RicScalar" begin
            _process_tensor_input("g^{ab} R_{ab}")
            result = _process_tensor_input("simplify %")
            @test result isa Tensor
            @test result.name === :RicScalar
        end

        @testset "Einstein definition" begin
            expr = _parse_and_resolve("R_{ab} - \\frac{1}{2} g_{ab} R")
            @test expr isa TSum
            @test length(expr.terms) == 2
        end
    end

    # ── Commands ─────────────────────────────────────────────────────

    @testset "commands" begin
        reg = _setup_repl_test()

        @testset "terms command" begin
            _process_tensor_input("R_{ab} + G_{ab}")
            # terms command prints but returns the expr
            result = _process_tensor_input("terms %")
            @test result isa TSum
        end

        @testset "latex command" begin
            _process_tensor_input("R_{abcd}")
            result = _process_tensor_input("latex %")
            @test result isa Tensor
        end

        @testset "indices command" begin
            _process_tensor_input("R_{abcd}")
            result = _process_tensor_input("indices %")
            @test result isa Tensor
        end

        @testset "% with no prior result" begin
            TensorREPL._last_result[] = nothing
            @test_throws ErrorException _process_tensor_input("simplify %")
        end
    end

    # ── Help ─────────────────────────────────────────────────────────

    @testset "help" begin
        _setup_repl_test()
        result = _process_tensor_input("help")
        @test result === nothing
        result = _process_tensor_input("?")
        @test result === nothing
    end

    # ── Empty input ──────────────────────────────────────────────────

    @testset "empty input" begin
        @test _process_tensor_input("") === nothing
        @test _process_tensor_input("   ") === nothing
    end

    # ── Display ──────────────────────────────────────────────────────

    @testset "display produces output" begin
        _setup_repl_test()
        expr = _parse_and_resolve("R_{abcd}")
        buf = IOBuffer()
        _display_tensor_result(buf, expr)
        s = String(take!(buf))
        @test occursin("Riem", s)
    end

    # ── Wald ground truth calculations ───────────────────────────────

    @testset "Wald identities via tensor mode" begin
        reg = _setup_repl_test()

        @testset "metric is symmetric: g_{ab} = g_{ba} after simplify" begin
            g_ab = _parse_and_resolve("g_{ab}")
            g_ba = _parse_and_resolve("g_{ba}")
            r1 = with_registry(reg) do; simplify(g_ab); end
            r2 = with_registry(reg) do; simplify(g_ba); end
            @test r1 == r2
        end

        @testset "Ricci is symmetric" begin
            R_ab = _parse_and_resolve("R_{ab}")
            R_ba = _parse_and_resolve("R_{ba}")
            # After canonicalize, both should be identical
            r1 = with_registry(reg) do; simplify(R_ab); end
            r2 = with_registry(reg) do; simplify(R_ba); end
            @test r1 == r2
        end

        @testset "trace of metric = dimension" begin
            _process_tensor_input("g^{ab} g_{ab}")
            result = _process_tensor_input("simplify %")
            @test result == TScalar(4 // 1)
        end

        @testset "Ricci scalar from trace" begin
            _process_tensor_input("g^{ab} R_{ab}")
            result = _process_tensor_input("simplify %")
            @test result isa Tensor
            @test result.name === :RicScalar
        end
    end
end
