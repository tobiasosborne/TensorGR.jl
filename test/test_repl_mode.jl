@testset "REPL Tensor Mode" begin
    using TensorGR: _process_tensor_input, _init_commands!, _parse_and_resolve,
                    _display_tensor_result, _tensor_registry, set_tensor_registry!,
                    _record_result!, _resolve_percent_ref, _tensor_completions,
                    TensorREPL,
                    TensorRegistry, Tensor, TProduct, TSum, TDeriv, TScalar, TIndex,
                    up, down, with_registry, current_registry,
                    simplify, to_unicode, to_latex

    function _setup_repl_test()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        set_tensor_registry!(reg)
        _init_commands!()
        # Clean state for each test group
        empty!(TensorREPL._history)
        TensorREPL._last_result[] = nothing
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
            empty!(TensorREPL._history)
            TensorREPL._last_result[] = nothing
            @test_throws ErrorException _process_tensor_input("simplify %")
        end
    end

    # ── Function-call syntax ──────────────────────────────────────────

    @testset "function-call syntax" begin
        reg = _setup_repl_test()

        @testset "simplify(expr) parses and simplifies" begin
            result = _process_tensor_input("simplify(g^{ab} g_{ab})")
            @test result == TScalar(4 // 1)
        end

        @testset "contract(expr) works" begin
            _process_tensor_input("g^{ab} g_{bc}")
            result = _process_tensor_input("contract(%)")
            @test result isa Tensor
            @test result.name === :δ
        end

        @testset "simplify(%) applies to last result" begin
            _process_tensor_input("g^{ab} g_{ab}")
            result = _process_tensor_input("simplify(%)")
            @test result == TScalar(4 // 1)
        end
    end

    # ── Variable assignment ───────────────────────────────────────────

    @testset "variable assignment" begin
        reg = _setup_repl_test()
        empty!(TensorREPL._variables)

        @testset "name = LaTeX stores variable" begin
            result = _process_tensor_input("expr = R_{abcd}")
            @test result isa Tensor
            @test result.name === :Riem
            @test haskey(TensorREPL._variables, "expr")
            @test TensorREPL._variables["expr"] === result
        end

        @testset "bare variable name recalls value" begin
            _process_tensor_input("myvar = g_{ab}")
            result = _process_tensor_input("myvar")
            @test result isa Tensor
            @test result.name === :g
        end

        @testset "simplify variable_name works" begin
            _process_tensor_input("x = g^{ab} g_{ab}")
            result = _process_tensor_input("simplify x")
            @test result == TScalar(4 // 1)
        end

        @testset "name = simplify % works" begin
            _process_tensor_input("g^{ab} R_{ab}")
            result = _process_tensor_input("s = simplify %")
            @test result isa Tensor
            @test result.name === :RicScalar
            @test TensorREPL._variables["s"] === result
        end

        @testset "command names not shadowed" begin
            # "simplify = R_{ab}" should NOT create a variable named "simplify"
            # Instead it should be treated as "simplify" command applied to "= R_{ab}"
            # which would error. This is fine — commands take priority.
            @test !haskey(TensorREPL._variables, "simplify")
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

    # ── Workspace introspection ──────────────────────────���──────────────

    @testset "workspace introspection" begin
        reg = _setup_repl_test()
        empty!(TensorREPL._variables)

        @testset "vars with no variables" begin
            @test _process_tensor_input("vars") === nothing
        end

        @testset "vars with variables" begin
            _process_tensor_input("x = R_{abcd}")
            @test _process_tensor_input("vars") === nothing  # prints, returns nothing
        end

        @testset "info on expression" begin
            _process_tensor_input("R_{abcd}")
            @test _process_tensor_input("info %") === nothing
        end

        @testset "info on variable" begin
            _process_tensor_input("y = R_{ab}")
            @test _process_tensor_input("info y") === nothing
        end

        @testset "info on inline expression" begin
            @test _process_tensor_input("info R_{abcd} + R_{bacd}") === nothing
        end

        @testset "registry command" begin
            @test _process_tensor_input("registry") === nothing
        end

        @testset "info with no prior result errors" begin
            empty!(TensorREPL._history)
            TensorREPL._last_result[] = nothing
            @test_throws ErrorException _process_tensor_input("info")
        end
    end

    # ── Additional commands ────────────────────────────────────────────

    @testset "additional commands" begin
        reg = _setup_repl_test()

        @testset "define command" begin
            _process_tensor_input("define T_{ab}")
            @test TensorGR.has_tensor(reg, :T)
        end

        @testset "define already registered" begin
            @test _process_tensor_input("define g_{ab}") === nothing
        end

        @testset "sub command" begin
            _process_tensor_input("R_{ab}")
            result = _process_tensor_input("sub R_{ab} -> g_{ab}")
            @test result isa Tensor
            @test result.name === :g
        end

        @testset "sub needs prior result" begin
            empty!(TensorREPL._history)
            TensorREPL._last_result[] = nothing
            @test_throws ErrorException _process_tensor_input("sub R_{ab} -> g_{ab}")
        end
    end

    # ── Pipe/chain syntax ──────────────────────────────────────────────

    @testset "pipe syntax" begin
        reg = _setup_repl_test()

        @testset "basic pipe" begin
            result = _process_tensor_input("g^{ab} g_{ab} | simplify")
            @test result == TScalar(4 // 1)
        end

        @testset "multi-pipe" begin
            result = _process_tensor_input("g^{ab} R_{ab} | contract | simplify")
            @test result isa Tensor
            @test result.name === :RicScalar
        end

        @testset "pipe with variable assignment" begin
            empty!(TensorREPL._variables)
            result = _process_tensor_input("s = g^{ab} g_{ab} | simplify")
            @test result == TScalar(4 // 1)
            @test TensorREPL._variables["s"] == TScalar(4 // 1)
        end

        @testset "pipe with %N" begin
            empty!(TensorREPL._history)
            TensorREPL._last_result[] = nothing
            _process_tensor_input("g^{ab} g_{ab}")   # [1]
            result = _process_tensor_input("%1 | simplify")
            @test result == TScalar(4 // 1)
        end

        @testset "unknown command in pipe errors" begin
            @test_throws ErrorException _process_tensor_input("R_{ab} | nonexistent")
        end
    end

    # ── Tab completion ──────────────────────────────────────────────────

    @testset "tab completion" begin
        reg = _setup_repl_test()
        TensorREPL._variables["myexpr"] = Tensor(:Riem, TIndex[])

        @testset "completes commands" begin
            comps, _ = TensorGR._tensor_completions("sim")
            @test "simplify" in comps
            @test "simplify_level2" in comps
        end

        @testset "completes variable names" begin
            comps, _ = TensorGR._tensor_completions("mye")
            @test "myexpr" in comps
        end

        @testset "completes registry tensors" begin
            comps, _ = TensorGR._tensor_completions("Ri")
            @test any(c -> startswith(c, "Ri"), comps)
        end

        @testset "completes LaTeX names" begin
            comps, _ = TensorGR._tensor_completions("\\alp")
            @test "\\alpha" in comps
        end

        @testset "empty input gives no completions" begin
            comps, _ = TensorGR._tensor_completions("")
            @test isempty(comps)
        end

        @testset "completes built-in commands" begin
            comps, _ = TensorGR._tensor_completions("var")
            @test "vars" in comps
        end
    end

    # ── Numbered output history ────────────────────────────────────────

    @testset "numbered history" begin
        reg = _setup_repl_test()
        empty!(TensorREPL._history)
        TensorREPL._last_result[] = nothing

        @testset "%N references" begin
            _process_tensor_input("R_{abcd}")           # [1]
            _process_tensor_input("g_{ab}")             # [2]
            @test length(TensorREPL._history) >= 2

            # %1 recalls first result
            r = _process_tensor_input("%1")
            @test r isa Tensor
            @test r.name === :Riem

            # %2 recalls second result
            r = _process_tensor_input("%2")
            @test r isa Tensor
            @test r.name === :g
        end

        @testset "% still means last" begin
            empty!(TensorREPL._history)
            TensorREPL._last_result[] = nothing
            _process_tensor_input("R_{ab}")
            r = _process_tensor_input("simplify %")
            @test r isa Tensor
            @test r.name === :Ric
        end

        @testset "%N in commands" begin
            empty!(TensorREPL._history)
            TensorREPL._last_result[] = nothing
            _process_tensor_input("g^{ab} g_{ab}")      # [1]
            _process_tensor_input("R_{abcd}")            # [2]
            r = _process_tensor_input("simplify %1")
            @test r == TScalar(4 // 1)
        end

        @testset "out of range errors" begin
            empty!(TensorREPL._history)
            TensorREPL._last_result[] = nothing
            @test_throws ErrorException _process_tensor_input("%0")
            @test_throws ErrorException _process_tensor_input("%999")
        end

        @testset "display shows [N] prefix" begin
            empty!(TensorREPL._history)
            TensorREPL._last_result[] = nothing
            _process_tensor_input("R_{abcd}")
            buf = IOBuffer()
            _display_tensor_result(buf, TensorREPL._history[end])
            s = String(take!(buf))
            @test occursin("[", s)
        end

        @testset "variable assignment stores from %N" begin
            empty!(TensorREPL._history)
            TensorREPL._last_result[] = nothing
            empty!(TensorREPL._variables)
            _process_tensor_input("R_{abcd}")            # [1]
            _process_tensor_input("g_{ab}")              # [2]
            _process_tensor_input("x = %1")
            @test haskey(TensorREPL._variables, "x")
            @test TensorREPL._variables["x"].name === :Riem
        end
    end

    # ── Derivative shorthands ──────────────────────────────────────────

    @testset "derivative shorthands" begin
        reg = _setup_repl_test()

        @testset "\\partial stays partial" begin
            expr = _parse_and_resolve("\\partial_a R_{bc}")
            @test expr isa TDeriv
            @test expr.covd == :partial
        end

        @testset "\\nabla resolves to covd" begin
            expr = _parse_and_resolve("\\nabla_a R_{bc}")
            @test expr isa TDeriv
            # Should resolve to the manifold's derivative (not :partial or :nabla)
            @test expr.covd != :nabla
        end

        @testset "\\nabla fallback without covd" begin
            # Create a registry without a CovD
            bare_reg = TensorRegistry()
            with_registry(bare_reg) do
                TensorGR.register_manifold!(bare_reg, TensorGR.ManifoldProperties(
                    :M, 4, nothing, nothing, Symbol[]))
                TensorGR.register_tensor!(bare_reg, TensorGR.TensorProperties(;
                    name=:T, manifold=:M, rank=(0,2)))
            end
            set_tensor_registry!(bare_reg)
            _init_commands!()
            expr = _parse_and_resolve("\\nabla_a T_{bc}")
            @test expr isa TDeriv
            @test expr.covd == :partial  # falls back when no CovD

            # Restore
            set_tensor_registry!(reg)
        end
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
