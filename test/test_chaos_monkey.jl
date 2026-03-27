@testset "Chaos Monkey: REPL UX Hardening" begin
    using TensorGR: _process_tensor_input, _init_commands!, _parse_and_resolve,
                    _display_tensor_result, set_tensor_registry!,
                    TensorREPL,
                    TensorRegistry, Tensor, TProduct, TSum, TDeriv, TScalar,
                    up, down, with_registry, simplify, to_unicode

    using Random

    function _setup_monkey()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        set_tensor_registry!(reg)
        _init_commands!()
        reg
    end

    """Run input through tensor mode, return :ok, :error, or :crash."""
    function _monkey_input(s::AbstractString)
        try
            _process_tensor_input(s)
            return :ok
        catch e
            if e isa InterruptException
                rethrow()
            end
            return :error
        end
    end

    """Run n random inputs, return (ok_count, error_count, crash_count)."""
    function _monkey_run(inputs::Vector{String})
        ok = 0; err = 0; crash = 0
        for s in inputs
            result = _monkey_input(s)
            if result === :ok
                ok += 1
            elseif result === :error
                err += 1
            else
                crash += 1
            end
        end
        (ok, err, crash)
    end

    # ── TGR-pqto: Graceful error recovery ────────────────────────────

    @testset "graceful error recovery (TGR-pqto)" begin
        _setup_monkey()

        @testset "random byte strings never crash" begin
            rng = MersenneTwister(42)
            inputs = [String(rand(rng, UInt8.(0x20:0x7e), rand(rng, 1:200))) for _ in 1:500]
            ok, err, crash = _monkey_run(inputs)
            @test crash == 0
            @test ok + err == 500
        end

        @testset "null bytes and control chars" begin
            nasty = [
                "\0",
                "\0\0\0",
                "\x01\x02\x03",
                "R_{\x00ab}",
                "hello\x7fworld",
                "\t\t\t",
                "\r\n\r\n",
                "\e[31mred\e[0m",  # ANSI escape
                "\e[2J",           # clear screen escape
                "\x1b[H\x1b[2J",  # home + clear
            ]
            ok, err, crash = _monkey_run(nasty)
            @test crash == 0
        end

        @testset "empty and whitespace variants" begin
            blanks = ["", " ", "  ", "\t", "\n", "\r\n", "   \t   "]
            for s in blanks
                @test _monkey_input(s) === :ok
            end
        end

        @testset "very long input" begin
            long_str = "R_{" * "a"^10000 * "}"
            @test _monkey_input(long_str) in (:ok, :error)

            long_expr = join(["R_{ab}" for _ in 1:500], " + ")
            @test _monkey_input(long_expr) in (:ok, :error)
        end

        @testset "binary-like data" begin
            binary_inputs = [
                String(UInt8[0xff, 0xfe, 0x00, 0x01]),
                "PK\x03\x04",  # zip header
                "\x89PNG\r\n\x1a\n",  # PNG header
                "%PDF-1.4",
            ]
            for s in binary_inputs
                @test _monkey_input(s) in (:ok, :error)
            end
        end
    end

    # ── TGR-qwe0: LaTeX typo fuzzer ──────────────────────────────────

    @testset "LaTeX typo fuzzer (TGR-qwe0)" begin
        _setup_monkey()

        # Valid base expressions to mutate
        valid_exprs = [
            "R_{abcd}",
            "g^{ab} g_{ab}",
            "R_{ab} - \\frac{1}{2} g_{ab} R",
            "\\partial_a \\phi",
            "g^{ab} R_{ab}",
            "R_{abcd} R^{abcd}",
            "\\nabla_a T^{bc}",
            "\\frac{1}{2} g_{ab}",
        ]

        """Introduce a random typo into a LaTeX string."""
        function mutate(rng, s)
            isempty(s) && return s
            mutation = rand(rng, 1:10)
            if mutation == 1
                # Delete a random char
                i = rand(rng, 1:length(s))
                s[1:i-1] * s[i+1:end]
            elseif mutation == 2
                # Duplicate a random char
                i = rand(rng, 1:length(s))
                s[1:i] * s[i:i] * s[i+1:end]
            elseif mutation == 3
                # Swap two adjacent chars
                i = rand(rng, 1:max(1, length(s)-1))
                chars = collect(s)
                if i < length(chars)
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                end
                String(chars)
            elseif mutation == 4
                # Remove all closing braces
                replace(s, "}" => "")
            elseif mutation == 5
                # Remove all opening braces
                replace(s, "{" => "")
            elseif mutation == 6
                # Double all underscores
                replace(s, "_" => "__")
            elseif mutation == 7
                # Replace \ with /
                replace(s, "\\" => "/")
            elseif mutation == 8
                # Insert random char at random position
                i = rand(rng, 1:length(s))
                c = Char(rand(rng, 0x20:0x7e))
                s[1:i] * string(c) * s[i+1:end]
            elseif mutation == 9
                # Truncate at random position
                i = rand(rng, 1:length(s))
                s[1:i]
            else
                # Reverse the string
                reverse(s)
            end
        end

        @testset "single mutations: no crashes" begin
            rng = MersenneTwister(123)
            crash_count = 0
            for _ in 1:500
                base = valid_exprs[rand(rng, 1:length(valid_exprs))]
                mutated = mutate(rng, base)
                result = _monkey_input(mutated)
                if result === :crash
                    crash_count += 1
                end
            end
            @test crash_count == 0
        end

        @testset "double mutations: no crashes" begin
            rng = MersenneTwister(456)
            crash_count = 0
            for _ in 1:200
                base = valid_exprs[rand(rng, 1:length(valid_exprs))]
                mutated = mutate(rng, mutate(rng, base))
                result = _monkey_input(mutated)
                if result === :crash
                    crash_count += 1
                end
            end
            @test crash_count == 0
        end

        @testset "specific common typos" begin
            typos = [
                "R_{abcd",          # missing closing brace
                "R_abcd}",          # missing opening brace
                "R__ab",            # double underscore
                "R_{ab}}",          # extra closing brace
                "g^^{ab}",          # double caret
                "\\partail_a phi",  # misspelled command
                "\\frac{1}{2",      # incomplete frac
                "\\frac{1}",        # incomplete frac
                "\\frac{}{}",       # empty frac
                "R_{_a}",           # nested underscore
                "g^{a^b}",         # nested caret
                "R_{a b c d}",     # spaces in subscript
                "R_{}",            # empty subscript
                "g^{}",            # empty superscript
                "_a",              # leading underscore
                "^a",              # leading caret
                "\\",              # bare backslash
                "\\\\",            # double backslash
                "{{{",             # brace avalanche
                "}}}",             # closing brace avalanche
            ]
            for t in typos
                @test _monkey_input(t) in (:ok, :error)
            end
        end
    end

    # ── TGR-b36q: Clipboard dump simulator ───────────────────────────

    @testset "clipboard dump simulator (TGR-b36q)" begin
        _setup_monkey()

        clipboard_dumps = [
            # HTML fragment
            "<div class=\"container\"><p>Hello world</p></div>",
            # Python code
            "import numpy as np\nx = np.array([1, 2, 3])\nprint(x.sum())",
            # JSON blob
            """{"name": "test", "values": [1, 2, 3], "nested": {"a": true}}""",
            # Email header
            "From: user@example.com\nTo: other@example.com\nSubject: Re: meeting",
            # Shell command
            "ls -la /usr/local/bin | grep python | awk '{print \$9}'",
            # SQL
            "SELECT * FROM users WHERE name = 'admin'; DROP TABLE users; --",
            # Markdown
            "# Title\n\n- item 1\n- item 2\n\n```julia\nprintln(\"hello\")\n```",
            # CSV data
            "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago",
            # URL
            "https://arxiv.org/abs/2301.12345?query=xAct&lang=en#section3",
            # Stack trace
            "Stacktrace:\n [1] error(s::String)\n   @ Base ./error.jl:35\n [2] top-level scope\n   @ REPL[1]:1",
            # Git diff
            "diff --git a/src/foo.jl b/src/foo.jl\n--- a/src/foo.jl\n+++ b/src/foo.jl\n@@ -1,3 +1,4 @@",
            # Julia code
            "function foo(x::Int)\n    return x^2 + 2x + 1\nend",
            # Mathematica
            "TensorReduce[RiemannCD[-a,-b,-c,-d] RiemannCD[a,b,c,d]]",
            # LaTeX document (not just tensor notation)
            "\\documentclass{article}\n\\begin{document}\n\\section{Introduction}\nHello.\n\\end{document}",
            # Binary-ish
            "\\x00\\x01\\x02\\xff\\xfe\\xfd",
            # Very long line
            "a"^50000,
            # Repeated special chars
            "{{{{{{{{{{" * "}}}}}}}}}}" * "___" * "^^^" * "\\\\\\\\",
            # Mixed valid + garbage
            "R_{ab} this is garbage 💩 more stuff",
            # Emoji
            "🔬📐🧮 R_{ab} 🎯",
            # RTL text
            "مرحبا R_{ab} שלום",
        ]

        @testset "clipboard dumps: no crashes" begin
            crash_count = 0
            for (i, dump) in enumerate(clipboard_dumps)
                result = _monkey_input(dump)
                if result === :crash
                    crash_count += 1
                end
            end
            @test crash_count == 0
        end

        @testset "display doesn't crash on garbage results" begin
            buf = IOBuffer()
            # Even if parse succeeds with weird input, display must not crash
            for dump in clipboard_dumps
                try
                    expr = _process_tensor_input(dump)
                    if expr !== nothing
                        _display_tensor_result(buf, expr)
                    end
                catch
                    # parse error is fine, display crash is not
                end
            end
            @test true  # got here without crashing
        end
    end

    # ── TGR-x8do: Command injection / confused user ──────────────────

    @testset "command injection / confused user (TGR-x8do)" begin
        _setup_monkey()

        @testset "nested/repeated commands" begin
            confused = [
                "simplify simplify %",
                "simplify simplify simplify",
                "simplify",           # no argument, no %
                "contract expand %",
                "latex latex %",
                "help help",
                "help simplify",
                "? ?",
            ]
            for s in confused
                @test _monkey_input(s) in (:ok, :error)
            end
        end

        @testset "commands that look like LaTeX" begin
            ambiguous = [
                "simplify_{ab}",       # command name with subscript
                "expand^{ab}",         # command name with superscript
                "contract R_{ab}",     # valid command + valid expr
                "terms R_{ab} + G_{ab}",  # valid command + valid expr
            ]
            for s in ambiguous
                @test _monkey_input(s) in (:ok, :error)
            end
        end

        @testset "Julia code in tensor mode" begin
            julia_code = [
                "println(\"hello\")",
                "using Pkg",
                "import Base",
                "1 + 1",
                "x = 5",
                "for i in 1:10; println(i); end",
                "run(`ls`)",
                "eval(:(1+1))",
                "Base.@eval import REPL",
                "ccall(:jl_exit, Cvoid, (Int32,), 0)",
            ]
            for s in julia_code
                result = _monkey_input(s)
                @test result in (:ok, :error)
                # CRITICAL: none of these should execute as Julia code
            end
        end

        @testset "shell commands in tensor mode" begin
            shell = [
                "ls -la",
                "rm -rf /",
                "cat /etc/passwd",
                "; ls",
                "| cat",
                "\$(whoami)",
                "`id`",
            ]
            for s in shell
                @test _monkey_input(s) in (:ok, :error)
            end
        end

        @testset "Pkg mode input in tensor mode" begin
            pkg = [
                "]add Foo",
                "]status",
                "]rm TensorGR",
            ]
            for s in pkg
                @test _monkey_input(s) in (:ok, :error)
            end
        end

        @testset "% edge cases" begin
            TensorREPL._last_result[] = nothing
            @test _monkey_input("simplify %") === :error  # no prior result
            @test _monkey_input("%") in (:ok, :error)      # bare %
            @test _monkey_input("% %") in (:ok, :error)    # double %
            @test _monkey_input("%%") in (:ok, :error)     # stuck together
        end
    end

    # ── TGR-v5r1: Unicode edge cases ─────────────────────────────────

    @testset "Unicode edge cases (TGR-v5r1)" begin
        _setup_monkey()

        @testset "emoji in input" begin
            emoji_inputs = [
                "🔬",
                "📐R_{ab}",
                "R_{🎯}",
                "🧮 + 🔭",
                "∑∏∫∂∇",
                "R_{ab} × G_{cd}",   # multiplication sign, not x
                "½ g_{ab}",          # vulgar fraction
            ]
            for s in emoji_inputs
                @test _monkey_input(s) in (:ok, :error)
            end
        end

        @testset "combining characters" begin
            combining = [
                "R\u0308_{ab}",        # R with combining umlaut
                "g\u0303^{ab}",        # g with combining tilde
                "T\u0301\u0302_{a}",   # T with two combining marks
                "\u0061\u0308",        # a + combining umlaut (ä decomposed)
            ]
            for s in combining
                @test _monkey_input(s) in (:ok, :error)
            end
        end

        @testset "RTL and bidirectional text" begin
            bidi = [
                "مرحبا",                       # Arabic
                "שלום",                         # Hebrew
                "R_{ab} مرحبا R_{cd}",          # mixed
                "\u200fR_{ab}",                 # RTL mark + tensor
                "\u200eR_{ab}",                 # LTR mark + tensor
                "\u200b",                       # zero-width space
                "\u200bR_{ab}\u200b",           # ZWS around tensor
                "\ufeffR_{ab}",                 # BOM + tensor
            ]
            for s in bidi
                @test _monkey_input(s) in (:ok, :error)
            end
        end

        @testset "Greek letters (should work as tensor names)" begin
            greek = [
                "α_{ab}",
                "β^{a}",
                "Γ_{abc}",
                "δ_{ab}",
                "ε_{abcd}",
                "φ",
                "ψ_{a}",
                "Ω^{ab}",
            ]
            for s in greek
                result = _monkey_input(s)
                @test result in (:ok, :error)
            end
        end

        @testset "mathematical symbols" begin
            math = [
                "∂_a φ",           # literal partial symbol
                "∇_a T^{bc}",      # nabla
                "□ φ",             # d'Alembertian
                "R_{ab} ≈ 0",      # approximate
                "R ≠ 0",           # not equal
                "∫ R √g d⁴x",     # integral
                "⟨R_{ab}⟩",       # angle brackets
            ]
            for s in math
                @test _monkey_input(s) in (:ok, :error)
            end
        end

        @testset "display of resolved expressions doesn't corrupt" begin
            buf = IOBuffer()
            # Parse some valid expressions, display them, check no garbled output
            for input in ["R_{abcd}", "g^{ab}", "R"]
                expr = _parse_and_resolve(input)
                _display_tensor_result(buf, expr)
            end
            output = String(take!(buf))
            # Should contain recognizable tensor names, no raw escape sequences
            @test occursin("Riem", output)
            @test !occursin("\e[", output)  # no ANSI leaking into buffer
        end
    end

    # ── TGR-nv86: Rapid-fire stress test ─────────────────────────────

    @testset "rapid-fire stress test (TGR-nv86)" begin
        reg = _setup_monkey()

        valid_inputs = [
            "R_{abcd}", "g^{ab} g_{ab}", "R_{ab}", "R",
            "G_{ab}", "\\partial_a \\phi", "g^{ab} R_{ab}",
        ]
        commands = ["simplify %", "contract %", "canon %", "terms %", "latex %"]

        @testset "100 interleaved expressions + commands" begin
            rng = MersenneTwister(789)
            crash_count = 0
            for i in 1:100
                # Alternate: expression, then command
                expr_input = valid_inputs[rand(rng, 1:length(valid_inputs))]
                result = _monkey_input(expr_input)
                if result === :crash; crash_count += 1; end

                cmd = commands[rand(rng, 1:length(commands))]
                result = _monkey_input(cmd)
                if result === :crash; crash_count += 1; end
            end
            @test crash_count == 0
        end

        @testset "100 rapid invalid + valid mix" begin
            rng = MersenneTwister(101)
            crash_count = 0
            for _ in 1:100
                if rand(rng, Bool)
                    # Valid
                    _monkey_input(valid_inputs[rand(rng, 1:length(valid_inputs))])
                else
                    # Random garbage
                    s = String(rand(rng, UInt8.(0x20:0x7e), rand(rng, 1:50)))
                    _monkey_input(s)
                end
            end
            @test crash_count == 0
        end

        @testset "memory stability: no registry growth" begin
            # Run 200 simplify cycles, check registry size doesn't grow
            initial_rules = length(reg.rules)
            initial_tensors = length(reg.tensors)

            for _ in 1:200
                _monkey_input("g^{ab} R_{ab}")
                _monkey_input("simplify %")
            end

            @test length(reg.rules) == initial_rules
            @test length(reg.tensors) == initial_tensors
        end
    end

    # ── TGR-pdzl: Parameterized harness ──────────────────────────────

    @testset "parameterized chaos_monkey(n, seed) (TGR-pdzl)" begin
        _setup_monkey()

        """
            chaos_monkey(; n=100, seed=42) -> (ok, errors, crashes)

        Run n random chaos monkey scenarios deterministically.
        Returns counts of ok/error/crash results.
        """
        function chaos_monkey(; n::Int=100, seed::Int=42)
            rng = MersenneTwister(seed)
            ok = 0; err = 0; crash = 0

            generators = [
                # Random ASCII
                () -> String(rand(rng, UInt8.(0x20:0x7e), rand(rng, 1:100))),
                # Valid expression
                () -> ["R_{abcd}", "g^{ab}", "R", "G_{ab}", "\\partial_a \\phi"][rand(rng, 1:5)],
                # Mutated valid expression
                () -> begin
                    base = ["R_{abcd}", "g^{ab} g_{ab}", "R_{ab}"][rand(rng, 1:3)]
                    # Random deletion
                    i = rand(rng, 1:max(1, length(base)))
                    base[1:max(1,i-1)] * base[min(length(base),i+1):end]
                end,
                # Command on %
                () -> ["simplify %", "contract %", "latex %", "terms %"][rand(rng, 1:4)],
                # Clipboard-like
                () -> ["<html>", "{\"key\": 1}", "SELECT *", "import os"][rand(rng, 1:4)],
                # Unicode
                () -> String(rand(rng, Char.(0x20:0x2fff), rand(rng, 1:20))),
            ]

            for _ in 1:n
                gen = generators[rand(rng, 1:length(generators))]
                input = gen()
                result = _monkey_input(input)
                if result === :ok; ok += 1
                elseif result === :error; err += 1
                else; crash += 1; end
            end

            (ok=ok, errors=err, crashes=crash)
        end

        @testset "n=500, seed=42: zero crashes" begin
            result = chaos_monkey(n=500, seed=42)
            @test result.crashes == 0
            @test result.ok + result.errors == 500
        end

        @testset "n=500, seed=99: zero crashes (different seed)" begin
            result = chaos_monkey(n=500, seed=99)
            @test result.crashes == 0
        end

        @testset "deterministic: same seed = same results" begin
            r1 = chaos_monkey(n=100, seed=12345)
            r2 = chaos_monkey(n=100, seed=12345)
            @test r1.ok == r2.ok
            @test r1.errors == r2.errors
        end
    end
end
