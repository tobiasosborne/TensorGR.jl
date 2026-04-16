#!/usr/bin/env julia
# regen.jl — TGR-bhs5.28
#
# One-shot driver that regenerates all committed golden-master JSON files
# from their wolframscript generators.
#
# Usage:
#   julia --project test/golden/regen.jl              # regen all
#   julia --project test/golden/regen.jl bianchi      # regen single
#   julia --project test/golden/regen.jl --dry-run    # list without running
#
# Requires `wolframscript` on PATH.

using Printf

const GENERATORS_DIR = joinpath(@__DIR__, "generators")

function collect_generators()
    gens = String[]
    for f in readdir(GENERATORS_DIR; join=true)
        endswith(f, ".wl") || continue
        base = basename(f)
        startswith(base, "_") && continue      # probe artifacts
        base == "common.wl"  && continue        # library, not a case
        push!(gens, f)
    end
    sort(gens)
end

function have_wolframscript()
    try
        read(`wolframscript -code "1+1"`, String)
        return true
    catch
        return false
    end
end

function main()
    args = ARGS
    dry_run = "--dry-run" in args
    filter_args = [a for a in args if !startswith(a, "-")]

    gens = collect_generators()
    if !isempty(filter_args)
        keep = Set(String.(filter_args))
        gens = [g for g in gens if basename(g) in keep ||
                replace(basename(g), ".wl" => "") in keep]
    end

    println("golden regen: $(length(gens)) generator(s)")
    for g in gens
        println("  - ", basename(g))
    end

    dry_run && return
    isempty(gens) && (println("(no generators matched — nothing to do)"); return)

    if !have_wolframscript()
        println(stderr, "ERROR: wolframscript not available on PATH.")
        exit(2)
    end

    ok = 0
    fail = 0
    for g in gens
        rel = relpath(g)
        print(@sprintf("  %-42s  ", basename(g)))
        try
            run(`wolframscript -f $g`)
            println("[OK]")
            ok += 1
        catch e
            println("[FAIL]")
            @error "generator failed" generator=rel exception=e
            fail += 1
        end
    end

    println()
    println(@sprintf("done: %d ok, %d fail", ok, fail))
    exit(fail == 0 ? 0 : 1)
end

main()
