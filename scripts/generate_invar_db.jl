#= Memory-safe generation of canonical RInv forms for the Invar pipeline.
#
# Usage:
#   julia --project scripts/generate_invar_db.jl <degree>
#   julia --project scripts/generate_invar_db.jl <degree> --dry-run
#   julia --project scripts/generate_invar_db.jl <degree> --output <file>
#
# Stream-enumerates ALL fixed-point-free involutions on [1..4k] for the given
# degree k, canonicalizing each immediately via BFS orbit enumeration.
# Only the distinct canonical forms (and their orbit sizes) are kept in memory.
#
# Memory usage: O(n_canonical * 4k) -- typically <100 KB even for degree 4.
# The recursion stack is O(4k) deep -- negligible.
#
# For degree 2: 105 involutions,   4 non-vanishing canonical forms (~instant)
# For degree 3: 10395 involutions, 13 non-vanishing canonical forms (~1 sec)
# For degree 4: 2027025 involutions, 57 non-vanishing canonical forms (~minutes)
#
# DO NOT run for degree >= 5 (654,729,075 involutions -- too slow for streaming).
=#

using TensorGR

# ---- Core: streaming enumeration with immediate canonicalization ----

"""
    stream_enumerate_canonical(k::Int; dry_run::Bool=false)

Stream-enumerate all fixed-point-free involutions on [1..4k] and canonicalize
each immediately.  Only keeps the Dict of canonical forms in memory.

Returns `(canonical_forms, vanishing_count, total_count)` where:
- `canonical_forms::Dict{Vector{Int}, Int}` maps canonical contraction => orbit size
- `vanishing_count::Int` is the number of involutions that canonicalize to zero
- `total_count::Int` is the total number of involutions enumerated

If `dry_run=true`, counts involutions without canonicalizing (instant).
"""
function stream_enumerate_canonical(k::Int; dry_run::Bool=false)
    nslots = 4k
    canonical_forms = Dict{Vector{Int}, Int}()
    vanishing_count = Ref(0)
    total_count = Ref(0)

    buffer = zeros(Int, nslots)

    println("Starting enumeration for degree $k ($nslots slots)...")
    flush(stdout)
    println(stderr, "Starting enumeration for degree $k ($nslots slots)...")
    flush(stderr)

    if dry_run
        _stream_count!(total_count, buffer, 1, nslots)
        return (canonical_forms, 0, total_count[])
    end

    t0 = time()
    _stream_enum!(canonical_forms, vanishing_count, total_count, buffer, 1, nslots, k, t0)

    (canonical_forms, vanishing_count[], total_count[])
end

"""
    _stream_count!(total_count, buffer, pos, nslots)

Count involutions without canonicalizing (dry-run mode).
"""
function _stream_count!(total_count::Ref{Int}, buffer::Vector{Int},
                         pos::Int, nslots::Int)
    # Find first unpaired position at or after pos
    while pos <= nslots && buffer[pos] != 0
        pos += 1
    end
    if pos > nslots
        total_count[] += 1
        return
    end
    for partner in (pos+1):nslots
        buffer[partner] == 0 || continue
        buffer[pos] = partner
        buffer[partner] = pos
        _stream_count!(total_count, buffer, pos + 1, nslots)
        buffer[pos] = 0
        buffer[partner] = 0
    end
end

"""
    _stream_enum!(canonical_forms, vanishing_count, total_count,
                   buffer, pos, nslots, degree, t0)

Recursive enumeration with immediate canonicalization.
Never stores more than one involution at a time.
"""
function _stream_enum!(canonical_forms::Dict{Vector{Int}, Int},
                        vanishing_count::Ref{Int},
                        total_count::Ref{Int},
                        buffer::Vector{Int},
                        pos::Int, nslots::Int, degree::Int,
                        t0::Float64)
    # Find first unpaired position at or after pos
    while pos <= nslots && buffer[pos] != 0
        pos += 1
    end
    if pos > nslots
        # Complete involution -- canonicalize immediately, don't store
        total_count[] += 1
        rinv = RInv(degree, copy(buffer))
        canon = canonicalize(rinv)
        if all(==(0), canon.contraction)
            vanishing_count[] += 1
        else
            c = canon.contraction
            canonical_forms[c] = get(canonical_forms, c, 0) + 1
        end
        # Progress reporting every 10k involutions
        if total_count[] % 10_000 == 0
            elapsed = time() - t0
            rate = total_count[] / elapsed
            n_invol = double_factorial(nslots - 1)
            eta = (n_invol - total_count[]) / max(rate, 1.0)
            msg = "  Processed $(total_count[])/$(n_invol) involutions, " *
                  "$(length(canonical_forms)) distinct canonical forms " *
                  "($(round(rate, digits=0))/sec, $(round(elapsed, digits=1))s elapsed, " *
                  "ETA $(round(eta, digits=0))s)"
            println(msg)
            flush(stdout)
            println(stderr, msg)
            flush(stderr)
        end
        return
    end
    for partner in (pos+1):nslots
        buffer[partner] == 0 || continue
        buffer[pos] = partner
        buffer[partner] = pos
        _stream_enum!(canonical_forms, vanishing_count, total_count,
                       buffer, pos + 1, nslots, degree, t0)
        buffer[pos] = 0
        buffer[partner] = 0
    end
end

# ---- Output formatting ----

"""
    format_rinv_list(canonical_forms::Dict{Vector{Int}, Int}, degree::Int) -> String

Format the canonical forms as Julia source code suitable for inclusion in
a degree database file (e.g., degree4.jl).
"""
function format_rinv_list(canonical_forms::Dict{Vector{Int}, Int}, degree::Int)
    # Sort by lexicographic contraction vector
    sorted = sort(collect(keys(canonical_forms)))

    lines = String[]
    push!(lines, "# $(length(sorted)) non-vanishing canonical RInv forms at degree $degree")
    push!(lines, "# Generated by scripts/generate_invar_db.jl")
    push!(lines, "RInv[")
    for (i, c) in enumerate(sorted)
        orbit_size = canonical_forms[c]
        comma = i < length(sorted) ? "," : ""
        cstr = join(c, ",")
        push!(lines, "    RInv($degree, [$cstr], true)$comma   # I$i (orbit size $orbit_size)")
    end
    push!(lines, "]")
    join(lines, "\n")
end

"""
    print_summary(canonical_forms, vanishing_count, total_count, degree, elapsed)

Print a human-readable summary of the enumeration results.
"""
function print_summary(canonical_forms, vanishing_count, total_count, degree, elapsed)
    n_nonzero = length(canonical_forms)
    n_zero = vanishing_count
    total_orbits = n_nonzero + (n_zero > 0 ? 1 : 0)
    orbit_sizes = sort(collect(values(canonical_forms)))

    println()
    println("=" ^ 70)
    println("Degree $degree RInv Enumeration Results")
    println("=" ^ 70)
    println("  Slots:              $(4 * degree)")
    println("  Total involutions:  $total_count  (($(4*degree)-1)!! = $(double_factorial(4*degree - 1)))")
    println("  Vanishing:          $n_zero")
    println("  Non-vanishing canonical: $n_nonzero")
    println("  Total classes:      $total_orbits (= $n_nonzero non-zero + $(n_zero > 0 ? 1 : 0) zero)")
    println("  Orbit sizes:        min=$(minimum(orbit_sizes)), max=$(maximum(orbit_sizes)), " *
            "median=$(orbit_sizes[length(orbit_sizes) ÷ 2 + 1])")
    println("  Sum check:          $(sum(orbit_sizes) + n_zero) == $total_count ? $(sum(orbit_sizes) + n_zero == total_count)")
    println("  Elapsed time:       $(round(elapsed, digits=2))s")
    println("=" ^ 70)
end

"""Double factorial (2k-1)!! = 1*3*5*...*(2k-1)."""
function double_factorial(n::Int)
    n <= 0 && return 1
    result = 1
    for i in 1:2:n
        result *= i
    end
    result
end

# ---- Verification against existing database ----

"""
    verify_against_database(canonical_forms::Dict{Vector{Int}, Int}, degree::Int)

Cross-check the enumerated canonical forms against the existing database.
Returns (matches, missing_from_db, extra_in_db).
"""
function verify_against_database(canonical_forms::Dict{Vector{Int}, Int}, degree::Int)
    # Get the existing database forms
    db_rinvs = if degree == 2
        degree2_canonical_rinvs()
    elseif degree == 3
        degree3_canonical_rinvs()
    elseif degree == 4
        degree4_canonical_rinvs()
    else
        println("  No database to verify against for degree $degree")
        return nothing
    end

    db_set = Set([r.contraction for r in db_rinvs])
    enum_set = Set(keys(canonical_forms))

    matches = intersect(db_set, enum_set)
    missing_from_db = setdiff(enum_set, db_set)
    extra_in_db = setdiff(db_set, enum_set)

    println()
    println("Database verification (degree $degree):")
    println("  Database forms:     $(length(db_set))")
    println("  Enumerated forms:   $(length(enum_set))")
    println("  Matches:            $(length(matches))")
    println("  New (not in DB):    $(length(missing_from_db))")
    println("  Missing (in DB only): $(length(extra_in_db))")

    if !isempty(missing_from_db)
        println("\n  NEW canonical forms found:")
        for c in sort(collect(missing_from_db))
            println("    $c  (orbit size $(canonical_forms[c]))")
        end
    end

    if !isempty(extra_in_db)
        println("\n  WARNING: forms in database but not found by enumeration:")
        for c in sort(collect(extra_in_db))
            println("    $c")
        end
    end

    (matches=matches, missing_from_db=missing_from_db, extra_in_db=extra_in_db)
end

# ---- Main entry point ----

function main()
    args = ARGS

    if isempty(args) || "--help" in args || "-h" in args
        println("Usage: julia --project scripts/generate_invar_db.jl <degree> [options]")
        println()
        println("Options:")
        println("  --dry-run     Count involutions without canonicalizing")
        println("  --output FILE Write canonical forms as Julia source to FILE")
        println("  --no-verify   Skip verification against existing database")
        println("  --help        Show this help")
        println()
        println("Examples:")
        println("  julia --project scripts/generate_invar_db.jl 2          # degree 2 (~instant)")
        println("  julia --project scripts/generate_invar_db.jl 3          # degree 3 (~1 sec)")
        println("  julia --project scripts/generate_invar_db.jl 4          # degree 4 (~5-10 min)")
        println("  julia --project scripts/generate_invar_db.jl 4 --dry-run  # just count")
        return
    end

    degree = parse(Int, args[1])
    dry_run = "--dry-run" in args
    no_verify = "--no-verify" in args
    output_idx = findfirst(==("--output"), args)
    output_file = output_idx !== nothing && output_idx < length(args) ? args[output_idx + 1] : nothing

    # Safety check
    if degree >= 5
        nslots = 4 * degree
        n_invol = double_factorial(nslots - 1)
        println("WARNING: degree $degree has $n_invol involutions.")
        println("This will take a VERY long time and is not recommended.")
        println("Press Ctrl+C to abort, or wait 5 seconds to continue...")
        sleep(5)
    end

    nslots = 4 * degree
    n_invol = double_factorial(nslots - 1)
    println("Degree $degree: $(nslots) slots, $n_invol involutions to enumerate")

    if dry_run
        println("Dry run mode: counting only (no canonicalization)")
    end

    t0 = time()
    (canonical_forms, vanishing_count, total_count) = stream_enumerate_canonical(degree; dry_run=dry_run)
    elapsed = time() - t0

    if dry_run
        println("Total involutions: $total_count (expected: $n_invol)")
        println("Match: $(total_count == n_invol)")
        println("Elapsed: $(round(elapsed, digits=3))s")
        return
    end

    print_summary(canonical_forms, vanishing_count, total_count, degree, elapsed)

    # Verify against database
    if !no_verify
        verify_against_database(canonical_forms, degree)
    end

    # Output canonical forms
    source = format_rinv_list(canonical_forms, degree)
    println()
    println("Canonical forms (Julia source):")
    println(source)

    if output_file !== nothing
        open(output_file, "w") do io
            println(io, source)
        end
        println("\nWritten to $output_file")
    end
end

main()
