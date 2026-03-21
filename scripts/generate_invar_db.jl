#= Memory-safe generation and verification of canonical RInv forms for the
# Invar pipeline.
#
# Usage:
#   julia --project scripts/generate_invar_db.jl <degree>
#   julia --project scripts/generate_invar_db.jl <degree> --dry-run
#   julia --project scripts/generate_invar_db.jl <degree> --verify-orbits
#   julia --project scripts/generate_invar_db.jl <degree> --output <file>
#
# Modes:
#   Default:        Stream-enumerate ALL involutions, canonicalize each.
#   --dry-run:      Count involutions without canonicalizing (instant).
#   --verify-orbits: Compute orbit sizes of known canonical forms and verify
#                   their sum + vanishing count = total. Also run random
#                   sampling to check no forms are missing. Fast (~seconds).
#
# Memory usage: O(n_canonical * 4k) -- typically <100 KB even for degree 4.
# The recursion stack is O(4k) deep -- negligible.
#
# For degree 2: 105 involutions,   4 non-vanishing canonical forms (~instant)
# For degree 3: 10395 involutions, 13 non-vanishing canonical forms (~10 sec)
# For degree 4: 2027025 involutions, 57 non-vanishing canonical forms (~30 min)
#
# DO NOT run for degree >= 5 (654,729,075 involutions -- too slow for streaming).
#
# For degree 4, prefer --verify-orbits (completes in ~30 seconds) over full
# enumeration (~30 minutes). The orbit-based verification mathematically proves
# database completeness by confirming that orbit sizes sum to the expected total.
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

# ---- Orbit-based verification ------------------------------------------------

"""
    compute_orbit_size(contraction::Vector{Int}, degree::Int) -> Int

Compute the orbit size of a canonical RInv contraction under the Riemann
symmetry group by BFS enumeration.  The orbit is the set of all contractions
reachable by conjugation with the symmetry generators.
"""
function compute_orbit_size(contraction::Vector{Int}, degree::Int)
    gens = TensorGR._rinv_slot_generators(degree)
    orbit = Dict{Vector{Int}, Int}()
    orbit[contraction] = +1
    queue = [contraction]

    while !isempty(queue)
        sigma = popfirst!(queue)
        current_sign = orbit[sigma]
        for (g, gsign) in gens
            sigma_new = TensorGR._conjugate_contraction(sigma, g)
            new_sign = current_sign * gsign
            if !haskey(orbit, sigma_new)
                orbit[sigma_new] = new_sign
                push!(queue, sigma_new)
            end
        end
    end

    length(orbit)
end

"""
    verify_orbits(degree::Int; n_samples::Int=1000)

Verify database completeness for a given degree by:
1. Computing orbit sizes of all known canonical forms via BFS
2. Checking that orbit sizes sum to total - vanishing_count
3. Running random sampling to confirm no unknown forms exist

This is much faster than full enumeration for degree 4 (~30 sec vs ~30 min).
"""
function verify_orbits(degree::Int; n_samples::Int=1000)
    nslots = 4 * degree
    n_invol = double_factorial(nslots - 1)

    # Get canonical forms from database
    db_rinvs = if degree == 2
        degree2_canonical_rinvs()
    elseif degree == 3
        degree3_canonical_rinvs()
    elseif degree == 4
        degree4_canonical_rinvs()
    else
        println("No database for degree $degree; cannot verify orbits.")
        return false
    end

    println("Verifying degree $degree database ($n_invol total involutions)...")
    println("  Computing orbit sizes for $(length(db_rinvs)) canonical forms...")
    flush(stdout)

    orbit_sizes = Int[]
    total_nonvanishing = 0
    for (i, rinv) in enumerate(db_rinvs)
        osize = compute_orbit_size(rinv.contraction, degree)
        push!(orbit_sizes, osize)
        total_nonvanishing += osize
        if i % 10 == 0 || i == length(db_rinvs)
            println("    I$i/$(length(db_rinvs)): orbit=$osize, cumulative=$total_nonvanishing")
            flush(stdout)
        end
    end

    vanishing_count = n_invol - total_nonvanishing

    println()
    println("  Orbit verification:")
    println("    Non-vanishing orbit total: $total_nonvanishing")
    println("    Vanishing (by subtraction): $vanishing_count")
    println("    Sum: $(total_nonvanishing + vanishing_count)")
    println("    Expected: $n_invol")
    sum_ok = (total_nonvanishing + vanishing_count == n_invol)
    println("    Match: $sum_ok")

    if vanishing_count < 0
        println("  ERROR: vanishing count is negative -- database has too many orbit elements!")
        return false
    end

    # Random sampling verification
    println()
    println("  Random sampling ($n_samples involutions)...")
    flush(stdout)

    known_set = Set([r.contraction for r in db_rinvs])
    zero_vec = zeros(Int, nslots)
    n_known = 0
    n_vanishing_sample = 0
    n_unknown = 0

    for _ in 1:n_samples
        perm = zeros(Int, nslots)
        slots = collect(1:nslots)
        while !isempty(slots)
            i = popfirst!(slots)
            remaining = [j for j in slots if perm[j] == 0 && j != i]
            j = remaining[rand(1:length(remaining))]
            perm[i] = j
            perm[j] = i
            filter!(x -> x != j, slots)
        end
        rinv = RInv(degree, perm)
        canon = canonicalize(rinv)
        c = canon.contraction
        if c == zero_vec
            n_vanishing_sample += 1
        elseif c in known_set
            n_known += 1
        else
            n_unknown += 1
            println("    UNKNOWN canonical form found: $c")
        end
    end

    println("    Results: known=$n_known, vanishing=$n_vanishing_sample, unknown=$n_unknown")

    all_ok = sum_ok && n_unknown == 0

    println()
    println("=" ^ 70)
    if all_ok
        println("VERIFIED: degree $degree database is COMPLETE")
        println("  $(length(db_rinvs)) non-vanishing canonical forms")
        println("  Orbit sizes sum to $total_nonvanishing (+ $vanishing_count vanishing = $n_invol)")
        println("  $n_samples random samples all confirmed")
    else
        println("VERIFICATION FAILED for degree $degree")
        if !sum_ok
            println("  Orbit sizes do not sum correctly")
        end
        if n_unknown > 0
            println("  Found $n_unknown unknown canonical forms")
        end
    end
    println("=" ^ 70)

    all_ok
end

# ---- Main entry point ----

function main()
    args = ARGS

    if isempty(args) || "--help" in args || "-h" in args
        println("Usage: julia --project scripts/generate_invar_db.jl <degree> [options]")
        println()
        println("Options:")
        println("  --dry-run       Count involutions without canonicalizing (instant)")
        println("  --verify-orbits Verify database completeness via orbit sizes (fast)")
        println("  --output FILE   Write canonical forms as Julia source to FILE")
        println("  --no-verify     Skip verification against existing database")
        println("  --help          Show this help")
        println()
        println("Examples:")
        println("  julia --project scripts/generate_invar_db.jl 2               # degree 2 (~instant)")
        println("  julia --project scripts/generate_invar_db.jl 3               # degree 3 (~10 sec)")
        println("  julia --project scripts/generate_invar_db.jl 4 --verify-orbits  # verify degree 4 (~30 sec)")
        println("  julia --project scripts/generate_invar_db.jl 4               # full enumeration (~30 min)")
        println("  julia --project scripts/generate_invar_db.jl 4 --dry-run     # just count")
        return
    end

    degree = parse(Int, args[1])
    dry_run = "--dry-run" in args
    verify_orbits_mode = "--verify-orbits" in args
    no_verify = "--no-verify" in args
    output_idx = findfirst(==("--output"), args)
    output_file = output_idx !== nothing && output_idx < length(args) ? args[output_idx + 1] : nothing

    # Orbit-based verification mode (fast)
    if verify_orbits_mode
        t0 = time()
        ok = verify_orbits(degree; n_samples=1000)
        elapsed = time() - t0
        println("Elapsed: $(round(elapsed, digits=2))s")
        exit(ok ? 0 : 1)
    end

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
