#= Segre classification of the Ricci tensor.

Characterizes the eigenvalue/Jordan structure of the mixed Ricci tensor
R^a_b = g^{ac} R_{cb}, providing an algebraic classification complementary
to the Petrov type. Together, Petrov + Segre fully classify local geometry.

References:
  Stephani et al, "Exact Solutions" (2003), Ch 5
  Hall, "Symmetries and Curvature Structure in GR" (2004), Ch 9
=#

using LinearAlgebra: eigvals, svdvals, I

"""
    SegreType

Result of the Segre classification of a symmetric 2-tensor (typically R_{ab}).

Fields:
- `notation::String`        -- Segre notation, e.g. "{(1,1)(11)}", "{1,(111)}"
- `eigenvalues::Vector`     -- distinct eigenvalues of R^a_b
- `multiplicities::Vector{Int}` -- multiplicity of each distinct eigenvalue
- `jordan_sizes::Vector{Vector{Int}}` -- Jordan block sizes per eigenvalue
- `is_degenerate::Bool`     -- true if any Jordan block has size > 1
"""
struct SegreType
    notation::String
    eigenvalues::Vector
    multiplicities::Vector{Int}
    jordan_sizes::Vector{Vector{Int}}
    is_degenerate::Bool
end

function Base.show(io::IO, st::SegreType)
    print(io, "SegreType(", st.notation, ")")
end

"""
    segre_classify(Ric::Matrix, ginv::Matrix; atol=1e-10) -> SegreType

Classify the Segre type of the Ricci tensor from components R_{ab}
and inverse metric g^{ab}. Computes R^a_b = g^{ac} R_{cb} and
analyzes its eigenvalue/Jordan structure.
"""
function segre_classify(Ric::Matrix, ginv::Matrix; atol=1e-10)
    dim = size(Ric, 1)
    @assert size(Ric) == (dim, dim)
    @assert size(ginv) == (dim, dim)
    # Compute mixed tensor R^a_b = g^{ac} R_{cb}
    R_mixed = ginv * Ric
    segre_classify(R_mixed; atol=atol)
end

"""
    segre_classify(R_mixed::Matrix; atol=1e-10) -> SegreType

Classify from the pre-computed mixed tensor R^a_b.
"""
function segre_classify(R_mixed::Matrix; atol=1e-10)
    dim = size(R_mixed, 1)
    @assert size(R_mixed) == (dim, dim)

    evals, mults, jsizes = _jordan_structure(R_mixed; atol=atol)
    is_deg = any(any(s > 1 for s in js) for js in jsizes)
    notation = _segre_notation(evals, mults, jsizes, dim; atol=atol)

    SegreType(notation, evals, mults, jsizes, is_deg)
end

"""
    _jordan_structure(M::Matrix; atol=1e-10)

Compute eigenvalues, multiplicities, and Jordan block sizes of matrix M.

Returns `(eigenvalues, multiplicities, jordan_sizes)` where:
- `eigenvalues` are the distinct eigenvalues (real parts kept if imaginary part < atol)
- `multiplicities[i]` is the algebraic multiplicity of eigenvalues[i]
- `jordan_sizes[i]` lists the Jordan block sizes for eigenvalues[i]
"""
function _jordan_structure(M::Matrix; atol=1e-10)
    dim = size(M, 1)
    raw_evals = eigvals(M)

    # Clean: snap small imaginary parts to zero
    cleaned = map(raw_evals) do ev
        if abs(imag(ev)) < atol
            real(ev)
        else
            ev
        end
    end

    # Group into distinct eigenvalues
    distinct = Any[]
    mults = Int[]
    for ev in cleaned
        found = false
        for (k, dev) in enumerate(distinct)
            if abs(ev - dev) < atol
                mults[k] += 1
                found = true
                break
            end
        end
        if !found
            push!(distinct, ev)
            push!(mults, 1)
        end
    end

    # Compute Jordan block sizes for each eigenvalue via rank analysis.
    # For eigenvalue lambda with algebraic multiplicity m:
    #   Let n_k = nullity((M - lambda*I)^k) = dim - rank((M - lambda*I)^k)
    #   n_0 = 0, n_1 = geometric multiplicity (number of Jordan blocks)
    #   Blocks of size >= k: n_k - n_{k-1}  (for k >= 1)
    #   Blocks of exactly size s: (n_s - n_{s-1}) - (n_{s+1} - n_s)
    #                            = 2*n_s - n_{s-1} - n_{s+1}
    jordan_sizes = Vector{Vector{Int}}(undef, length(distinct))
    M_float = Float64.(M)

    for (i, lam) in enumerate(distinct)
        m = mults[i]
        if m == 1
            jordan_sizes[i] = [1]
            continue
        end

        # Compute nullities of (M - lam*I)^k for k = 0, 1, ..., m
        N = M_float - Float64(real(lam)) * I
        nullities = Int[0]  # nullity of (M-lam*I)^0 = 0
        Nk = Matrix{Float64}(I, dim, dim)
        for k in 1:m
            Nk = Nk * N
            sv = svdvals(Nk)
            tol = max(atol, atol * sv[1])
            r = count(s -> s > tol, sv)
            push!(nullities, dim - r)
        end

        # Compute block sizes from nullity sequence
        # Number of blocks of size >= k is nullities[k+1] - nullities[k]
        # Number of blocks of exactly size s:
        #   (nullities[s+1] - nullities[s]) - (nullities[s+2] - nullities[s+1])
        #   = 2*nullities[s+1] - nullities[s] - nullities[s+2]
        blocks = Int[]
        for s in 1:m
            n_prev = nullities[s]     # nullity((M-lam*I)^{s-1})
            n_curr = nullities[s+1]   # nullity((M-lam*I)^s)
            n_next = s + 1 <= m ? nullities[s+2] : n_curr  # stabilized
            count_s = 2 * n_curr - n_prev - n_next
            for _ in 1:count_s
                push!(blocks, s)
            end
        end

        # Fallback: if rank analysis gave empty or inconsistent result
        if isempty(blocks) || sum(blocks) != m
            blocks = fill(1, m)
        end

        sort!(blocks, rev=true)
        jordan_sizes[i] = blocks
    end

    (distinct, mults, jordan_sizes)
end

"""
    _segre_notation(eigenvalues, multiplicities, jordan_sizes, dim; atol=1e-10)

Format the Segre notation string. In 4D Lorentzian signature (-,+,+,+),
the convention is `{timelike, spacelike}` with parentheses grouping
degenerate eigenvalues.

For purely algebraic (eigenvalue-based) classification we group by
multiplicity, placing the timelike sector first (separated by comma from
the spacelike sector). When all eigenvalues are degenerate, the timelike
and spacelike labels are grouped together.
"""
function _segre_notation(eigenvalues, multiplicities, jordan_sizes, dim; atol=1e-10)
    n_ev = length(eigenvalues)

    # Build block-size strings per eigenvalue group.
    # Each eigenvalue contributes its Jordan block sizes as digits.
    # If multiplicity > 1 (all blocks size 1), group with parentheses.
    parts = String[]
    for (i, ev) in enumerate(eigenvalues)
        m = multiplicities[i]
        js = jordan_sizes[i]
        digit_str = join(string.(js))
        if m > 1 && all(s == 1 for s in js)
            push!(parts, "(" * digit_str * ")")
        else
            # Each block is a separate digit (possibly multi-digit for big Jordan blocks)
            push!(parts, digit_str)
        end
    end

    # Determine timelike vs spacelike assignment.
    # Convention: in 4D, one eigenvalue direction is timelike, rest are spacelike.
    # We separate the first eigenvalue (timelike) from the rest with a comma.
    # If the first eigenvalue is degenerate with others, they share parentheses.

    # For a standard 4D classification:
    # Check if all eigenvalues are the same (fully degenerate)
    if n_ev == 1
        # All eigenvalues equal -> (1,111) notation
        m = multiplicities[1]
        if m == dim
            return "{(" * join(fill("1", dim), "") * ")}"
        else
            return "{" * parts[1] * "}"
        end
    end

    # Multiple distinct eigenvalues.
    # Assign: the eigenvalue with multiplicity 1 that comes first is timelike.
    # Standard approach: the first entry is timelike, rest are spacelike,
    # separated by comma.

    # Simple heuristic: look for natural timelike-spacelike split.
    # In the general case, one eigenvalue is timelike (mult >= 1),
    # and the rest are spacelike.

    # Find if there's a natural 1 + (d-1) split
    for (i, m) in enumerate(multiplicities)
        if m == 1 && sum(multiplicities) - 1 == dim - 1
            # This eigenvalue is the "1" (timelike sector)
            timelike_part = parts[i]
            spacelike_parts = [parts[j] for j in 1:n_ev if j != i]
            return "{" * timelike_part * "," * join(spacelike_parts) * "}"
        end
    end

    # For 1+1 + 2 split (e.g., vacuum Schwarzschild: two pairs of degenerate eigenvalues)
    # or any other pattern: list all parts separated by commas between groups
    # that don't share parentheses.

    # General formatting: join all parts, inserting comma between
    # the first non-parenthesized digit and the rest.
    all_str = join(parts)

    # If the total character count of digits equals dim, insert comma after first digit/group
    # to separate timelike from spacelike.
    _insert_segre_comma(all_str, dim)
end

"""
    _insert_segre_comma(s, dim)

Insert the timelike/spacelike comma in a Segre notation string.
The comma goes after the first block (representing the timelike sector).
"""
function _insert_segre_comma(s::String, dim::Int)
    # Parse the string to find block boundaries
    # A block is either a single digit or a parenthesized group (...)
    blocks = String[]
    i = 1
    while i <= length(s)
        if s[i] == '('
            j = findnext(')', s, i)
            push!(blocks, s[i:j])
            i = j + 1
        else
            push!(blocks, string(s[i]))
            i += 1
        end
    end

    if length(blocks) <= 1
        return "{" * s * "}"
    end

    # Count total digit count in each block
    function block_digit_count(b)
        count(isdigit, b)
    end

    # The first block is timelike. If it has only 1 digit, the comma
    # goes after it. If it's a parenthesized group, comma after the group.
    timelike = blocks[1]
    spacelike = join(blocks[2:end])

    "{" * timelike * "," * spacelike * "}"
end
