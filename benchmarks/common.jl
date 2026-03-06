if !@isdefined(_BENCH_COMMON_LOADED)

const _BENCH_COMMON_LOADED = true

using TensorGR, Test

"""Count the number of terms in a TensorExpr (after simplification)."""
count_terms(e::TSum) = length(e.terms)
count_terms(e::TScalar) = e.val == 0 ? 0 : 1
count_terms(::TensorExpr) = 1

"""Time a computation, return (result, time_s, alloc_bytes)."""
function timed_compute(f)
    GC.gc()
    stats = @timed f()
    (result=stats.value, time=stats.time, bytes=stats.bytes)
end

const PERF_MODE = get(ENV, "TENSORGR_BENCH_PERF", "0") == "1"

end # if !@isdefined
