"""
    run_erosion_stats(L, rule; ...)

Erosion-time distribution for a rule's deterministic-favored phase. Each
trial starts from a `clean` (bias-aligned) background with a centered
circular minority domain of fractional radius `domain_size`, then evolves
under `mc_sweep!` at the given `p`, `η` until the minority domain is fully
absorbed (state becomes uniformly bias-aligned again) or `max_time` MC
sweeps elapse, whichever happens first. Saves the **per-trial vector** of
erosion times.

Per-trial randomness comes from the random update order in `mc_sweep!`
(asynchronous updates) — even with `p = 0` this produces a non-trivial
distribution of erosion times.

Output file: `data/{rule}_erosion_stats_L{L}_n_samples{n}_p{p}_ds{ds}.jld2`.
"""
function run_erosion_stats(L::Int, rule::AbstractString;
                           p::Float64 = 0.0,
                           domain_size::Float64 = 0.1,
                           n_samples::Int = 500,
                           max_time::Union{Nothing,Int} = nothing,
                           η::Float64 = 0.0,
                           use_or_probability::Float64 = 0.5,
                           init_cond::AbstractString = "circ")

    @assert n_samples >= 1            "n_samples must be ≥ 1"
    @assert 0.0 < domain_size < 0.5   "domain_size must be in (0, 0.5)"

    R_or, R_and = get_rule(rule)
    tmax = something(max_time, 50 * L)

    # Erosion target: the bias-aligned clean state. For η ≤ 0 the background
    # is all 1s, so the trial completes when `all(state)`; for η > 0 it's all
    # 0s, so the trial completes when `!any(state)`.
    erosion_done(state) = η > 0 ? !any(state) : all(state)

    erosion_times = zeros(Int, n_samples)
    n_timeouts    = Threads.Atomic{Int}(0)

    nthreads = Threads.nthreads()
    println("running $n_samples erosion trials on $nthreads thread(s); " *
            "L=$L, domain_size=$domain_size (R=$(round(Int, domain_size * L))), " *
            "p=$(round(p; sigdigits=4)), tmax=$tmax")

    done = Threads.Atomic{Int}(0)
    Threads.@threads for s in 1:n_samples
        state = init_state_with_domain(L, η, domain_size)
        t = 0
        @inbounds while t < tmax
            mc_sweep!(state, L, R_or, R_and, p, η;
                      use_or_probability=use_or_probability,
                      rule=rule)
            t += 1
            erosion_done(state) && break
        end
        if t == tmax
            Threads.atomic_add!(n_timeouts, 1)
        end
        erosion_times[s] = t
        k = Threads.atomic_add!(done, 1) + 1
        k % max(1, div(n_samples, 10)) == 0 &&
            println("  trial $k/$n_samples done")
    end

    nt = n_timeouts[]
    if nt > 0
        @warn "$nt / $n_samples trials hit max_time=$tmax (erosion incomplete; treat those as lower bounds)"
    end

    mean_t = Float64(sum(erosion_times)) / n_samples
    std_t  = n_samples > 1 ?
             sqrt(sum((erosion_times .- mean_t).^2) / (n_samples - 1)) :
             NaN

    return Dict{String, Any}(
        "rule"                => rule,
        "mode"                => "erosion_stats",
        "L"                   => L,
        "p"                   => p,
        "domain_size"         => domain_size,
        "n_samples"           => n_samples,
        "max_time"            => tmax,
        "η"                   => η,
        "use_or_probability"  => use_or_probability,
        "init_cond"           => String(init_cond),
        "erosion_times"       => erosion_times,
        "n_timeouts"          => nt,
        "mean_erosion_time"   => mean_t,
        "std_erosion_time"    => std_t,
    )
end
