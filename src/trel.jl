"""
    first_passage_time(state, R_or, R_and, p, η, max_time;
                       use_or_probability=0.5, M_threshold=0.0)

Evolve `state` (modified in place) one MC sweep at a time and return the
number of sweeps before its magnetization first crosses zero (the "sign flip"
first-passage time).

The initial sign of `m = magnetization_value(state)` is captured on entry;
the return value is the first sweep at which `sign(m) ≠ sign(m_init)` (or
`|m| ≤ M_threshold` if a positive threshold is supplied). If no crossing
occurs within `max_time` sweeps, returns `max_time` (timeout sentinel).

Returns `0` if the initial state already satisfies the stopping criterion.
"""
function first_passage_time(state::AbstractMatrix{Bool}, R_or, R_and,
                            p::Float64, η::Float64, max_time::Int;
                            use_or_probability::Float64=0.5,
                            M_threshold::Float64=0.0,
                            rule::AbstractString="")
    L = size(state, 1)
    m0 = magnetization_value(state)
    init_sign = sign(m0)
    # already at / past threshold: no evolution needed
    if init_sign == 0 || abs(m0) <= M_threshold
        return 0
    end
    for t in 1:max_time
        if t % round(Int,max_time/10) == 0 println("$(t/max_time)...") end 
        mc_sweep!(state, L, R_or, R_and, p, η;
                  use_or_probability=use_or_probability, rule=rule)
        m = magnetization_value(state)
        if sign(m) != init_sign || abs(m) <= M_threshold
            return t
        end
    end
    return max_time  # timeout
end

"""
    run_trel(L, rule; ...)

Relaxation-time (first-passage) driver.

For each sweep point, `n_trials` independent trials are launched: each is
initialized in the `"clean"` state (aligned against the noise bias η, so for
η > 0 the system starts all-down, for η ≤ 0 all-up), then evolved one MC
sweep at a time until its magnetization crosses zero. Average crossing time
= the relaxation time `τ_rel(p)`.

- If `vary_L = false` (default): sweep `n_ps` linearly spaced `p ∈ [pmin, pmax]`
  at fixed `L`.
- If `vary_L = true`: sweep `n_Ls` log-spaced `L ∈ [Lmin, Lmax]` at fixed `p`
  (defaults to the rule's `pc`).

Trials within each sweep point are threaded; sweep points are run sequentially
so the progress-logged `[i/n]` lines stay interleaved sensibly.

Returns a Dict matching the `trel`-mode JLD2 schema. The first-passage
times are written under keys `t_rel`, `t_rel_err`, `t_rel_median`,
`t_rel_timeouts`, and `t_rel_times` — distinct from `t_auto`, the
magnetization autocorrelation time produced by `--mode=stats`.
"""
function run_trel(L::Int, rule::AbstractString;
                  pmin::Union{Nothing,Float64}=nothing,
                  pmax::Union{Nothing,Float64}=nothing,
                  n_ps::Int=12,
                  vary_L::Bool=false,
                  Lmin::Int=12, Lmax::Int=96, n_Ls::Int=7,
                  p::Union{Nothing,Float64}=nothing,
                  pc::Union{Nothing,Float64}=nothing,
                  log_spacing::Bool=false,
                  n_trials::Int=100,
                  max_time::Int=1_000_000,
                  M_threshold::Float64=0.0,
                  η::Float64=0.0,
                  use_or_probability::Float64=0.5,
                  init_cond::AbstractString="clean")

    R_or, R_and = get_rule(rule)
    pmin_d, pmax_d, pc_d = default_p_window(rule)
    pmin_v = something(pmin, pmin_d)
    pmax_v = something(pmax, pmax_d)
    pc_v   = something(pc, pc_d)
    p_v    = something(p, pc_v)

    if vary_L
        Ls = round.(Int, exp.(range(log(Lmin), log(Lmax), length=n_Ls)))
        ps = fill(p_v, n_Ls)
        n  = n_Ls
    else
        Ls = fill(L, n_ps)
        ps = _generate_ps(pmin_v, pmax_v, n_ps, pc_v, log_spacing)
        n  = n_ps
    end

    t_rel_mean      = zeros(Float64, n)
    t_rel_sem       = fill(NaN, n)
    t_rel_median    = fill(NaN, n)
    t_rel_timeouts  = zeros(Int,   n)
    # Raw per-trial times, shape (n_trials, n) so Python sees (n, n_trials).
    t_rel_times_mat = zeros(Int, n_trials, n)

    nthreads = Threads.nthreads()
    println("running $n sweep points × $n_trials trials on $nthreads thread(s); max_time = $max_time")

    for i in 1:n
        Li = Ls[i]; pi = ps[i]
        trial_times = zeros(Int, n_trials)
        Threads.@threads for t in 1:n_trials
            state = init_state(Li, η, init_cond)
            trial_times[t] = first_passage_time(state, R_or, R_and, pi, η, max_time;
                                                use_or_probability=use_or_probability,
                                                M_threshold=M_threshold,
                                                rule=rule)
        end
        t_rel_times_mat[:, i] .= trial_times
        t_rel_mean[i]     = mean(trial_times)
        t_rel_sem[i]      = n_trials > 1 ? std(trial_times) / sqrt(n_trials) : NaN
        t_rel_median[i]   = median(trial_times)
        t_rel_timeouts[i] = count(==(max_time), trial_times)

        status = t_rel_timeouts[i] > 0 ?
                 " ($(t_rel_timeouts[i])/$n_trials timed out — τ_rel is a lower bound)" : ""
        println("  [$i/$n] rule=$rule L=$Li p=$(round(pi; sigdigits=4)) " *
                "⟨τ_rel⟩=$(round(t_rel_mean[i]; sigdigits=4)) " *
                "median=$(round(t_rel_median[i]; sigdigits=4))$status")
    end

    return Dict{String, Any}(
        "rule"               => rule,
        "mode"               => "trel",
        "L"                  => L,
        "Ls"                 => Ls,
        "ps"                 => ps,
        "t_rel"              => t_rel_mean,
        "t_rel_err"          => t_rel_sem,
        "t_rel_median"       => t_rel_median,
        "t_rel_timeouts"     => t_rel_timeouts,
        "t_rel_times"        => t_rel_times_mat,
        "n_trials"           => n_trials,
        "max_time"           => max_time,
        "M_threshold"        => M_threshold,
        "η"                  => η,
        "use_or_probability" => use_or_probability,
        "vary_L"             => vary_L,
        "init_cond"          => String(init_cond),
    )
end
