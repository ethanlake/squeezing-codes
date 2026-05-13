"""
    mc_sweep!(state, L, R_or, R_and, p, η;
              use_or_probability=0.5, rule="")

One Monte Carlo sweep: `L²` asynchronous site updates. Dispatches at sweep
level (one branch per sweep, not per site) between:

- `rule == "Toom"`: `update_site_toom!` — 3-site majority vote `maj(self,
  n₁, n₂)` with neighbor offsets from `R_or` (R_and is unused), noise `p`.
- `rule == "R3"`: `update_site_r3!` — R-rule with the site itself included
  in both the OR and the AND.
- `rule == "Ising"`: `update_site_ising!` — zero-T Glauber on the 2D NN
  Ising model. Offsets are inlined (4 NN); `R_or`/`R_and` ignored.
- all other rules: squeezing-rule `update_site!` — OR/AND coin flip on
  `R_or` / `R_and`, noise `p` per site.
"""
function mc_sweep!(state::AbstractMatrix{Bool}, L::Int, R_or, R_and,
                   p::Float64, η::Float64;
                   use_or_probability::Float64=0.5,
                   rule::AbstractString="")
    N = L * L
    if rule == "Toom"
        @inbounds for _ in 1:N
            i = rand(1:L); j = rand(1:L)
            update_site_toom!(state, i, j, L, R_or, p, η)
        end
    elseif rule == "R3"
        @inbounds for _ in 1:N
            i = rand(1:L); j = rand(1:L)
            update_site_r3!(state, i, j, L, R_or, R_and, p, η;
                            use_or_probability=use_or_probability)
        end
    elseif rule == "Ising"
        @inbounds for _ in 1:N
            i = rand(1:L); j = rand(1:L)
            update_site_ising!(state, i, j, L, p, η)
        end
    else
        @inbounds for _ in 1:N
            i = rand(1:L); j = rand(1:L)
            update_site!(state, i, j, L, R_or, R_and, p, η;
                         use_or_probability=use_or_probability)
        end
    end
    return state
end

# Keys written to the per-run output dict for a single observable label.
# `t_auto_<obs>` is the magnetization autocorrelation time (the exponential
# decay time of the connected autocorrelation function of |obs|), in MC
# sweeps. It is distinct from `t_rel`, the first-passage / relaxation time
# produced by `--mode=trel` and stored in trel-mode JLD2 files.
_obs_scalar_keys(obs::AbstractString) = (
    obs, "chi_$obs", "bind_$obs", "t_auto_$obs",
    "$(obs)_err", "chi_$(obs)_err", "bind_$(obs)_err", "t_auto_$(obs)_err",
)

"""
    monte_carlo!(state, R_or, R_and, p, η, n_steps; ...)

Run `n_steps` MC sweeps on `state` (modified in place).

If `take_data`, samples each observable defined in `OBSERVABLES` every
`data_taking_ratio` sweeps and returns a Dict with, for each observable
`obs` in `OBSERVABLES`:

    obs, chi_obs, bind_obs, t_auto_obs,
    obs_err, chi_obs_err, bind_obs_err, t_auto_obs_err,
    obs_hist   (per-site time series, Vector{Float64})

plus `corr_x`, `corr_y`, `corr_t` (spatial/temporal correlators of the bit
state). `t_auto_obs` is the magnetization autocorrelation time of |obs|,
in MC sweeps — distinct from `t_rel` (first-passage time, written by
`--mode=trel`).
"""
function monte_carlo!(state::AbstractMatrix{Bool}, R_or, R_and,
                      p::Float64, η::Float64, n_steps::Int;
                      take_data::Bool=true,
                      spatial_corr::Bool=true,
                      temporal_corr::Bool=true,
                      data_taking_ratio::Int=20,
                      maxrfact::Float64=1/2.5,
                      maxtaufact::Float64=2.0,
                      use_or_probability::Float64=0.5,
                      show_progress::Bool=true,
                      rule::AbstractString="")

    L, L2 = size(state)
    @assert L == L2 "state must be square"
    N       = L * L
    max_r   = round(Int, L * maxrfact)
    max_tau = round(Int, L * maxtaufact)
    if temporal_corr && take_data
        @assert n_steps > max_tau "need n_steps > $max_tau for temporal correlator"
    end

    n_data = take_data ? div(n_steps, data_taking_ratio) : 0

    # One time series per observable.
    hists = Dict(obs => (take_data ? zeros(Float64, n_data) : Float64[])
                 for obs in OBSERVABLES)

    corr_x = zeros(max_r)
    corr_y = zeros(max_r)
    corr_t = zeros(max_tau)

    buf = (take_data && temporal_corr) ? [falses(L, L) for _ in 1:max_tau] : nothing
    xcorr_samps = 0
    tcorr_samps = 0

    prog = show_progress ? Progress(n_steps, dt=1) : nothing
    for t in 1:n_steps
        mc_sweep!(state, L, R_or, R_and, p, η;
                  use_or_probability=use_or_probability, rule=rule)

        if take_data && t % data_taking_ratio == 0
            k = div(t, data_taking_ratio)
            for obs in OBSERVABLES
                hists[obs][k] = OBSERVABLE_VALUES[obs](state)
            end
            if spatial_corr
                accumulate_spatial!(corr_x, corr_y, state)
                xcorr_samps += 1
            end
        end

        if take_data && temporal_corr
            idx = mod1(t, max_tau)
            buf[idx] .= state
            if idx == max_tau
                for τ in 1:max_tau
                    diff = hamming(buf[τ], buf[1])
                    corr_t[τ] += (N - 2 * diff) / N
                end
                tcorr_samps += 1
            end
        end

        show_progress && next!(prog)
    end
    show_progress && finish!(prog)

    data = Dict{String, Any}(
        "corr_x" => corr_x,
        "corr_y" => corr_y,
        "corr_t" => corr_t,
    )
    for obs in OBSERVABLES
        data["$(obs)_hist"] = hists[obs]
    end

    if take_data
        for obs in OBSERVABLES
            mom = compute_moments(hists[obs], N; data_taking_ratio=data_taking_ratio)
            data[obs]                      = mom.mean
            data["chi_$obs"]               = mom.chi
            data["bind_$obs"]              = mom.bind
            data["t_auto_$obs"]            = mom.t_auto
            data["$(obs)_err"]             = mom.mean_err
            data["chi_$(obs)_err"]         = mom.chi_err
            data["bind_$(obs)_err"]        = mom.bind_err
            data["t_auto_$(obs)_err"]      = mom.t_auto_err
        end
        if spatial_corr && xcorr_samps > 0
            corr_x ./= xcorr_samps
            corr_y ./= xcorr_samps
        end
        if temporal_corr && tcorr_samps > 0
            corr_t ./= tcorr_samps
        end
    else
        for obs in OBSERVABLES, k in _obs_scalar_keys(obs)
            data[k] = NaN
        end
    end

    return data
end

"""
    default_p_window(rule)

Starting guesses for (pmin, pmax, pc) per rule, for unbiased (η = 0)
dynamics. pc values come from mag-quench runs on large systems under this
repo's async-Poisson semantics; override with `--p=...` or `--pmin/--pmax`
for biased runs or pre-measured refinements.
"""
function default_p_window(rule::AbstractString)
    if rule == "R"
        return (0.028, 0.050, 0.038425)
    elseif rule == "R3"
        return (0.020, 0.040, 0.02876)
    elseif rule == "M"
        return (0.0020, 0.0045, 0.0032875)
    elseif rule == "F"
        return (0.0085, 0.0145, 0.01165)
    elseif rule == "Toom"
        return (0.115, 0.155, 0.13395)
    elseif rule == "Ising"
        # 2D zero-T Glauber + noise. pc ≈ 0.141 from earlier measurements
        # (MemoryNCA/Ethan/ca_statistics.jl, `zeroT_glauber` rule).
        return (0.10, 0.17, 0.141)
    else
        error("Unknown rule $rule")
    end
end

# Combine per-sample outputs into the aggregate scalar vectors for a single
# sweep point. Means/chi/bind are simple averages; their errors combine across
# n_samples as σ_tot = √(Σ σ_s²) / n_samples, which requires every sample to
# have returned a finite error. τ_exp is averaged only over samples whose fit
# converged; its error combines the same way over those same samples.
function _combine_sample_scalars!(scalars::Dict{String, Vector{Float64}},
                                  i::Int, obs::AbstractString,
                                  sample_outs::AbstractVector)
    n = length(sample_outs)
    n == 0 && return

    for key in (obs, "chi_$obs", "bind_$obs")
        scalars[key][i] = sum(s[key] for s in sample_outs) / n
    end

    τ_vals = [s["t_auto_$obs"] for s in sample_outs if !isnan(s["t_auto_$obs"])]
    scalars["t_auto_$obs"][i] = isempty(τ_vals) ? NaN : mean(τ_vals)

    for base in (obs, "chi_$obs", "bind_$obs")
        ek   = "$(base)_err"
        errs = [s[ek] for s in sample_outs]
        scalars[ek][i] = all(!isnan, errs) ?
                         sqrt(sum(e^2 for e in errs)) / n : NaN
    end
    τ_errs = [s["t_auto_$(obs)_err"] for s in sample_outs
              if !isnan(s["t_auto_$(obs)_err"])]
    scalars["t_auto_$(obs)_err"][i] = isempty(τ_errs) ? NaN :
                                      sqrt(sum(e^2 for e in τ_errs)) / length(τ_errs)
end

# Combine per-sample correlator vectors into the aggregate mean + SEM at a
# given sweep point. Works for both spatial (max_r) and temporal (max_tau)
# correlators. Across-sample SEM is undefined at n_samples=1 and left NaN.
function _combine_sample_vectors!(mean_mat::Matrix{Float64},
                                  err_mat::Matrix{Float64},
                                  i::Int,
                                  samples::AbstractVector{<:AbstractVector{<:Real}})
    n = length(samples)
    n == 0 && return
    len = length(samples[1])
    s1  = zeros(Float64, len)
    s2  = zeros(Float64, len)
    for v in samples
        @assert length(v) == len "per-sample correlator length mismatch"
        s1 .+= v
        s2 .+= v .^ 2
    end
    means = s1 ./ n
    mean_mat[1:len, i] .= means
    if n > 1
        vars = max.(0.0, (s2 .- n .* means .^ 2) ./ (n - 1))
        err_mat[1:len, i] .= sqrt.(vars ./ n)
    end
end

"""
    run_stats(L, rule; ...)

Main driver for stats-mode data collection.

- If `vary_L=false` (default): sweep `n_ps` linearly spaced values of `p` in
  [pmin, pmax] at fixed L.
- If `vary_L=true`: sweep `n_Ls` log-spaced values of L in [Lmin, Lmax] at
  fixed p (defaults to the rule's pc).

For each (L, p) pair, thermalize for `thermalizing_steps` sweeps, then run
`data_steps` sweeps of measurement, averaging over `n_samples` independent
initializations.

Computes, for each observable in `OBSERVABLES` (currently `"m"` and `"D"`):
⟨|x|⟩, χ, Binder cumulant, τ_exp, and 1σ block-jackknife errors on each.

Returns a Dict matching the JLD2 schema.
"""
function run_stats(L::Int, rule::AbstractString;
                   pmin::Union{Nothing,Float64}=nothing,
                   pmax::Union{Nothing,Float64}=nothing,
                   n_ps::Int=12,
                   vary_L::Bool=false,
                   Lmin::Int=12, Lmax::Int=96, n_Ls::Int=7,
                   p::Union{Nothing,Float64}=nothing,
                   pc::Union{Nothing,Float64}=nothing,
                   log_spacing::Bool=false,
                   n_samples::Int=1,
                   thermalizing_steps::Int=5000,
                   data_steps::Int=500_000,
                   data_taking_ratio::Int=20,
                   save_corrs::Bool=false,
                   η::Float64=0.0,
                   use_or_probability::Float64=0.5,
                   init_cond::AbstractString="rand",
                   maxrfact::Float64=1/2.5,
                   maxtaufact::Float64=2.0)

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

    Lmax_eff = maximum(Ls)
    max_r    = round(Int, Lmax_eff * maxrfact)
    max_tau  = round(Int, Lmax_eff * maxtaufact)

    # Pre-allocate scalar result vectors for every (observable × key) pair.
    scalars = Dict{String, Vector{Float64}}()
    for obs in OBSERVABLES, key in _obs_scalar_keys(obs)
        scalars[key] = fill(NaN, n)
    end

    # Correlator matrices stored as (max_r/max_tau, n) so that h5py in Python
    # sees the documented shape (n, max_r/max_tau).
    corr_x_mat     = zeros(Float64, max_r,   n)
    corr_y_mat     = zeros(Float64, max_r,   n)
    corr_t_mat     = zeros(Float64, max_tau, n)
    corr_x_err_mat = fill(NaN,      max_r,   n)
    corr_y_err_mat = fill(NaN,      max_r,   n)
    corr_t_err_mat = fill(NaN,      max_tau, n)

    nthreads = Threads.nthreads()
    # One thread gets a live ProgressMeter bar for its share of the sweep;
    # the others stay silent to avoid clobbering the bar's carriage-return
    # redraws. Thread 1 is a natural choice — with Threads.@threads's static
    # partitioning it always receives a chunk of iterations.
    bar_thread = 1
    # Per-iteration sweep count handled by the "bar thread" once assigned its
    # first iteration (set below); used to size the Progress bar correctly.
    println("running $n sweep points on $nthreads thread(s) " *
            "(live progress bar shown for thread $bar_thread; " *
            "other threads print a line per completed sample)")

    # Sweep points are independent; thread over i. Each iteration writes only
    # to its own slot in the result vectors/matrices, so no locking is needed.
    # Julia's default per-task RNG (Xoshiro, v1.7+) gives each thread its own
    # random stream automatically.
    done         = Threads.Atomic{Int}(0)   # completed sweep points
    samples_done = Threads.Atomic{Int}(0)   # completed (point, sample) pairs
    n_pairs      = n * n_samples
    t0 = time()
    # Stable bar state for `bar_thread` across its successive sweep points
    # (created on its first iteration, then `update!`'d after each sample).
    bar_ref::Ref{Union{Nothing, Progress}} = Ref{Union{Nothing, Progress}}(nothing)
    bar_done = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:n
        Li = Ls[i]; pi = ps[i]
        is_bar_thread = Threads.threadid() == bar_thread

        sample_outs = Vector{Dict{String, Any}}(undef, n_samples)
        for s in 1:n_samples
            state = init_state(Li, η, init_cond)
            monte_carlo!(state, R_or, R_and, pi, η, thermalizing_steps;
                         take_data=false,
                         spatial_corr=false, temporal_corr=false,
                         use_or_probability=use_or_probability,
                         show_progress=false,
                         rule=rule)
            sample_outs[s] = monte_carlo!(state, R_or, R_and, pi, η, data_steps;
                                          take_data=true,
                                          spatial_corr=save_corrs,
                                          temporal_corr=save_corrs,
                                          data_taking_ratio=data_taking_ratio,
                                          maxrfact=maxrfact, maxtaufact=maxtaufact,
                                          use_or_probability=use_or_probability,
                                          show_progress=false,
                                          rule=rule)
            # On the bar thread, advance a single stable ProgressMeter bar
            # that tracks samples completed *on that thread* out of the total
            # it's expected to do. Other threads print a one-line text update.
            Threads.atomic_add!(samples_done, 1)
            if is_bar_thread
                # Thread 1's workload under static @threads partitioning:
                # ⌈n / nthreads⌉ sweep points × n_samples samples each.
                if bar_ref[] === nothing
                    per_thread_n = cld(n, nthreads)
                    bar_ref[] = Progress(per_thread_n * n_samples, dt=1.0,
                                         desc="thread $bar_thread: ",
                                         showspeed=true)
                end
                Threads.atomic_add!(bar_done, 1)
                ProgressMeter.update!(bar_ref[], bar_done[])
            else
                kp = samples_done[]
                elapsed = time() - t0
                eta     = elapsed * (n_pairs - kp) / max(kp, 1)
                println("    [$kp/$n_pairs samples] L=$Li p=$(round(pi; sigdigits=4)) " *
                        "sample $s/$n_samples done  " *
                        "elapsed=$(round(elapsed; digits=1))s  " *
                        "eta=$(round(eta; digits=1))s")
            end
        end

        for obs in OBSERVABLES
            _combine_sample_scalars!(scalars, i, obs, sample_outs)
        end

        if save_corrs
            _combine_sample_vectors!(corr_x_mat, corr_x_err_mat, i,
                                     [out["corr_x"] for out in sample_outs])
            _combine_sample_vectors!(corr_y_mat, corr_y_err_mat, i,
                                     [out["corr_y"] for out in sample_outs])
            _combine_sample_vectors!(corr_t_mat, corr_t_err_mat, i,
                                     [out["corr_t"] for out in sample_outs])
        end

        k = Threads.atomic_add!(done, 1) + 1
        println("  [$k/$n] rule=$rule L=$Li p=$(round(pi; sigdigits=4)) " *
                "m=$(round(scalars["m"][i]; sigdigits=4)) " *
                "bind_m=$(round(scalars["bind_m"][i]; sigdigits=4)) " *
                "D=$(round(scalars["D"][i]; sigdigits=4)) " *
                "bind_D=$(round(scalars["bind_D"][i]; sigdigits=4))")
    end
    if bar_ref[] !== nothing
        ProgressMeter.finish!(bar_ref[])
    end

    # Assemble the output dict: metadata + all observable scalars + correlators.
    out = Dict{String, Any}(
        "rule"               => rule,
        "mode"               => "stats",
        "L"                  => L,
        "Ls"                 => Ls,
        "ps"                 => ps,
        "corr_x"             => corr_x_mat,
        "corr_y"             => corr_y_mat,
        "corr_t"             => corr_t_mat,
        "corr_x_err"         => corr_x_err_mat,
        "corr_y_err"         => corr_y_err_mat,
        "corr_t_err"         => corr_t_err_mat,
        "max_r"              => max_r,
        "max_tau"            => max_tau,
        "thermalizing_steps" => thermalizing_steps,
        "data_steps"         => data_steps,
        "data_taking_ratio"  => data_taking_ratio,
        "n_samples"          => n_samples,
        "η"                  => η,
        "use_or_probability" => use_or_probability,
        "save_corrs"         => save_corrs,
        "vary_L"             => vary_L,
        "init_cond"          => String(init_cond),
        "observables"        => collect(OBSERVABLES),
    )
    for (k, v) in scalars
        out[k] = v
    end
    return out
end
