"""
    run_simulation(mode, L, rule; kwargs...)

Dispatch a single simulation and optionally save its output to JLD2.

Supported modes:

- `"stats"` — long Monte Carlo runs measuring ⟨|m|⟩, χ, Binder, τ_exp, and
  spatial/temporal correlators for each observable in `OBSERVABLES` (currently
  `m`, `D`). p-sweep at fixed L, or L-sweep at fixed p via `vary_L=true`.
- `"trel"` — relaxation-time / first-passage-time runs. Each trial starts
  aligned against the noise bias (`init_cond="clean"`) and is evolved until
  its magnetization crosses zero; `τ_rel(p)` is the trial-average crossing
  time. p-sweep at fixed L, or L-sweep at fixed p via `vary_L=true`.
- `"quench"` — quench-from-uniform runs. Each trajectory starts aligned
  against the noise bias (`init_cond="clean"`), evolves at fixed `p`
  (default `pc`) for `T` MC sweeps, and records `m(t)` and `D(t)` every
  `data_taking_ratio` sweeps. Trajectory-averaged decay curves
  `⟨|x|⟩(t) ∝ t^{-β/(ν z)}` are used downstream to extract `z`.

Keyword arguments forwarded to the mode-specific driver are a subset of the
full kwargs; the file-naming / control kwargs `save`, `out_adj`, and
`provided_params` are intercepted here.
"""
function run_simulation(mode::AbstractString, L::Int, rule::AbstractString; kwargs...)
    @assert mode in ("stats", "trel", "quench", "erosion_stats", "coarsening") (
        "unknown mode: $mode (supported: stats, trel, quench, erosion_stats, coarsening)")

    control_keys = (:save, :out_adj, :test, :provided_params, :_filename_ref)

    # Select only the kwargs each driver actually accepts (avoids accidental
    # pass-through of cross-mode flags set via `--key=value`).
    stats_keys        = (:pmin, :pmax, :n_ps, :vary_L, :Lmin, :Lmax, :n_Ls, :p,
                         :pc, :log_spacing,
                         :n_samples, :thermalizing_steps, :data_steps,
                         :data_taking_ratio, :save_corrs, :η, :use_or_probability,
                         :init_cond, :maxrfact, :maxtaufact)
    trel_keys         = (:pmin, :pmax, :n_ps, :vary_L, :Lmin, :Lmax, :n_Ls, :p,
                         :pc, :log_spacing,
                         :n_trials, :max_time, :M_threshold, :η,
                         :use_or_probability, :init_cond)
    quench_keys       = (:p, :T, :n_samples, :data_taking_ratio,
                         :η, :use_or_probability, :init_cond)
    erosion_stats_keys = (:p, :n_samples, :domain_size, :max_time,
                          :η, :use_or_probability, :init_cond)
    coarsening_keys   = (:p, :T, :n_samples, :data_taking_ratio,
                         :η, :use_or_probability, :init_cond)

    allowed = if mode == "stats";              stats_keys
              elseif mode == "trel";           trel_keys
              elseif mode == "quench";         quench_keys
              elseif mode == "erosion_stats";  erosion_stats_keys
              else;                            coarsening_keys
              end
    mode_kwargs = Dict{Symbol, Any}()
    for (k, v) in kwargs
        k in control_keys && continue
        k in allowed || continue
        mode_kwargs[k] = v
    end

    data = if mode == "stats"
        run_stats(L, rule; mode_kwargs...)
    elseif mode == "trel"
        run_trel(L, rule; mode_kwargs...)
    elseif mode == "quench"
        run_quench(L, rule; mode_kwargs...)
    elseif mode == "erosion_stats"
        run_erosion_stats(L, rule; mode_kwargs...)
    else  # coarsening
        run_coarsening(L, rule; mode_kwargs...)
    end

    save = get(kwargs, :save, true)
    if save !== false && save !== nothing
        filename = save_results(data, rule, L, mode, kwargs)
        ref = get(kwargs, :_filename_ref, nothing)
        ref isa Ref && (ref[] = filename)
    end

    return data
end

"""
    save_results(data, rule, L, mode, kwargs)

Serialize `data` to `data/{rule}_{mode}{specifier}.jld2`. The specifier is
built from parameters the user actually provided on the CLI (tracked via the
`:provided_params::Set{Symbol}` kwarg), so default-valued parameters do not
clutter filenames. Existing files are never overwritten — `_1`, `_2`, … are
appended on collision.
"""
function save_results(data::Dict{String, Any}, rule::AbstractString, L::Int,
                      mode::AbstractString, kwargs)
    provided = get(kwargs, :provided_params, Set{Symbol}())
    out_adj  = get(kwargs, :out_adj, "")

    if get(kwargs, :test, false) === true
        isdir("data") || mkdir("data")
        filename = "data/test.jld2"
        println("Saving to $filename ...")
        jldsave(filename; (Symbol(k) => v for (k, v) in data)...)
        println("Done.")
        return filename
    end

    parts = String["_L$L"]

    if :pmin in provided && :pmax in provided
        pmin_str = @sprintf("%.4g", data["ps"][1])
        pmax_str = @sprintf("%.4g", data["ps"][end])
        push!(parts, "_p$(pmin_str)to$(pmax_str)")
    end
    if :n_ps in provided
        push!(parts, "_n_ps$(get(kwargs, :n_ps, 12))")
    end
    if :log_spacing in provided && get(kwargs, :log_spacing, false)
        push!(parts, "_logp")
    end
    if :pc in provided
        pc_val = get(kwargs, :pc, 0.0)
        if pc_val !== nothing
            push!(parts, "_pc$(@sprintf("%.4g", pc_val))")
        end
    end
    if :vary_L in provided && get(kwargs, :vary_L, false)
        push!(parts, "_varyL$(data["Ls"][1])to$(data["Ls"][end])")
    end
    if :p in provided && get(kwargs, :vary_L, false)
        p_val = get(kwargs, :p, 0.0)
        push!(parts, "_p$(@sprintf("%.4g", p_val))")
    end
    # stats-mode params
    if :n_samples in provided
        push!(parts, "_n_samples$(get(kwargs, :n_samples, 1))")
    end
    if :data_steps in provided
        push!(parts, "_data_steps$(get(kwargs, :data_steps, 500_000))")
    end
    if :thermalizing_steps in provided
        push!(parts, "_therm$(get(kwargs, :thermalizing_steps, 5000))")
    end
    # trel-mode params
    if :n_trials in provided
        push!(parts, "_n_trials$(get(kwargs, :n_trials, 100))")
    end
    if :max_time in provided && mode != "erosion_stats"
        push!(parts, "_max_time$(get(kwargs, :max_time, 1_000_000))")
    end
    if :M_threshold in provided
        thr = get(kwargs, :M_threshold, 0.0)
        if thr != 0.0
            push!(parts, "_Mthr$(@sprintf("%.4g", thr))")
        end
    end
    # quench-mode params. `:p` is already handled above under the vary_L
    # branch; for quench it's always a scalar so we emit it unconditionally
    # when provided.
    if :T in provided
        push!(parts, "_T$(get(kwargs, :T, 0))")
    end
    if mode == "quench" && :p in provided
        p_val = get(kwargs, :p, 0.0)
        # only add if not already appended (vary_L=true branch does the same)
        if !any(startswith.(parts, "_p"))
            push!(parts, "_p$(@sprintf("%.4g", p_val))")
        end
    end
    # erosion_stats-mode params. `:p` always-scalar like quench.
    if mode == "erosion_stats" && :p in provided
        p_val = get(kwargs, :p, 0.0)
        if !any(startswith.(parts, "_p"))
            push!(parts, "_p$(@sprintf("%.4g", p_val))")
        end
    end
    # coarsening-mode params. Same p-handling as quench.
    if mode == "coarsening" && :p in provided
        p_val = get(kwargs, :p, 0.0)
        if !any(startswith.(parts, "_p"))
            push!(parts, "_p$(@sprintf("%.4g", p_val))")
        end
    end
    if mode == "erosion_stats" && :domain_size in provided
        push!(parts, "_ds$(@sprintf("%.4g", get(kwargs, :domain_size, 0.1)))")
    end
    if mode == "erosion_stats" && :max_time in provided
        mt = get(kwargs, :max_time, nothing)
        if mt !== nothing
            push!(parts, "_tmax$(mt)")
        end
    end
    # shared
    if :η in provided
        push!(parts, "_eta$(@sprintf("%.4g", get(kwargs, :η, 0.0)))")
    end
    if :use_or_probability in provided
        push!(parts, "_uor$(@sprintf("%.2g", get(kwargs, :use_or_probability, 0.5)))")
    end
    if :init_cond in provided
        push!(parts, "_init$(get(kwargs, :init_cond, "rand"))")
    end
    if !isempty(out_adj)
        push!(parts, "_$(out_adj)")
    end

    specifier = join(parts, "")
    base = "data/$(rule)_$(mode)$(specifier).jld2"

    isdir("data") || mkdir("data")

    filename = base
    if isfile(filename)
        stem = base[1:end-5]
        k = 1
        while true
            cand = "$(stem)_$(k).jld2"
            if !isfile(cand)
                filename = cand; break
            end
            k += 1
        end
    end

    println("Saving to $filename ...")
    jldsave(filename; (Symbol(k) => v for (k, v) in data)...)
    println("Done.")
    try
        clipboard(filename)
        println("(copied filename to clipboard)")
    catch e
        @warn "could not copy filename to clipboard" exception=(e, catch_backtrace())
    end
    return filename
end
