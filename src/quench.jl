"""
    run_quench(L, rule; ...)

Relaxation-from-uniform driver. Initialise the system in `init_cond="clean"`
(all spins aligned against the noise bias — `m = +1` for η ≤ 0, `m = -1`
for η > 0), evolve at fixed `p` (default the rule's `pc`) for `T` MC
sweeps per trajectory, and record `m(t)` and `D(t)` each sweep. The
trajectory-averaged `⟨m⟩(t)`, `⟨|m|⟩(t)`, `⟨D⟩(t)`, `⟨|D|⟩(t)` and their
SEMs are what the plotter uses to extract the dynamical exponent `z` via a
log-log power-law fit of `⟨|x|⟩(t) ∝ t^{−β/(ν z)}`.

Trajectories are run in parallel across threads; each seeds its own
`init_state` and produces its own `(n_t,)` time series. At the end the
`n_t × n_samples` matrix per observable is reduced to mean + SEM over the
trajectory axis.

Output file: `data/{rule}_quench_L{L}_...jld2`.
"""
function run_quench(L::Int, rule::AbstractString;
                    p::Union{Nothing,Float64}=nothing,
                    T::Union{Nothing,Int}=nothing,
                    n_samples::Int=50,
                    data_taking_ratio::Int=1,
                    η::Float64=0.0,
                    use_or_probability::Float64=0.5,
                    init_cond::AbstractString="clean")

    R_or, R_and = get_rule(rule)
    _, _, pc    = default_p_window(rule)
    p_v = something(p, pc)
    T_v = something(T, max(500, 20 * L))

    @assert T_v >= data_taking_ratio "T must be ≥ data_taking_ratio"
    @assert n_samples >= 1            "n_samples must be ≥ 1"

    n_t  = div(T_v, data_taking_ratio)
    # Measurement-time axis in MC sweeps (1-based: first sample after 1 sweep).
    ts   = collect(data_taking_ratio:data_taking_ratio:data_taking_ratio * n_t)

    # Per-observable (n_t × n_samples) buffer of per-trajectory time series.
    trajs = Dict(obs => zeros(Float64, n_t, n_samples) for obs in OBSERVABLES)

    nthreads = Threads.nthreads()
    println("running $n_samples trajectories on $nthreads thread(s); " *
            "L=$L, T=$T_v, p=$(round(p_v; sigdigits=4))")

    done = Threads.Atomic{Int}(0)
    Threads.@threads for s in 1:n_samples
        state = init_state(L, η, init_cond)
        @inbounds for t in 1:T_v
            mc_sweep!(state, L, R_or, R_and, p_v, η;
                      use_or_probability=use_or_probability,
                      rule=rule)
            if t % data_taking_ratio == 0
                k = div(t, data_taking_ratio)
                for obs in OBSERVABLES
                    trajs[obs][k, s] = OBSERVABLE_VALUES[obs](state)
                end
            end
        end
        k = Threads.atomic_add!(done, 1) + 1
        # One line per completed trajectory so the user has *some* progress
        # signal without the multi-line `Progress` spam across threads.
        k % max(1, div(n_samples, 10)) == 0 &&
            println("  trajectory $k/$n_samples done")
    end

    # Reduce (n_t × n_samples) → (n_t,) mean + SEM per observable, for both
    # the signed and abs series.
    out = Dict{String, Any}(
        "rule"               => rule,
        "mode"               => "quench",
        "L"                  => L,
        "p"                  => p_v,
        "T"                  => T_v,
        "ts"                 => ts,
        "n_samples"          => n_samples,
        "data_taking_ratio"  => data_taking_ratio,
        "η"                  => η,
        "use_or_probability" => use_or_probability,
        "init_cond"          => String(init_cond),
        "observables"        => collect(OBSERVABLES),
    )
    for obs in OBSERVABLES
        A = trajs[obs]                          # (n_t, n_samples)
        μ  = vec(mean(A;         dims=2))
        μa = vec(mean(abs.(A);   dims=2))
        if n_samples > 1
            σ  = vec(std(A;       dims=2, corrected=true)) ./ sqrt(n_samples)
            σa = vec(std(abs.(A); dims=2, corrected=true)) ./ sqrt(n_samples)
        else
            σ  = fill(NaN, n_t)
            σa = fill(NaN, n_t)
        end
        out["$(obs)_t"]         = μ
        out["abs_$(obs)_t"]     = μa
        out["$(obs)_t_err"]     = σ
        out["abs_$(obs)_t_err"] = σa

        # Per-time Binder cumulant, `B(t) = (3 − ⟨x⁴⟩(t) / ⟨x²⟩(t)²) / 2`,
        # with the across-trajectory ensemble taking the role of the steady-
        # state ensemble used in stats mode. Same definition (symmetric
        # convention reaching 1 in the ordered limit, 0 in the disordered
        # limit) as `compute_moments` in helpers.jl.
        m2 = vec(mean(A .^ 2; dims=2))
        m4 = vec(mean(A .^ 4; dims=2))
        # m2 = 0 → 0/0 → NaN naturally; downstream plotters already handle NaN.
        out["bind_$(obs)_t"] = (3 .- m4 ./ m2 .^ 2) ./ 2

        # Across-trajectory error on B(t): jackknife with n_samples blocks
        # (leave-one-trajectory-out). Cheap (n_samples × n_t small reductions)
        # and exact for `n_samples > 1`; left as NaN for the single-trajectory
        # case where it's undefined.
        if n_samples > 1
            sum2 = sum(A .^ 2; dims=2)
            sum4 = sum(A .^ 4; dims=2)
            bind_jk = zeros(Float64, n_t, n_samples)
            @inbounds for s in 1:n_samples
                jk2 = (sum2 .- A[:, s] .^ 2) ./ (n_samples - 1)
                jk4 = (sum4 .- A[:, s] .^ 4) ./ (n_samples - 1)
                bind_jk[:, s] .= vec((3 .- jk4 ./ jk2 .^ 2) ./ 2)
            end
            jk_mean = vec(mean(bind_jk; dims=2))
            jk_var  = vec(sum((bind_jk .- jk_mean) .^ 2; dims=2)) .*
                      (n_samples - 1) / n_samples
            out["bind_$(obs)_t_err"] = sqrt.(max.(jk_var, 0.0))
        else
            out["bind_$(obs)_t_err"] = fill(NaN, n_t)
        end
    end
    return out
end
