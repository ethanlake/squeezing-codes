"""
    run_coarsening(L, rule; ...)

Coarsening-mode driver. Initialises in `init_cond="rand"` (a disordered
state — the natural starting point for coarsening: random domains then
grow / merge under the rule) and evolves at fixed `p` for `T` MC sweeps
per trajectory, recording the **arithmetic mean cluster size in cells**
every `data_taking_ratio` sweeps. The mean is taken over clusters of
*both* spin species (matching `get_domain_areas(state, weighted=false)`
from MemoryNCA/Ethan/helper_functions.jl). The reduction across
`n_samples` trajectories is to mean + std (population, with Bessel
correction); only these summary statistics are saved — per-trial
trajectories are not.

Output file: `data/{rule}_coarsening_L{L}_...jld2` with keys `area_t` and
`area_t_err`.
"""
function run_coarsening(L::Int, rule::AbstractString;
                        p::Union{Nothing,Float64}=nothing,
                        T::Union{Nothing,Int}=nothing,
                        n_samples::Int=50,
                        data_taking_ratio::Int=1,
                        η::Float64=0.0,
                        use_or_probability::Float64=0.5,
                        init_cond::AbstractString="rand")

    R_or, R_and = get_rule(rule)
    _, _, pc    = default_p_window(rule)
    p_v = something(p, pc)
    T_v = something(T, max(500, 20 * L))

    @assert T_v >= data_taking_ratio "T must be ≥ data_taking_ratio"
    @assert n_samples >= 1            "n_samples must be ≥ 1"

    n_t = div(T_v, data_taking_ratio)
    ts  = collect(data_taking_ratio:data_taking_ratio:data_taking_ratio * n_t)

    trajs = zeros(Float64, n_t, n_samples)

    nthreads = Threads.nthreads()
    println("running $n_samples coarsening trajectories on $nthreads thread(s); " *
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
                sizes_t = hoshen_kopelman(state, true)
                sizes_f = hoshen_kopelman(state, false)
                ntot = length(sizes_t) + length(sizes_f)
                trajs[k, s] = ntot == 0 ? 0.0 :
                              (sum(sizes_t) + sum(sizes_f)) / ntot
            end
        end
        kk = Threads.atomic_add!(done, 1) + 1
        kk % max(1, div(n_samples, 10)) == 0 &&
            println("  trajectory $kk/$n_samples done")
    end

    μ = vec(mean(trajs; dims=2))
    σ = n_samples > 1 ?
        vec(std(trajs; dims=2, corrected=true)) :
        fill(NaN, n_t)

    return Dict{String, Any}(
        "rule"               => rule,
        "mode"               => "coarsening",
        "L"                  => L,
        "p"                  => p_v,
        "T"                  => T_v,
        "ts"                 => ts,
        "n_samples"          => n_samples,
        "data_taking_ratio"  => data_taking_ratio,
        "η"                  => η,
        "use_or_probability" => use_or_probability,
        "init_cond"          => String(init_cond),
        "area_t"             => μ,
        "area_t_err"         => σ,
    )
end
