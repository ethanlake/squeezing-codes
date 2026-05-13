#!/usr/bin/env julia
#=
Single-run driver for squeezing-criticality simulations.

Usage from the command line:
    julia --project=. simulation_driver.jl [--param=value ...]

    Examples:
    julia --project=. simulation_driver.jl --rule=R --L=24                 # stats mode
    julia --project=. simulation_driver.jl --mode=trel --rule=R --L=24     # relaxation time
    julia --project=. simulation_driver.jl --rule=M --L=48 --pmin=0.0025 --pmax=0.004
    julia --project=. simulation_driver.jl --rule=F --vary_L=true --Lmin=12 --Lmax=96

Usage from the REPL:
    using Pkg; Pkg.activate(".")
    using Revise
    include("simulation_driver.jl")
=#

try
    using Revise
catch
    @warn "Revise.jl not loaded: restart the REPL to pick up module changes."
end

using SqueezingCriticality

### common defaults ###

mode               = "stats"
rule               = "R"
L                  = 24
pmin               = nothing         # falls back to default_p_window(rule)
pmax               = nothing
n_ps               = 12
vary_L             = false
Lmin               = 12
Lmax               = 96
n_Ls               = 7
p                  = nothing          # only used with vary_L=true
pc                 = nothing          # override default_p_window's pc
log_spacing        = false            # log-uniform in (p − pc) when true
η                  = 0.0
use_or_probability = 0.5
out_adj            = ""
save               = true
test               = false

# stats-mode defaults
n_samples          = 1
thermalizing_steps = 5000
data_steps         = 500_000
data_taking_ratio  = 20
save_corrs         = false     # correlators are expensive; opt in explicitly

# trel-mode defaults
n_trials           = 100
max_time           = 1_000_000
M_threshold        = 0.0

# quench-mode defaults (n_samples is shared with stats mode and parsed below).
# Use `0` as a "not-provided" sentinel so parse_arg types it as Int; we map
# 0 -> nothing below so run_quench falls back to its internal default
# `max(500, 20*L)`.
T                  = 0

# erosion_stats-mode defaults. `domain_size` is a fraction of L; the runner
# uses `max_time = 50L` when --max_time is not provided (handled below by
# checking `:max_time in provided_params`).
domain_size        = 0.1

### CLI parser ###

function parse_arg(key::AbstractString, default_value)
    for arg in ARGS
        if startswith(arg, "--$key=")
            value_str = split(arg, "=", limit=2)[2]
            if lowercase(value_str) == "nothing" || value_str == ""
                return nothing
            elseif default_value isa Bool
                return lowercase(value_str) in ["true", "1", "yes"]
            elseif default_value isa Int
                return parse(Int, value_str)
            elseif default_value isa Float64
                return parse(Float64, value_str)
            elseif default_value isa AbstractString
                return String(value_str)
            elseif default_value === nothing
                fv = tryparse(Float64, value_str)
                fv !== nothing && return fv
                iv = tryparse(Int, value_str)
                iv !== nothing && return iv
                return String(value_str)
            else
                return String(value_str)
            end
        end
    end
    return default_value
end

function arg_provided(keys::AbstractString...)
    for k in keys, arg in ARGS
        startswith(arg, "--$k=") && return true
    end
    return false
end

# parse --mode first so we can set mode-dependent defaults before other parses.
mode = parse_arg("mode", mode)
mode in ("stats", "trel", "quench", "erosion_stats", "coarsening") ||
    error("Unknown mode: $mode (supported: stats, trel, quench, erosion_stats, coarsening)")

# mode-dependent default for init_cond:
# - stats / coarsening: random (we want the full distribution / disordered
#   start so domains can grow under the dynamics).
# - trel / quench: "clean" — aligned against the noise bias.
# - erosion_stats: "circ" — a clean background with a centered minority disk.
init_cond = if mode in ("trel", "quench")
    "clean"
elseif mode == "erosion_stats"
    "circ"
else
    "rand"
end

rule               = parse_arg("rule", rule)
L                  = parse_arg("L", L)
pmin               = parse_arg("pmin", pmin)
pmax               = parse_arg("pmax", pmax)
n_ps               = parse_arg("n_ps", n_ps)
vary_L             = parse_arg("vary_L", vary_L)
Lmin               = parse_arg("Lmin", Lmin)
Lmax               = parse_arg("Lmax", Lmax)
n_Ls               = parse_arg("n_Ls", n_Ls)
p                  = parse_arg("p", p)
pc                 = parse_arg("pc", pc)
log_spacing        = parse_arg("log_spacing", log_spacing)
η                  = parse_arg("η", η)
η                  = parse_arg("eta", η)    # ASCII alias
use_or_probability = parse_arg("use_or_probability", use_or_probability)
init_cond          = parse_arg("init_cond", init_cond)
out_adj            = parse_arg("out_adj", out_adj)
save               = parse_arg("save", save)
test               = parse_arg("test", test)

# stats-mode-only
n_samples          = parse_arg("n_samples", n_samples)
thermalizing_steps = parse_arg("thermalizing_steps", thermalizing_steps)
data_steps         = parse_arg("data_steps", data_steps)
data_taking_ratio  = parse_arg("data_taking_ratio", data_taking_ratio)
save_corrs         = parse_arg("save_corrs", save_corrs)

# trel-mode-only
n_trials           = parse_arg("n_trials", n_trials)
max_time           = parse_arg("max_time", max_time)
M_threshold        = parse_arg("M_threshold", M_threshold)

# quench-mode-only
T                  = parse_arg("T", T)

# erosion_stats-mode-only
domain_size        = parse_arg("domain_size", domain_size)

### track which parameters the user actually provided ###

provided_params = Set{Symbol}()
for (sym, cli_keys) in (
        (:pmin,               ("pmin",)),
        (:pmax,               ("pmax",)),
        (:n_ps,               ("n_ps",)),
        (:n_samples,          ("n_samples",)),
        (:thermalizing_steps, ("thermalizing_steps",)),
        (:data_steps,         ("data_steps",)),
        (:data_taking_ratio,  ("data_taking_ratio",)),
        (:save_corrs,         ("save_corrs",)),
        (:n_trials,           ("n_trials",)),
        (:max_time,           ("max_time",)),
        (:M_threshold,        ("M_threshold",)),
        (:T,                  ("T",)),
        (:domain_size,        ("domain_size",)),
        (:vary_L,             ("vary_L",)),
        (:Lmin,               ("Lmin",)),
        (:Lmax,               ("Lmax",)),
        (:n_Ls,               ("n_Ls",)),
        (:p,                  ("p",)),
        (:pc,                 ("pc",)),
        (:log_spacing,        ("log_spacing",)),
        (:η,                  ("η", "eta")),
        (:use_or_probability, ("use_or_probability",)),
        (:init_cond,          ("init_cond",)),
        (:out_adj,            ("out_adj",)),
        (:save,               ("save",)),
        (:test,               ("test",)))
    if arg_provided(cli_keys...)
        push!(provided_params, sym)
    end
end

### assemble kwargs and run ###

kwargs = Dict{Symbol, Any}(
    :pmin               => pmin,
    :pmax               => pmax,
    :n_ps               => n_ps,
    :vary_L             => vary_L,
    :Lmin               => Lmin,
    :Lmax               => Lmax,
    :n_Ls               => n_Ls,
    :p                  => p,
    :pc                 => pc,
    :log_spacing        => log_spacing,
    :η                  => η,
    :use_or_probability => use_or_probability,
    :init_cond          => init_cond,
    :out_adj            => out_adj,
    :save               => save,
    :test               => test,
    :provided_params    => provided_params,
    :_filename_ref      => Ref{Union{String, Nothing}}(nothing),
)
if mode == "stats"
    kwargs[:n_samples]          = n_samples
    kwargs[:thermalizing_steps] = thermalizing_steps
    kwargs[:data_steps]         = data_steps
    kwargs[:data_taking_ratio]  = data_taking_ratio
    kwargs[:save_corrs]         = save_corrs
elseif mode == "trel"
    kwargs[:n_trials]           = n_trials
    kwargs[:max_time]           = max_time
    kwargs[:M_threshold]        = M_threshold
elseif mode == "quench"
    kwargs[:T]                  = T == 0 ? nothing : T
    kwargs[:n_samples]          = n_samples
    kwargs[:data_taking_ratio]  = data_taking_ratio
elseif mode == "erosion_stats"
    kwargs[:p]                  = p === nothing ? 0.0 : p
    kwargs[:n_samples]          = n_samples
    kwargs[:domain_size]        = domain_size
    # `:max_time` is shared with trel mode; only forward as Int when the user
    # explicitly passed --max_time, otherwise let the runner default to 50L.
    kwargs[:max_time]           = (:max_time in provided_params) ? max_time : nothing
else  # coarsening
    kwargs[:T]                  = T == 0 ? nothing : T
    kwargs[:n_samples]          = n_samples
    kwargs[:data_taking_ratio]  = data_taking_ratio
end

if mode == "stats"
    println("Running stats: rule=$rule, L=$L, n_ps=$n_ps, data_steps=$data_steps, " *
            "η=$η, save_corrs=$save_corrs, vary_L=$vary_L")
elseif mode == "trel"
    println("Running trel: rule=$rule, L=$L, n_ps=$n_ps, n_trials=$n_trials, " *
            "max_time=$max_time, η=$η, M_threshold=$M_threshold, vary_L=$vary_L")
elseif mode == "quench"
    T_show = T == 0 ? "default" : string(T)
    p_show = p === nothing ? "pc" : string(p)
    println("Running quench: rule=$rule, L=$L, T=$T_show, " *
            "n_samples=$n_samples, η=$η, p=$p_show")
elseif mode == "erosion_stats"
    p_show    = p === nothing ? "0.0" : string(p)
    tmax_show = (:max_time in provided_params) ? string(max_time) : "50L"
    println("Running erosion_stats: rule=$rule, L=$L, domain_size=$domain_size, " *
            "n_samples=$n_samples, η=$η, p=$p_show, max_time=$tmax_show")
else  # coarsening
    T_show = T == 0 ? "default" : string(T)
    p_show = p === nothing ? "pc" : string(p)
    println("Running coarsening: rule=$rule, L=$L, T=$T_show, " *
            "n_samples=$n_samples, η=$η, p=$p_show")
end
run_simulation(mode, L, rule; kwargs...)

saved = kwargs[:_filename_ref][]
if saved !== nothing && Sys.isapple()
    msg = "Saved $(basename(saved))"
    try
        run(`osascript -e "display notification \"$msg\" with title \"simulation_driver\""`)
    catch e
        @warn "could not display notification" exception=(e, catch_backtrace())
    end
end

println("Done.")
