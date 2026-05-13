module SqueezingCriticality

__precompile__(true)

using Printf, Random, Statistics, LinearAlgebra, JLD2, ProgressMeter,
      InteractiveUtils   # for `clipboard(filename)` after save

include("helpers.jl")
include("stats.jl")
include("trel.jl")
include("quench.jl")
include("erosion_stats.jl")
include("coarsening.jl")
include("simulation.jl")

export run_simulation, run_stats, run_trel, run_quench,
       run_erosion_stats, run_coarsening,
       monte_carlo!, mc_sweep!, first_passage_time, hoshen_kopelman,
       get_rule, update_site!, update_site_ising!, init_state,
       init_state_with_domain,
       magnetization_value, anisotropy_value, compute_moments,
       OBSERVABLES, OBSERVABLE_VALUES,
       autocorr_window, t_auto_from_corr, accumulate_spatial!,
       block_jackknife_stats, t_auto_jackknife_err,
       default_p_window

end  # module SqueezingCriticality
