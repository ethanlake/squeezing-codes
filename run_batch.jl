#!/usr/bin/env julia
#=
Batch runner for parameter sweeps.

Edit the parameter vectors (all equal length) to define a sweep. Each index is
one job; jobs run sequentially locally, or one-per-array-task under SLURM.

Usage:
    julia --project=. run_batch.jl                    # local, sequential
    sbatch run_batch_slurm.sh                         # SLURM array
=#

using SqueezingCriticality

# fixed defaults shared by all jobs
mode               = "stats"
thermalizing_steps = 5000
data_taking_ratio  = 20
save_corrs         = true
η                  = 0.0
use_or_probability = 0.5
init_cond          = "rand"

# per-job parameter vectors (all must have the same length)
rules       = ["R",    "R",    "R",    "M",    "M",    "F",    "F"   ]
Ls          = [16,      24,     32,     24,     32,     24,     32   ]
n_ps_vec    = [12,      12,     12,     12,     12,     12,     12   ]
n_samples_v = [1,       1,      1,      1,      1,      1,      1    ]
data_steps_v = [200_000, 200_000, 200_000, 200_000, 200_000, 200_000, 200_000]

m = length(rules)
@assert length(Ls)           == m "Ls must have length $m"
@assert length(n_ps_vec)     == m "n_ps_vec must have length $m"
@assert length(n_samples_v)  == m "n_samples_v must have length $m"
@assert length(data_steps_v) == m "data_steps_v must have length $m"

jobs = [(rule       = rules[i],
         L          = Ls[i],
         n_ps       = n_ps_vec[i],
         n_samples  = n_samples_v[i],
         data_steps = data_steps_v[i]) for i in 1:m]

println("Total jobs: $m")

function run_job(job)
    # These are CLI-provided parameters; the filename specifier will reflect them.
    provided = Set{Symbol}([:n_ps, :n_samples, :data_steps])
    run_simulation(mode, job.L, job.rule;
                   n_ps               = job.n_ps,
                   n_samples          = job.n_samples,
                   data_steps         = job.data_steps,
                   thermalizing_steps = thermalizing_steps,
                   data_taking_ratio  = data_taking_ratio,
                   save_corrs         = save_corrs,
                   η                  = η,
                   use_or_probability = use_or_probability,
                   init_cond          = init_cond,
                   save               = true,
                   provided_params    = provided)
end

if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
    id > m && error("SLURM_ARRAY_TASK_ID $id exceeds number of jobs ($m)")
    job = jobs[id]
    println("Running SLURM job $id: $job")
    run_job(job)
    println("Job $id complete.")
else
    for (i, job) in enumerate(jobs)
        println("\n=== Job $i/$m: $job ===")
        try
            run_job(job)
            println("Job $i complete.")
        catch e
            println("Job $i failed: $e")
        end
    end
    println("\nAll jobs complete.")
end
