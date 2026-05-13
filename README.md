# squeezing-criticality

Code companion to [*Squeezing codes: robust fluctuation-stabilized memories*](https://arxiv.org/abs/2509.20730) by myself and Sunghan Ro. This repository contains everything needed to reproduce the numerical results in that paper: a Julia Monte Carlo simulator for the squeezing rules R, F, M (plus Toom's rule and zero-temperature Glauber as baselines), a Python finite-size-scaling pipeline with bootstrap uncertainty quantification, and the raw data and figures used in Sec. VI of the paper.

The simulator runs on a square torus with i.i.d. bit-flip noise at rate `p`. It sweeps `p` (and optionally `L`), and records moments and correlators of the magnetization `m` and an anisotropy observable `D`. The Python plotter performs interactive and automated FSS collapses to extract `(p_c, ν, β, γ, z)` with statistical and systematic error budgets.

## Citation

If you use this code or data, please cite the paper. A machine-readable `CITATION.cff` is included; the BibTeX entry is:

```bibtex
@article{lake2025squeezing,
    title  = {Squeezing codes: robust fluctuation-stabilized memories},
    author = {Lake, Ethan and Ro, Sunghan},
    year   = {2025},
    eprint = {2509.20730},
    archivePrefix = {arXiv},
    primaryClass  = {cond-mat.stat-mech},
    url    = {https://arxiv.org/abs/2509.20730},
}
```

## Quickstart

```bash
# Julia deps
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Reproduce the joint R-rule FSS collapse from the paper
python3 plotter.py --rule R --plot binds --joint-fit --just-bm \
  --files data/R_stats_L48_*_r2rerun5x.jld2 \
          data/R_stats_L64_*_r2rerun5x.jld2 \
          data/R_stats_L96_*_r2rerun5x.jld2 \
          data/R_stats_L128_*_r2rerun5x.jld2 \
          data/R_stats_L192_*_r2rerun5x.jld2
```

The exact commands used for every figure in the paper live in
[`reproducing_paper_plots.md`](reproducing_paper_plots.md).

## Running simulations

The driver runs a single `(rule, L)` configuration in a chosen mode:

```bash
# Default p-sweep at fixed L (stats mode, the standard FSS workflow)
julia --project=. simulation_driver.jl --rule=R --L=24 --n_ps=12

# Relaxation-time sweep (trel mode): 100 trials per p, timeout at 100k sweeps
julia --project=. simulation_driver.jl --mode=trel --rule=R --L=24 --n_trials=100

# Initial-condition quench at pc (gives z from m(t) decay)
julia --project=. simulation_driver.jl --mode=quench --rule=R --L=48 --n_samples=50

# Erosion-time measurement 
julia --project=. simulation_driver.jl --mode=erosion_stats --rule=R --L=64 \
    --η=1.0 --domain_size=0.1

# Coarsening dynamics (mean cluster area vs time at fixed p)
julia --project=. simulation_driver.jl --mode=coarsening --rule=R --L=48 --p=0.04
```

One MC sweep is `L²` asynchronous site updates, and all times (`t_auto`,
`t_rel`, …) are reported in units of sweeps.

**Two distinct timescales.** This repo carefully distinguishes:

- `t_auto` — the *magnetization autocorrelation time*: the exponential
  decay time of the connected autocorrelation function of `|m|`,
  measured in equilibrium (stats mode). Written as
  `t_auto_<obs>` and exposed in the plotter as `--plot t_autos`.
- `t_rel` — the *first-passage / relaxation time*: starting from a clean
  initial state aligned against the noise bias, the time for the
  magnetization to first cross zero (trel mode). Written to disk as
  `t_rel`, `t_rel_err`, … and exposed in the plotter as
  `--plot t_rels`.

Both scale as `τ ~ L^z` near criticality and share the same
`τ → L^z / τ` collapse, but they are physically distinct measurements
on different runs; the names are kept separate everywhere. Output is written as `data/{rule}_{mode}{specifier}.jld2`,
where the specifier is built only from parameters that were explicitly set
on the command line; existing files are never overwritten (`_1`, `_2`, …
are appended on collision).

### Modes

- **`stats`** — long Monte Carlo runs measuring `⟨|m|⟩`, `χ`, the normalized Binder cumulant of `|m|`, the magnetization autocorrelation time, and (optionally) spatial / temporal correlators for every observable. p-sweep at fixed L (default) or L-sweep at fixed p (`--vary_L=true`).
- **`trel`** — first-passage / relaxation time. Each trial starts in the `"clean"` state (aligned against the noise bias `η`) and is evolved until its magnetization crosses zero. Reports the trial-average crossing time `τ_rel(p)`, along with median, SEM, and a timeout count.
- **`quench`** — initial-condition quench. Each trajectory starts in the `"clean"` state and is evolved at fixed `p` (default `pc`) for `T` MC sweeps. A scaling collapse of the time-dependent Binder cumulant and a fit of the trajectory-averaged curves `⟨|m|⟩(t) ∝ t^{-β/(νz)}` are used by the Python plotter to extract `z`.
- **`erosion_stats`** — time-to-absorption of a centered, bias-aligned minority disk on an otherwise clean background. Used to characterize the erosion speeds of each automaton. 
- **`coarsening`** — mean cluster area as a function of time, starting from a disordered initial condition at fixed `p`; tracks coarsening dynamics.

### Batch runs

```bash
# Sequential (edit parameter vectors at the top of run_batch.jl)
julia --project=. run_batch.jl

# SLURM job array
sbatch run_batch_slurm.sh
```

## Plotting

The Python plotter auto-discovers files by rule and mode, but you can
also feed it an explicit `--files` list. Dependencies: `numpy`, `h5py`,
`scipy`, `matplotlib`.

### Interactive scaling collapse

```bash
# Auto-discover all data/R_stats_*.jld2 and show the Binder collapse for m
python3 plotter.py --rule R --plot binds

# Anisotropy observable D instead of magnetization
python3 plotter.py --rule R --observable D --plot binds

# Override pc for comparison
python3 plotter.py --rule M --plot mags --pc 0.00323

# Unscaled curves (sanity check before attempting collapse)
python3 plotter.py --rule F --plot t_autos --raw

# trel collapse: 1/t_rel ~ L^(-z) · g((p − pc)·L^(1/ν))
python3 plotter.py --mode=trel --rule=R --plot=t_rels

# Restrict to a subset of system sizes
python3 plotter.py --rule R --plot binds --Ls 16 24 32

# Explicit file list
python3 plotter.py --plot binds --files data/R_stats_L16.jld2 data/R_stats_L24.jld2
```

`--mode` selects `stats` (default), `trel`, `quench`, `erosion_stats`, or
`coarsening`; for `stats` and `quench`, `--observable` selects between
`m` (magnetization) and `D` (x/y anisotropy of `m` nearest-neighbor two-point correlator); `--plot` selects which
moment to show: `mags = ⟨|x|⟩`, `chis = χ`, `binds` = Binder,
`t_autos = 1/t_auto` (stats mode only), or `t_rels = 1/t_rel` (trel mode
only). The `t_autos` and `t_rels` paths share collapse math but plot
distinct physical quantities. Quench mode has a
single fixed figure type and does not use `--plot`.

Interactive keypress tuning of exponents (once the figure window is focused):
- `left` / `right` — decrement / increment `ν`
- `up` / `down` — increment / decrement `γ`
- `,` / `.` — increment / decrement `β`
- `l` / `:` — decrement / increment `z`

The starting `(p_c, ν, β, γ, z)` seeds per rule live in `exponents.py`.
They have been updated to the joint-fit values reported in the paper
(R: `β = 0.165(5)`, `ν = 0.952(11)`; F: `β = 0.1826(20)`, `ν = 0.972(16)`;
M: `β = 0.227(5)`, `ν = 0.99(4)`); the per-rule (β, ν) used to convert
the post-quench `m(t)` slope into a `z` estimate are encoded directly in
`plotter.py` as `_QUENCH_RULE_EXPONENTS`. Toom and Glauber baselines use
the exact 2D Ising values `β = 1/8`, `ν = 1`.

### Automated scaling collapse (`--fit`, `--joint-fit`)

The plotter auto-minimises a Houdayer–Hartmann-style
reduced-χ² collapse cost and reports bootstrap 1σ uncertainties plus a
leave-one-L-out systematic estimate (see the paper for details).

```bash
# Single-observable fit: best (pc, ν) from Binder, with bootstrap σ
python3 plotter.py --rule R --plot binds --fit

# Joint fit across {binds, mags, chis, t_autos} sharing (pc, ν)
python3 plotter.py --rule R --plot binds --joint-fit

# "just B and m" joint fit (drops χ and t_auto; safer when the latter are
# noisier)
python3 plotter.py --rule R --plot binds --joint-fit --just-bm

# Fit + save a corner-style 2D-uncertainty PNG
python3 plotter.py --rule R --plot binds --joint-fit --contours

# Restricted-range FSS check: refit on {L ≥ L_min} for each L_min and plot
# the drift of each fitted exponent
python3 plotter.py --rule R --plot binds --joint-fit --Lmin-sweep

# Print exponents and exit without opening the interactive figure
python3 plotter.py --rule R --plot binds --fit --fit-only
```

Uncertainty pipeline in brief: `collapse_fit.collapse_cost` computes a
leave-one-curve-out χ² on the scaled data, `fit_collapse` / `fit_joint`
minimise it with SciPy Nelder–Mead, `bootstrap_fit` draws
`--n-bootstrap` synthetic datasets from Normal(mean, σ) at every
(L, p), σ being the block-jackknife error already stored in the JLD2
file. It re-fits each, and reports the 16 / 50 / 84-th percentiles per
parameter. `jackknife_L` does a leave-one-L-out refit; its spread is
combined in quadrature with the bootstrap σ into σ_total. `Lmin_sweep`
refits on `{L ≥ L_min}` for each `L_min` in turn and renders the
fractional drift of each fitted exponent. A stable trajectory means that the 
asymptotic regime has been reached; monotone drift means that confluent corrections still
dominant.

### Quench-relaxation mode for the dynamical exponent z

Alternative initial-condition-based estimator of `z`: start from the
uniform-magnetization state, evolve at `p = p_c` for `T` MC sweeps, average
many trajectories, and fit the power-law decay `⟨|x|⟩(t) ∝ t^{-β/(νz)}`.

```bash
# Sweep L, rule R at pc (default T = max(500, 20·L), 50 trajectories per L)
for L in 16 24 32 48; do
  julia --project=. simulation_driver.jl --mode=quench --rule=R --L=$L --n_samples=50
done

# Log-log m(t) decay with power-law fit + per-L z readout
python3 plotter.py --mode=quench --rule=R

# Instantaneous decay exponent θ(t) = log_10(⟨m(t/10)⟩ / ⟨m(t)⟩)
python3 plotter.py --mode=quench --rule=R --plot-theta

# Multi-rule comparison: load files from several rules at once. The
# plotter auto-detects heterogeneous rules and switches to ca_plotter-style
# per-rule colors / labels.
python3 plotter.py --mode=quench --files \
    data/sqztest_quench_rsqz_h_300_0.038425.jld2 \
    data/ca_fsqz_h_quench_L300_p0.01165_eta0.0_alpha0.0.jld2 \
    data/sqztest_quench_msqz_h_300_0.0032875.jld2 \
    data/sqztest_quench_toom_300_0.13395.jld2 \
    data/sqztest_quench_zeroT_glauber_300_0.141294.jld2
```

`z = -β / (ν · slope)` is reported per L with σ_z from propagating the (β, ν) uncertainties.

## Data layout

Simulation outputs live in [`data/`](data/) as JLD2 (HDF5) files named
`{rule}_{mode}_L{L}_{specifier}.jld2`. The specifier encodes every
parameter set explicitly on the CLI in a fixed order; defaults are
omitted to keep filenames short. The Python plotter loads any of these
via `--files` (or auto-discovers them via `--rule` + `--mode`). The full
per-mode schema is documented under **Output schema** below.

## Full parameter reference

All arguments use `--key=value` syntax.

**Global**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | String | `"stats"` | Simulation mode: `stats`, `trel`, `quench`, `erosion_stats`, or `coarsening` |
| `rule` | String | `"R"` | Rule: `R`, `R3`, `M`, `F`, `Toom`, or `Ising`. `R3` is `R` with `s(i,j)` added to both the OR and the AND. `Ising` is 2D zero-T Glauber on the NN Ising model, with the standard bit-flip noise `p` applied on top. |
| `L` | Int | `24` | Side length of the square torus |
| `save` | Bool | `true` | Save results to JLD2 file in `data/` |
| `out_adj` | String | `""` | Optional string appended to the filename before `.jld2` |
| `η` | Float | `0.0` | Noise bias: flipped spins drawn from Bernoulli((1+η)/2). Accepts `--η=...` or `--eta=...` |
| `use_or_probability` | Float | `0.5` | Probability a site update applies the OR rule (vs. AND) |

**Sweep (shared between `stats` and `trel`)**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pmin` | Float | rule default | Lower end of noise sweep (default from `default_p_window(rule)`) |
| `pmax` | Float | rule default | Upper end of noise sweep |
| `n_ps` | Int | `12` | Number of linearly spaced `p` values (when `vary_L=false`) |
| `vary_L` | Bool | `false` | If `true`, sweep `L` at fixed `p` instead of sweeping `p` at fixed `L` |
| `Lmin` | Int | `12` | Min `L` value (when `vary_L=true`; log-spaced) |
| `Lmax` | Int | `96` | Max `L` value (when `vary_L=true`) |
| `n_Ls` | Int | `7` | Number of log-spaced `L` values (when `vary_L=true`) |
| `p` | Float | rule `pc` | Fixed noise (when `vary_L=true`); defaults to the rule's critical point |
| `init_cond` | String | stats: `"rand"` / trel: `"clean"` | Initial state: `rand`, `balanced_rand`, or `clean` (`clean` = aligned against bias `η`) |

**`stats` mode (long-run moment measurement)**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | Int | `1` | Independent runs averaged over per (L, p) point |
| `thermalizing_steps` | Int | `5000` | MC sweeps discarded before measurement |
| `data_steps` | Int | `500000` | MC sweeps over which measurements accumulate |
| `data_taking_ratio` | Int | `20` | Take a sample every this many MC sweeps |
| `save_corrs` | Bool | `false` | Record spatial (`corr_x`, `corr_y`) and temporal (`corr_t`) correlators. Off by default because it roughly doubles the per-sample wall time; pass `--save_corrs=true` to opt in. |

**`trel` mode (relaxation-time / first-passage)**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_trials` | Int | `100` | Independent first-passage trials per (L, p) point (threaded) |
| `max_time` | Int | `1000000` | Max MC sweeps per trial before timing out |
| `M_threshold` | Float | `0.0` | Stop when `|m| ≤ M_threshold` instead of strict zero-crossing (0.0 = first sign flip) |

**`quench` mode**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p` | Float | rule `pc` | Fixed noise during the quench |
| `T` | Int | `max(500, 20·L)` | Total MC sweeps per trajectory (pass `--T=3000` etc. to extend) |
| `n_samples` | Int | `50` | Independent trajectories (shared flag with stats mode) |
| `data_taking_ratio` | Int | `1` | Record observables every this many sweeps |

**`erosion_stats` mode**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p` | Float | `0.0` | Noise during the erosion (typically 0 for the deterministic phase boundary) |
| `domain_size` | Float | `0.1` | Initial minority-disk radius as a fraction of `L` |
| `n_samples` | Int | `1` | Independent trials |
| `max_time` | Int | `50·L` | Max sweeps per trial before timing out |

**`coarsening` mode**

Shares the `quench`-mode flags (`p`, `T`, `n_samples`, `data_taking_ratio`).

### Observables

Each `stats` run measures moments of two per-configuration scalar
observables in parallel:

| Name | Definition | Range | What it measures |
|------|------------|-------|------------------|
| `m`  | `(1/N) Σᵢ σᵢ` with `σᵢ = 2 sᵢ − 1` | [-1, 1] | Magnetization (standard Z₂ order parameter) |
| `D`  | `(1/N) Σᵢ (σᵢ σ_{i+x̂} − σᵢ σ_{i+ŷ})` | [-2, 2] | x-vs-y anisotropy of nearest-neighbor spin correlations (vanishes for isotropic or fully aligned states) |

For every observable `obs` ∈ `{m, D}` the JLD2 output contains eight
scalar time-series summaries: `obs`, `chi_obs`, `bind_obs`,
`t_auto_obs`, plus their `_err` counterparts. The Python plotter selects
which observable to analyze via `--observable` (default `m`).

## Output schema

Each run writes one JLD2 file. Top-level keys (all h5py-compatible).

**Common to all modes:**

| Key | Shape | Description |
|-----|-------|-------------|
| `rule` | String | One of `R`, `R3`, `M`, `F`, `Toom`, `Ising` |
| `mode` | String | One of `stats`, `trel`, `quench`, `erosion_stats`, `coarsening` |
| `L` | Int | Primary system size |
| `Ls` | Vector{Int} | Per-sweep-point system size (length `n_ps` or `n_Ls`; sweeping modes only) |
| `ps` | Vector{Float64} | Per-sweep-point noise (sweeping modes only) |
| `η`, `use_or_probability` | Float64 | Dynamics parameters |
| `vary_L` | Bool | Whether this is an L-sweep |
| `init_cond` | String | Initial condition used |

**Stats-mode-only:**

| Key | Shape | Description |
|-----|-------|-------------|
| `observables` | Vector{String} | List of observable short names present in the file (currently `["m", "D"]`) |
| `m`, `D` | Vector{Float64} | ⟨\|observable\|⟩, averaged over `n_samples` |
| `chi_m`, `chi_D` | Vector{Float64} | Susceptibility `N · Var(\|observable\|)` |
| `bind_m`, `bind_D` | Vector{Float64} | Binder cumulant `(3 − ⟨x⁴⟩/⟨x²⟩²)/2` |
| `t_auto_m`, `t_auto_D` | Vector{Float64} | Magnetization autocorrelation time of `\|observable\|` (MC sweeps) — extracted from a fit to the connected autocorrelation function. Distinct from `t_rel` below. *(Legacy stats files written before 2026-05 stored this under `tau_exp_m`, `tau_exp_D`; the Python loader falls back to those names.)* |
| `m_err`, `chi_m_err`, `bind_m_err`, and `D` / `chi_D` / `bind_D` counterparts | Vector{Float64} | 1σ statistical error from a 32-block jackknife per sample, combined as `σ_tot = √(Σ σ_s²) / n_samples` across `n_samples`. `NaN` if any sample's series was too short to block. |
| `t_auto_m_err`, `t_auto_D_err` | Vector{Float64} | 1σ error on `t_auto`, leave-one-block-out jackknife (8 blocks). `NaN` if fewer than two replicas converged. *(Legacy: `tau_exp_m_err`, `tau_exp_D_err`.)* |
| `corr_x`, `corr_y` | Matrix{Float64} | Spatial correlators of the raw bit state, shape `(n_ps, max_r)` in Python (`h5py` swaps axes vs. Julia) |
| `corr_t` | Matrix{Float64} | Temporal correlator, shape `(n_ps, max_tau)` in Python |
| `corr_x_err`, `corr_y_err`, `corr_t_err` | Matrix{Float64} | Across-sample SEM of the per-sample correlators (same shape as `corr_*`). All `NaN` when `n_samples = 1`. |
| `max_r`, `max_tau` | Int | Correlator extents (`max_r = L/2.5`, `max_tau = 2L`) |
| `thermalizing_steps`, `data_steps`, `data_taking_ratio`, `n_samples` | Int | Run configuration |
| `save_corrs` | Bool | Whether correlators were recorded |

**Trel-mode-only:**

| Key | Shape | Description |
|-----|-------|-------------|
| `t_rel` | Vector{Float64} | Mean first-passage time ⟨t_rel⟩ per sweep point, in MC sweeps — the *relaxation time*, distinct from the autocorrelation time `t_auto` in stats-mode files. *(Legacy trel files written before 2026-05 stored this under `trel`; the Python loader falls back.)* |
| `t_rel_err` | Vector{Float64} | SEM of `t_rel` across `n_trials` trials. *(Legacy: `trel_err`.)* |
| `t_rel_median` | Vector{Float64} | Median of the per-trial crossing times (robust against heavy tails / timeouts). *(Legacy: `trel_median`.)* |
| `t_rel_timeouts` | Vector{Int} | Number of trials (out of `n_trials`) that hit `max_time` per sweep point. Nonzero ⇒ `t_rel` is a lower bound; increase `max_time`. *(Legacy: `trel_timeouts`.)* |
| `t_rel_times` | Matrix{Int} | Raw per-trial crossing times, shape `(n_trials, n_ps)` in Python. *(Legacy: `trel_times`.)* |
| `n_trials`, `max_time` | Int | Run configuration |
| `M_threshold` | Float64 | Stopping threshold on \|m\| (0.0 = strict sign flip) |

**Quench-mode-only:**

| Key | Shape | Description |
|-----|-------|-------------|
| `p` | Float64 | Fixed noise during the quench (defaults to the rule's `pc`) |
| `T` | Int | Number of MC sweeps per trajectory |
| `n_samples` | Int | Number of independent trajectories averaged over |
| `ts` | Vector{Int} | Measurement times in MC sweeps |
| `m_t`, `D_t` | Vector{Float64} | Trajectory-averaged signed ⟨x⟩(t), one entry per measurement time |
| `abs_m_t`, `abs_D_t` | Vector{Float64} | Trajectory-averaged ⟨\|x\|⟩(t) |
| `m_t_err`, `D_t_err`, `abs_m_t_err`, `abs_D_t_err` | Vector{Float64} | SEM across trajectories (NaN when `n_samples=1`) |
| `bind_m_t`, `bind_D_t`, and their `_err` versions | Vector{Float64} | Time-dependent Binder cumulant `B(t) = (3 − ⟨x⁴⟩/⟨x²⟩²)/2` and SEM. `NaN` for pre-2026 files. |

**Erosion / coarsening modes** record the per-trial absorption times and
per-time mean cluster area respectively; see the module sources
[`src/erosion_stats.jl`](src/erosion_stats.jl) and
[`src/coarsening.jl`](src/coarsening.jl) for the exact key set.
