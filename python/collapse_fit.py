"""
Automated scaling-collapse fit + exponent uncertainties.

Given the per-L observable curves measured by `run_stats` (already packaged
into the `thermo` dict by `plotter.assemble_thermo_data`), this module

1. defines a Houdayer-Hartmann-style chi-square quality-of-collapse metric
   (`collapse_cost`),
2. minimises it with SciPy's Nelder-Mead (`fit_collapse` for one observable,
   `fit_joint` for a shared-(pc, nu) fit across {binds, mags, chis, t_autos}),
3. estimates uncertainties by parametric bootstrap of the fit, resampling
   each data point from Normal(mean, sigma) using the already-computed per-
   point errors (`bootstrap_fit`), and
4. provides a leave-one-L-out jackknife diagnostic (`jackknife_L`) plus a
   corner-style plot of the bootstrap marginals + 2D covariance ellipses
   (`plot_bootstrap_corner`).

All of this is consumer-side — it reads the JLD2 files that the Julia
simulation already writes and asks for no further MC data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

# Parameter names per observable (order matters: (pc, nu) always first for
# stats-mode moments). `t_autos` and `t_rels` use identical scaling forms
# (τ ~ L^z) and share the inverse-time scaling code path; the distinct names
# preserve the physical distinction between the magnetization autocorrelation
# time (stats files) and the first-passage / relaxation time (trel files).
# `binds_t` is the quench-mode Binder cumulant whose collapse is
# U(t, L) = f(t / L^z) — only `z` is fit.
_OBS_PARAMS = {
    "binds":   ("pc", "nu"),
    "mags":    ("pc", "nu", "beta"),
    "chis":    ("pc", "nu", "gamma"),
    "t_autos": ("pc", "nu", "z"),
    "t_rels":  ("pc", "nu", "z"),
    "binds_t": ("z",),
}

# Corresponding error-array key in `thermo` for each observable.
_ERR_KEY = {"binds": "binds_err", "mags": "mags_err",
            "chis": "chis_err",
            "t_autos": "t_autos_err", "t_rels": "t_rels_err",
            "binds_t": "binds_t_err"}

# LaTeX labels for fitted parameters, for axis labels on figures.
_PARAM_LABEL = {
    "pc":    r"$p_c$",
    "nu":    r"$\nu$",
    "beta":  r"$\beta$",
    "gamma": r"$\gamma$",
    "z":     r"$z$",
}

# Bounds enforced by returning +inf outside — Nelder-Mead is unconstrained.
_BOUNDS = {
    "pc":    (1e-6, 1.0),
    "nu":    (1e-3, 4.0),
    "beta":  (0.0,  4.0),
    "gamma": (0.0,  4.0),
    "z":     (0.0,  6.0),
}


# ---------------------------------------------------------------------------
# Scaling transforms


def _scale_y(plot: str, L: np.ndarray, y: np.ndarray,
             yerr: Optional[np.ndarray], params: Mapping[str, float]):
    """
    Apply the observable's multiplicative prefactor to `y` (and linearly
    propagate `yerr` through it). `L` is broadcast to `y.shape`.

    binds            : y_scaled = y
    mags             : y_scaled = y * L^( beta/nu)
    chis             : y_scaled = y * L^(-gamma/nu)
    t_autos / t_rels : y_scaled = (1/y) * L^z ; err = (L^z / y^2) * yerr
    binds_t          : y_scaled = y                            (quench)
    """
    if plot in ("binds", "binds_t"):
        return y, yerr
    nu = params["nu"]
    if plot == "mags":
        sf = L ** (params["beta"] / nu)
        return y * sf, None if yerr is None else yerr * np.abs(sf)
    if plot == "chis":
        sf = L ** (-params["gamma"] / nu)
        return y * sf, None if yerr is None else yerr * np.abs(sf)
    if plot in ("t_autos", "t_rels"):
        sf = L ** params["z"]
        scaled = sf / y
        scaled_err = None if yerr is None else sf * yerr / (y ** 2)
        return scaled, scaled_err
    raise ValueError(f"unknown plot kind: {plot}")


def _scale_x(plot: str, L: np.ndarray, p: np.ndarray,
             params: Mapping[str, float]):
    """Scaling variable on the x-axis.

    Stats moments  : X = ((p - pc) / pc) * L^(1/nu).
    Quench binds_t : X = t / L^z   (here `p` is the time array).
    """
    if plot == "binds_t":
        return p / (L ** params["z"])
    return (p - params["pc"]) / params["pc"] * L ** (1.0 / params["nu"])


# ---------------------------------------------------------------------------
# Collapse cost (Houdayer-Hartmann with chi-square weighting)


def _out_of_bounds(params: Mapping[str, float]) -> bool:
    for k, v in params.items():
        lo, hi = _BOUNDS[k]
        if not (lo < v < hi):
            return True
    return False


def _linear_interp_with_err_vec(x_query: np.ndarray,
                                xs_sorted: np.ndarray,
                                ys_sorted: np.ndarray,
                                es_sorted: np.ndarray):
    """
    Vectorized linear interpolation of (y, yerr) over pre-sorted reference
    points. `xs_sorted` must be ascending (caller's responsibility). Query
    points outside the reference range come out as NaN for both arrays.

    Batching the whole query vector at once is the hot-path fix that used to
    dominate profiler output — the original scalar version re-sorted on every
    call, which is ~10⁶ argsorts per joint-fit bootstrap iteration.
    """
    n_ref = len(xs_sorted)
    if n_ref < 2:
        return (np.full_like(x_query, np.nan, dtype=float),
                np.full_like(x_query, np.nan, dtype=float))
    in_range = (x_query >= xs_sorted[0]) & (x_query <= xs_sorted[-1])
    j = np.clip(np.searchsorted(xs_sorted, x_query, side="right") - 1,
                0, n_ref - 2)
    x0 = xs_sorted[j];     x1 = xs_sorted[j + 1]
    y0 = ys_sorted[j];     y1 = ys_sorted[j + 1]
    e0 = es_sorted[j];     e1 = es_sorted[j + 1]
    dx = x1 - x0
    # Guard against dx == 0 (duplicate reference x) by placing the query at
    # the lower endpoint.
    t  = np.where(dx > 0, (x_query - x0) / np.where(dx > 0, dx, 1.0), 0.0)
    y  = (1 - t) * y0 + t * y1
    e2 = (1 - t) ** 2 * e0 ** 2 + t ** 2 * e1 ** 2
    e  = np.sqrt(np.maximum(e2, 0.0))
    y = np.where(in_range, y, np.nan)
    e = np.where(in_range, e, np.nan)
    return y, e


def collapse_cost(theta: Sequence[float],
                  thermo: Mapping,
                  plot: str,
                  param_names: Sequence[str],
                  y_override: Optional[np.ndarray] = None) -> float:
    """
    Reduced chi-square of the collapse for a single observable.

    Parameters
    ----------
    theta        : values for each parameter named in `param_names`.
    thermo       : output of `plotter.assemble_thermo_data`.
    plot         : one of 'binds', 'mags', 'chis', 't_autos', 't_rels'.
    param_names  : parameter order in `theta`. Must match `_OBS_PARAMS[plot]`
                   for single-observable fits; joint fits pass a superset.
    y_override   : optional (n_L, n_p) array replacing `thermo[plot]`, used
                   by `bootstrap_fit` to substitute resampled observable
                   values while keeping shape and error structure intact.

    Returns +inf outside parameter bounds.
    """
    params = dict(zip(param_names, theta))
    if _out_of_bounds({k: params[k] for k in _OBS_PARAMS[plot]}):
        return np.inf

    Ls = np.asarray(thermo["Ls"],  dtype=float)
    xs = np.asarray(thermo["xs"],  dtype=float)          # (n_L, n_p)
    ys = np.asarray(thermo[plot] if y_override is None else y_override,
                    dtype=float)                          # (n_L, n_p)
    es = np.asarray(thermo[_ERR_KEY[plot]], dtype=float) # (n_L, n_p)

    n_L = len(Ls)
    if n_L < 2:
        return np.inf

    # precompute scaled (X, Y, sigma) per curve
    X = [None] * n_L
    Y = [None] * n_L
    S = [None] * n_L
    for l in range(n_L):
        L = Ls[l]
        x_raw = xs[l]; y_raw = ys[l]; e_raw = es[l]
        finite = np.isfinite(y_raw) & np.isfinite(e_raw) & (e_raw > 0)
        if finite.sum() < 2:
            return np.inf
        Xl = _scale_x(plot, L, x_raw[finite], params)
        Yl, El = _scale_y(plot, L, y_raw[finite], e_raw[finite], params)
        if Yl is None or El is None:
            return np.inf
        if not np.all(np.isfinite(Yl)) or not np.all(np.isfinite(El)):
            return np.inf
        X[l] = Xl; Y[l] = Yl; S[l] = El

    # Pre-build and pre-sort each "others" union once per cost evaluation.
    # (Before this change, the per-query `_linear_interp_with_err` re-sorted
    # the reference array on every call — dominating the profile at ~60%.)
    others_sorted = [None] * n_L
    for l in range(n_L):
        Xo = np.concatenate([X[m] for m in range(n_L) if m != l])
        Yo = np.concatenate([Y[m] for m in range(n_L) if m != l])
        So = np.concatenate([S[m] for m in range(n_L) if m != l])
        order = np.argsort(Xo)
        others_sorted[l] = (Xo[order], Yo[order], So[order])

    chi2 = 0.0
    n_eff = 0
    for l in range(n_L):
        # "other curves" as the union of points from every L' != l, linearly
        # interpolated. Using their union rather than a per-curve minimum is
        # a standard Houdayer-Hartmann choice and reduces the influence of
        # any single L.
        Xo, Yo, So = others_sorted[l]
        y_hat, s_hat = _linear_interp_with_err_vec(X[l], Xo, Yo, So)
        ok = np.isfinite(y_hat) & np.isfinite(s_hat)
        if not np.any(ok):
            continue
        yi = Y[l][ok];        si = S[l][ok]
        yh = y_hat[ok];       sh = s_hat[ok]
        denom = si * si + sh * sh
        good = denom > 0
        if not np.any(good):
            continue
        chi2  += float(np.sum((yi[good] - yh[good]) ** 2 / denom[good]))
        n_eff += int(good.sum())

    if n_eff == 0:
        return np.inf
    return chi2 / n_eff


# ---------------------------------------------------------------------------
# Fits


@dataclass
class FitResult:
    params: dict                 # {name: best value}
    cost:   float                # minimised cost
    converged: bool              # from scipy.optimize.OptimizeResult.success
    param_order: list = field(default_factory=list)  # stable ordering

    def __str__(self):
        pieces = [f"{k}={self.params[k]:.5g}" for k in self.param_order]
        tag = "" if self.converged else "  [!not converged!]"
        return f"fit(cost={self.cost:.4f}{tag}, " + ", ".join(pieces) + ")"


def fit_collapse(thermo: Mapping, plot: str, *,
                 x0: Optional[Mapping[str, float]] = None,
                 fixed: Optional[Mapping[str, float]] = None,
                 maxiter: int = 2000) -> FitResult:
    """
    Single-observable fit. `x0` may supply seed values for any subset of
    `_OBS_PARAMS[plot]`; missing entries fall back to reasonable defaults.
    `fixed` holds any subset of `_OBS_PARAMS[plot]` at the supplied value
    (excluded from the Nelder-Mead search); `fixed` values take precedence
    over `x0`.
    """
    from scipy.optimize import minimize
    param_names = list(_OBS_PARAMS[plot])
    defaults = {"pc": 0.04, "nu": 1.0, "beta": 0.2, "gamma": 1.6, "z": 2.0}
    fixed = dict(fixed or {})
    free_names = [n for n in param_names if n not in fixed]
    if not free_names:
        # Degenerate case: everything fixed. Just evaluate the cost once.
        theta_full = [fixed[n] for n in param_names]
        cost = collapse_cost(theta_full, thermo, plot, param_names)
        return FitResult(params=dict(zip(param_names, theta_full)),
                         cost=float(cost),
                         converged=True,
                         param_order=param_names)

    seed = [float((x0 or {}).get(n, defaults[n])) for n in free_names]

    def fn(free_theta):
        merged = dict(fixed)
        merged.update(dict(zip(free_names, free_theta)))
        theta_full = [merged[n] for n in param_names]
        return collapse_cost(theta_full, thermo, plot, param_names)

    res = minimize(fn, seed, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-6,
                            "maxiter": maxiter, "adaptive": True})
    merged = dict(fixed)
    merged.update(dict(zip(free_names, res.x)))
    return FitResult(params={n: merged[n] for n in param_names},
                     cost=float(res.fun),
                     converged=bool(res.success),
                     param_order=param_names)


@dataclass
class JointFitResult:
    params: dict                     # flat {pc, nu, beta, gamma, z}
    per_plot_cost: dict              # {plot: cost after optimisation}
    total_cost: float
    converged: bool
    param_order: list = field(default_factory=list)

    def __str__(self):
        pieces = [f"{k}={self.params[k]:.5g}" for k in self.param_order]
        per = ", ".join(f"{p}:{c:.3g}" for p, c in self.per_plot_cost.items())
        tag = "" if self.converged else "  [!not converged!]"
        return (f"joint-fit(total={self.total_cost:.4f}{tag}, " +
                ", ".join(pieces) + f") [{per}]")


def fit_joint(thermo: Mapping,
              plots: Sequence[str] = ("binds", "mags", "chis", "t_autos"),
              *,
              x0: Optional[Mapping[str, float]] = None,
              fixed: Optional[Mapping[str, float]] = None,
              maxiter: int = 3000) -> JointFitResult:
    """
    Joint fit across multiple observables sharing (pc, nu). Each observable
    other than 'binds' contributes one additional exponent.

    Observables whose data is entirely NaN (e.g. a stats file with no t_auto
    populated, or a trel file without t_autos) are
    silently dropped before fitting.

    `fixed` holds any subset of {pc, nu, beta, gamma, z} at the supplied
    value (excluded from the Nelder-Mead search). Typical use: fix pc and
    nu to literature / prior-fit values and fit only {beta, gamma, z}.
    """
    from scipy.optimize import minimize

    # Drop observables that are not usable in this dataset.
    kept = []
    for plot in plots:
        ys = np.asarray(thermo[plot], dtype=float)
        es = np.asarray(thermo[_ERR_KEY[plot]], dtype=float)
        if np.any(np.isfinite(ys) & np.isfinite(es) & (es > 0)):
            kept.append(plot)
    if "binds" not in kept and len(kept) == 0:
        raise ValueError("no usable observables for joint fit")

    # Build param layout: always pc, nu; then per-observable exponent.
    param_names = ["pc", "nu"]
    for plot in kept:
        for n in _OBS_PARAMS[plot]:
            if n not in param_names:
                param_names.append(n)

    defaults = {"pc": 0.04, "nu": 1.0, "beta": 0.2, "gamma": 1.6, "z": 2.0}
    fixed = dict(fixed or {})
    free_names = [n for n in param_names if n not in fixed]

    # Helper: build the full params dict from a free-parameter vector.
    def _merge(free_theta):
        out = dict(fixed)
        out.update(dict(zip(free_names, free_theta)))
        return out

    # Evaluate S_tot = Σ_plot S_plot.
    def _total_cost(params):
        total = 0.0
        for plot in kept:
            sub_names = list(_OBS_PARAMS[plot])
            sub_theta = [params[n] for n in sub_names]
            c = collapse_cost(sub_theta, thermo, plot, sub_names)
            total += c
            if not np.isfinite(total):
                return np.inf
        return total

    if not free_names:
        # All parameters fixed — just evaluate the cost once.
        params = dict(fixed)
        total = _total_cost(params)
        per_cost = {}
        for plot in kept:
            sub_names = list(_OBS_PARAMS[plot])
            per_cost[plot] = collapse_cost([params[n] for n in sub_names],
                                            thermo, plot, sub_names)
        return JointFitResult(params={n: params[n] for n in param_names},
                              per_plot_cost=per_cost,
                              total_cost=float(total),
                              converged=True,
                              param_order=param_names)

    seed = [float((x0 or {}).get(n, defaults[n])) for n in free_names]

    def fn(free_theta):
        return _total_cost(_merge(free_theta))

    res = minimize(fn, seed, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-6,
                            "maxiter": maxiter, "adaptive": True})
    best = _merge(res.x)
    per_cost = {}
    for plot in kept:
        sub_names = list(_OBS_PARAMS[plot])
        per_cost[plot] = collapse_cost([best[n] for n in sub_names],
                                        thermo, plot, sub_names)
    return JointFitResult(params={n: best[n] for n in param_names},
                          per_plot_cost=per_cost,
                          total_cost=float(res.fun),
                          converged=bool(res.success),
                          param_order=param_names)


# ---------------------------------------------------------------------------
# Bootstrap + jackknife


def _resample_thermo(thermo: Mapping, plots: Sequence[str],
                     rng: np.random.Generator) -> dict:
    """
    Return a copy of `thermo` whose per-observable mean arrays have been
    replaced by Normal(mean, sigma) draws at each (L, p). NaN errors leave
    the original value untouched. Other keys (Ls, xs, *_err) are shared.
    """
    out = dict(thermo)
    for plot in plots:
        y = np.asarray(thermo[plot], dtype=float)
        e = np.asarray(thermo[_ERR_KEY[plot]], dtype=float)
        mask = np.isfinite(e) & (e > 0)
        sampled = y.copy()
        sampled[mask] = y[mask] + rng.standard_normal(int(mask.sum())) * e[mask]
        out[plot] = sampled
    return out


@dataclass
class BootstrapResult:
    central: dict                    # mapping param name -> central estimate
    samples: np.ndarray              # (n_boot, n_params)
    param_order: list
    per_param: dict                  # {name: {"median","p16","p84","std"}}
    n_fail: int                      # how many fits did not converge
    joint: bool

    def format_summary(self) -> str:
        rows = []
        for n in self.param_order:
            stats = self.per_param[n]
            mu = stats["median"]
            lo = mu - stats["p16"]
            hi = stats["p84"] - mu
            cen = self.central.get(n, np.nan)
            rows.append(f"  {n:>5s} = {cen:.5g}  "
                        f"[boot median {mu:.5g}, +{hi:.2g} / -{lo:.2g} (1σ)]")
        head = f"Bootstrap over {self.samples.shape[0]} resamples "
        head += f"({self.n_fail} fits failed)"
        return head + "\n" + "\n".join(rows)


def bootstrap_fit(thermo: Mapping,
                  plot_or_plots,
                  *,
                  n_boot: int = 50,
                  joint: bool = False,
                  x0: Optional[Mapping[str, float]] = None,
                  fixed: Optional[Mapping[str, float]] = None,
                  seed: int = 0,
                  progress: bool = False) -> BootstrapResult:
    """
    Parametric bootstrap using the per-point errors in `thermo[_ERR_KEY[...]]`.

    - Single-observable:  bootstrap_fit(thermo, "binds")
    - Joint:              bootstrap_fit(thermo, ("binds", "mags"), joint=True)

    `fixed` (same semantics as fit_collapse / fit_joint) holds a subset of
    parameters at the supplied values in both the central fit and every
    bootstrap replica; fixed parameters show up in the result with zero
    scatter across replicas.
    """
    rng = np.random.default_rng(seed)

    if joint:
        plots = list(plot_or_plots) if not isinstance(plot_or_plots, str) \
                else list(_OBS_PARAMS.keys())
        central_fit = fit_joint(thermo, plots=plots, x0=x0, fixed=fixed)
        central_params = dict(central_fit.params)
        param_order = list(central_fit.param_order)
        warmstart = central_params
        def _fit(resampled):
            return fit_joint(resampled, plots=plots, x0=warmstart, fixed=fixed)
    else:
        plot = plot_or_plots if isinstance(plot_or_plots, str) else plot_or_plots[0]
        central_fit = fit_collapse(thermo, plot, x0=x0, fixed=fixed)
        central_params = dict(central_fit.params)
        param_order = list(central_fit.param_order)
        plots = [plot]
        warmstart = central_params
        def _fit(resampled):
            return fit_collapse(resampled, plot, x0=warmstart, fixed=fixed)

    samples = np.full((n_boot, len(param_order)), np.nan)
    n_fail = 0
    import time as _time
    t0 = _time.time()
    # Print progress at 5% increments or every 30s, whichever is more often.
    # Shows wall-clock rate + ETA so the user can distinguish slow-but-
    # progressing from actually stuck.
    step = max(1, n_boot // 20)
    next_print = _time.time() + 30.0
    for k in range(n_boot):
        resampled = _resample_thermo(thermo, plots, rng)
        try:
            fk = _fit(resampled)
        except Exception:
            n_fail += 1
            continue
        if not fk.converged:
            n_fail += 1
        samples[k] = [fk.params[n] for n in param_order]
        if progress and ((k + 1) % step == 0 or _time.time() >= next_print
                         or k + 1 == n_boot):
            done = k + 1
            elapsed = _time.time() - t0
            rate = done / elapsed if elapsed > 0 else float("inf")
            eta = (n_boot - done) / rate if rate > 0 else float("inf")
            print(f"  bootstrap {done}/{n_boot}  "
                  f"({elapsed:.1f}s elapsed, {rate:.2f}/s, "
                  f"ETA {eta:.0f}s, {n_fail} fails so far)")
            next_print = _time.time() + 30.0

    per_param = {}
    for j, n in enumerate(param_order):
        col = samples[:, j]
        valid = col[np.isfinite(col)]
        if len(valid) == 0:
            per_param[n] = {"median": np.nan, "p16": np.nan,
                            "p84": np.nan, "std": np.nan}
        else:
            per_param[n] = {
                "median": float(np.median(valid)),
                "p16":    float(np.percentile(valid, 16)),
                "p84":    float(np.percentile(valid, 84)),
                "std":    float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            }
    return BootstrapResult(
        central=central_params,
        samples=samples,
        param_order=param_order,
        per_param=per_param,
        n_fail=n_fail,
        joint=joint,
    )


def Lmin_sweep(thermo: Mapping, plot_or_plots, *,
               joint: bool = False,
               x0: Optional[Mapping[str, float]] = None,
               fixed: Optional[Mapping[str, float]] = None,
               min_remaining: int = 2) -> dict:
    """
    Restricted-range FSS sweep: refit on `{L : L ≥ L_min}` for each
    successive `L_min` in the sorted list of system sizes, dropping the
    smallest `L` one at a time. Returns a dict mapping `L_min`
    (the *smallest* size kept) → {param: value}. The trajectory of the
    fitted parameters as `L_min` rises is the standard test for whether
    the asymptotic regime has been reached: if successive entries fall
    inside one statistical σ of each other the fit has plateaued; if
    they drift monotonically, finite-size corrections still dominate
    at every kept size.

    `min_remaining` (default 2) is the smallest number of L values left
    after dropping; entries beyond that bound are skipped.
    """
    Ls = np.asarray(thermo["Ls"])
    order = np.argsort(Ls)
    Ls_sorted = Ls[order]
    keys = ("Ls", "xs") + tuple(_OBS_PARAMS.keys()) + tuple(_ERR_KEY.values())

    out = {}
    for k_drop in range(0, len(Ls_sorted) - min_remaining + 1):
        kept_idx = np.sort(order[k_drop:])           # positions in original layout
        L_min = int(Ls_sorted[k_drop])
        sub = dict(thermo)
        for k in keys:
            if k in thermo and np.asarray(thermo[k]).ndim >= 1:
                arr = np.asarray(thermo[k])
                if arr.shape[0] == len(Ls):
                    sub[k] = arr[kept_idx]
        if joint:
            plots = (list(plot_or_plots) if not isinstance(plot_or_plots, str)
                     else list(_OBS_PARAMS.keys()))
            fit = fit_joint(sub, plots=plots, x0=x0, fixed=fixed)
        else:
            plot = plot_or_plots if isinstance(plot_or_plots, str) \
                   else plot_or_plots[0]
            fit = fit_collapse(sub, plot, x0=x0, fixed=fixed)
        out[L_min] = dict(fit.params)
    return out


def jackknife_L(thermo: Mapping, plot_or_plots, *,
                joint: bool = False,
                x0: Optional[Mapping[str, float]] = None,
                fixed: Optional[Mapping[str, float]] = None) -> dict:
    """
    Leave-one-L-out finite-size systematic diagnostic. For each L in
    `thermo["Ls"]`, refit with that row removed and return a dict
    mapping L (the *excluded* size) -> {param: value}. The spread of the
    fitted parameters across exclusions estimates the systematic shift
    that would result from confluent corrections beyond the leading
    scaling ansatz, which the parametric bootstrap on the per-point
    jackknife errors does not capture.

    `joint`, `x0`, and `fixed` are forwarded to the underlying fitter and
    must match the call that produced the central estimate (so the
    leave-one-L spreads are comparable to the bootstrap σ on equal terms).
    For joint fits, `plot_or_plots` may be a sequence of plot names; the
    same list is used for every refit.
    """
    Ls = np.asarray(thermo["Ls"])
    out = {}
    keys = ("Ls", "xs") + tuple(_OBS_PARAMS.keys()) + tuple(_ERR_KEY.values())
    for i, L in enumerate(Ls):
        sub = dict(thermo)
        for k in keys:
            if k in thermo and np.asarray(thermo[k]).ndim >= 1:
                arr = np.asarray(thermo[k])
                if arr.shape[0] == len(Ls):
                    sub[k] = np.delete(arr, i, axis=0)
        if joint:
            plots = (list(plot_or_plots) if not isinstance(plot_or_plots, str)
                     else list(_OBS_PARAMS.keys()))
            fit = fit_joint(sub, plots=plots, x0=x0, fixed=fixed)
        else:
            plot = plot_or_plots if isinstance(plot_or_plots, str) \
                   else plot_or_plots[0]
            fit = fit_collapse(sub, plot, x0=x0, fixed=fixed)
        out[int(L)] = dict(fit.params)
    return out


# ---------------------------------------------------------------------------
# Corner-style uncertainty plot


def _conf_ellipse(xs: np.ndarray, ys: np.ndarray, level: float):
    """
    Return (cx, cy, width, height, angle_deg) describing a 2D confidence
    ellipse from the empirical covariance of (xs, ys). `level` is the chi-
    square critical value for the desired probability — for the 2-DOF case,
    use `2.30` (68%) or `5.99` (95%). Width/height are full axes lengths.
    """
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]; ys = ys[mask]
    if len(xs) < 3:
        return None
    cx = np.mean(xs); cy = np.mean(ys)
    cov = np.cov(xs, ys, ddof=1)
    evals, evecs = np.linalg.eigh(cov)
    # eigh yields ascending eigenvalues; we want major axis first
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    if np.any(evals <= 0):
        return None
    width  = 2.0 * np.sqrt(level * evals[0])
    height = 2.0 * np.sqrt(level * evals[1])
    angle  = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    return cx, cy, width, height, angle


def plot_bootstrap_corner(boot: BootstrapResult, *,
                          save: Optional[str] = None,
                          show: bool = True,
                          title: str = ""):
    """
    Corner-style figure of the bootstrap distribution.

    Diagonal: 1D histograms per parameter with 16/50/84th-percentile lines.
    Off-diagonal: scatter of bootstrap samples with 68% and 95% empirical
    covariance ellipses overlaid.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    samples = boot.samples
    names = boot.param_order
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n),
                             squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                col = samples[:, i]
                col = col[np.isfinite(col)]
                if len(col) == 0:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                else:
                    ax.hist(col, bins=30, density=True, color="#4477AA",
                            alpha=0.75, edgecolor="black", linewidth=0.5)
                    for q, style in [(16, "--"), (50, "-"), (84, "--")]:
                        ax.axvline(np.percentile(col, q), color="k",
                                   linestyle=style, linewidth=0.8)
                ax.set_yticks([])
            elif j < i:
                xs = samples[:, j]; ys = samples[:, i]
                mask = np.isfinite(xs) & np.isfinite(ys)
                ax.scatter(xs[mask], ys[mask], s=3, alpha=0.35, color="#4477AA")
                for level, ls, lw in [(2.30, "-", 1.3), (5.99, "--", 0.9)]:
                    el = _conf_ellipse(xs, ys, level)
                    if el is not None:
                        cx, cy, w, h, ang = el
                        ax.add_patch(Ellipse((cx, cy), w, h, angle=ang,
                                             fill=False, edgecolor="k",
                                             linestyle=ls, linewidth=lw))
            else:
                ax.axis("off")
            # Corner-plot label convention: diagonal histograms carry the
            # parameter name as a *title* above the cell (works uniformly for
            # every row, including the top), bottom-row off-diagonals carry
            # the xlabel, and left-column non-(0,0) off-diagonals carry the
            # ylabel.
            if i == j:
                ax.set_title(_PARAM_LABEL.get(names[j], names[j]), pad=4)
            elif i == n - 1:
                ax.set_xlabel(_PARAM_LABEL.get(names[j], names[j]))
            elif j < i:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(_PARAM_LABEL.get(names[i], names[i]))
            elif j < i:
                ax.set_yticklabels([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=140, bbox_inches="tight")
        print(f"saved {save}")
    if show:
        import matplotlib.pyplot as _plt
        _plt.show()
    return fig
