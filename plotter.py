#!/usr/bin/env python3
"""
Interactive finite-size-scaling plotter for squeezing-criticality data.

Supports two simulation modes via --mode (default: stats):

  stats  Auto-discovers data/{rule}_stats_*.jld2. --observable picks the
         measured quantity to analyze:
           m  — magnetization  ⟨|m|⟩,  χ_m,  bind_m,  t_auto_m
           D  — anisotropy     ⟨|D|⟩,  χ_D,  bind_D,  t_auto_D
               (D = (1/N) Σᵢ σᵢσ_{i+x̂} − σᵢσ_{i+ŷ})
         --plot picks which moment: mags, chis, binds, t_autos. Here
         `t_auto` is the magnetization autocorrelation time of |obs|
         (the exponential decay time of its connected autocorrelation
         function, in MC sweeps) — NOT to be confused with `t_rel`,
         the first-passage / relaxation time produced by --mode=trel.

  trel   Auto-discovers data/{rule}_trel_*.jld2 (aligned-against-bias first-
         passage / relaxation times). Only --plot=t_rels is meaningful in
         this mode; the collapse code path is the same as for `t_autos` in
         stats mode (both scale as τ ~ L^z), only the loaded quantity and
         axis labels differ. --observable is ignored.

Examples:
    python3 plotter.py --rule R --plot binds                         # Binder of m
    python3 plotter.py --rule R --observable D --plot binds          # Binder of D
    python3 plotter.py --mode=trel --rule=R --plot=t_rels            # collapse t_rel(p, L)
    python3 plotter.py --mode=trel --rule=R --plot=t_rels --raw      # raw t_rel vs p
    python3 plotter.py --rule M --plot mags --pc 0.00323
    python3 plotter.py --rule R --plot binds --Ls 16 24 32
    python3 plotter.py --plot binds --files data/R_stats_L16.jld2 data/R_stats_L24.jld2
"""

import argparse
import glob
import os
import sys

# Helper modules live in ./python/ — add it to sys.path so the bare imports
# below (and the deferred `import collapse_fit as cf` inside fit functions)
# resolve without callers having to set PYTHONPATH.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "python"))

import h5py
import numpy as np

# Shim for matplotlib >= 3.9, which removed matplotlib.cbook._Stack that older
# interactive widgets (including ours) depend on. Must run before scaling_plotter
# (and therefore matplotlib.widgets) is imported.
try:
    import matplotlib.cbook
    if not hasattr(matplotlib.cbook, "_Stack"):
        class _Stack(list):
            def push(self, item):
                self.append(item)
                return item
            def pop(self):
                return super().pop() if self else None
            def current(self):
                return self[-1] if self else None
            def forward(self):
                pass
            def back(self):
                pass
        matplotlib.cbook._Stack = _Stack
except Exception:
    pass

import matplotlib.cm as _cm
from exponents import get_defaults
from scaling_plotter import scaling_plotter

# Per-rule colormap — matches the convention from ca_plotter.py in the
# MemoryNCA/Ethan repo so cross-paper figures stay visually consistent.
# R3 (self-included variant of R) uses Oranges to match ca_plotter's colour
# for the three-neighbour squeezing variant.
_RULE_CMAP = {"R":  _cm.Purples,
              "R3": _cm.Oranges,
              "M":  _cm.Blues,
              "F":  _cm.Reds}


# Solid per-rule colours and labels matching ca_plotter.py's `get_labels()`,
# used by plot_quench when the loaded files cover multiple distinct rules
# (e.g. comparing magnetization decays across R / F / M / Toom / Ising on the
# same axes — the use case in MemoryNCA/Ethan/ca_plotter.py -plot mt).
def _quench_rule_style(rule_str, threesqz=False):
    """Return (color, label) for a given rule name string. Falls back to a
    neutral dark grey when no match is found.
    """
    r = (rule_str or "").lower()
    if "toom" in r:
        return ("#6BF3B3", r"$\mathsf{Toom}$")
    if "glauber" in r or "ising" in r:
        return ("#D3D3D3", r"$\mathsf{Ising}$")
    if "rsqz" in r or r == "r" or "r3" in r:
        if threesqz or "r3" in r:
            return ("#FFA500", r"$\mathsf{R}_3$")
        return ("#9086F8", r"$\mathsf{R}_2$")
    if "msqz" in r or r == "m":
        return ("#7BC1FC", r"$\mathsf{M}$")
    if "fsqz" in r or r == "f":
        return ("#FF4B62", r"$\mathsf{F}$")
    return ("#4A4A4A", rf"$\mathsf{{{rule_str}}}$")


def _brighten(color, factor=1.2):
    """Brighten (factor>1) or darken (factor<1) a colour by scaling its HLS
    lightness. Matches `brighten()` in ca_plotter.py so the dashed fit line
    is drawn in the same darker shade of the curve colour.
    """
    import colorsys
    import matplotlib.colors as mcolors
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    return colorsys.hls_to_rgb(h, min(1.0, l * factor), s)


# Per-rule (β, σ_β, ν, σ_ν) used to convert the fitted log-log slope of m(t)
# into a dynamical exponent z = -β / (ν · slope) in multi-rule mode. β, ν
# values for R₂, F, M come from the joint-fit results reported in the paper
# (numbers in parentheses are 1σ on the last digit). Toom and Glauber/Ising
# use the exact 2D-Ising values β = 1/8, ν = 1 with no uncertainty.
_QUENCH_RULE_EXPONENTS = {
    "R":    (0.165,  0.005,  0.952, 0.011),
    "R3":   (0.17,   0.0,    0.95,  0.0  ),
    "F":    (0.1826, 0.0020, 0.972, 0.016),
    "M":    (0.227,  0.005,  0.99,  0.04 ),
    "Toom": (0.125,  0.0,    1.0,   0.0  ),
    "Ising":(0.125,  0.0,    1.0,   0.0  ),
}


def _fmt_value_uncert(value, sigma, sig_sigma=1):
    """Format a measurement as '1.89(6)' — value rounded to the same decimal
    as σ, which itself is rounded to `sig_sigma` significant figures; the
    parenthesised number is σ expressed in units of the last quoted digit.
    Falls back to a plain `%.3g` when σ is zero / non-finite.
    """
    import math
    if not (np.isfinite(value) and np.isfinite(sigma)) or sigma <= 0:
        return f"{value:.3g}"
    # decimal place at which σ's first significant digit sits
    exp_sigma = math.floor(math.log10(abs(sigma)))
    decimals = -(exp_sigma - (sig_sigma - 1))   # = -exp_sigma when sig_sigma=1
    rounded_sigma = round(sigma, decimals)
    if rounded_sigma == 0:
        return f"{value:.3g}"
    paren  = int(round(rounded_sigma * 10 ** decimals))
    val_fmt = f"{{:.{max(0, decimals)}f}}"
    return f"{val_fmt.format(round(value, decimals))}({paren})"


def _quench_rule_exponents(rule_str, threesqz=False):
    """Return (β, σ_β, ν, σ_ν) for a given rule name."""
    r = (rule_str or "").lower()
    if "toom"    in r: return _QUENCH_RULE_EXPONENTS["Toom"]
    if "glauber" in r or "ising" in r: return _QUENCH_RULE_EXPONENTS["Ising"]
    if "rsqz" in r or r == "r" or "r3" in r:
        return _QUENCH_RULE_EXPONENTS["R3"] if (threesqz or "r3" in r) else _QUENCH_RULE_EXPONENTS["R"]
    if "msqz" in r or r == "m": return _QUENCH_RULE_EXPONENTS["M"]
    if "fsqz" in r or r == "f": return _QUENCH_RULE_EXPONENTS["F"]
    return (np.nan, np.nan, np.nan, np.nan)


def _decode(x):
    """JLD2 strings come through h5py as bytes; decode to str."""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray) and x.dtype.kind in ("O", "S"):
        try:
            return x.item().decode("utf-8")
        except Exception:
            return str(x)
    return x


def _read_key(f, key):
    """Read a top-level key from a JLD2/HDF5 file, stripping JLD2 metadata wrappers."""
    if key not in f:
        return None
    val = f[key]
    if isinstance(val, h5py.Group):
        # JLD2 sometimes wraps typed objects in a group
        if "data" in val:
            return val["data"][()]
        return None
    return val[()]


def _read_err(f, key, like):
    """Read an error array if present, else an all-NaN array matching `like`."""
    val = _read_key(f, key)
    if val is None:
        return np.full_like(np.asarray(like, dtype=float), np.nan)
    return np.asarray(val, dtype=float)


def _nan_like(x):
    return np.full_like(np.asarray(x, dtype=float), np.nan)


def _require_key(f, key, path, expected_mode):
    """Read a required key; raise a clear error if it's missing. Helps
    diagnose mode/schema mismatches (e.g. loading a trel file as stats)."""
    val = _read_key(f, key)
    if val is None:
        file_mode = _decode(_read_key(f, "mode")) or "?"
        raise KeyError(
            f"{os.path.basename(path)}: missing required key '{key}' "
            f"(expected under mode='{expected_mode}', file has mode='{file_mode}'). "
            f"Try --mode={file_mode} to plot this file."
        )
    return val


def load_stats_file(path, observable="m"):
    """Load a stats JLD2 file and return a dict of the fields for `observable`.

    The observable-specific keys (m / D, chi_m / chi_D, ...) are mapped onto a
    uniform set of dict keys (m, chi, bind, t_auto, *_err) so that downstream
    code doesn't need to know which observable was selected. `t_auto` is the
    magnetization autocorrelation time (the exponential decay time of the
    connected autocorrelation function of |observable|, in MC sweeps); it is
    distinct from `t_rel`, the first-passage / relaxation time produced by
    `--mode=trel` and exposed under the dict key `t_rel` by `load_trel_file`.

    Legacy compatibility: pre-2026-05 stats files stored this quantity as
    `tau_exp_<obs>` on disk. If the new `t_auto_<obs>` key is missing, fall
    back to the legacy name so existing data still loads.
    """
    with h5py.File(path, "r") as f:
        m    = np.asarray(_require_key(f, observable,               path, "stats"))
        chi  = np.asarray(_require_key(f, f"chi_{observable}",      path, "stats"))
        bind = np.asarray(_require_key(f, f"bind_{observable}",     path, "stats"))
        # Read the autocorrelation time under its current name; fall back to
        # the legacy `tau_exp_<obs>` for older files.
        t_auto_raw = _read_key(f, f"t_auto_{observable}")
        if t_auto_raw is None:
            t_auto_raw = _require_key(f, f"tau_exp_{observable}", path, "stats")
        t_auto = np.asarray(t_auto_raw)
        t_auto_err_raw = _read_key(f, f"t_auto_{observable}_err")
        if t_auto_err_raw is None:
            t_auto_err_raw = _read_key(f, f"tau_exp_{observable}_err")
        t_auto_err = (np.asarray(t_auto_err_raw, dtype=float)
                      if t_auto_err_raw is not None
                      else np.full_like(t_auto, np.nan, dtype=float))
        ps = np.asarray(_read_key(f, "ps"))
        out = {
            "rule":         _decode(_read_key(f, "rule")),
            "L":            int(_read_key(f, "L")),
            "ps":           ps,
            "m":            m,
            "chi":          chi,
            "bind":         bind,
            "t_auto":       t_auto,
            "m_err":        _read_err(f, f"{observable}_err",         m),
            "chi_err":      _read_err(f, f"chi_{observable}_err",     chi),
            "bind_err":     _read_err(f, f"bind_{observable}_err",    bind),
            "t_auto_err":   t_auto_err,
            # stats mode has no `max_time` / first-passage semantics, so no
            # censoring concept — all zeros.
            "timeout_frac": np.zeros_like(ps, dtype=float),
            "n_trials":     0,
            "path":         path,
        }
    return out


def load_trel_file(path):
    """Load a trel JLD2 file. The first-passage time `t_rel` is exposed
    under the dict key `t_rel` (and its SEM as `t_rel_err`). It scales as
    `t_rel ~ L^z` near criticality with the same form as the autocorrelation
    time `t_auto` from stats mode — the scaling code path (`--plot t_rels`
    here vs. `--plot t_autos` for stats) handles both with shared math but
    distinct axis labels and dict keys.

    Fields not measured in trel mode (m, chi, bind) come back as all-NaN
    arrays of the right shape.

    Also populates `timeout_frac`: the fraction of trials that hit `max_time`
    per sweep point. Downstream consumers flag points with `timeout_frac >
    CENSORED_FRAC_THRESHOLD` because their `t_rel` is a lower bound, not a
    measurement.

    Legacy compatibility: pre-2026-05 trel files stored these quantities
    under the keys `trel`, `trel_err`, `trel_timeouts`. The loader falls
    back to those names when the new `t_rel_*` keys are absent so existing
    data still loads.
    """
    with h5py.File(path, "r") as f:
        ps = np.asarray(_read_key(f, "ps"))
        # New on-disk name `t_rel`, falling back to the legacy `trel`.
        t_rel_raw = _read_key(f, "t_rel")
        if t_rel_raw is None:
            t_rel_raw = _require_key(f, "trel", path, "trel")
        t_rel = np.asarray(t_rel_raw)
        t_rel_err_raw = _read_key(f, "t_rel_err")
        if t_rel_err_raw is None:
            t_rel_err_raw = _read_key(f, "trel_err")
        t_rel_err = (np.asarray(t_rel_err_raw, dtype=float)
                     if t_rel_err_raw is not None
                     else _nan_like(t_rel))
        timeouts_raw = _read_key(f, "t_rel_timeouts")
        if timeouts_raw is None:
            timeouts_raw = _read_key(f, "trel_timeouts")
        n_trials = _read_key(f, "n_trials")
        if timeouts_raw is not None and n_trials is not None and int(n_trials) > 0:
            timeout_frac = np.asarray(timeouts_raw, dtype=float) / float(n_trials)
        else:
            timeout_frac = _nan_like(ps)
        out = {
            "rule":          _decode(_read_key(f, "rule")),
            "L":             int(_read_key(f, "L")),
            "ps":            ps,
            "m":             _nan_like(ps),
            "chi":           _nan_like(ps),
            "bind":          _nan_like(ps),
            "t_rel":         t_rel,
            "m_err":         _nan_like(ps),
            "chi_err":       _nan_like(ps),
            "bind_err":      _nan_like(ps),
            "t_rel_err":     t_rel_err,
            "timeout_frac":  timeout_frac,
            "n_trials":      int(n_trials) if n_trials is not None else 0,
            "path":          path,
        }
    return out


# Points with more than this fraction of trials hitting `max_time` have a
# censored τ_rel (lower bound, not a measurement). Flag them visually and
# in stderr.
CENSORED_FRAC_THRESHOLD = 0.10


def _warn_censored(Ls, xs_all, timeout_frac, n_trials=None):
    """Print a stderr warning line for each (L, p) point whose timeout
    fraction exceeds `CENSORED_FRAC_THRESHOLD`. Called by the trel plotters
    just before drawing."""
    if timeout_frac is None:
        return
    to_frac = np.asarray(timeout_frac)
    if to_frac.size == 0 or np.all(to_frac <= CENSORED_FRAC_THRESHOLD):
        return
    lines = []
    for l, L in enumerate(Ls):
        if l >= to_frac.shape[0]:
            break
        xs = np.asarray(xs_all[l])
        fs = np.asarray(to_frac[l])
        nt = (int(n_trials[l])
              if (n_trials is not None
                  and l < len(n_trials)
                  and n_trials[l] is not None)
              else None)
        for k in range(min(xs.size, fs.size)):
            if fs[k] > CENSORED_FRAC_THRESHOLD:
                if nt:
                    ttl = int(round(fs[k] * nt))
                    lines.append(
                        f"  L={int(L)} p={xs[k]:.5g}: "
                        f"{ttl}/{nt} trials timed out "
                        f"({100*fs[k]:.0f}% — τ_rel is a lower bound)")
                else:
                    lines.append(
                        f"  L={int(L)} p={xs[k]:.5g}: "
                        f"{100*fs[k]:.0f}% of trials timed out "
                        "(τ_rel is a lower bound)")
    if lines:
        print(f"warning: {len(lines)} point(s) have > "
              f"{int(100*CENSORED_FRAC_THRESHOLD)}% trials censored at max_time:",
              file=sys.stderr)
        for ln in lines:
            print(ln, file=sys.stderr)
        print("  (these points are drawn as open red ▲ markers.)",
              file=sys.stderr)


def discover_files(rule, mode="stats", pattern_dir="data"):
    """Glob data/{rule}_{mode}_*.jld2."""
    pat = os.path.join(pattern_dir, f"{rule}_{mode}_*.jld2")
    return sorted(glob.glob(pat))


def load_quench_file(path):
    """Load a quench-mode JLD2 file. Returns a dict with:
      rule, L, p, T, ts, n_samples,
      per-observable trajectory-mean time series ({obs}_t, abs_{obs}_t) +
      SEMs ({obs}_t_err, abs_{obs}_t_err), and per-observable time-dependent
      Binder cumulants (bind_{obs}_t, bind_{obs}_t_err) when present (older
      pre-Binder files won't have these — they come back as NaN).

    Also supports the legacy memoryNCA schema (`mt`/`mabst`, no `ts`, no
    error keys, no D observable) by detecting the absence of `m_t` and
    falling back: ts is fabricated as 1..len(mt), errors come back NaN, and
    D-observable keys are filled with NaN.
    """
    with h5py.File(path, "r") as f:
        legacy = (_read_key(f, "m_t") is None) and (_read_key(f, "mt") is not None)

        n_samp = _read_key(f, "n_samples")
        if n_samp is None:
            n_samp = _read_key(f, "samps")  # legacy memoryNCA name

        # threesqz flag (legacy memoryNCA sqztest files) distinguishes the
        # R_2 (two-neighbour) and R_3 (three-neighbour) variants of the
        # squeezing rule, which use different ca_plotter rule colours.
        # Absent in the present repo's native quench files; default False.
        threesqz_raw = _read_key(f, "threesqz")
        threesqz = bool(threesqz_raw) if threesqz_raw is not None else False
        out = {
            "rule":      _decode(_read_key(f, "rule")),
            "L":         int(_read_key(f, "L")),
            "p":         float(_read_key(f, "p")),
            "T":         int(_read_key(f, "T")),
            "n_samples": int(n_samp) if n_samp is not None else 0,
            "threesqz":  threesqz,
            "path":      path,
            "mode_effective": "quench",
        }

        if legacy:
            mt   = np.asarray(_require_key(f, "mt", path, "quench"), dtype=float)
            mabs = _read_key(f, "mabst")
            mabs = np.asarray(mabs, dtype=float) if mabs is not None else np.abs(mt)
            ts = _read_key(f, "ts")
            out["ts"] = (np.asarray(ts) if ts is not None
                         else np.arange(1, len(mt) + 1, dtype=int))
            out["m_t"]         = mt
            out["abs_m_t"]     = mabs
            out["m_t_err"]     = np.full(mt.shape, np.nan)
            out["abs_m_t_err"] = np.full(mabs.shape, np.nan)
            for obs in ("D",):
                out[f"{obs}_t"]         = np.full(mt.shape, np.nan)
                out[f"abs_{obs}_t"]     = np.full(mt.shape, np.nan)
                out[f"{obs}_t_err"]     = np.full(mt.shape, np.nan)
                out[f"abs_{obs}_t_err"] = np.full(mt.shape, np.nan)
            for obs in ("m", "D"):
                out[f"bind_{obs}_t"]     = np.full(mt.shape, np.nan)
                out[f"bind_{obs}_t_err"] = np.full(mt.shape, np.nan)
            return out

        out["ts"] = np.asarray(_read_key(f, "ts"))
        for obs in ("m", "D"):
            out[f"{obs}_t"]         = np.asarray(_require_key(f, f"{obs}_t",         path, "quench"))
            out[f"abs_{obs}_t"]     = np.asarray(_require_key(f, f"abs_{obs}_t",     path, "quench"))
            out[f"{obs}_t_err"]     = np.asarray(_require_key(f, f"{obs}_t_err",     path, "quench"))
            out[f"abs_{obs}_t_err"] = np.asarray(_require_key(f, f"abs_{obs}_t_err", path, "quench"))
            # Optional (added 2026-04-27): time-dependent Binder cumulant.
            bind_t     = _read_key(f, f"bind_{obs}_t")
            bind_t_err = _read_key(f, f"bind_{obs}_t_err")
            ts = out["ts"]
            out[f"bind_{obs}_t"]     = (np.asarray(bind_t,     dtype=float)
                                       if bind_t     is not None
                                       else np.full(ts.shape, np.nan))
            out[f"bind_{obs}_t_err"] = (np.asarray(bind_t_err, dtype=float)
                                       if bind_t_err is not None
                                       else np.full(ts.shape, np.nan))
    return out


def load_coarsening_file(path):
    """Load a coarsening-mode JLD2 file. Returns a dict with metadata
    (rule, L, p, T, ts, n_samples, η) plus `area_t` (mean cluster size in
    cells over trials) and `area_t_err` (population std across trials).
    """
    with h5py.File(path, "r") as f:
        out = {
            "rule":               _decode(_read_key(f, "rule")),
            "L":                  int(_read_key(f, "L")),
            "p":                  float(_read_key(f, "p")),
            "T":                  int(_read_key(f, "T")),
            "ts":                 np.asarray(_require_key(f, "ts", path, "coarsening")),
            "n_samples":          int(_read_key(f, "n_samples")),
            "data_taking_ratio":  int(_read_key(f, "data_taking_ratio") or 1),
            "η":                  float(_read_key(f, "η")),
            "init_cond":          _decode(_read_key(f, "init_cond")),
            "area_t":             np.asarray(_require_key(f, "area_t",     path, "coarsening"),
                                             dtype=float),
            "area_t_err":         np.asarray(_require_key(f, "area_t_err", path, "coarsening"),
                                             dtype=float),
            "path":               path,
            "mode_effective":     "coarsening",
        }
    return out


def load_erosion_stats_file(path):
    """Load an erosion_stats-mode JLD2 file. Returns a dict with the
    metadata (rule, L, p, domain_size, n_samples, max_time, η, init_cond)
    plus the **per-trial vector** `erosion_times` (length n_samples) and
    summary scalars `mean_erosion_time`, `std_erosion_time`, `n_timeouts`.
    """
    with h5py.File(path, "r") as f:
        out = {
            "rule":               _decode(_read_key(f, "rule")),
            "L":                  int(_read_key(f, "L")),
            "p":                  float(_read_key(f, "p")),
            "domain_size":        float(_read_key(f, "domain_size")),
            "n_samples":          int(_read_key(f, "n_samples")),
            "max_time":           int(_read_key(f, "max_time")),
            "η":                  float(_read_key(f, "η")),
            "init_cond":          _decode(_read_key(f, "init_cond")),
            "erosion_times":      np.asarray(_require_key(f, "erosion_times",
                                                          path, "erosion_stats"),
                                             dtype=np.int64),
            "n_timeouts":         int(_read_key(f, "n_timeouts") or 0),
            "mean_erosion_time":  float(_read_key(f, "mean_erosion_time")),
            "std_erosion_time":   float(_read_key(f, "std_erosion_time")),
            "path":               path,
            "mode_effective":     "erosion_stats",
        }
    return out


def _compute_theta(ts, ys, b=10):
    """Instantaneous magnetization-decay exponent
    θ(t) = log_b(⟨m(t/b)⟩ / ⟨m(t)⟩),
    averaged over b consecutive samples to damp sample noise. Matches the
    ca_plotter.py estimator and the screenshot convention: θ(t) is
    **positive** for a decay, with θ(t) ≈ β/(νz) in the asymptotic regime.

    For each k, θ_k = (1/ln b) · ⟨ ln(m[k] / m[k*b + j]) ⟩_{j=0..b-1}, with
    m[k] taken as the "early" time and m[k*b + j] as the "late" time a
    factor of b later.

    Returns `(t_out, θ_out)` with `t_out[k] = ts[k] * b`. Empty arrays are
    returned when there are fewer than b+1 input samples.
    """
    ts = np.asarray(ts, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n  = len(ts)
    # Need at least b+1 points to form one estimate at the earliest time.
    if n <= b:
        return np.array([]), np.array([])
    t_out, theta_out = [], []
    log_b = np.log(b)
    for k in range(1, n // b):      # start at k=1: m0 = ys[k]; need k ≥ 1
        base = k * b
        if base + b > n:
            break
        m0 = ys[k]
        block = ys[base:base + b]
        if m0 <= 0 or np.any(block <= 0):
            continue
        # Mean over b offsets of log_b(m_early / m_late). For decay this
        # is > 0, and asymptotically equals β/(νz).
        ratios = np.log(m0 / block) / log_b
        theta_k = np.mean(ratios)
        t_out.append(ts[k] * b)
        theta_out.append(theta_k)
    return np.asarray(t_out), np.asarray(theta_out)


def plot_quench(entries, observable, rule, *, use_abs=True,
                fit_window=None, beta=None, nu=None,
                plot_theta=False, theta_b=10,
                title="", cmap=None):
    """Log-log plot of ⟨|x|⟩(t) vs t across L (one curve per size) for
    quench-mode data. Fits a least-squares power law on each curve over
    `fit_window = (t_lo, t_hi)` (default: middle 50% in log-t) and, given
    β and ν, reports z_L = -β / (ν · slope) plus the trajectory-mean z.

    With `use_abs=True` (default) the abs form ⟨|x|⟩(t) is plotted — required
    for observables that cross zero (D starts at 0, m can sign-flip at late t).
    With `use_abs=False` the signed ⟨x⟩(t) is plotted (positive before the
    decay hits its finite-size floor).

    With `plot_theta=True` the figure switches to the instantaneous decay
    exponent θ(t) = d ln ⟨|x|⟩ / d ln t (log-spaced finite-difference with
    step `theta_b`), on a semilog-x axis. A clean power-law regime shows up
    as a plateau in θ(t). The plateau value is also printed, along with
    z_plateau = -β / (ν · θ̄) when β, ν are available.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Match ca_plotter.py aesthetics: serif / Computer Modern Roman, larger
    # square figure, thin lines with fat markers, no minor ticks.
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif']  = ['Computer Modern Roman'] + plt.rcParams.get(
        'font.serif', [])
    if cmap is None:
        cmap = cm.coolwarm

    lw = 1.2
    ms = 7

    entries = sorted(entries, key=lambda e: e["L"])
    y_key  = f"abs_{observable}_t" if use_abs else f"{observable}_t"
    e_key  = f"{y_key}_err"

    # Detect multi-rule input (e.g. comparing R / F / M / Toom / Ising on the
    # same axes — the use case in ca_plotter.py -plot mt). In that case we
    # switch to ca_plotter-style aesthetics: one curve per rule with the
    # per-rule colour and label, no black marker edges, fit window
    # [1%, 75%] of the index range, and a plain `$m(t)$` y-label.
    rules_seen = {(ent.get("rule") or "").strip() for ent in entries}
    rule_mode  = len(rules_seen) > 1

    fig, ax = plt.subplots(figsize=(10, 8))
    if not rule_mode:
        # Rule-mode keeps minor ticks on so the y-axis log scale renders
        # decimal labels (1, .9, .8, …) via ScalarFormatter below.
        ax.minorticks_off()

    # ------------------------------------------------------------------
    # θ(t) branch: plot the instantaneous log-log slope per curve and
    # print its mean over the same `fit_window` used for the power-law fit.
    if plot_theta:
        theta_means = []
        # Post-_compute_theta binning width: group this many consecutive
        # θ(t) samples and average to damp residual sample noise (a single
        # _compute_theta call already averages over `theta_b` offsets but
        # one curve per time-step is still jagged for short-T runs).
        theta_bin = 10

        # X-axis upper limit: the smallest final time across the loaded
        # curves, so every rule has data over the full plotted range.
        t_maxes = [float(np.asarray(ent["ts"], dtype=float)[-1])
                   for ent in entries]
        x_right = min(t_maxes) if t_maxes else None
        for li, ent in enumerate(entries):
            L  = ent["L"]
            ts = np.asarray(ent["ts"],  dtype=float)
            ys = np.asarray(ent[y_key], dtype=float)
            mask = np.isfinite(ys) & (ys > 0) & (ts > 0)
            ts_m, ys_m = ts[mask], ys[mask]
            if len(ts_m) <= theta_b:
                continue
            t_th, theta_th = _compute_theta(ts_m, ys_m, b=theta_b)
            if t_th.size == 0:
                continue

            # Bin the (t_th, theta_th) output by `theta_bin` consecutive
            # samples, averaging each group. Drops the trailing partial bin
            # (so all bins represent the same number of underlying samples).
            if theta_bin > 1 and t_th.size >= theta_bin:
                nfull = (t_th.size // theta_bin) * theta_bin
                t_th = t_th[:nfull].reshape(-1, theta_bin).mean(axis=1)
                theta_th = theta_th[:nfull].reshape(-1, theta_bin).mean(axis=1)

            if rule_mode:
                col, lab = _quench_rule_style(ent.get("rule"),
                                              ent.get("threesqz", False))
                mec, line_lw, point_ms = col, 0, ms * 0.75
                ent_beta, ent_beta_err, ent_nu, ent_nu_err = \
                    _quench_rule_exponents(ent.get("rule"),
                                            ent.get("threesqz", False))
            else:
                col = cmap((li + 1) / max(len(entries), 1))
                lab, mec, line_lw, point_ms = rf"${int(L)}$", col, lw, ms
                ent_beta, ent_nu = beta, nu
                ent_beta_err = ent_nu_err = 0.0

            # Plateau θ̄: average of the latter 50% of the binned curve
            # within the displayed time window (overrides any explicit
            # --fit-window in plot-theta mode, which is fine since fit-window
            # was originally only used for the mt-branch power-law fit).
            in_window = (t_th <= x_right) if x_right is not None else slice(None)
            t_in = t_th[in_window]
            theta_in = theta_th[in_window]
            if t_in.size >= 2:
                half = t_in.size // 2
                theta_bar = float(np.mean(theta_in[half:]))
            else:
                theta_bar = np.nan

            # Combine the rule label with the plateau θ̄ to three decimal
            # places into a single legend entry (the dashed line itself is
            # unlabelled to avoid a duplicate entry).
            if rule_mode and np.isfinite(theta_bar):
                inner = lab.strip("$")
                curve_lab = rf"${inner}\;\;{theta_bar:.3f}$"
            else:
                curve_lab = lab

            ax.plot(t_th, theta_th, c=col, lw=line_lw, marker='o',
                    ms=point_ms, mew=lw, mec=mec, label=curve_lab,
                    alpha=0.8 if rule_mode else 0.85)

            if np.isfinite(theta_bar):
                line_col = _brighten(col, 0.5) if rule_mode else col
                ax.axhline(theta_bar, color=line_col, linestyle='--',
                           linewidth=2 * lw, alpha=0.6)

            theta_means.append((L, theta_bar, lab,
                                ent_beta, ent_beta_err, ent_nu, ent_nu_err))

        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\theta(t)$')
        ax.minorticks_off()
        if x_right is not None:
            ax.set_xlim(left=0, right=x_right)
        else:
            ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        if rule_mode:
            # Multi-column legend at the bottom-centre so it doesn't
            # overlap the rising portions of the θ(t) curves.
            ax.legend(loc='lower center', ncols=3, frameon=False)
        else:
            ax.legend(title=r'$L$', ncols=2)
            ax.set_title(title)
        fig.tight_layout()

        # Per-curve table: plateau mean θ → z = β / (ν · θ̄).
        # Propagate σ_β, σ_ν (θ̄ treated as fixed): (σ_z/|z|)² = (σ_β/β)² + (σ_ν/ν)².
        print()
        if rule_mode:
            print(f"{'rule':>12} {'L':>5} {'mean θ':>10} {'z':>14}")
        else:
            print(f"{'L':>5} {'mean θ':>10} {'z':>10}")
        zs = []
        for L, th, lab, b_, be_, n_, ne_ in theta_means:
            if b_ is not None and n_ is not None and th > 0 and np.isfinite(th):
                z_L = b_ / (n_ * th)
                if b_ > 0 and n_ > 0:
                    ze_L = abs(z_L) * np.sqrt((be_ / b_) ** 2 + (ne_ / n_) ** 2)
                else:
                    ze_L = np.nan
            else:
                z_L, ze_L = float('nan'), float('nan')
            zs.append(z_L)
            if rule_mode:
                z_str = _fmt_value_uncert(z_L, ze_L)
                print(f"{lab:>12} {L:>5d} {th:>10.4g} {z_str:>14}")
            else:
                print(f"{L:>5d} {th:>10.4g} {z_L:>10.4g}")
        finite_z = [z for z in zs if np.isfinite(z)]
        if finite_z and not rule_mode:
            mn = float(np.mean(finite_z))
            sd = float(np.std(finite_z, ddof=1)) if len(finite_z) > 1 else 0.0
            print(f"\nz = {mn:.4g} ± {sd:.4g}  "
                  f"(mean ± std of β/(ν·θ̄) across {len(finite_z)} L values, "
                  f"β={beta}, ν={nu})")
        plt.show()
        return

    # ------------------------------------------------------------------
    # Default branch: log-log ⟨|x|⟩(t) curves + power-law fit overlay.
    z_vals = []
    for li, ent in enumerate(entries):
        L  = ent["L"]
        ts = np.asarray(ent["ts"],    dtype=float)
        ys = np.asarray(ent[y_key],   dtype=float)
        es = np.asarray(ent[e_key],   dtype=float)

        # drop points with non-positive y (log-scale) or non-finite values
        mask = np.isfinite(ys) & (ys > 0) & (ts > 0)
        ts_m, ys_m = ts[mask], ys[mask]
        es_m = np.where(np.isfinite(es[mask]) & (es[mask] > 0), es[mask], np.nan)

        if rule_mode:
            col, lab = _quench_rule_style(ent.get("rule"),
                                          ent.get("threesqz", False))
            mec      = col
            ent_beta, ent_beta_err, ent_nu, ent_nu_err = \
                _quench_rule_exponents(ent.get("rule"),
                                       ent.get("threesqz", False))
        else:
            col = cmap((li + 1) / max(len(entries), 1))
            lab = rf"${int(L)}$"
            mec = 'k'
            ent_beta, ent_nu = beta, nu
            ent_beta_err = ent_nu_err = 0.0

        # error ribbon: shaded band at ±1 SEM, falling back to no band where
        # SEM is NaN (e.g. n_samples=1 run).
        if not np.all(np.isnan(es_m)):
            ax.fill_between(ts_m, ys_m - es_m, ys_m + es_m,
                             color=col, alpha=0.25, linewidth=0)
        # Rule-mode: smaller markers and no connecting line (the dashed
        # power-law fit is the only line drawn). FSS-mode unchanged.
        plot_lw = 0 if rule_mode else lw
        plot_ms = ms * 0.75 if rule_mode else ms
        ax.plot(ts_m, ys_m, c=col, lw=plot_lw, marker='o', ms=plot_ms,
                mew=lw, mec=mec,
                label=lab, alpha=0.8 if rule_mode else 0.85)

        # fit window. In rule-mode we match ca_plotter.py: indices [1%, 75%]
        # of the curve. Otherwise (FSS-style single-rule data) keep the
        # historical middle-50%-in-log-t window.
        if fit_window is None:
            if rule_mode:
                n = len(ts_m)
                i1 = max(1, int(round(0.01 * n)))
                i2 = max(i1 + 2, int(round(0.75 * n)))
                lo, hi = ts_m[i1], ts_m[min(i2, n - 1)]
            else:
                log_t = np.log(ts_m)
                lo = np.exp(log_t[0] + 0.25 * (log_t[-1] - log_t[0]))
                hi = np.exp(log_t[0] + 0.75 * (log_t[-1] - log_t[0]))
        else:
            lo, hi = fit_window
        fit_mask = (ts_m >= lo) & (ts_m <= hi)
        if fit_mask.sum() < 2:
            continue
        # log-log slope via np.polyfit
        slope, intercept = np.polyfit(np.log(ts_m[fit_mask]),
                                       np.log(ys_m[fit_mask]), 1)
        # z from β, ν, slope:  ⟨|x|⟩(t) ∝ t^(-β/(νz))  ⇒  z = -β/(ν·slope).
        # Propagate σ_β, σ_ν (slope treated as fixed):
        #   (σ_z / |z|)² = (σ_β / β)² + (σ_ν / ν)².
        if ent_beta is not None and ent_nu is not None \
                and np.isfinite(ent_beta) and np.isfinite(ent_nu) \
                and slope < 0:
            z_L = -ent_beta / (ent_nu * slope)
            if ent_beta > 0 and ent_nu > 0:
                rel_var = (ent_beta_err / ent_beta) ** 2 \
                          + (ent_nu_err / ent_nu) ** 2
                z_err = abs(z_L) * np.sqrt(rel_var)
            else:
                z_err = np.nan
        else:
            z_L = np.nan
            z_err = np.nan
        z_vals.append((L, slope, z_L, z_err, lab if rule_mode else None))

        if rule_mode:
            # ca_plotter style: dashed darker-shade fit line drawn over the
            # whole curve from index 5 onward, labelled with the magnetization
            # decay exponent (= -slope) to three significant figures.
            t_line = ts_m[5:] if len(ts_m) > 5 else ts_m
            y_line = np.exp(intercept + slope * np.log(t_line))
            ax.plot(t_line, y_line, c=_brighten(col, 0.5),
                    ls='--', lw=2 * lw, alpha=1.0,
                    label=r"$%.3g$" % (-slope))
        else:
            # FSS style: dashed fit line confined to the fit window.
            t_line = np.geomspace(ts_m[fit_mask][0], ts_m[fit_mask][-1], 20)
            y_line = np.exp(intercept + slope * np.log(t_line))
            ax.plot(t_line, y_line, c=col, ls='--', lw=2 * lw, alpha=0.9)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$t$')
    if rule_mode:
        ax.set_ylabel(r'$m(t)$')
        ax.set_ylim(bottom=0.3, top=1.1)
        # Render the log-scale y ticks as plain decimals (1, .9, .8, …), the
        # same trick ca_plotter uses on its mt plot.
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.legend()
    else:
        if use_abs:
            ax.set_ylabel(rf'$\langle | {observable} | (t) \rangle$')
        else:
            ax.set_ylabel(rf'$\langle {observable}(t) \rangle$')
        ax.legend(title=r'$L$', ncols=2)
        ax.set_title(title)
    fig.tight_layout()

    # Per-curve table + summary. In rule-mode the first column is the rule
    # label (per-rule β, ν used for z, with σ_z from error propagation);
    # otherwise L (with shared β, ν, no σ_z).
    print()
    if rule_mode:
        print(f"{'rule':>12} {'L':>5} {'slope':>10} {'z':>14}")
        for L, s, z, ze, lab in z_vals:
            z_str = _fmt_value_uncert(z, ze)
            print(f"{lab:>12} {L:>5d} {s:>10.4g} {z_str:>14}")
    else:
        print(f"{'L':>5} {'slope':>10} {'z':>10}")
        for L, s, z, _, _ in z_vals:
            print(f"{L:>5d} {s:>10.4g} {z:>10.4g}")
    finite_z = [z for _, _, z, _, _ in z_vals if np.isfinite(z)]
    if finite_z and not rule_mode:
        mean = float(np.mean(finite_z))
        std  = float(np.std(finite_z, ddof=1)) if len(finite_z) > 1 else 0.0
        print(f"\nz = {mean:.4g} ± {std:.4g}  "
              f"(mean ± std across {len(finite_z)} L values, "
              f"β={beta}, ν={nu})")
    elif not rule_mode and (beta is None or nu is None):
        print("\nz not computed — pass --beta and --nu (or their defaults "
              "in exponents.py) to infer z from the fitted slope.")
    plt.show()


def plot_trel(thermo, pc, title="", raw=False, cmap=None):
    """Direct τ_rel vs (p − pc) plot for trel-mode data.

    One curve per system size, log-log axes when `raw=False` (shows power-law
    τ_rel ∝ (p − pc)^(−νz)). With `raw=True`, plot τ_rel vs p on semilogy axes
    for a sanity check when pc isn't trusted.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif']  = ['Computer Modern Roman'] + plt.rcParams.get(
        'font.serif', [])
    if cmap is None:
        cmap = cm.coolwarm

    Ls    = np.asarray(thermo["Ls"])
    xs_all = np.asarray(thermo["xs"])        # (nL, n_ps)
    tau   = np.asarray(thermo["t_rels"])     # (nL, n_ps), first-passage time
    tau_e = np.asarray(thermo["t_rels_err"]) # (nL, n_ps), may be all NaN
    to_frac = np.asarray(thermo.get("timeout_frac",
                                    np.zeros_like(tau)))  # (nL, n_ps)

    _warn_censored(Ls, xs_all, to_frac, thermo.get("n_trials"))

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    any_plotted = False
    min_p_seen = np.inf
    # Pool uncensored (p − pc, τ) across all L for the power-law overlay.
    pool_dp, pool_t = [], []
    for l, _ in enumerate(Ls):
        col = cmap((l + 1) / max(len(Ls), 1))
        raw_xs = xs_all[l]
        xs = raw_xs if raw else raw_xs - pc
        ys = tau[l]
        ye = tau_e[l]
        cen = to_frac[l] if l < len(to_frac) else np.zeros_like(ys)
        # Drop non-positive x (only relevant with `not raw` if some p ≤ pc),
        # and non-finite y values.
        mask = np.isfinite(ys) & (np.asarray(xs) > 0 if not raw
                                  else np.ones_like(xs, bool))
        if raw_xs.size:
            min_p_seen = min(min_p_seen, float(raw_xs.min()))
        if not np.any(mask):
            continue
        xs_m, ys_m = xs[mask], ys[mask]
        ye_m = ye[mask]
        ye_m = None if np.all(np.isnan(ye_m)) else np.where(np.isnan(ye_m), 0.0, ye_m)
        # Split into "good" and "censored" (timeout frac > threshold). Censored
        # points get an open red-edged marker and a vertical up-arrow to signal
        # that τ_rel is a lower bound.
        cen_mask = cen[mask] > CENSORED_FRAC_THRESHOLD
        good_mask = ~cen_mask
        if good_mask.any():
            ye_g = None if ye_m is None else ye_m[good_mask]
            ax.errorbar(xs_m[good_mask], ys_m[good_mask], yerr=ye_g,
                        c=col, marker='o', mec='k', mew=0.8, ms=5, ls='-',
                        lw=1.5, capsize=2, label=None)
            if not raw:
                pool_dp.extend(xs_m[good_mask].tolist())
                pool_t.extend(ys_m[good_mask].tolist())
        if cen_mask.any():
            ax.errorbar(xs_m[cen_mask], ys_m[cen_mask], yerr=None,
                        c=col, marker='^', mec='red', mew=1.2, ms=7,
                        mfc='white', ls='', lw=0,
                        label=None)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        raise RuntimeError(
            f"plot_trel: no points survived the p > pc filter "
            f"(pc = {pc:.5g}, min p in data = {min_p_seen:.5g}). "
            "Pass a correct --pc, or use --raw to plot τ_rel vs p directly."
        )

    # Power-law overlay (log-log axes only): τ = A · (p − pc)^(−a). Two-param
    # linear fit of log τ on log(p − pc) over the pooled uncensored points.
    fit_label = None
    if not raw and len(pool_dp) >= 2:
        pool_dp_a = np.asarray(pool_dp)
        pool_t_a  = np.asarray(pool_t)
        slope, intercept = np.polyfit(np.log(pool_dp_a), np.log(pool_t_a), 1)
        a_fit = -slope
        A_fit = np.exp(intercept)
        xs_line = np.geomspace(pool_dp_a.min(), pool_dp_a.max(), 80)
        ys_line = A_fit * xs_line ** (-a_fit)
        fit_label = rf"$a = {a_fit:.2f}$"
        ax.plot(xs_line, ys_line, "k--", lw=1.2, label=fit_label)
        print(f"power-law fit  τ_mem = A · (p − pc)^(−a)  with pc = {pc:.5g}:",
              file=sys.stderr)
        print(f"  a = {a_fit:.3f}, A = {A_fit:.3g}", file=sys.stderr)

    ax.set_xscale('log' if not raw else 'linear')
    ax.set_yscale('log')
    ax.set_xlabel(r'$p - p_c$' if not raw else r'$p$')
    ax.set_ylabel(r'$\tau_{\sf mem}$')
    if not raw:
        ax.set_title((title + r"  $|$  " if title else "") +
                     rf"$p_c = {pc:.5g}$")
    else:
        ax.set_title(title)
    if fit_label is not None:
        # Only the fit line goes in the legend, no frame, no title.
        ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    plt.show()


def plot_trel_nucleation(thermo, pc, title="", cmap=None, a_fixed=None):
    """Test first-order / nucleation-style scaling
        τ_rel  ∝  exp( b · (p − pc)^(−a) )
    on trel-mode data. This is the natural ansatz if the transition is
    nucleation-driven: the activation barrier scales as a power of the
    supersaturation, giving an exp-of-power relaxation time.

    Two complementary fits are reported:

    1. **Linearized fit** (log-log-log): `log log τ = log b − a · log(p − pc)`.
       Slope in log-log-log space gives −a directly, independent of the
       prefactor. Stable and unique; uses only the points with τ > 1.
       Skipped if `a_fixed` is supplied; in that case the left panel still
       shows the data but overlays the user's `a` as the fit line.

    2. **Rectified fit** (pooled, fix a): fit
       `log τ = log C + b · (p − pc)^(−a)` as a linear regression in
       (p − pc)^(−a). Gives (b, C) without the three-way degeneracy that
       plagues a joint (a, b, C) non-linear fit. `a` is either taken from
       fit 1 or from the user-supplied `a_fixed`.

    Two diagnostic panels:

    Left: log log τ  vs  log (p − pc), with the slope-−a fit line. Straight
        line ⇒ the nucleation ansatz is a good description.

    Right: log τ  vs  (p − pc)^(−a), with the rectified linear fit.
        Straight line ⇒ the ansatz holds even accounting for the prefactor.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif']  = ['Computer Modern Roman'] + plt.rcParams.get(
        'font.serif', [])
    if cmap is None:
        cmap = cm.coolwarm

    Ls     = np.asarray(thermo["Ls"])
    xs_all = np.asarray(thermo["xs"])
    tau    = np.asarray(thermo["t_rels"])    # first-passage time
    to_frac = np.asarray(thermo.get("timeout_frac",
                                    np.zeros_like(tau)))

    _warn_censored(Ls, xs_all, to_frac, thermo.get("n_trials"))

    # Pool (p−pc, τ) across L. Exclude censored points (timeout_frac >
    # threshold) from the fit — their τ is a lower bound, including them as
    # a measurement would bias the fit.
    pool_dp, pool_logt = [], []
    n_excluded = 0
    for l in range(len(Ls)):
        dp = xs_all[l] - pc
        t  = tau[l]
        cen = to_frac[l] if l < len(to_frac) else np.zeros_like(t)
        mask = ((dp > 0) & np.isfinite(t) & (t > 0)
                & (cen <= CENSORED_FRAC_THRESHOLD))
        pool_dp.extend(dp[mask].tolist())
        pool_logt.extend(np.log(t[mask]).tolist())
        n_excluded += int(((dp > 0) & np.isfinite(t) & (t > 0)
                           & (cen > CENSORED_FRAC_THRESHOLD)).sum())
    pool_dp   = np.asarray(pool_dp)
    pool_logt = np.asarray(pool_logt)
    if n_excluded:
        print(f"  (note: excluded {n_excluded} censored point(s) from the fit; "
              "plotted as open red ▲.)", file=sys.stderr)

    if len(pool_dp) < 3:
        raise RuntimeError(
            f"plot_trel_nucleation: need ≥ 3 uncensored points with p > pc; "
            f"got {len(pool_dp)}. Check --pc or widen the sweep upward."
        )

    # ---- Fit 1: log log τ = log b − a · log(p − pc) on points with log τ > 0.
    loglog_mask = pool_logt > 0
    if loglog_mask.sum() >= 3:
        logdp_ll = np.log(pool_dp[loglog_mask])
        loglt    = np.log(pool_logt[loglog_mask])
        (slope_ll, intercept_ll), cov1 = np.polyfit(logdp_ll, loglt, 1, cov=True)
        a_from_ll = -slope_ll
        a_ll_err  =  np.sqrt(cov1[0, 0])
        logb_ll   =  intercept_ll
        logb_ll_err = np.sqrt(cov1[1, 1])
        resid_ll  = loglt - (slope_ll * logdp_ll + intercept_ll)
        rms_ll    = float(np.sqrt(np.mean(resid_ll ** 2)))
        have_fit1 = True
    elif a_fixed is None:
        raise RuntimeError(
            "plot_trel_nucleation: need ≥ 3 points with τ > 1 for the "
            "log-log-log linearization, or supply --a to fix a by hand."
        )
    else:
        have_fit1 = False

    # Decide which `a` to use for the rectified fit.
    if a_fixed is not None:
        a_fit  = float(a_fixed)
        a_err  = 0.0
        a_src  = "user-supplied"
    else:
        a_fit  = a_from_ll
        a_err  = a_ll_err
        a_src  = "fit (1)"

    # ---- Fit 2: given a_fit, linear fit log τ = log C + b · (p − pc)^(−a).
    inv_pow = pool_dp ** (-a_fit)
    (b_fit, logC_fit), cov2 = np.polyfit(inv_pow, pool_logt, 1, cov=True)
    b_err    = np.sqrt(cov2[0, 0])
    logC_err = np.sqrt(cov2[1, 1])
    resid_r  = pool_logt - (logC_fit + b_fit * inv_pow)
    rms_r    = float(np.sqrt(np.mean(resid_r ** 2)))

    print(f"Nucleation fit, pc = {pc:.5g}:")
    if have_fit1:
        print(f"  (1) Linearized  log log τ = log b − a · log(p − pc)")
        print(f"       a       = {a_from_ll:.3f}  ± {a_ll_err:.3f}  "
              f"{'[NOT USED; --a override]' if a_fixed is not None else ''}")
        print(f"       b       = {np.exp(logb_ll):.3g}   "
              f"(log b = {logb_ll:.3f} ± {logb_ll_err:.3f})")
        print(f"       RMS(log log τ) residual = {rms_ll:.3f} over {loglog_mask.sum()} points")
    else:
        print(f"  (1) Linearized fit skipped (too few points with τ > 1).")
    print(f"  (2) Rectified   log τ = log C + b · (p − pc)^(−a)   "
          f"[a = {a_fit:.3f} from {a_src}]")
    print(f"       b       = {b_fit:.3g}   ± {b_err:.2g}")
    print(f"       C       = {np.exp(logC_fit):.3g}   (log C = {logC_fit:.3f} ± {logC_err:.3f})")
    print(f"       RMS(log τ) residual = {rms_r:.3f} over {len(pool_dp)} points")

    # Single-panel rectified plot: τ (log y) vs (p − pc)^(−a) (linear x).
    # Straight line on this axes ⇒ τ = C · exp(b · (p − pc)^(−a)).
    fig, ax = plt.subplots(figsize=(5, 4))

    for l, L in enumerate(Ls):
        col = cmap((l + 1) / max(len(Ls), 1))
        dp  = xs_all[l] - pc
        t   = tau[l]
        cen = to_frac[l] if l < len(to_frac) else np.zeros_like(t)
        censored = cen > CENSORED_FRAC_THRESHOLD
        m_all = (dp > 0) & np.isfinite(t) & (t > 0)
        m    = m_all & ~censored
        m_c  = m_all & censored
        if m.any():
            ax.plot(dp[m] ** (-a_fit), t[m],
                    "o", mec="k", mew=0.8, mfc=col, ms=5,
                    label=rf"${int(L)}$")
        if m_c.any():
            ax.plot(dp[m_c] ** (-a_fit), t[m_c],
                    "^", mec="red", mew=1.2, mfc="white", ms=7,
                    label=(None if m.any() else rf"${int(L)}$"))

    # Overlay the rectified fit: τ = exp(log C + b · x).
    xs_line = np.linspace(0.0, inv_pow.max() * 1.02, 80)
    ax.plot(xs_line, np.exp(logC_fit + b_fit * xs_line), "k--", lw=1.2,
            label=rf"$b={b_fit:.3g}$")

    ax.set_yscale("log")
    ax.set_xlabel(rf"$(p - p_c)^{{-{a_fit:.1f}}}$")
    ax.set_ylabel(r"$\tau_{\sf mem}$")
    # Title: just the rule name, as passed in by the caller.
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def assemble_quench_collapse(entries, observable="m"):
    """Pack quench entries into a `thermo`-shaped dict suitable for the
    `binds_t` collapse fit (`U(t, L) = f(t / L^z)`).

    Returns `{Ls, xs, binds_t, binds_t_err}` with `xs` shape `(n_L, n_t)`.
    Trims all `ts` arrays to the shortest common length so each row has the
    same number of columns (required by `collapse_cost`).
    """
    entries = sorted(entries, key=lambda e: e["L"])
    n_t = min(len(e["ts"]) for e in entries)
    Ls = np.array([e["L"] for e in entries], dtype=float)
    xs = np.array([np.asarray(e["ts"][:n_t], dtype=float) for e in entries])
    bind_key     = f"bind_{observable}_t"
    bind_err_key = f"bind_{observable}_t_err"
    if bind_key not in entries[0]:
        raise KeyError(
            f"quench file is missing '{bind_key}'. Re-run the simulation "
            "with the updated `run_quench` (now writes per-time Binder "
            "cumulants).")
    binds_t     = np.array([np.asarray(e[bind_key][:n_t],     dtype=float)
                            for e in entries])
    binds_t_err = np.array([np.asarray(e[bind_err_key][:n_t], dtype=float)
                            for e in entries])
    return {
        "Ls":          Ls,
        "xs":          xs,
        "binds_t":     binds_t,
        "binds_t_err": binds_t_err,
    }


def plot_quench_bind_raw(entries, observable, *, title="", cmap=None):
    """Plot the time-dependent Binder cumulant U_obs(t) for each L on the
    same axes — semilog-x, linear-y. No rescaling, no fit. One curve per L.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif']  = ['Computer Modern Roman'] + plt.rcParams.get(
        'font.serif', [])
    if cmap is None:
        cmap = cm.coolwarm

    entries = sorted(entries, key=lambda e: e["L"])
    bind_key     = f"bind_{observable}_t"
    bind_err_key = f"bind_{observable}_t_err"

    # Marker / line aesthetics matched to scaling_plotter's raw branch so the
    # time-dependent Binder figure reads visually the same as the time-
    # independent stats-mode plots.
    mew = 1.0
    ms  = 6.5 * 0.75
    lw  = 2.2

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.minorticks_off()
    asymptotes = []
    for li, ent in enumerate(entries):
        L  = ent["L"]
        ts = np.asarray(ent["ts"], dtype=float)
        b  = np.asarray(ent[bind_key], dtype=float)
        be = np.asarray(ent[bind_err_key], dtype=float)
        if not np.any(np.isfinite(b)):
            print(f"warning: L={int(L)} has no Binder data — re-run the "
                  "simulation with the updated `run_quench`.", file=sys.stderr)
            continue
        col = cmap((li + 1) / max(len(entries), 1))
        yerr = np.where(np.isnan(be), 0.0, be)
        ax.errorbar(ts, b, yerr=yerr, c=col, label=r'$%d$' % int(L),
                    marker='o', mec='k', mew=mew, ms=ms, lw=lw,
                    elinewidth=0.9, capsize=2)
        finite = np.isfinite(b) & (ts > 0)
        ts_f = ts[finite]
        b_f  = b[finite]
        if b_f.size >= 3:
            n_late = max(3, b_f.size // 2)
            t_late = ts_f[-n_late:]
            b_late = b_f[-n_late:]
            coefs = np.polyfit(1.0 / t_late, b_late, 1)
            asymptotes.append(float(coefs[-1]))
            # Debug overlay: show the fit curve on the fit window, extended
            # to the right so the approach to the constant term is visible.
            t_plot = np.linspace(t_late.min(), 1.5 * t_late.max(), 300)
            b_plot = np.polyval(coefs, 1.0 / t_plot)
            ax.plot(t_plot, b_plot, c=col, linestyle='-', linewidth=1.0,
                    alpha=0.9, zorder=5)
    if asymptotes:
        B_inf = float(np.mean(asymptotes))
        ax.axhline(B_inf, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.text(0.08, B_inf, f'{B_inf:.2f}',
                transform=ax.get_yaxis_transform(),
                va='bottom', ha='left', fontsize=11, color='k')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$B(t)$")
    ax.set_title(title)
    ax.legend(title=r'$L$', title_fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_quench_bind_collapse(entries, observable, z, *,
                              z_uncert=None, title="", cmap=None):
    """Plot the quench Binder cumulant rescaled as U(t/L^z) for each L,
    with all curves collapsed onto a single x-axis. Static figure (no
    interactive z-tuning); the printed fit value is what gets used.

    `z_uncert`: optional `(lo, hi)` tuple from a bootstrap, displayed in the
    legend if provided.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif']  = ['Computer Modern Roman'] + plt.rcParams.get(
        'font.serif', [])
    if cmap is None:
        cmap = cm.coolwarm

    entries = sorted(entries, key=lambda e: e["L"])
    bind_key     = f"bind_{observable}_t"
    bind_err_key = f"bind_{observable}_t_err"

    # Marker aesthetics matched to scaling_plotter's scaled branch: filled
    # coloured circles, no connecting line, faint alpha, small square figure.
    mew = 1.0
    ms  = 6.5 * 0.75 * 0.8
    a   = 0.8

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.minorticks_off()
    for li, ent in enumerate(entries):
        L  = ent["L"]
        ts = np.asarray(ent["ts"], dtype=float)
        b  = np.asarray(ent[bind_key], dtype=float)
        be = np.asarray(ent[bind_err_key], dtype=float)
        col = cmap((li + 1) / max(len(entries), 1))
        x = ts / (L ** z)
        # Skip the very first point (t = 1) where U ≈ 1 across all L is
        # uninformative and dominated by the initial condition.
        m = x > 0
        ax.errorbar(x[m], b[m], yerr=np.where(np.isnan(be[m]), 0.0, be[m]),
                    c=col, label=r'$%d$' % int(L),
                    marker='o', mfc=col, mew=mew, ms=ms, lw=0, alpha=a,
                    elinewidth=0.8, capsize=2)
    ax.set_xlabel(rf"$t / L^{{{z:.3f}}}$")
    ax.set_ylabel(r"$B(t)$")
    # Match scaling_plotter's title convention: symmetric (hi ≈ lo) renders
    # as " \pm σ"; asymmetric falls back to ^{+hi}_{-lo}. Symmetric is the
    # common case once σ_total = √(σ_stat² + σ_sys²) has been folded in.
    if z_uncert is None:
        unc_tex = ""
    else:
        lo, hi = z_uncert
        tol = 0.01 * max(abs(lo), abs(hi), 1e-300)
        if abs(hi - lo) <= tol:
            unc_tex = r" \pm %.2g" % (0.5 * (hi + lo))
        else:
            unc_tex = r"^{+%.2g}_{-%.2g}" % (hi, lo)
    rule_tex = (title[1:-1]
                if title.startswith("$") and title.endswith("$") else title)
    ax.set_title(r"$%s\, \, | \, \, z = %.3f%s$"
                 % (rule_tex, z, unc_tex))
    ax.legend(title=r'$L$', title_fontsize=12)
    fig.tight_layout()
    plt.show()


def _rule_style(rule):
    """Map a rule string to (color, latex_label). Mirrors the substring
    matching in ca_plotter.py:90-161 so heterogeneous-rule figures are
    visually consistent with MemoryNCA conventions, and accepts the
    squeezing-criticality short names ("R", "M", "F") as well.
    """
    s = (rule or "").lower()
    if "toom" in s:
        return "#6BF3B3", r"${\sf Toom}$"
    if "glauber" in s or "ising" in s:
        return "#888888", r"${\sf Glauber}$"
    if "rsqz" in s or s == "r":
        return "#9086F8", r"${\sf R}$"
    if "msqz" in s or s == "m":
        return "#7BC1FC", r"${\sf M}$"
    if "fsqz" in s or s == "f":
        return "#FF4B62", r"${\sf F}$"
    return "k", rf"${{\sf {rule or '?'}}}$"


def plot_quench_mt(entries, *, fit_window=(0.01, 0.75), title=""):
    """Plot ⟨|m(t)|⟩ vs t on log–log axes for a heterogeneous-rule set of
    quench files. Each entry is colored / labeled by rule via `_rule_style`;
    a power-law fit on the fit window gives the long-time decay exponent θ
    (m(t) ∝ t^{-θ}), and a jackknife over the points in the fit window
    yields the standard error on θ. θ and its SE are shown in the legend.

    `fit_window`: (lo, hi) fractions of the per-curve time series.
    Default (0.01, 0.75) matches the convention of ca_plotter.py:765-766.
    """
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif']  = ['Computer Modern Roman'] + plt.rcParams.get(
        'font.serif', [])
    if fit_window is None:
        fit_window = (0.01, 0.75)
    lo, hi = fit_window

    mew = 1.0
    ms  = 6.5
    lw  = 2.2

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.minorticks_off()
    for ent in entries:
        rule = ent.get("rule", "")
        col, lab = _rule_style(rule)
        ts = np.asarray(ent["ts"], dtype=float)
        y  = np.asarray(ent["abs_m_t"], dtype=float)
        good = np.isfinite(ts) & np.isfinite(y) & (ts > 0) & (y > 0)
        ts, y = ts[good], y[good]
        if ts.size == 0:
            print(f"warning: rule={rule}: no usable data; skipping.",
                  file=sys.stderr)
            continue

        n = ts.size
        i_lo = max(0, int(lo * n))
        i_hi = min(n, max(i_lo + 2, int(hi * n)))
        t_fit, y_fit = ts[i_lo:i_hi], y[i_lo:i_hi]

        theta, se_theta = float("nan"), float("nan")
        if t_fit.size >= 3:
            logt, logy = np.log(t_fit), np.log(y_fit)
            coefs = np.polyfit(logt, logy, 1)
            theta = -float(coefs[0])
            t_ov = np.array([t_fit[0], t_fit[-1]])
            ax.plot(t_ov, np.exp(coefs[1]) * t_ov ** coefs[0],
                    c=col, ls='--', lw=lw, alpha=0.5)
            if t_fit.size >= 5:
                slopes = np.empty(t_fit.size)
                idx = np.arange(t_fit.size)
                for i in range(t_fit.size):
                    mask = idx != i
                    sub = np.polyfit(logt[mask], logy[mask], 1)
                    slopes[i] = -sub[0]
                jk_mean = slopes.mean()
                se_theta = float(np.sqrt(
                    (t_fit.size - 1) / t_fit.size *
                    np.sum((slopes - jk_mean) ** 2)))
            else:
                print(f"warning: rule={rule}: {t_fit.size} pts in fit window; "
                      "skipping jackknife.", file=sys.stderr)

        rule_inner = lab.strip("$")
        if np.isfinite(theta) and np.isfinite(se_theta) and se_theta > 0:
            # Russian-doll convention "0.0936(1)" — show θ to enough decimals
            # that the SE's first significant digit lands in the last shown
            # place, then put that digit in parentheses.
            sig = int(np.floor(np.log10(se_theta)))
            n_decimals = max(2, -sig)
            err_digits = max(1, int(round(se_theta / (10.0 ** sig))))
            full_lab = (rf"${rule_inner},\, "
                        rf"\theta = {theta:.{n_decimals}f}({err_digits})$")
        elif np.isfinite(theta):
            full_lab = rf"${rule_inner},\, \theta = {theta:.3f}$"
        else:
            full_lab = lab

        ax.plot(ts, y, c=col, lw=lw, marker='o', ms=ms, mew=mew, mec=col,
                ls='-', alpha=0.85, label=full_lab)

        print(f"rule={rule}: θ = {theta:.4f} ± {se_theta:.4f}  "
              f"(jackknife, n_pts={t_fit.size})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\langle |m(t)| \rangle$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    plt.show()


def plot_erosion_distribution(entries, *, log_y=True, n_bins=40, title=""):
    """Overlay per-rule erosion-time PDFs from erosion_stats files. Each
    curve is plotted in mean-normalized units: x = t_erode / ⟨t_erode⟩, so
    all distributions are centered at 1 and their relative concentration
    is directly comparable. Color/label via `_rule_style`. The legend
    reports the rule, mean μ, and relative spread σ/μ — the diagnostic for
    "exponential concentration about the mean." y-axis defaults to log.
    """
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif']  = ['Computer Modern Roman'] + plt.rcParams.get(
        'font.serif', [])

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.minorticks_off()

    # Build a common bin range across all rules in the mean-normalized x-axis
    # so the histograms are comparable.
    rescaled = []
    for ent in entries:
        t = np.asarray(ent["erosion_times"], dtype=float)
        if t.size == 0:
            continue
        mu = float(t.mean())
        if mu > 0:
            rescaled.append(t / mu)
    if not rescaled:
        print("warning: no erosion_times found in any entry.", file=sys.stderr)
        return
    all_x = np.concatenate(rescaled)
    bins = np.linspace(float(all_x.min()), float(all_x.max()), n_bins + 1)

    for ent in entries:
        rule = ent.get("rule", "")
        col, lab = _rule_style(rule)
        t = np.asarray(ent["erosion_times"], dtype=float)
        if t.size == 0:
            print(f"warning: rule={rule}: empty erosion_times.", file=sys.stderr)
            continue
        if ent.get("n_timeouts", 0) > 0:
            print(f"warning: rule={rule}: {ent['n_timeouts']}/{ent['n_samples']} "
                  "trials hit max_time (μ and σ are biased low — these "
                  "trials only contribute lower-bound times).",
                  file=sys.stderr)

        mu = float(t.mean())
        sd = float(t.std(ddof=1)) if t.size > 1 else float("nan")
        rel = (sd / mu) if (np.isfinite(sd) and mu > 0) else float("nan")

        rule_inner = lab.strip("$")
        if np.isfinite(rel):
            full_lab = (rf"${rule_inner},\, \mu = {mu:.0f},\, "
                        rf"\sigma/\mu = {rel:.2f}$")
        else:
            full_lab = rf"${rule_inner},\, \mu = {mu:.0f}$"

        ax.hist(t / mu, bins=bins, density=True, histtype='step',
                color=col, lw=1.8, label=full_lab)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$t_{\sf erode}\, /\, \langle t_{\sf erode} \rangle$")
    ax.set_ylabel(r"${\sf P}(t_{\sf erode})$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    plt.show()


def plot_coarsening_areas(entries, *, log_x=False, log_y=True, title=""):
    """Overlay per-rule mean-cluster-size trajectories from coarsening files.
    For each entry: x = t / L, y = ⟨A(t)⟩ / L², ±σ band shaded. Color/label
    via `_rule_style`. Defaults to log-log axes (coarsening dynamics are
    typically power-law in t).
    """
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif']  = ['Computer Modern Roman'] + plt.rcParams.get(
        'font.serif', [])

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.minorticks_off()

    for ent in entries:
        rule = ent.get("rule", "")
        col, lab = _rule_style(rule)
        L  = float(ent["L"])
        ts = np.asarray(ent["ts"], dtype=float)
        a  = np.asarray(ent["area_t"], dtype=float)
        ae = np.asarray(ent["area_t_err"], dtype=float)
        n_samples = max(int(ent.get("n_samples", 1)), 1)

        x  = ts / L
        y  = a / (L * L)
        # `area_t_err` on disk is the population std across trials; convert to
        # SEM (uncertainty on the mean curve) by dividing by √n_samples. This
        # is what `run_quench` already does for its `*_err` keys, and for
        # n_samples ≥ a few hundred the band stays inside [0, 1] in normal
        # regimes (A is bounded above by L²).
        sem = ae / np.sqrt(n_samples)
        ye  = sem / (L * L)

        # Shade the ±SEM band; clip the lower edge at a small positive value
        # so log-y doesn't choke on non-positive entries.
        lo = np.maximum(y - ye, 1.0 / (L * L * n_samples))
        hi = y + ye
        ax.fill_between(x, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(x, y, c=col, lw=1.8, label=lab)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$t / L$")
    ax.set_ylabel(r"$A(t) / L^2$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    plt.show()


def assemble_thermo_data(entries):
    """Pack per-L entries into the dict layout expected by scaling_plotter.

    The single L-dependent timescale carried in the dict is either
    `t_autos` (when the entries came from stats files — the magnetization
    autocorrelation time) or `t_rels` (when the entries came from trel
    files — the first-passage / relaxation time). Whichever is present
    on the entries is plumbed through; the other key is absent. Downstream
    code that wants the L-dependent τ should look up by the plot keyword
    (`t_autos` or `t_rels`), which the CLI keeps in `args.plot`.
    """
    entries = sorted(entries, key=lambda e: e["L"])
    n_ps_set = {len(e["ps"]) for e in entries}
    if len(n_ps_set) != 1:
        raise ValueError(
            f"files have different n_ps values {n_ps_set}; "
            "use --files or --Ls to pick a compatible subset"
        )
    out = {
        "Ls":               np.array([e["L"]            for e in entries]),
        "xs":               np.array([e["ps"]           for e in entries]),
        "mags":             np.array([e["m"]            for e in entries]),
        "chis":             np.array([e["chi"]          for e in entries]),
        "binds":            np.array([e["bind"]         for e in entries]),
        "mags_err":         np.array([e["m_err"]        for e in entries]),
        "chis_err":         np.array([e["chi_err"]      for e in entries]),
        "binds_err":        np.array([e["bind_err"]     for e in entries]),
        # Per-point fraction of trials that hit `max_time` (trel mode only;
        # zero for stats mode). Consumers use this to flag censored t_rel.
        "timeout_frac":     np.array(
            [e.get("timeout_frac", np.zeros_like(np.asarray(e["ps"], float)))
             for e in entries]),
        "n_trials":         np.array([e.get("n_trials", 0) for e in entries]),
    }
    # Stats entries carry `t_auto`; trel entries carry `t_rel`. Emit the
    # corresponding plural key for the scaling pipeline to pick up.
    first = entries[0]
    if "t_auto" in first:
        out["t_autos"]     = np.array([e["t_auto"]     for e in entries])
        out["t_autos_err"] = np.array([e["t_auto_err"] for e in entries])
    if "t_rel" in first:
        out["t_rels"]      = np.array([e["t_rel"]      for e in entries])
        out["t_rels_err"]  = np.array([e["t_rel_err"]  for e in entries])
    return out


def _run_finite_size_diagnostics(thermo, plot_or_plots, *,
                                 joint, fit_params, boot, fit_uncert,
                                 seed_for_boot, fixed,
                                 do_jackknife, do_lmin, title):
    """Leave-one-L-out jackknife (σ_sys) and restricted-range L_min sweep
    diagnostics, factored out so the same block applies to both the
    stats-mode joint/single fits and the quench-mode `binds_t` fit.

    `fit_uncert` is mutated in place: when both the jackknife and a
    bootstrap have run, each parameter's entry is replaced with the
    symmetric (σ_tot, σ_tot) so the figure title reports the
    systematic-aware uncertainty.
    """
    import collapse_fit as cf

    n_Ls_avail = len(np.asarray(thermo["Ls"]))
    if do_jackknife and n_Ls_avail < 3:
        print(f"\n--jackknife-L: only {n_Ls_avail} L value(s) in this "
              "dataset; need ≥ 3 for the leave-one-out subset to retain "
              "≥ 2 curves. Skipping.", file=sys.stderr)
    if do_jackknife and n_Ls_avail >= 3:
        jk = cf.jackknife_L(
            thermo,
            plot_or_plots=plot_or_plots,
            joint=joint,
            x0=seed_for_boot,
            fixed=fixed,
        )
        n_excluded = len(jk)
        param_names = list(fit_params.keys())
        print(f"\nLeave-one-L-out finite-size systematic "
              f"({n_excluded} refits):")
        header = f"  {'excl L':>6} " + " ".join(f"{n:>11s}"
                                                for n in param_names)
        print(header)
        for L in sorted(jk.keys()):
            row = f"  {L:>6d} " + " ".join(f"{jk[L][n]:>11.5g}"
                                           for n in param_names)
            print(row)
        print("\n  σ_sys (leave-one-L-out, sqrt[(n-1)/n · Σ(θ - θ̄)²]):")
        for n in param_names:
            vals = np.array([jk[L][n] for L in jk])
            mean = vals.mean()
            sigma_sys = np.sqrt((n_excluded - 1) / n_excluded
                                 * ((vals - mean) ** 2).sum())
            cen = fit_params.get(n, np.nan)
            if boot is not None and n in boot.per_param:
                s = boot.per_param[n]
                sigma_stat = 0.5 * (s["p84"] - s["p16"])
                sigma_tot  = np.sqrt(sigma_stat ** 2 + sigma_sys ** 2)
                print(f"    {n:>5s} = {cen:.5g}  "
                      f"σ_stat={sigma_stat:.3g}  σ_sys={sigma_sys:.3g}  "
                      f"σ_tot={sigma_tot:.3g}")
                # Override the plot-title uncertainty: bootstrap-only
                # asymmetric (+lo/-hi) gets replaced by symmetric σ_total.
                if fit_uncert is not None and n in fit_uncert:
                    fit_uncert[n] = (sigma_tot, sigma_tot)
            else:
                print(f"    {n:>5s} = {cen:.5g}  "
                      f"σ_sys={sigma_sys:.3g}  "
                      f"(σ_stat unavailable — pair with bootstrap)")

    if do_lmin:
        sweep = cf.Lmin_sweep(
            thermo,
            plot_or_plots=plot_or_plots,
            joint=joint,
            x0=seed_for_boot,
            fixed=fixed,
        )
        param_names = list(fit_params.keys())
        print(f"\nLmin-sweep restricted-range refits "
              f"({len(sweep)} L_min values, smallest L kept first):")
        header = (f"  {'L_min':>6} {'#L':>3} " +
                  " ".join(f"{n:>11s}" for n in param_names))
        print(header)
        Ls_full = sorted(set(int(L) for L in np.asarray(thermo["Ls"])))
        for L_min in sorted(sweep.keys()):
            n_kept = sum(1 for L in Ls_full if L >= L_min)
            row = f"  {L_min:>6d} {n_kept:>3d} " + \
                  " ".join(f"{sweep[L_min][n]:>11.5g}" for n in param_names)
            print(row)
        print("\n  trend with L_min:")
        sweep_keys_sorted = sorted(sweep.keys())
        for n in param_names:
            ys = np.array([sweep[k][n] for k in sweep_keys_sorted])
            diffs = np.diff(ys)
            spread = float(ys.max() - ys.min())
            if len(diffs) == 0 or np.all(diffs == 0):
                verdict = "constant"
            elif np.all(diffs >= 0) and np.any(diffs > 0):
                verdict = "monotone ↑  (push to bigger L)"
            elif np.all(diffs <= 0) and np.any(diffs < 0):
                verdict = "monotone ↓  (push to bigger L)"
            else:
                verdict = "non-monotone (likely noise)"
            print(f"    {n:>5s}: spread={spread:.3g}  {verdict}")

        # Drift plot: each parameter shown as fractional drift from the
        # smallest-L_min (all-L) fit so curves share a dimensionless axis.
        try:
            import matplotlib.pyplot as plt
            ref_key = sweep_keys_sorted[0]
            ref = sweep[ref_key]
            marker_for = {"pc": "o", "nu": "s", "beta": "D", "gamma": "^",
                          "z": "v"}
            color_for  = {"pc": "#4A3FA8", "nu": "#1f77b4", "beta": "#e1812c",
                          "gamma": "#3aa55b", "z": "#d62728"}
            label_for  = {"pc": r"$p_c$", "nu": r"$\nu$",
                          "beta": r"$\beta$", "gamma": r"$\gamma$",
                          "z": r"$z$"}

            fig_d, ax_d = plt.subplots(figsize=(4.5, 3.5))
            ax_d.axhline(0.0, c="k", lw=0.6, alpha=0.4)
            xs_plot = np.array(sweep_keys_sorted, dtype=float)
            ylab_extra = ""
            for n in param_names:
                ref_val = float(ref[n])
                if ref_val == 0:
                    ys = np.array([sweep[k][n] - ref_val
                                   for k in sweep_keys_sorted])
                    ylab_extra = " (absolute, ref=0)"
                else:
                    ys = np.array([sweep[k][n] / ref_val - 1.0
                                    for k in sweep_keys_sorted])
                ax_d.plot(xs_plot, ys,
                          marker=marker_for.get(n, "o"),
                          color=color_for.get(n, "k"),
                          ms=7, mew=1.0, mec="k",
                          lw=1.2, label=label_for.get(n, n))
                if boot is not None and n in boot.per_param and ref_val != 0:
                    s = boot.per_param[n]
                    sigma_stat_frac = 0.5 * (s["p84"] - s["p16"]) / abs(ref_val)
                    for sign in (+1, -1):
                        ax_d.axhline(sign * sigma_stat_frac,
                                     color=color_for.get(n, "k"),
                                     lw=0.8, ls="--", alpha=0.45)
            ax_d.set_xlabel(r"$L_{\min}$")
            ax_d.set_ylabel(r"$\theta(L_{\min})\,/\,\theta_{\rm full} - 1$"
                            + ylab_extra)
            ax_d.legend(frameon=False, ncols=max(1, len(param_names)))
            ax_d.set_title(title)
            fig_d.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"  could not render Lmin drift plot: {exc}",
                  file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", type=str, default=None,
                    choices=["stats", "trel", "quench", "erosion_stats", "coarsening"],
                    help="which simulation mode's output to plot. If omitted: "
                         "auto-detect per file (via its 'mode' key) when --files "
                         "is used, else default to 'stats' for globbing.")
    ap.add_argument("--rule", type=str, default=None,
                    help="rule name (R / M / F). Required if --files is omitted.")
    ap.add_argument("--observable", type=str, default="m",
                    choices=["m", "D"],
                    help="(stats mode only) which measured quantity to analyze "
                         "(m = magnetization, D = xy nearest-neighbor anisotropy).")
    ap.add_argument("--plot", type=str, default=None,
                    choices=["mags", "chis", "binds", "t_autos", "t_rels", "mt"],
                    help="which moment of the chosen observable to plot. "
                         "mags=⟨|x|⟩, chis=χ, binds=Binder, mt=log–log ⟨|m(t)|⟩ "
                         "across rules. The two distinct timescales are "
                         "kept separate: `t_autos` (stats mode only) is "
                         "the magnetization autocorrelation time; "
                         "`t_rels` (trel mode only) is the first-passage / "
                         "relaxation time. Both scale as τ ~ L^z and share "
                         "the same collapse code path. "
                         "Required for stats/trel modes; ignored for quench.")
    ap.add_argument("--pc", type=float, default=None,
                    help="Critical point pc. Two roles: (a) always overrides "
                         "the exponents.py default used for the plotter's "
                         "x-axis reference; (b) with --fit/--joint-fit, "
                         "*fixes* pc to this value in the Nelder-Mead search "
                         "(the optimizer only varies the remaining exponents).")
    ap.add_argument("--Ls", type=int, nargs="+", default=None,
                    help="restrict to these system sizes.")
    ap.add_argument("--files", type=str, nargs="+", default=None,
                    help="explicit file list (overrides auto-discovery).")
    ap.add_argument("--raw", action="store_true",
                    help="plot unscaled data instead of interactive collapse.")
    ap.add_argument("--linear-y", dest="log_y", action="store_false",
                    default=True,
                    help="(erosion_stats mode) plot the erosion-time PDF on "
                         "linear y. Default is log-y so exponential tails "
                         "appear linear.")
    ap.add_argument("--nucleation", action="store_true",
                    help="(trel mode only) test the first-order/nucleation ansatz "
                         "τ_rel = C · exp( b · (p − pc)^(−a) ) instead of the "
                         "default power-law plot. Fits (a, b, C) jointly and "
                         "shows log log τ vs log (p − pc) plus the rectified "
                         "log τ vs (p − pc)^(−a) panels.")
    ap.add_argument("--a", type=float, default=None,
                    help="Exponent `a` for nucleation (trel) mode: fixes a in "
                         "τ_rel = C · exp(b · (p − pc)^(−a)); fits only (b, C) "
                         "by linear regression of log τ on (p − pc)^(−a).")
    ap.add_argument("--data-dir", type=str, default="data",
                    help="directory to glob for jld2 files (default: data).")
    ap.add_argument("--fit", action="store_true",
                    help="run the single-observable auto-collapse fit for --plot, "
                         "print best-fit exponents with bootstrap uncertainties, "
                         "then seed the interactive plotter with the result.")
    ap.add_argument("--joint-fit", action="store_true",
                    help="joint auto-collapse fit across {binds, mags, chis, t_autos}"
                         "sharing (pc, nu). Opens the interactive plotter for --plot "
                         "seeded with the joint-fit values.")
    ap.add_argument("--just-bm", action="store_true",
                    help="(with --joint-fit) restrict the joint fit to only "
                         "the Binder cumulant and magnetization observables; "
                         "χ and τ_exp are dropped. Useful when only m and B "
                         "have reliable statistics (e.g. short data_steps "
                         "truncating the τ_exp fit window) or when you want "
                         "to isolate (pc, ν, β) without the other exponents.")
    ap.add_argument("--just-bmc", action="store_true",
                    help="(with --joint-fit) restrict the joint fit to the "
                         "Binder cumulant, magnetization, and susceptibility "
                         "(B, m, χ); τ_exp is dropped. Fits (pc, ν, β, γ).")
    ap.add_argument("--n-bootstrap", type=int, default=50,
                    help="number of parametric-bootstrap resamples for the fit "
                         "uncertainty (default 50; bump to ~500 for tight "
                         "percentile estimates once you've settled on a fit). "
                         "See --no-bootstrap to skip.")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="skip the bootstrap uncertainty step (fit only).")
    ap.add_argument("--fit-only", action="store_true",
                    help="print fit results and exit without opening the plotter.")
    ap.add_argument("--contours", action="store_true",
                    help="after --fit / --joint-fit, also draw the corner-style "
                         "bootstrap uncertainty plot (saved as PNG).")
    ap.add_argument("--jackknife-L", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="after --fit / --joint-fit, also do a leave-one-L-out "
                         "jackknife: refit with each system size removed in "
                         "turn and report the spread as a finite-size "
                         "*systematic* uncertainty σ_sys. Combined as "
                         "σ_total = √(σ_stat² + σ_sys²). σ_stat is the "
                         "bootstrap σ from --no-bootstrap=False (so don't pair "
                         "this flag with --no-bootstrap unless you only want "
                         "σ_sys). Default ON; pass --no-jackknife-L to skip "
                         "(e.g. on the rare runs where you only have one or "
                         "two L values and the leave-one-out refit is "
                         "ill-conditioned).")
    ap.add_argument("--Lmin-sweep", action="store_true",
                    help="after --fit / --joint-fit, also do a restricted-"
                         "range FSS sweep: refit on {L ≥ L_min} for each L_min "
                         "in turn (dropping the smallest L progressively). "
                         "Look for a plateau in the fitted exponents to judge "
                         "whether the asymptotic regime has been reached. "
                         "Monotone drift ⇒ confluent corrections still "
                         "dominate, push to bigger L. Stable trajectory "
                         "⇒ small-L data are safe to keep, σ_sys is just "
                         "noise.")
    # --- quench-mode-only flags ---
    ap.add_argument("--no-use-abs", dest="use_abs", action="store_false",
                    default=True,
                    help="(quench) plot the signed ⟨x⟩(t) instead of ⟨|x|⟩(t). "
                         "Off by default — |x| is safer since D starts at 0 "
                         "and m can sign-flip at late t.")
    ap.add_argument("--fit-window", type=str, default=None,
                    help="(quench) 't_lo,t_hi' time range for the power-law "
                         "fit. Default: middle 50%% in log-t of each curve.")
    ap.add_argument("--beta", type=float, default=None,
                    help="β. Two roles: (a) in quench mode, used to invert "
                         "the fitted decay slope to z; (b) with stats-mode "
                         "--fit/--joint-fit, *fixes* β to this value in the "
                         "Nelder-Mead search. Falls back to exponents.py "
                         "default (as a seed only) when not supplied.")
    ap.add_argument("--nu", type=float, default=None,
                    help="ν. Two roles: (a) in quench mode, used to invert "
                         "the fitted decay slope to z; (b) with stats-mode "
                         "--fit/--joint-fit, *fixes* ν to this value in the "
                         "Nelder-Mead search. Falls back to exponents.py "
                         "default (as a seed only) when not supplied.")
    ap.add_argument("--z", type=float, default=None,
                    help="z. With --fit/--joint-fit, *fixes* z to this value "
                         "in the Nelder-Mead search (applies to the quench "
                         "binds_t collapse and to stats joint fits that "
                         "include z). Falls back to exponents.py default "
                         "(as a seed only) when not supplied.")
    ap.add_argument("--plot-theta", action="store_true",
                    help="(quench) plot the instantaneous decay exponent "
                         "θ(t) = d ln ⟨|x|⟩ / d ln t instead of the decay curves.")
    ap.add_argument("--theta-b", type=int, default=10,
                    help="(quench) log-spacing step for the θ(t) estimator "
                         "(default 10; matches ca_plotter.py).")
    args = ap.parse_args()

    # --fit-only / --contours / --no-bootstrap are modifiers of an auto-fit;
    # require --fit or --joint-fit to actually run one.
    if (args.fit_only or args.contours or args.no_bootstrap) and \
            not (args.fit or args.joint_fit):
        ap.error("--fit-only / --contours / --no-bootstrap require --fit "
                 "or --joint-fit")
    if args.just_bm and not args.joint_fit:
        ap.error("--just-bm requires --joint-fit")
    if args.just_bmc and not args.joint_fit:
        ap.error("--just-bmc requires --joint-fit")
    if args.just_bm and args.just_bmc:
        ap.error("--just-bm and --just-bmc are mutually exclusive")

    # --plot is meaningless for quench (single hard-coded figure type) but
    # required for stats/trel. Bail early only when the user explicitly
    # asked for stats/trel; with auto-detect (args.mode is None) we defer
    # the check until after file loading.
    if args.mode in ("stats", "trel") and args.plot is None:
        ap.error("--plot is required for --mode=stats or --mode=trel")

    # Mode selection:
    #   --files + explicit --mode  -> honor it (clean error on mismatch)
    #   --files + no --mode        -> auto-detect per file from its `mode` key
    #   no --files + --mode        -> glob {rule}_{mode}_*.jld2
    #   no --files + no --mode     -> glob stats files (backwards-compatible default)
    mode_explicit = args.mode is not None
    glob_mode = args.mode if mode_explicit else "stats"

    if mode_explicit and args.mode == "trel" and args.plot != "t_rels":
        print(f"warning: --plot={args.plot} is not meaningful for trel mode; "
              "only --plot=t_rels has data.", file=sys.stderr)

    def _peek_mode(path):
        """Return the file's `mode` key value (e.g. 'stats' or 'trel'), or None
        if not present (e.g. older files pre-dating the mode field)."""
        try:
            with h5py.File(path, "r") as f:
                return _decode(_read_key(f, "mode"))
        except Exception:
            return None

    def _load(path):
        # Per-file mode: use explicit --mode if given, else auto-detect from
        # the file; fall back to 'stats' if the file has no `mode` key.
        mode_for_file = args.mode if mode_explicit else (_peek_mode(path) or "stats")
        if mode_for_file == "trel":
            return load_trel_file(path)
        if mode_for_file == "quench":
            return load_quench_file(path)
        if mode_for_file == "erosion_stats":
            return load_erosion_stats_file(path)
        if mode_for_file == "coarsening":
            return load_coarsening_file(path)
        return load_stats_file(path, observable=args.observable)

    if args.files is not None:
        paths = args.files
    else:
        if args.rule is None:
            ap.error("--rule is required when --files is not given")
        paths = discover_files(args.rule, glob_mode, pattern_dir=args.data_dir)
        if not paths:
            print(f"no files found matching {args.data_dir}/{args.rule}_{glob_mode}_*.jld2",
                  file=sys.stderr)
            sys.exit(1)

    entries = []
    for p in paths:
        try:
            e = _load(p)
        except Exception as exc:
            print(f"skipping {p}: {exc}", file=sys.stderr)
            continue
        entries.append(e)

    if args.rule is not None:
        entries = [e for e in entries if e["rule"] == args.rule]
    if args.Ls is not None:
        keep = set(args.Ls)
        entries = [e for e in entries if e["L"] in keep]

    if not entries:
        print("no entries survived filtering.", file=sys.stderr)
        sys.exit(1)

    rule = entries[0]["rule"] or args.rule or "?"
    # Effective mode of what we actually loaded (may mix if --files was heterogeneous).
    loaded_modes = {e.get("mode_effective", _peek_mode(e["path"]) or "stats")
                    for e in entries}
    effective_mode = (args.mode if mode_explicit
                      else (loaded_modes.pop() if len(loaded_modes) == 1 else "mixed"))
    label_text = ("τ_rel"          if effective_mode == "trel"
                  else args.observable)
    print(f"mode = {effective_mode}, rule = {rule}, label = {label_text}, "
          f"Ls = {[e['L'] for e in entries]}, "
          f"paths = {[os.path.basename(e['path']) for e in entries]}")

    defaults = get_defaults(rule)
    pc = args.pc if args.pc is not None else defaults["pc"]

    # Title is just the rule name — both for --raw (left as-is by the raw
    # branch in scaling_plotter) and for scaled plots (the scaling_plotter
    # appends "| ν = ..." or "| β = ..." etc. for the active exponent).
    # For joint-fit plots of the R rule, render as R₂ to match the
    # MemoryNCA/Ethan convention (our "R" = the 2-neighbour rsqz variant;
    # the 3-neighbour variant is our separate "R3" rule).
    if rule == "R" and args.joint_fit:
        title = r"$\mathsf{R}_2$"
    else:
        title = r"$\mathsf{" + rule + r"}$"

    # erosion_stats mode: overlay per-rule erosion-time PDFs from the
    # per-trial vectors. Heterogeneous rules expected (one file per rule).
    if effective_mode == "erosion_stats":
        if args.fit or args.joint_fit:
            print("note: --fit / --joint-fit are stats-mode features; ignored "
                  "in erosion_stats mode.", file=sys.stderr)
        plot_erosion_distribution(entries, log_y=args.log_y)
        return

    # coarsening mode: per-rule ⟨A(t)⟩ / L² vs t/L with ±σ shaded bands.
    # Heterogeneous rules expected (one file per rule).
    if effective_mode == "coarsening":
        if args.fit or args.joint_fit:
            print("note: --fit / --joint-fit are stats-mode features; ignored "
                  "in coarsening mode.", file=sys.stderr)
        plot_coarsening_areas(entries, log_y=args.log_y)
        return

    # quench mode: two paths.
    #   default                       : log-log ⟨|x|⟩(t) vs t (plot_quench)
    #   --fit (single-observable)     : Binder collapse U(t,L) = f(t/L^z),
    #                                   fits exponent z via fit_collapse,
    #                                   optionally bootstrapped.
    if effective_mode == "quench":
        if args.joint_fit:
            print("note: --joint-fit is a stats-mode feature; ignored in "
                  "quench mode (use --fit for the U(t,L) collapse).",
                  file=sys.stderr)
        if args.fit:
            import collapse_fit as cf
            try:
                qthermo = assemble_quench_collapse(entries, observable=args.observable)
            except KeyError as e:
                print(f"error: {e}", file=sys.stderr)
                sys.exit(1)
            z_seed = defaults["z"]
            # --z fixes z during the fit (and bootstrap / diagnostics);
            # otherwise z is the single free parameter.
            fixed_quench = {}
            if args.z is not None:
                fixed_quench["z"] = float(args.z)
                print(f"fixing parameters during fit: {fixed_quench}")
            sfit = cf.fit_collapse(qthermo, "binds_t",
                                    x0={"z": z_seed}, fixed=fixed_quench,
                                    maxiter=2000)
            print("\n" + str(sfit))
            z_fit = float(sfit.params["z"])
            fit_params = {"z": z_fit}
            fit_uncert = None
            boot = None
            if not args.no_bootstrap:
                print(f"running parametric bootstrap "
                      f"({args.n_bootstrap} resamples)...")
                boot = cf.bootstrap_fit(qthermo, "binds_t",
                                         joint=False,
                                         n_boot=args.n_bootstrap,
                                         x0={"z": z_fit},
                                         fixed=fixed_quench,
                                         progress=True)
                print(boot.format_summary())
                fit_uncert = {n: (s["median"] - s["p16"],
                                  s["p84"] - s["median"])
                              for n, s in boot.per_param.items()}
                if args.contours:
                    out_png = os.path.join(args.data_dir,
                                           f"{rule}_corner_quench_binds_t.png")
                    cf.plot_bootstrap_corner(boot, save=out_png,
                                              show=not args.fit_only,
                                              title=r"$\mathsf{" + rule + r"}$")
            # Same finite-size diagnostics as the stats-mode fit: jackknife
            # rolls σ_sys into σ_tot in fit_uncert; --Lmin-sweep prints the
            # drift table and renders the drift plot.
            _run_finite_size_diagnostics(
                qthermo, "binds_t",
                joint=False,
                fit_params=fit_params, boot=boot, fit_uncert=fit_uncert,
                seed_for_boot={"z": z_fit}, fixed=fixed_quench,
                do_jackknife=args.jackknife_L,
                do_lmin=args.Lmin_sweep,
                title=title,
            )
            z_uncert = fit_uncert["z"] if fit_uncert and "z" in fit_uncert else None
            if args.fit_only:
                return
            plot_quench_bind_collapse(entries, args.observable, z_fit,
                                       z_uncert=z_uncert,
                                       title=title,
                                       cmap=_RULE_CMAP.get(rule))
            return
        # No fit. Choose the figure based on --plot:
        #   --plot binds   : raw U_obs(t) vs t, one curve per L (no rescaling).
        #   --plot mt      : log–log ⟨|m(t)|⟩ across heterogeneous rules with
        #                    per-curve power-law fit and jackknife θ error.
        #   anything else (or unset) : the default decay-of-⟨|x|⟩(t) view.
        if args.plot == "binds":
            plot_quench_bind_raw(entries, args.observable,
                                 title=title,
                                 cmap=_RULE_CMAP.get(rule))
            return
        if args.plot == "mt":
            fit_window = None
            if args.fit_window is not None:
                fit_window = tuple(float(x) for x in args.fit_window.split(","))
            # Skip the rule-derived title — `--plot mt` is for heterogeneous
            # rules, so the first entry's rule is not representative.
            plot_quench_mt(entries, fit_window=fit_window)
            return
        fit_window = None
        if args.fit_window is not None:
            fit_window = tuple(float(x) for x in args.fit_window.split(","))
        beta = args.beta if args.beta is not None else defaults["beta"]
        nu   = args.nu   if args.nu   is not None else defaults["nu"]
        plot_quench(entries, args.observable, rule,
                    use_abs=args.use_abs,
                    fit_window=fit_window,
                    beta=beta, nu=nu,
                    plot_theta=args.plot_theta,
                    theta_b=args.theta_b,
                    title=title,
                    cmap=_RULE_CMAP.get(rule))
        return

    thermo = assemble_thermo_data(entries)

    # trel mode: direct τ_rel vs (p − pc) plot. Skip the scaling_plotter /
    # auto-fit machinery (those target data-collapse of stats-mode moments).
    if effective_mode == "trel":
        if args.fit or args.joint_fit:
            print("note: --fit / --joint-fit are stats-mode features; ignored "
                  "in trel mode.", file=sys.stderr)
        if args.nucleation:
            if args.raw:
                print("note: --raw is ignored in --nucleation mode.",
                      file=sys.stderr)
            # Nucleation plot uses just the rule name as the title.
            nuc_title = r"$\mathsf{" + rule + r"}$"
            plot_trel_nucleation(thermo, pc, title=nuc_title, a_fixed=args.a)
        else:
            if args.a is not None:
                print("note: --a is only used with --nucleation; ignored.",
                      file=sys.stderr)
            plot_trel(thermo, pc, title=title, raw=args.raw)
        return

    # ------------------------------------------------------------------
    # Optional auto-fit path (replaces hand-entered defaults in exponents.py).
    fit_seed = {"pc": pc, "nu": defaults["nu"], "beta": defaults["beta"],
                "gamma": defaults["gamma"], "z": defaults["z"]}
    fit_params = None   # best-fit exponent values if we ran a fit
    fit_uncert = None   # {name: (lo_err, hi_err)} (1σ, from bootstrap)
    if args.fit or args.joint_fit:
        import collapse_fit as cf

        # Joint-fit observable subset:
        # * --just-bm                  → {binds, mags}        (fits pc, nu, beta)
        # * --just-bmc                 → {binds, mags, chis}  (fits pc, nu, beta, gamma)
        # * --plot=t_autos             → {binds, mags, t_autos} (fits pc, nu, beta, z)
        #                                (χ dropped; when the user is focused on
        #                                the dynamic exponent, γ isn't what we
        #                                need, and excluding χ keeps the fit
        #                                parsimonious).
        # * default                    → all four observables (fits all five
        #                                exponents pc, nu, beta, gamma, z).
        # Stats-mode observables only — the quench-mode `binds_t` key is also
        # in cf._OBS_PARAMS and would explode the joint cost with a missing
        # data key here.
        #
        # `joint_plots` drives the joint cost; `--plot` is allowed to request
        # an additional plot (e.g. `--just-bm --plot t_autos` fits on B,m
        # only and then displays τ_auto scaled with the resulting (pc, nu) —
        # z is taken from defaults / `--z` since it isn't fit).
        # Stats-mode observables only: `t_autos` is the autocorrelation
        # time (NOT the first-passage time `t_rels`, which only appears in
        # trel-mode files and never enters the joint fit alongside stats
        # observables).
        _STATS_OBS = ["binds", "mags", "chis", "t_autos"]
        if args.just_bm:
            joint_plots = ["binds", "mags"]
        elif args.just_bmc:
            joint_plots = ["binds", "mags", "chis"]
        elif args.plot == "t_autos":
            joint_plots = ["binds", "mags", "t_autos"]
        else:
            joint_plots = list(_STATS_OBS)

        # --pc, --nu, --beta, --z double as "hold this parameter fixed during
        # the fit" constraints when explicitly passed. Empty dict → fully
        # free fit (the historical behaviour). These take precedence over x0
        # seeds.
        fixed_params = {}
        if args.pc is not None:
            fixed_params["pc"] = float(args.pc)
        if args.nu is not None:
            fixed_params["nu"] = float(args.nu)
        if args.beta is not None:
            fixed_params["beta"] = float(args.beta)
        if args.z is not None:
            fixed_params["z"] = float(args.z)
        if fixed_params:
            print(f"fixing parameters during fit: {fixed_params}")

        if args.joint_fit:
            jfit = cf.fit_joint(thermo, plots=joint_plots, x0=fit_seed,
                                fixed=fixed_params)
            print("\n" + str(jfit))
            fit_params = dict(jfit.params)
            seed_for_boot = fit_params
        else:
            sfit = cf.fit_collapse(thermo, args.plot, x0=fit_seed,
                                   fixed=fixed_params)
            print("\n" + str(sfit))
            fit_params = dict(sfit.params)
            seed_for_boot = fit_params

        boot = None
        if not args.no_bootstrap:
            print(f"running parametric bootstrap ({args.n_bootstrap} resamples)...")
            boot = cf.bootstrap_fit(
                thermo, fixed=fixed_params,
                plot_or_plots=(joint_plots if args.joint_fit
                               else args.plot),
                joint=bool(args.joint_fit),
                n_boot=args.n_bootstrap,
                x0=seed_for_boot,
                # Progress so the user can see the bootstrap isn't stuck —
                # joint fits are ~20× slower per resample than single.
                progress=True,
            )
            print(boot.format_summary())
            fit_uncert = {n: (s["median"] - s["p16"], s["p84"] - s["median"])
                          for n, s in boot.per_param.items()}

        # ------------------------------------------------------------------
        # Optional leave-one-L-out (σ_sys) and restricted-range L_min sweep
        # diagnostics. See `_run_finite_size_diagnostics` for the math; the
        # jackknife mutates fit_uncert in place to fold σ_sys into σ_tot.
        _run_finite_size_diagnostics(
            thermo,
            plot_or_plots=(joint_plots if args.joint_fit else args.plot),
            joint=bool(args.joint_fit),
            fit_params=fit_params, boot=boot, fit_uncert=fit_uncert,
            seed_for_boot=seed_for_boot, fixed=fixed_params,
            do_jackknife=args.jackknife_L,
            do_lmin=args.Lmin_sweep,
            title=title,
        )

        # Seed the interactive plotter's exponents with the fit values. Only
        # parameters actually in the fit are overridden; others stay at
        # exponents.py defaults.
        for k in ("pc", "nu", "beta", "gamma", "z"):
            if k in fit_params:
                fit_seed[k] = float(fit_params[k])
        pc = fit_seed["pc"]

        if args.contours:
            if boot is None:
                print("note: --contours requires bootstrap; re-run without "
                      "--no-bootstrap. Skipping corner plot.")
            else:
                tag = (
                    "just_bm"   if (args.joint_fit and args.just_bm) else
                    "just_bmc"  if (args.joint_fit and args.just_bmc) else
                    "joint_bmt" if (args.joint_fit and args.plot == "t_autos") else
                    "joint"     if args.joint_fit else
                    args.plot
                )
                out_png = os.path.join(args.data_dir,
                                       f"{rule}_corner_{tag}.png")
                rule_tex = (r"$\mathsf{R}_2$" if rule == "R" and args.joint_fit
                            else r"$\mathsf{" + rule + r"}$")
                cf.plot_bootstrap_corner(boot, save=out_png,
                                          show=not args.fit_only,
                                          title=rule_tex)

        # Fit-quality summary — the very last thing we print before either
        # returning (`--fit-only`) or opening the interactive plot. The
        # Houdayer–Hartmann cost S is a reduced-χ²-like statistic: at the
        # true optimum, with the scaling ansatz exact and the per-point
        # jackknife errors honest, each per-plot S fluctuates about unity;
        # the joint total is the *sum* across observables (≈ n_obs at the
        # optimum). S ≫ expected → scaling violation or underestimated
        # errors; S ≪ expected → inflated errors.
        print("\n" + "=" * 60)
        if args.joint_fit:
            print("Fit quality (Houdayer–Hartmann cost S = (1/N_eff) Σ "
                  "(Y − Ŷ)² / (σ² + σ̂²)):")
            for p, c in jfit.per_plot_cost.items():
                print(f"  S_{p:<6s} = {c:.4f}")
            print(f"  total S  = {jfit.total_cost:.4f}   "
                  f"(sum over {len(jfit.per_plot_cost)} observable(s); "
                  f"≈ n_obs at the true optimum)")
        else:
            print("Fit quality (Houdayer–Hartmann cost S = (1/N_eff) Σ "
                  "(Y − Ŷ)² / (σ² + σ̂²)):")
            print(f"  S_{args.plot:<6s} = {sfit.cost:.4f}   "
                  f"(≈ 1 at the true optimum)")
        print("=" * 60)

        if args.fit_only:
            return

    # When the user asked for a joint fit, walk through each observable that
    # actually entered the joint cost function and show its scaled-data
    # window in turn (close one to advance to the next). For single-obs
    # --fit (or no fit) just show the requested --plot.
    if args.fit or args.joint_fit:
        if args.joint_fit:
            # Always show every observable that entered the joint cost. If the
            # user explicitly asked for an extra `--plot` outside that subset
            # (e.g. `--just-bm --plot t_autos`), tack it on so they can see the
            # fitted (pc, nu) overlaid on that scaling axis as well.
            plots_to_show = list(joint_plots)
            if args.plot is not None and args.plot not in plots_to_show:
                plots_to_show.append(args.plot)
        else:
            plots_to_show = [args.plot]
    else:
        plots_to_show = [args.plot]
    for plot_name in plots_to_show:
        scaling_plotter(
            thermo, plot_name, pc,
            nu0=fit_seed["nu"],
            gamma0=fit_seed["gamma"],
            beta0=fit_seed["beta"],
            z0=fit_seed["z"],
            raw=args.raw,
            d=2,
            title=title,
            cmap=_RULE_CMAP.get(rule, _cm.coolwarm),
            fit_params=fit_params,
            fit_uncert=fit_uncert,
        )


if __name__ == "__main__":
    main()
