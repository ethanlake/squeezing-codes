"""
Per-rule starting guesses for the critical point pc and the critical exponents
(nu, beta, gamma, z).

pc values below are the current η = 0 critical points for this repo's
async-Poisson dynamics (OR-vs-AND coin flip at each site update). They were
measured from mag-quench runs on large systems. Override with --pc on the CLI
for biased (η ≠ 0) runs.
"""

RULE_DEFAULTS = {
    "R":    {"pc": 0.038425, "nu": 0.92, "beta": 0.17, "gamma": 1.6, "z": 2.17},
    "R3":   {"pc": 0.02876,  "nu": 0.92, "beta": 0.17, "gamma": 1.6, "z": 2.17},
    "M":    {"pc": 0.0032875,"nu": 1.00, "beta": 0.20, "gamma": 1.6, "z": 2.17},
    "F":    {"pc": 0.01165,  "nu": 0.95, "beta": 0.17, "gamma": 1.6, "z": 2.17},
    # Toom universality class is 2D Ising; exponents seeded from there.
    "Toom":  {"pc": 0.13395, "nu": 1.00, "beta": 0.125, "gamma": 1.75, "z": 2.17},
    # Ising: 2D nearest-neighbor Ising with zero-T Glauber + bit-flip noise.
    # pc seeded from earlier MemoryNCA measurements (zeroT_glauber).
    "Ising": {"pc": 0.141,   "nu": 1.00, "beta": 0.125, "gamma": 1.75, "z": 2.17},
}


def get_defaults(rule):
    """Return a dict {pc, nu, beta, gamma, z} for the given rule."""
    return RULE_DEFAULTS.get(rule, {
        "pc": 0.08, "nu": 1.0, "beta": 0.125, "gamma": 1.75, "z": 2.17,
    })
