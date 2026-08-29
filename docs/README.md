# Interactive demo

A browser demo of the squeezing codes, published with GitHub Pages at

> https://ethanlake.github.io/squeezing-codes/

Pick one of the rules `R2`, `R3`, `F`, `M`, `T` studied in the paper; each step, every site updates
(with probability `α`) by applying the OR half of the rule with probability given by the rule
selection probability slider and the AND half otherwise, on top of i.i.d. noise of strength
`p` and bias `η`. Draw on the canvas with the brush to perturb the state.

## Running locally

Any static file server works; the page fetches `ca_rules.json`, so `file://` will not do.

```bash
cd docs && python3 -m http.server 8000   # then open http://localhost:8000
```

## Enabling the public link

In the GitHub repo, go to Settings → Pages and set the source to branch `main`, folder
`/docs`.

## Contents

| file | |
| --- | --- |
| `index.html` | page layout, controls, and the closing blurb |
| `ca.js` | simulation and rendering logic; `RULES` at the top lists the dropdown entries |
| `ca_rules.json` | the OR/AND 512-bit codes for the five rules |
| `swissgl.js` | [SwissGL](https://github.com/google/swissgl), the WebGL wrapper doing the GPU work |
| `style.css`, `images/` | styling and UI assets |

## Provenance

Adapted from the [Memory NCA demo](https://memorynca.github.io/2D/floq/)
([paper](https://arxiv.org/abs/2508.15726)). Changes from the original: assets are served
from this directory rather than a parent; the analytics and `polyfill.io` scripts are
removed; the two rule dropdowns are collapsed into one selector that loads a matched OR/AND
pair; the editable rule bitmap, the current-rule readout, and the custom-rule form are gone;
the rule library is trimmed to the paper's rules; and the title and text describe the
squeezing codes.
