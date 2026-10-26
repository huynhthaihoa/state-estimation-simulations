# generator/

Scripts that draw the figures, GIFs and SVGs of this repo. Run everything from the repo root.

| Script | Writes | Used by |
| --- | --- | --- |
| [make_figures.py](make_figures.py) | `assets/<name>.png`: one static figure per simulation (and a `_concept` figure for the docs that have one), from the simulations' own output | the docs' result figures |
| `animate_<simulation>.py` (14) | an animated GIF of one simulation's default run, `--out assets/<simulation>.gif` | the matching doc, next to its static figure |
| [make_frontend_backend_svgs.py](make_frontend_backend_svgs.py) | `docs/images/frontend_backend.svg`, `docs/images/frontend_backend_loop_closure.svg` | [docs/frontend_backend.md](../docs/frontend_backend.md) |
| [make_factor_graph_svgs.py](make_factor_graph_svgs.py) | `docs/images/factor_graph.svg`, `docs/images/factor_graph_vs_pose_graph.svg` | [docs/optimization/factor_graph.md](../docs/optimization/factor_graph.md) |
| [make_banner_svg.py](make_banner_svg.py) | `assets/banner.svg` | the top of [README.md](../README.md) |
| [_paths.py](_paths.py) | nothing: imported by the animation scripts to put `use_numpy/` and the repo root on `sys.path` | |

```bash
uv run python generator/make_figures.py                          # all PNG figures, or name one: pose_graph
uv run python generator/animate_gait_phase_ekf.py --out assets/gait_phase_ekf.gif
uv run python generator/make_frontend_backend_svgs.py            # writes into docs/images/ (or give a directory)
uv run python generator/make_factor_graph_svgs.py              # writes into docs/images/ (or give a directory)
uv run python generator/make_banner_svg.py                       # rewrites assets/banner.svg (or give a file)
```

The SVG generators are deterministic: running one again rewrites its files byte for byte. `make_banner_svg.py` was written to reproduce the banner that had been edited by hand, so its output matches that banner exactly; the picture in the ESTIMATE box is scene data in the script. If the banner's content changes, update its `DESC` and the README's image alt text with it.
