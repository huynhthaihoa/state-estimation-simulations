# generator/

Scripts that draw the figures, GIFs and SVGs of this repo. Run everything from the repo root.

| Script | Writes | Used by |
| --- | --- | --- |
| [make_figures.py](make_figures.py) | `assets/<name>.png`: one static figure per simulation (and a `_concept` figure for the docs that have one), from the simulations' own output | the docs' result figures |
| `animate_<simulation>.py` (23) | an animated GIF of one simulation's default run, `--out assets/<simulation>.gif` | the matching doc, next to its static figure |
| [animate_left_right_invariant.py](animate_left_right_invariant.py), [animate_left_right_invariant_angular_velocity.py](animate_left_right_invariant_angular_velocity.py) | `assets/left_right_invariant.gif`, `assets/left_right_invariant_angular_velocity.gif`: the frame-change pictures (no simulation) | [docs/filtering/left_right_invariant.md](../docs/filtering/left_right_invariant.md) §3, §3.1 |
| [animate_sparse_cholesky_fill_in.py](animate_sparse_cholesky_fill_in.py) | `assets/sparse_cholesky_fill_in.gif`: the Cholesky factor of the bundle-adjustment scene filling in under three elimination orders (symbolic, no simulation run) | [docs/optimization/sparse_cholesky_factorization.md](../docs/optimization/sparse_cholesky_factorization.md) §5 |
| [animate_quaternion_double_cover.py](animate_quaternion_double_cover.py), [animate_kf_predict_update.py](animate_kf_predict_update.py), [animate_triangulation_baseline.py](animate_triangulation_baseline.py) | `assets/quaternion_double_cover.gif` (exact math), `assets/kf_predict_update.gif` (the 1D Kalman example, later cycles illustrative), `assets/triangulation_baseline.gif` (`triangulate_landmark` solves): concept GIFs, no simulation run | [quaternion.md](../docs/foundations/quaternion.md) §10, [kf_ekf_iekf.md](../docs/filtering/kf_ekf_iekf.md) §1, [triangulation_pnp.md](../docs/frontend/triangulation_pnp.md) §2 |
| [make_frontend_backend_svgs.py](make_frontend_backend_svgs.py) | `docs/images/frontend_backend.svg`, `docs/images/frontend_backend_loop_closure.svg` | [docs/frontend_backend.md](../docs/frontend_backend.md) |
| [make_factor_graph_svgs.py](make_factor_graph_svgs.py) | `docs/images/factor_graph.svg`, `docs/images/factor_graph_vs_pose_graph.svg` | [docs/optimization/factor_graph.md](../docs/optimization/factor_graph.md) |
| [make_banner_svg.py](make_banner_svg.py) | `assets/banner.svg` | the top of [README.md](../README.md) |

```bash
uv run python generator/make_figures.py                          # all PNG figures, or name one: pose_graph
uv run python generator/animate_gait_phase_ekf.py --out assets/gait_phase_ekf.gif
uv run python generator/make_frontend_backend_svgs.py            # writes into docs/images/ (or give a directory)
uv run python generator/make_factor_graph_svgs.py              # writes into docs/images/ (or give a directory)
uv run python generator/make_banner_svg.py                       # rewrites assets/banner.svg (or give a file)
```

The SVG generators are deterministic: running one again rewrites its files byte for byte. `make_banner_svg.py` was written to reproduce the banner that had been edited by hand, so its output matches that banner exactly; the picture in the ESTIMATE box is scene data in the script. If the banner's content changes, update its `DESC` and the README's image alt text with it.
