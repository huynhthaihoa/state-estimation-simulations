'''
Animated version of part 2 (propagation) of use_numpy/probability_basics.py: a
range sensor measures an object 10 m straight ahead as a range r (std 0.2 m)
and a bearing angle theta (std sigma_bearing). The position (x, y) = (r cos
theta, r sin theta) is a curved (nonlinear) function of (r, theta). The
straight-line approximation of the Kalman filter family (linearization: replace
the curve by its tangent at the mean, so the Gaussian stays a Gaussian with
covariance G P G^T) is compared with the real answer (200 000 Monte Carlo
samples pushed through the true curve).

The frames sweep sigma_bearing from 1 deg to 35 deg in steps of 1 deg, then
hold the last frame. Two blocks:
  1. left: the sample cloud (a random subset of the samples), the true 10 m arc
     and its straight tangent, the linearized 1 sigma / 2 sigma ellipses (red,
     dashed) and the 1 sigma / 2 sigma ellipses of the sample covariance (blue).
     The view zooms with sigma_bearing so the SHAPE is comparable from frame to
     frame (the axis numbers show the scale); the text under the title gives the
     arc's departure from its tangent at +-1 sigma, s = r (1 - cos sigma_bearing),
     next to the radial noise (0.2 m), the thickness of the cloud;
  2. right: the two mismatch metrics of the script versus sigma_bearing with a
     moving marker: the fit score (script: "sample NEES", 2 = the straight-line
     Gaussian describes the cloud, log scale) and the covariance error (script:
     "cov err", relative Frobenius norm of P_lin - P_sample, drawn in %).
     A faint vertical line marks where s equals the radial noise.

Data: the script's own run_propagation with its default scene and seed 0; the
random stream is the script's spawned "propagation" stream, so the rows at 1,
5, 15 and 30 deg equal the table printed by use_numpy/probability_basics.py (the
generator asserts this). run_propagation also builds the linear-map part (a),
which the animation does not show; neutral dummy inputs are passed for it.
The SAME standard-normal draws are reused at every frame (common random
numbers: only the bearing scale changes), so the cloud morphs smoothly.

Message (doc docs/foundations/probability_basics.md, propagation section): the
straight-line Gaussian matches the cloud while the arc barely leaves its
tangent within +-1 sigma, and fails once the bend is larger than the cloud's
thickness: the cloud turns into a crescent, the real mean slides away from the
tangent point and the fit score leaves 2.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed). About 30 s.

Run: python generator/animate_probability_basics.py --out assets/probability_basics.gif
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from state_estimation_simulations.use_numpy.probability_basics import PARTS, ellipse_points, run_propagation

# Table rows printed by use_numpy/probability_basics.py (seed 0, 200000 samples):
# bearing std (deg) -> (cov err, sample NEES), rounded as printed there.
SCRIPT_TABLE = {1: (0.002, 2.002), 5: (0.007, 2.094), 15: (0.076, 10.214), 30: (0.324, 114.755)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--range", type=float, default=10.0, help="Mean range to the object (m)")
    parser.add_argument("--range-std", type=float, default=0.2, help="Range standard deviation (m)")
    parser.add_argument("--bearing-deg", type=float, default=0.0, help="Mean bearing (deg)")
    parser.add_argument("--sweep-deg", type=float, nargs=2, default=[1.0, 35.0], metavar=("FIRST", "LAST"),
                        help="Bearing standard deviation sweep, first and last value (deg), in steps of 1 deg")
    parser.add_argument("--n-samples", type=int, default=200000, help="Monte Carlo samples per frame")
    parser.add_argument("--n-draw", type=int, default=3000, help="Samples drawn as dots (a subset of the Monte Carlo ones)")
    parser.add_argument("--hold", type=int, default=6, help="Extra frames that hold the last sweep value")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (as use_numpy/probability_basics.py)")
    parser.add_argument("--fps", type=int, default=5, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=70, help="GIF resolution")
    parser.add_argument("--out", type=str, default="probability_basics.gif", help="Output GIF path")
    args = parser.parse_args()

    # Same stream as the script's main(): the "propagation" child of SeedSequence(seed).spawn(3).
    ss = dict(zip(PARTS, np.random.SeedSequence(args.seed).spawn(len(PARTS))))
    sweep_deg = np.arange(args.sweep_deg[0], args.sweep_deg[1] + 0.5, 1.0)
    theta0 = np.radians(args.bearing_deg)
    res = run_propagation(np.zeros(2), np.eye(2), np.eye(2), args.range, theta0, args.range_std,
                          np.radians(sweep_deg), args.n_samples, ss["propagation"])["polar"]
    cases = [{k: v for k, v in c.items() if k != "samples"} | {"dots": c["samples"][:args.n_draw]} for c in res]
    del res
    nees = np.array([c["sample_nees"] for c in cases])
    cov_err = np.array([c["cov_err"] for c in cases]) * 100.0
    # bend of the arc away from its tangent at +-1 sigma (m), and the sigma where it equals the radial noise
    bend = args.range * (1.0 - np.cos(np.radians(sweep_deg)))
    sigma_eq = np.degrees(np.arccos(1.0 - args.range_std / args.range))

    if args.n_samples == 200000 and args.seed == 0 and args.range == 10.0 and args.range_std == 0.2 and args.bearing_deg == 0.0:
        for deg, (ce, ne) in SCRIPT_TABLE.items():
            i = int(np.argmin(np.abs(sweep_deg - deg)))
            if abs(sweep_deg[i] - deg) < 1e-9:
                assert round(cases[i]["cov_err"], 3) == ce and round(cases[i]["sample_nees"], 3) == ne, \
                    f"{deg} deg row differs from the script table: {cases[i]['cov_err']:.3f}, {cases[i]['sample_nees']:.3f}"
        print("Rows at 1, 5, 15, 30 deg match the table printed by use_numpy/probability_basics.py")
    print("bearing std (deg)  bend at +-1 sigma (m)  cov err (%)  fit score (sample NEES)")
    for i in range(0, len(sweep_deg), 2):
        print(f"{sweep_deg[i]:17.0f}  {bend[i]:21.4f}  {cov_err[i]:11.2f}  {nees[i]:23.3f}")
    print(f"arc bend at +-1 sigma equals the range noise ({args.range_std} m) at {sigma_eq:.1f} deg")

    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0], left=0.07, right=0.97,
                          top=0.80, bottom=0.25, hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1], sharex=ax_f)
    fig.suptitle("Is a straight-line approximation good enough? A range + bearing sensor locating an object",
                 fontsize=12.5, y=0.985)
    status = fig.text(0.5, 0.925, "", ha="center", fontsize=12, fontweight="bold")
    sub = fig.text(0.5, 0.88, "", ha="center", fontsize=10.5)

    # --- left: the scene
    th = np.linspace(-np.pi / 2, np.pi / 2, 400)
    arc = args.range * np.stack([np.cos(th + theta0), np.sin(th + theta0)], axis=1)
    ax.plot(arc[:, 0], arc[:, 1], color="0.35", lw=1.4, label=f"true curve: all points {args.range:g} m from the sensor")
    tan_dir = np.array([-np.sin(theta0), np.cos(theta0)])
    p0 = args.range * np.array([np.cos(theta0), np.sin(theta0)])
    tl = np.stack([p0 - 40 * tan_dir, p0 + 40 * tan_dir])
    ax.plot(tl[:, 0], tl[:, 1], color="tab:red", lw=1.0, ls=":", label="straight-line approximation (tangent at the mean)")
    dots = ax.scatter([], [], s=2.5, color="tab:green", alpha=0.35, lw=0, label="true cloud (Monte Carlo samples)")
    ell_lin = [ax.plot([], [], color="tab:red", lw=1.8, ls="--")[0] for _ in range(2)]
    ell_lin[0].set_label("straight-line Gaussian: 1σ and 2σ ellipses")
    ell_smp = [ax.plot([], [], color="tab:blue", lw=1.8)[0] for _ in range(2)]
    ell_smp[0].set_label("actual spread of the cloud: 1σ and 2σ ellipses")
    mean_lin, = ax.plot([p0[0]], [p0[1]], "x", color="tab:red", ms=8, mew=2)
    mean_smp, = ax.plot([], [], "+", color="tab:blue", ms=10, mew=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x: straight ahead of the sensor (m)")
    ax.set_ylabel("y: sideways (m)")
    ax.set_title("Where the object may be (view zooms with σ)", fontsize=10.5)
    ax.grid(alpha=0.25)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.065), fontsize=8.5, ncol=3,
               frameon=False, markerscale=3)

    # --- right: the two metrics
    ax_f.set_yscale("log")
    ax_f.plot(sweep_deg, nees, color="tab:purple", lw=1.0, alpha=0.35)
    ax_f.axhline(2.0, color="black", lw=1.0)
    ax_f.text(sweep_deg[-1], 2.0 * 1.15, "2 = the straight-line Gaussian describes the cloud", ha="right", va="bottom", fontsize=8)
    ax_f.set_ylim(1.0, nees.max() * 2.0)
    ax_f.set_ylabel("Fit score (sample NEES)\nlower is better, 2 is perfect", fontsize=9)
    ax_f.set_title("How well the straight-line Gaussian describes the cloud", fontsize=10.5)
    fit_line, = ax_f.plot([], [], color="tab:purple", lw=2.2)
    fit_dot, = ax_f.plot([], [], "o", color="tab:purple", ms=8)

    ax_c.plot(sweep_deg, cov_err, color="tab:orange", lw=1.0, alpha=0.35)
    ax_c.set_ylim(0.0, cov_err.max() * 1.15)
    ax_c.set_ylabel("Spread error (%)\nlower is better", fontsize=9)
    ax_c.set_xlabel("Bearing uncertainty σ (degrees): how noisy the angle sensor is")
    ax_c.set_title("How wrong the straight-line covariance is (relative error)", fontsize=10.5)
    cov_line, = ax_c.plot([], [], color="tab:orange", lw=2.2)
    cov_dot, = ax_c.plot([], [], "o", color="tab:orange", ms=8)
    for a in (ax_f, ax_c):
        a.axvline(sigma_eq, color="0.5", lw=1.0, ls="--")
        a.grid(alpha=0.25)
        a.set_xlim(sweep_deg[0], sweep_deg[-1])
    ax_c.text(sigma_eq + 0.4, ax_c.get_ylim()[1] * 0.97, f"dashed line: bend = range noise (at {sigma_eq:.1f}°)",
              fontsize=7.5, color="0.35", va="top")
    plt.setp(ax_f.get_xticklabels(), visible=False)

    fig.text(0.5, 0.012,
             "Red x and dashed ellipses: the Kalman-filter family's straight-line (linearized) answer. Blue + and ellipses: the cloud's real mean and spread.\n"
             "Fit score: the average, over the cloud, of the squared distance to the red mean measured in red-ellipse units (2 if the red Gaussian were right).",
             ha="center", fontsize=8)

    n_sweep = len(sweep_deg)
    frames = list(range(n_sweep)) + [n_sweep - 1] * args.hold

    def update(k):
        c = cases[k]
        s_deg = sweep_deg[k]
        dots.set_offsets(c["dots"])
        for j, n_sig in enumerate((1.0, 2.0)):
            e = ellipse_points(c["mu_lin"], c["P_lin"], n_sig)
            ell_lin[j].set_data(e[:, 0], e[:, 1])
            e = ellipse_points(c["mu_s"], c["P_s"], n_sig)
            ell_smp[j].set_data(e[:, 0], e[:, 1])
        mean_smp.set_data([c["mu_s"][0]], [c["mu_s"][1]])
        half = min(2.6 * max(args.range * np.radians(s_deg), 2.0 * args.range_std), 11.0)
        ax.set_xlim(args.range - 0.9 * half - 0.5 * c["offset"], args.range + 0.9 * half)
        ax.set_ylim(-half, half)
        fit_line.set_data(sweep_deg[:k + 1], nees[:k + 1])
        fit_dot.set_data([s_deg], [nees[k]])
        cov_line.set_data(sweep_deg[:k + 1], cov_err[:k + 1])
        cov_dot.set_data([s_deg], [cov_err[k]])
        small = bend[k] < args.range_std
        status.set_text(f"Bearing uncertainty σ = {s_deg:.0f}°:  fit score {nees[k]:.1f} (2 = perfect)")
        sub.set_text(f"Within ±σ the curve leaves its straight tangent by {bend[k] * 100:.1f} cm; the range noise (cloud thickness) is "
                     f"{args.range_std * 100:.0f} cm → the bend is {'smaller' if small else 'larger'}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
