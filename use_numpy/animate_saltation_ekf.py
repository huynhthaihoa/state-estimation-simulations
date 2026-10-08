'''
Animated version of saltation_matrix_ekf_v2.py's default run: a point mass
dropped from 5 m bounces on the ground while two EKFs track it from noisy
position measurements. The filters differ only in how they carry the
covariance through a bounce (naive: the reset map's own Jacobian; saltation:
the saltation matrix, see docs/filtering/hybrid_saltation_ekf.md).

Their estimates stay close, so the animation shows what actually differs, the
uncertainty:
  1. side view: ground truth, measurements and both estimates;
  2. zoomed error view: each filter's error in height and vertical velocity,
     with its 2-sigma ellipse (a consistent filter's error usually sits inside);
  3. the vertical-velocity standard deviation of each filter over time.

At a bounce the saltation update scales the height (guard-normal) variance by
e^2 and hands a smaller vertical-velocity variance to the next tick; the naive
update carries the height variance through unchanged. Neither is claimed
better here: doc §6 shows both are slightly underconfident with exact contact
detection.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse

from saltation_matrix_ekf_v2 import HybridEKF, generate_ground_truth_and_data, detect_bounce_ticks


def run_filters(x_true, z, x_init, P_init, dt, e, g, accel_noise_std, impact_noise_std, R):
    """Steps a naive and a saltation-corrected HybridEKF in lockstep over the
    same measurements.
    Returns:
        dict use_saltation -> (x_hist (N,6), P_hist (N,6,6)), including the
        initial state at index 0
    """
    out = {}
    for use_saltation in (False, True):
        ekf = HybridEKF(x_init, P_init, dt, e, g, accel_noise_std, impact_noise_std, R, use_saltation)
        x, P = ekf.state()
        xs, Ps = [x], [P]
        for k in range(len(x_true) - 1):
            ekf.step(z[k + 1])
            x, P = ekf.state()
            xs.append(x)
            Ps.append(P)
        out[use_saltation] = (np.array(xs), np.array(Ps))
    return out


def error_ellipse(P, n_sigma=2.0):
    """n-sigma ellipse of the (height, vertical velocity) block of P, with
    height in cm and velocity in m/s (the error view's axis units).
    Returns:
        width, height, angle_deg for matplotlib.patches.Ellipse
    """
    D = np.diag([100.0, 1.0])
    C = D @ P[np.ix_([2, 5], [2, 5])] @ D
    eigval, eigvec = np.linalg.eigh(C)
    angle = np.degrees(np.arctan2(eigvec[1, 1], eigvec[0, 1]))
    width, height = 2.0 * n_sigma * np.sqrt(eigval[1]), 2.0 * n_sigma * np.sqrt(eigval[0])
    return width, height, angle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=5.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Filter/measurement step interval in seconds")
    parser.add_argument("--restitution", type=float, default=0.85, help="Coefficient of restitution e")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration (m/s^2)")
    parser.add_argument("--drop-height", type=float, default=5.0, help="Initial height (m)")
    parser.add_argument("--init-horizontal-vel", type=float, nargs=2, default=[1.0, 0.5],
                        help="Initial horizontal velocity (vx, vy) in m/s")
    parser.add_argument("--pos-noise-std", type=float, default=0.03, help="Position measurement noise std-dev (m)")
    parser.add_argument("--process-noise-std", type=float, default=0.3,
                        help="Assumed acceleration-disturbance noise std-dev (m/s^2)")
    parser.add_argument("--impact-noise-std", type=float, default=0.005,
                        help="Regularizing covariance floor added at each bounce (see step_hybrid)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.1, help="Initial position error std-dev (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.2, help="Initial velocity error std-dev (m/s)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second (one frame per filter tick)")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, default="saltation_matrix_ekf.gif", help="Output GIF path")
    args = parser.parse_args()

    e, g, dt = args.restitution, args.gravity, args.dt
    rng = np.random.default_rng(args.seed)
    x_true, z = generate_ground_truth_and_data(args.duration, dt, e, g, args.drop_height,
                                               args.init_horizontal_vel, args.pos_noise_std, rng)
    x_true = np.array(x_true)
    init_std = np.array([args.init_pos_noise_std] * 3 + [args.init_vel_noise_std] * 3)
    x_init = x_true[0] + rng.normal(0.0, 1.0, 6) * init_std
    P_init = np.diag(init_std ** 2)
    R = args.pos_noise_std ** 2 * np.eye(3)

    hist = run_filters(x_true, z, x_init, P_init, dt, e, g, args.process_noise_std,
                       args.impact_noise_std, R)
    bounces = detect_bounce_ticks(list(x_true))
    t = np.arange(len(x_true)) * dt
    dist = lambda X: np.hypot(X[:, 0], X[:, 1])  # horizontal distance travelled from the start

    styles = {False: dict(color="tab:orange", label="EKF, naive bounce update"),
              True: dict(color="tab:blue", label="EKF, saltation bounce update")}

    # Fixed axis limits for the error and sigma panels, ignoring the first
    # ticks (the initial-guess transient would otherwise dwarf everything) and,
    # for the error view, the few most extreme ticks (points beyond the limits
    # are pinned to the edge).
    k_skip, k_settled = 10, 25
    err, sig = {}, {}
    for s, (X, P) in hist.items():
        err[s] = np.stack([(X[:, 2] - x_true[:, 2]) * 100.0, X[:, 5] - x_true[:, 5]], axis=1)
        sig[s] = np.sqrt(P[:, [2, 5], [2, 5]]) * np.array([100.0, 1.0])
    lim = 1.15 * np.max([np.percentile(np.abs(err[s][k_skip:]) + 2.0 * sig[s][k_skip:], 95, axis=0)
                         for s in hist], axis=0)
    sig_max = 1.4 * max(sig[s][k_settled:, 1].max() for s in hist)

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.38, wspace=0.28)
    ax_side = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_sig = fig.add_subplot(gs[1, 1])

    # --- side view ---
    ax_side.axhline(0.0, color="saddlebrown", linewidth=3)
    ax_side.plot(dist(x_true), x_true[:, 2], color="lightgray", linewidth=1, zorder=0)
    ax_side.set_xlim(-0.1, dist(x_true).max() + 0.2)
    ax_side.set_ylim(-0.3, args.drop_height + 0.4)
    ax_side.set_xlabel("Horizontal distance (m)")
    ax_side.set_ylabel("Height (m)")
    meas_dots, = ax_side.plot([], [], ".", color="gray", markersize=4, alpha=0.6, label="Measurements")
    truth_ball, = ax_side.plot([], [], "o", color="green", markersize=13, label="Ground truth")
    est_marks = {s: ax_side.plot([], [], "x", markersize=9, mew=2, **styles[s])[0] for s in hist}
    bounce_text = ax_side.text(0.5, 0.9, "", transform=ax_side.transAxes, ha="center",
                               fontsize=13, color="crimson", fontweight="bold")
    title = ax_side.set_title("")
    ax_side.legend(loc="upper right", fontsize=8)

    # --- zoomed error view ---
    ax_err.axhline(0.0, color="black", linewidth=0.5)
    ax_err.axvline(0.0, color="black", linewidth=0.5)
    ax_err.set_xlim(-lim[0], lim[0])
    ax_err.set_ylim(-lim[1], lim[1])
    ax_err.set_xlabel("Height error (cm)")
    ax_err.set_ylabel("Vertical-velocity error (m/s)")
    ax_err.set_title("Error and 2σ ellipse (truth at the origin)", fontsize=10)
    ellipses = {s: Ellipse((0, 0), 0, 0, fill=False, linewidth=2, color=styles[s]["color"]) for s in hist}
    for s in hist:
        ax_err.add_patch(ellipses[s])
    err_dots = {s: ax_err.plot([], [], "o", color=styles[s]["color"], markersize=6)[0] for s in hist}

    # --- vertical-velocity sigma over time ---
    for k in bounces:
        ax_sig.axvline(t[k], color="crimson", linestyle=":", linewidth=1)
    ax_sig.set_xlim(0.0, t[-1])
    ax_sig.set_ylim(0.0, sig_max)
    ax_sig.set_xlabel("Time (s)")
    ax_sig.set_ylabel("σ of vertical velocity (m/s)")
    ax_sig.set_title("Reported uncertainty (dotted: bounces)", fontsize=10)
    sig_lines = {s: ax_sig.plot([], [], linewidth=2, color=styles[s]["color"])[0] for s in hist}

    fig.text(0.5, 0.01, f"At a bounce the saltation update scales the height variance by e² = {e ** 2:.2f} "
             "and hands on a smaller vertical-velocity variance; the naive update carries the height variance through.",
             ha="center", fontsize=8.5, wrap=True)

    n_trail = 15

    def update(k):
        title.set_text(f"Bouncing point mass, two EKFs   t = {t[k]:.2f} s")
        lo = max(0, k - n_trail)
        meas_dots.set_data(np.hypot(z[lo:k + 1, 0], z[lo:k + 1, 1]), z[lo:k + 1, 2])
        truth_ball.set_data([dist(x_true[k:k + 1])[0]], [x_true[k, 2]])
        near_bounce = any(0 <= k - b < 6 for b in bounces)
        bounce_text.set_text("BOUNCE" if near_bounce else "")
        for s, (X, P) in hist.items():
            est_marks[s].set_data([dist(X[k:k + 1])[0]], [X[k, 2]])
            w, h, a = error_ellipse(P[k])
            ellipses[s].set_center(tuple(np.clip(err[s][k], -lim, lim)))
            ellipses[s].set_width(w)
            ellipses[s].set_height(h)
            ellipses[s].set_angle(a)
            err_dots[s].set_data([np.clip(err[s][k, 0], -lim[0], lim[0])], [np.clip(err[s][k, 1], -lim[1], lim[1])])
            sig_lines[s].set_data(t[:k + 1], sig[s][:k + 1, 1])
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(x_true), interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(x_true)} frames to {args.out}")


if __name__ == "__main__":
    main()
