'''
Animated version of the "Tiny example" in docs/filtering/kf_ekf_iekf.md §1
(figure kf_update_concept): a 1D Kalman filter tracking a car's position over
6 predict / update cycles.

Panels:
  1. left: the filter's Gaussian belief about the car's position x. Each cycle
     has three beats. PREDICT: the previous estimate (grey-green) shifts by the
     5 m the car moves in 1 s and widens by the process noise Q (blue =
     prediction, variance P-). The GPS reading (orange, variance R) appears.
     UPDATE: the blend (green, variance P) is narrower than both. A dashed
     black line marks the true position;
  2. top right: this cycle's counters x, P-, K, P;
  3. bottom right: P- and P per cycle against the steady-state values.

Message: cycle 1 is the doc's example (prediction 15 m with variance 3, GPS
14 m with variance 7, K = 0.3, estimate 14.7 m, variance 2.1). Predict grows
the variance, update shrinks it, and after a few cycles the two balance: P-
settles near 3.19 and P near 2.19. The steady state depends only on Q and R,
not on the readings.

Data source: the scalar KF equations of the doc (F = 1, H = 1):
  x- = x + 5,  P- = P + Q,  K = P-/(P- + R),  x = x- + K (z - x-),  P = (1 - K) P-.
The repo has no scalar KF script, so the loop is written here and cycle 1 is
asserted against the doc's numbers (the same ones make_figures.py's
fig_kf_update_concept plots). The steady state P-* = (Q + sqrt(Q^2 + 4QR))/2
is computed in code and the last cycle is asserted to be within 0.02 of it.

Illustrative (labelled on every frame; not from the doc):
  - the car moves 5 m per cycle (the doc: "moving at 5 m/s", "after 1 second"),
    and the true car moves exactly that, so the model is right and Q only
    states the filter's modelling doubt;
  - Q = 1 and P0 = 2 (previous variance), chosen so P- of cycle 1 = 2 + 1 = 3,
    the doc's value; R = 7 from the doc, the previous estimate x0 = 10 m from
    the doc;
  - GPS readings of cycles 2-6 are drawn once with seed 0 from N(truth, R)
    (truth = 10 m + 5 m per cycle); cycle 1 uses the doc's 14 m, not a draw;
  - the shift / widen / blend motion between beats is interpolated for
    readability (only the end of each beat is a computed value).
Seed 0 by default; the draws are not tuned.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

TRUTH = "#1f1f1f"
PRIOR = "#9a9a9a"
C = {"blue": "#2e86de", "green": "#2a9d8f", "orange": "#f4a261"}

X0, P0 = 10.0, 2.0          # previous estimate (doc: car at 10 m); variance chosen so P- of cycle 1 = 3
MOVE, Q, R = 5.0, 1.0, 7.0  # m per cycle (doc: 5 m/s, 1 s), process noise (chosen), GPS variance (doc)
Z1 = 14.0                   # cycle 1 GPS reading (doc)
N_CYCLES = 6
BEATS = (4, 2, 4, 2)        # frames: predict, GPS appears, update, hold
PER_CYCLE = sum(BEATS)


def gauss_pdf(x, mean, var):
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def run_filter(seed):
    """Scalar KF over N_CYCLES. Returns per-cycle arrays; asserts cycle 1 against the doc."""
    rng = np.random.default_rng(seed)
    truth = X0 + MOVE * np.arange(1, N_CYCLES + 1)
    z = truth + np.sqrt(R) * rng.normal(size=N_CYCLES)
    z[0] = Z1                                    # cycle 1 is the doc's reading, not a draw
    x_prev, P_prev = X0, P0
    rows = []
    for k in range(N_CYCLES):
        x_pred, P_pred = x_prev + MOVE, P_prev + Q
        K = P_pred / (P_pred + R)
        x_post, P_post = x_pred + K * (z[k] - x_pred), (1 - K) * P_pred
        rows.append((x_prev, P_prev, x_pred, P_pred, K, x_post, P_post))
        x_prev, P_prev = x_post, P_post
    out = dict(zip(("x_prev", "P_prev", "x_pred", "P_pred", "K", "x_post", "P_post"), np.array(rows).T))
    out.update(z=z, truth=truth)
    # The doc's tiny example (also make_figures.py fig_kf_update_concept): 15 m / 3, GPS 14 m / 7.
    assert np.allclose([out[n][0] for n in ("x_pred", "P_pred", "K", "x_post", "P_post")], [15.0, 3.0, 0.3, 14.7, 2.1])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the GPS draws of cycles 2-6 (see the module docstring)")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", required=True, help="Output GIF path")
    args = parser.parse_args()

    d = run_filter(args.seed)
    P_pred_ss = (Q + np.sqrt(Q ** 2 + 4 * Q * R)) / 2     # steady-state P- for these Q, R
    P_post_ss = P_pred_ss - Q                              # P = P- - Q at the fixed point (P- = P + Q)
    assert abs(d["P_pred"][-1] - P_pred_ss) < 0.02 and abs(d["P_post"][-1] - P_post_ss) < 0.02

    n_frames = N_CYCLES * PER_CYCLE
    cycles = np.arange(1, N_CYCLES + 1)
    xs = np.linspace(0, 50, 1000)

    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], height_ratios=[1, 1.15],
                          left=0.07, right=0.97, top=0.83, bottom=0.27, hspace=0.55, wspace=0.22)
    ax = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])
    fig.suptitle("1D Kalman filter: each cycle predict (widen) then update (narrow)", fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.915, f"ILLUSTRATIVE except cycle 1 (the doc's example): car moves {MOVE:g} m per cycle, Q = {Q:g}, P0 = {P0:g}, "
             f"R = {R:g}, GPS of cycles 2-6 drawn with seed {args.seed}", ha="center", fontsize=9.5, color="#a02030")
    fig.text(0.5, 0.05, "Predict grows the variance by Q, update shrinks it by the factor 1 - K. After a few cycles the two balance: "
             f"P- settles near {P_pred_ss:.2f}, P near {P_post_ss:.2f}\n(a property of Q = {Q:g}, R = {R:g}, "
             "not of the readings). Motion between beats is interpolated; only the end of each beat is computed.",
             ha="center", va="center", fontsize=9.5)

    def update(f):
        k, i = divmod(f, PER_CYCLE)
        e0, e1, e2 = np.cumsum(BEATS)[:3]
        xp, Pp, xq, Pq, K, xu, Pu = (d[n][k] for n in ("x_prev", "P_prev", "x_pred", "P_pred", "K", "x_post", "P_post"))
        z, truth = d["z"][k], d["truth"][k]

        ax.clear()
        centre = xp + MOVE / 2
        s_pred = min((i + 1) / BEATS[0], 1.0)
        show_pred = True
        if i < e0:
            mean, var, phase = xp + MOVE * s_pred, Pp + Q * s_pred, f"PREDICT: shift by {MOVE:g} m, widen by Q = {Q:g}"
        elif i < e1:
            mean, var, phase = xq, Pq, "GPS reading arrives (variance R)"
        elif i < e2:
            phase = "UPDATE: blend prediction and GPS with the gain K"
        else:
            phase = "updated estimate (becomes the next previous estimate)"
        ax.fill_between(xs, gauss_pdf(xs, xp, Pp), color=C["green"], alpha=0.10 if i >= BEATS[0] else 0.2)
        ax.plot(xs, gauss_pdf(xs, xp, Pp), color=C["green"], lw=1.5, ls="--", alpha=0.6,
                label=f"previous estimate: {xp:.1f} m, variance {Pp:.2f}")
        if show_pred:
            if i >= e0:
                mean, var = xq, Pq
            ax.plot(xs, gauss_pdf(xs, mean, var), color=C["blue"], lw=2.4,
                    label=f"prediction: {mean:.1f} m, variance {var:.2f}")
            ax.fill_between(xs, gauss_pdf(xs, mean, var), color=C["blue"], alpha=0.12)
        if i >= e0:
            ax.plot(xs, gauss_pdf(xs, z, R), color=C["orange"], lw=2.2, label=f"GPS reading: {z:.1f} m, variance {R:g}")
            ax.fill_between(xs, gauss_pdf(xs, z, R), color=C["orange"], alpha=0.12)
        if i >= e1:
            s_up = min((i - e1 + 1) / BEATS[2], 1.0)
            m_up, v_up = xq + s_up * (xu - xq), Pq + s_up * (Pu - Pq)
            ax.plot(xs, gauss_pdf(xs, m_up, v_up), color=C["green"], lw=2.8,
                    label=f"update: {m_up:.1f} m, variance {v_up:.2f}")
            ax.fill_between(xs, gauss_pdf(xs, m_up, v_up), color=C["green"], alpha=0.25)
        ax.axvline(truth, color=TRUTH, lw=1.4, ls=":", label=f"true position: {truth:.1f} m")
        ax.set(xlim=(centre - 11, centre + 11), ylim=(0, 0.34), xlabel="car position x (m)", ylabel="probability density")
        ax.set_title(f"Cycle {k + 1} of {N_CYCLES}  -  {phase}", fontsize=10.5)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8.5)

        ax_t.clear()
        ax_t.axis("off")
        pred_done, gain_known, upd_done = i >= e0 - 1, i >= e0, i >= e2 - 1
        rows = [("cycle", f"{k + 1}  (doc's example)" if k == 0 else f"{k + 1}"),
                ("P- (prediction variance)", f"{Pq:.2f}" if pred_done else "-"),
                ("K (gain)", f"{K:.2f}" if gain_known else "-"),
                ("x (estimate, m)", f"{xu:.2f}" if upd_done else "-"),
                ("P (updated variance)", f"{Pu:.2f}" if upd_done else "-")]
        for j, (name, val) in enumerate(rows):
            ax_t.text(0.0, 0.92 - j * 0.2, name, fontsize=10, va="center", transform=ax_t.transAxes)
            ax_t.text(1.0, 0.92 - j * 0.2, val, fontsize=11, va="center", ha="right", fontweight="bold", transform=ax_t.transAxes)

        ax_p.clear()
        ax_p.axhline(P_pred_ss, color=C["blue"], lw=1, ls="--")
        ax_p.axhline(P_post_ss, color=C["green"], lw=1, ls="--")
        ax_p.text(N_CYCLES + 0.45, P_pred_ss, f"{P_pred_ss:.2f}", color=C["blue"], fontsize=8.5, va="bottom", ha="right")
        ax_p.text(N_CYCLES + 0.45, P_post_ss, f"{P_post_ss:.2f}", color=C["green"], fontsize=8.5, va="top", ha="right")
        n_pred = k + 1 if pred_done else k
        n_upd = k + 1 if upd_done else k
        ax_p.plot(cycles[:n_pred], d["P_pred"][:n_pred], "o-", color=C["blue"], label="P- (prediction)")
        ax_p.plot(cycles[:n_upd], d["P_post"][:n_upd], "o-", color=C["green"], label="P (update)")
        ax_p.set(xlim=(0.5, N_CYCLES + 0.5), ylim=(1.2, 3.5), xlabel="cycle", ylabel="variance",
                 title="Variance per cycle (dashed: steady state)", xticks=cycles)
        ax_p.title.set_fontsize(10)
        ax_p.legend(loc="lower center", fontsize=8, ncol=2)
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)

    print(f"KF cycles (seed {args.seed}; cycle 1 = doc's example, asserted):")
    for k in range(N_CYCLES):
        print(f"  cycle {k + 1}: x- {d['x_pred'][k]:.2f}, P- {d['P_pred'][k]:.3f}, z {d['z'][k]:.2f}, "
              f"K {d['K'][k]:.3f}, x {d['x_post'][k]:.2f}, P {d['P_post'][k]:.3f}, truth {d['truth'][k]:.1f}")
    print(f"steady state (Q = {Q:g}, R = {R:g}): P- {P_pred_ss:.3f}, P {P_post_ss:.3f}")
    print(f"Saved {n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
