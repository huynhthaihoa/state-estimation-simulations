'''
Animated version of fig_ekf_linearization_concept (generator/make_figures.py), the
figure ekf_linearization_concept.png of docs/foundations/jacobian.md section 6 and
docs/filtering/kf_ekf_iekf.md section 2: the unicycle's x-step per metre travelled,
f(theta) = cos(theta), linearized at the estimate theta_hat = 30 deg.

The heading uncertainty sigma sweeps from 1 deg to 40 deg (log-spaced, 41 values:
equal steps in sigma-ratio, so the small-sigma regime where the tangent is fine gets
as much screen time as the large-sigma one; 5 deg, the doc's "small" case, is one of
the values and is held for a moment), then holds at 40 deg.

Two panels:
  1. left: cos(theta) with its tangent at theta_hat and the band theta_hat +- sigma.
     The vertical segment at theta_hat + sigma is the gap between curve and tangent,
     its length printed in cm;
  2. right: Monte Carlo histogram of cos(theta) for theta ~ N(theta_hat, sigma^2)
     against the EKF's Gaussian N(cos theta_hat, (|sin theta_hat| sigma)^2), with the
     true Monte Carlo mean and the EKF mean marked and their difference printed.

Message: for small sigma the tangent is an excellent stand-in and the EKF's Gaussian
matches the histogram; as sigma grows the gap grows like sigma^2, the true mean slides
away from the EKF's mean (which stays at cos theta_hat), and the shape stops being a bell.

Data: pure math, no simulation, exactly the static figure's setup: 200 000 standard-
normal draws from default_rng(seed=0), theta = theta_hat + sigma * z, and the same
formulas for the tangent gap, the Monte Carlo mean/std and the EKF mean/std. The SAME
draws z are reused at every frame (common random numbers), scaled by that frame's
sigma, so the histogram morphs smoothly instead of flickering. The generator asserts
that the numbers shown at sigma = 5 deg and 40 deg reproduce the figure captions in the
docs (gap 0.3 cm / 17.5 cm; means 0.863 vs 0.866 m at 5 deg, 0.678 vs 0.866 m at 40 deg).

Illustrative: the right panel's x-axis window and bin width rescale every frame to
follow the distribution (so the tiny sigma = 1 deg histogram is visible); the density
axis rescales with it. The sweep itself (sigma moving) is an animation device, not a
process in time.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

TH_HAT = np.radians(30.0)
F0, SLOPE = np.cos(TH_HAT), -np.sin(TH_HAT)   # f(theta_hat) in m; slope in m per rad
N_SAMPLES = 200_000                           # as in make_figures.py
N_BINS = 80
TRUTH, PRIOR = "#1f1f1f", "#9a9a9a"
RED, BLUE, ORANGE, GOLD = "#d1495b", "#2e86de", "#f4a261", "#e9c46a"


def stats(z, sigma_deg):
    """Everything shown at one sigma: tangent gap (m) at theta_hat + sigma, Monte Carlo and EKF mean/std."""
    s = np.radians(sigma_deg)
    gap = float(np.cos(TH_HAT + s) - (F0 + SLOPE * s))       # negative: tangent above the curve
    samp = np.cos(TH_HAT + s * z)
    return dict(gap=gap, samp=samp, m_true=float(samp.mean()), s_true=float(samp.std()),
                m_ekf=float(F0), s_ekf=float(abs(SLOPE) * s))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed of the fixed draws (default 0, as make_figures.py)")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", required=True, help="Output GIF path")
    args = parser.parse_args()

    z = np.random.default_rng(args.seed).standard_normal(N_SAMPLES)

    # verification against the docs' captions (only meaningful for the default seed 0)
    if args.seed == 0:
        a, b = stats(z, 5.0), stats(z, 40.0)
        assert f"{abs(a['gap']) * 100:.1f}" == "0.3" and f"{abs(b['gap']) * 100:.1f}" == "17.5"
        assert f"{a['m_true']:.3f}" == "0.863" and f"{a['m_ekf']:.3f}" == "0.866"
        assert f"{b['m_true']:.3f}" == "0.678" and f"{b['m_ekf']:.3f}" == "0.866"

    sweep = np.unique(np.append(np.geomspace(1.0, 40.0, 40), 5.0))     # 41 values, 5 deg included
    i5 = int(np.argmin(np.abs(sweep - 5.0)))
    sigmas = np.concatenate([sweep[:i5 + 1], np.full(8, 5.0), sweep[i5 + 1:], np.full(20, 40.0)])
    n_frames = len(sigmas)

    deg = np.linspace(-30, 120, 600)
    fig = plt.figure(figsize=(13, 6.6))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.30, wspace=0.18)
    ax, ar = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    for a_ in (ax, ar):
        a_.grid(alpha=0.25)
        a_.spines[["top", "right"]].set_visible(False)

    # left panel: static curve + tangent, dynamic band / gap
    band = ax.axvspan(29, 31, color=BLUE, alpha=0.25, lw=0, label="band θ̂ ± σ")
    ax.plot(deg, np.cos(np.radians(deg)), color=TRUTH, lw=2.2, label="true f(θ) = cos θ")
    ax.plot(deg, F0 + SLOPE * (np.radians(deg) - TH_HAT), color=RED, lw=1.8, ls="--",
            label="EKF tangent: cos θ̂ − sin θ̂·(θ − θ̂)")
    ax.scatter([30], [F0], s=70, color=RED, zorder=4, label=f"estimate θ̂ = 30°, f = {F0:.3f} m")
    gap_line, = ax.plot([], [], color=ORANGE, lw=4, zorder=3, solid_capstyle="butt", label="gap at θ̂ + σ")
    gap_txt = ax.text(0.97, 0.97, "", transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#b8672a",
                      fontweight="bold")
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set(xlim=(-30, 120), ylim=(-0.65, 1.2), xlabel="heading θ (deg)", ylabel="x-step f(θ) (m)")
    ax.set_xticks(range(-30, 121, 30))
    ax.set_title("Linearize at the estimate", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9, frameon=False)

    # right panel: histogram bars + Gaussian + the two means
    edges0 = np.linspace(0, 1, N_BINS + 1)
    bars = ar.bar(edges0[:-1], np.zeros(N_BINS), width=np.diff(edges0), align="edge", color=GOLD, alpha=0.85,
                  label="true (Monte Carlo)")
    gline, = ar.plot([], [], color=RED, lw=2, label="EKF Gaussian")
    mt_line = ar.axvline(0, color=TRUTH, lw=1.4, ls=":", label="true mean")
    ar.axvline(F0, color=RED, lw=1.4, ls=":", label="EKF mean = cos θ̂")
    mean_txt = ar.text(0.03, 0.97, "", transform=ar.transAxes, ha="left", va="top", fontsize=10.5, color="#333333",
                       linespacing=1.5)
    ar.set(xlabel="x-step f(θ) (m)", ylabel="density (1/m)")
    ar_title = ar.set_title("", fontweight="bold")
    ar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9, frameon=False)

    fig.suptitle("EKF linearization: the tangent is fine for small σ and fails for large σ",
                            fontsize=15, fontweight="bold", x=0.01, ha="left", y=0.985)
    fig.text(0.5, 0.012,
             "f(θ) = cos θ, the unicycle's x-step per metre travelled; θ ~ N(30°, σ²), σ swept 1° to 40° (log steps). "
             "Same 200 000 draws (seed 0) at every σ.\n"
             "σ = 5°: gap 0.3 cm, means 0.863 vs 0.866 m. σ = 40°: gap 17.5 cm, true mean 0.678 m vs the EKF's 0.866 m. "
             "Right-panel axes rescale each frame (illustrative).",
             ha="center", va="bottom", fontsize=9.5, color="#333333")

    cache = {}

    def update(i):
        sd = float(sigmas[i])
        if sd not in cache:
            cache.clear()
            cache[sd] = stats(z, sd)
        st = cache[sd]
        s = np.radians(sd)
        # left
        band.set_x(30 - sd)
        band.set_width(2 * sd)
        x = 30 + sd
        gap_line.set_data([x, x], [np.cos(np.radians(x)), F0 + SLOPE * s])
        gap_txt.set_text(f"σ = {sd:.1f}°\ngap at θ̂ + σ: {abs(st['gap']) * 100:.1f} cm\n(tangent above the curve)")
        # right: window follows the distribution
        samp = st["samp"]
        lo = max(-1.05, min(F0 - 4 * st["s_ekf"], float(np.quantile(samp, 0.001))) - 0.01)
        hi = min(1.05, F0 + 4 * st["s_ekf"])
        edges = np.linspace(lo, hi, N_BINS + 1)
        dens = np.histogram(samp, bins=edges)[0] / (N_SAMPLES * np.diff(edges))   # density over ALL samples
        for r, e0, w, h in zip(bars, edges[:-1], np.diff(edges), dens):
            r.set_x(e0)
            r.set_width(w)
            r.set_height(h)
        xs = np.linspace(lo, hi, 500)
        gauss = np.exp(-0.5 * ((xs - st["m_ekf"]) / st["s_ekf"]) ** 2) / (st["s_ekf"] * np.sqrt(2 * np.pi))
        gline.set_data(xs, gauss)
        mt_line.set_xdata([st["m_true"]] * 2)
        ar.set_xlim(lo, hi)
        ar.set_ylim(0, 1.3 * max(dens.max(), gauss.max()))
        ar_title.set_text(f"θ ~ N(30°, ({sd:.1f}°)²): output of f")
        mean_txt.set_text(f"true mean {st['m_true']:.3f} m, std {st['s_true']:.3f} m\n"
                          f"EKF mean {st['m_ekf']:.3f} m, std {st['s_ekf']:.3f} m\n"
                          f"mean off by {abs(st['m_true'] - st['m_ekf']) * 100:.1f} cm")
        return ()

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)

    for sd in (5.0, 40.0):
        st = stats(z, sd)
        print(f"sigma={sd:.0f} deg: gap {abs(st['gap']) * 100:.2f} cm | true mean {st['m_true']:.4f} std {st['s_true']:.4f}"
              f" | EKF mean {st['m_ekf']:.4f} std {st['s_ekf']:.4f}")
    print(f"{n_frames} frames, saved {args.out}")


if __name__ == "__main__":
    main()
