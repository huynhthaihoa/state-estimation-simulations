'''
Animated version of imu_allan_variance.py (Part 1): the Allan deviation of a
motionless gyroscope axis and accelerometer axis built up as the recording
grows, with the EuRoC ADIS16448 noise values of the script's preset.

Three columns, one row per sensor, the record time T growing on a log scale:
  1. the raw readings recorded so far (white noise on top of a slowly wandering
     bias; the bias is hard to see by eye);
  2. the Allan deviation of the first T seconds (cluster times up to T / 9)
     against the model curve sqrt(N^2/tau + K^2 tau/3) (dashed), with the
     slope -1/2 and +1/2 fit lines and their read-offs at 1 s and 3 s once the
     automatic fit finds a region;
  3. the read-off N_hat / N and K_hat / K against T (1 = exact).

Message (doc docs/foundations/noise_identification.md, results section): the
white-noise density is read from the first minute, the random walk only once
the curve has turned upward and follows slope +1/2 over several points; the
script prints the record length at which that first happens for each sensor
(the gyroscope needs a much longer record than the accelerometer).

The records are the script's own simulate_static_axis outputs; each frame's
curve and fit are allan_deviation and fit_noise_parameters on the first T
seconds. Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from state_estimation_simulations.use_numpy.imu_allan_variance import (
    PRESETS, SENSORS, UNITS, allan_deviation, cluster_sizes, fit_noise_parameters, model_allan_deviation,
    simulate_static_axis)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=1.0, help="Final record length (h)")
    parser.add_argument("--start-seconds", type=float, default=60.0, help="Shortest record shown (s)")
    parser.add_argument("--n-frames", type=int, default=45, help="Record lengths (log-spaced) = GIF frames")
    parser.add_argument("--per-decade", type=int, default=12, help="Cluster sizes per decade of tau")
    parser.add_argument("--slope-band", type=float, default=0.1, help="Local-slope window for the fit regions")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--fps", type=int, default=8, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=60, help="GIF resolution")
    parser.add_argument("--out", type=str, default="imu_allan_variance.gif", help="Output GIF path")
    args = parser.parse_args()

    preset = PRESETS["euroc"]
    dt = 1.0 / preset["rate_hz"]
    n_max = int(round(args.hours * 3600.0 / dt))
    t_axis = np.arange(1, n_max + 1) * dt
    lengths = np.unique(np.round(np.geomspace(args.start_seconds / dt, n_max, args.n_frames)).astype(int))
    root = np.random.SeedSequence(args.seed).spawn(2 * len(SENSORS))
    y = {s: simulate_static_axis(n_max, dt, preset[s]["density"], preset[s]["walk"],
                                 np.random.default_rng(root[2 * j]), np.random.default_rng(root[2 * j + 1]))
         for j, s in enumerate(SENSORS)}

    curves = {s: [] for s in SENSORS}
    for n in lengths:
        for s in SENSORS:
            tau, adev = allan_deviation(y[s][:n], dt, cluster_sizes(n, args.per_decade))
            curves[s].append((tau, adev, fit_noise_parameters(tau, adev, args.slope_band)))
    print(f"{len(lengths)} frames, records from {lengths[0] * dt:.0f} s to {lengths[-1] * dt / 3600.0:.2f} h")
    first_k = {}
    for s in SENSORS:
        first = next((n for n, c in zip(lengths, curves[s]) if not np.isnan(c[2]["walk"])), None)
        first_k[s] = first
        print(f"  {s}: K first readable at T = {'never' if first is None else f'{first * dt:.0f} s'}; final "
              f"N_hat/N = {curves[s][-1][2]['density'] / preset[s]['density']:.3f}, "
              f"K_hat/K = {curves[s][-1][2]['walk'] / preset[s]['walk']:.3f}")

    show = max(1, n_max // 3000)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), gridspec_kw=dict(width_ratios=[1.0, 1.25, 1.0]))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.09, wspace=0.28, hspace=0.32)
    fig.suptitle("Allan deviation of a static IMU built up as the record grows (EuRoC ADIS16448 values)", fontsize=12)
    status = fig.text(0.5, 0.925, "", ha="center", fontsize=12, fontweight="bold")
    lines = {}
    for row, s in enumerate(SENSORS):
        p = preset[s]
        ax_raw, ax_ad, ax_err = axes[row]
        ax_raw.set_xlim(0.0, t_axis[-1] / 60.0)
        ax_raw.set_xlabel("record time (min)")
        ax_raw.set_ylabel(f"{s} reading ({UNITS[s]['adev']})")
        ax_raw.set_title(f"{s}: raw readings", fontsize=9)
        lines[s, "raw"], = ax_raw.plot([], [], lw=0.4, color="tab:blue")
        tau_all = np.logspace(np.log10(dt), np.log10(n_max * dt / 9.0), 200)
        ax_ad.loglog(tau_all, model_allan_deviation(tau_all, p["density"], p["walk"]), "k--", lw=1, label="model")
        ax_ad.set_xlim(dt, n_max * dt / 9.0 * 1.2)
        ax_ad.set_ylim(0.6 * model_allan_deviation(tau_all, p["density"], p["walk"]).min(),
                       1.6 * model_allan_deviation(tau_all, p["density"], p["walk"]).max())
        ax_ad.set_xlabel(r"cluster time $\tau$ (s)")
        ax_ad.set_ylabel(f"Allan deviation ({UNITS[s]['adev']})")
        ax_ad.set_title(f"{s}: Allan deviation of the record so far", fontsize=9)
        ax_ad.grid(True, which="both", alpha=0.3)
        lines[s, "ad"], = ax_ad.loglog([], [], "o-", ms=3, color="tab:blue", label="simulated")
        lines[s, "wfit"], = ax_ad.loglog([], [], color="tab:green", lw=1.6, label="slope -1/2 fit")
        lines[s, "kfit"], = ax_ad.loglog([], [], color="tab:red", lw=1.6, label="slope +1/2 fit")
        lines[s, "wpt"], = ax_ad.loglog([], [], "s", color="tab:green", ms=8)
        lines[s, "kpt"], = ax_ad.loglog([], [], "s", color="tab:red", ms=8)
        ax_ad.legend(fontsize=7, loc="upper right")
        ax_err.set_xscale("log")
        ax_err.set_xlim(lengths[0] * dt, lengths[-1] * dt)
        ax_err.set_ylim(0.5, 1.6)
        ax_err.axhline(1.0, color="k", lw=1)
        ax_err.set_xlabel("record length T (s)")
        ax_err.set_ylabel("estimate / true")
        ax_err.set_title(f"{s}: read-off against the truth (1 = exact)", fontsize=9)
        ax_err.grid(True, which="both", alpha=0.3)
        lines[s, "N"], = ax_err.plot([], [], "o-", ms=3, color="tab:green", label=r"$\hat N / N$")
        lines[s, "K"], = ax_err.plot([], [], "o-", ms=3, color="tab:red", label=r"$\hat K / K$")
        ax_err.legend(fontsize=8, loc="upper right")
    when = {s: "never" if first_k[s] is None else f"{first_k[s] * dt:,.0f} s" for s in SENSORS}
    fig.text(0.5, 0.012,
             "N is readable from the first minute. K appears once the curve has turned upward and follows slope +1/2 "
             f"over at least 4 points: in this run after T = {when['accel']} (accelerometer) and {when['gyro']} (gyroscope).",
             ha="center", fontsize=8.5)

    def update(i):
        n = lengths[i]
        status.set_text(f"record length T = {n * dt:,.0f} s  ({n * dt / 3600.0:.2f} h)")
        for s in SENSORS:
            tau, adev, fit = curves[s][i]
            lines[s, "raw"].set_data(t_axis[:n:show] / 60.0, y[s][:n:show])
            lo, hi = y[s][:n].min(), y[s][:n].max()
            axes[SENSORS.index(s), 0].set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            lines[s, "ad"].set_data(tau, adev)
            if fit["white_range"] is not None:
                t = np.geomspace(fit["white_range"][0], max(1.0, fit["white_range"][1]), 30)
                lines[s, "wfit"].set_data(t, fit["density"] / np.sqrt(t))
                lines[s, "wpt"].set_data([1.0], [fit["density"]])
            else:
                lines[s, "wfit"].set_data([], [])
                lines[s, "wpt"].set_data([], [])
            if fit["walk_range"] is not None:
                t = np.geomspace(min(3.0, fit["walk_range"][0]), fit["walk_range"][1], 30)
                lines[s, "kfit"].set_data(t, fit["walk"] * np.sqrt(t / 3.0))
                lines[s, "kpt"].set_data([3.0], [fit["walk"]])
            else:
                lines[s, "kfit"].set_data([], [])
                lines[s, "kpt"].set_data([], [])
            T = lengths[:i + 1] * dt
            nn = np.array([curves[s][j][2]["density"] for j in range(i + 1)]) / preset[s]["density"]
            kk = np.array([curves[s][j][2]["walk"] for j in range(i + 1)]) / preset[s]["walk"]
            lines[s, "N"].set_data(T, nn)
            lines[s, "K"].set_data(T, kk)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(lengths), interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(lengths)} frames to {args.out}")


if __name__ == "__main__":
    main()
