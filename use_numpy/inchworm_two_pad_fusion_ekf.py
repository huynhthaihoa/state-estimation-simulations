'''
Follow-up to inchworm_two_pad_ekf.py, whose filter reads only the anchored
pad's optical sensor (as on the inchworm robot of Kubik & Faigl, CTU, ICRA
2026) and gives the moving pad no motion model. Here the moving pad's sensor
is read too, the moving pad can be predicted from the commanded stroke, and
the sensors can fail. The numbers are not that robot's, and the filter
design is this script's, not the paper's.

Ground truth: the gait, turning, micro-slip and slip pulses of
inchworm_two_pad_ekf.py, plus

  - stroke error: in each phase the body executes the commanded stroke
    rate times a gain 1 + N(0, stroke_err_std^2), so the command is not
    exact (the encoders still measure the real length). With
    `--stroke-err-per-cycle`, the contract phase reuses the extend phase's
    gain, so the body returns to its starting length every cycle.
  - sensor faults: each pad's sensor, in each phase, has an episode of
    length `t_fault` with probability `p_fault`, either a spike (the
    reading gains a backward displacement at speed `v_spike`, like a slip;
    with `--random-spike-speed`, a speed in [0.5, 1.5] * v_spike and a
    random direction) or a dropout (the reading is 0, as when the sensor
    loses the surface).

When the anchored pad slips, the whole body slides with it: both pads move
by the slip, and the distance between them does not change. So a real slip
makes the two sensors agree, and a disagreement points to a faulty sensor.

Sensors, every tick: gyro; both pad sensors (body frame, white noise; the
anchored pad's reading d_a, the moving pad's d_m); kinematic closure from
the encoders (front pad relative to back pad, body frame).

Filter state x = [b_x, b_y, f_x, f_y, theta], as in inchworm_two_pad_ekf.py.
Predict: theta integrates the gyro; each pad moves by a body-frame increment
rotated to the world frame. The anchored pad always moves by d_a (the
`pad_fused` policy). The moving pad, per variant:

  - random_walk: no motion model, a loose random walk (`move_noise_std` per
    tick); identical to inchworm_two_pad_ekf.py's pad_fused.
  - encoder_model: d_a + s * e, where e is the commanded change of the
    front-minus-back vector over the tick, in the body frame (stroke and
    turn), and s = +1 if the front pad moves, -1 if the back pad moves.
  - moving_sensor: d_m.
  - both: the inverse-variance combination of d_m and d_a + s * e.
  - cross_check: as `both`, after a disagreement test on
    r = d_m - (d_a + s * e). If r is too large, one sensor is treated as
    faulty: the one whose reading implies the larger motion of the anchored
    pad (d_a for the anchored sensor, d_m - s * e for the moving one, each
    normalized by its noise), i.e. the one that disagrees more with "the
    anchored pad is still". If that is the anchored sensor, the anchored pad
    moves by d_m - s * e instead of d_a; otherwise the moving pad follows the
    encoder model. Under the two hypotheses both pads' increments differ by
    exactly r, so on a disagreement r r^T is added to their covariances (and
    cross-covariance): the filter admits it may have blamed the wrong sensor
    (`--no-cross-check-inflate` turns this off).

Update: kinematic closure, as in inchworm_two_pad_ekf.py.

Slip flags (one per tick): `threshold` flags |d_a| above the slip threshold
(the paper's detector); `cross_check` flags the anchored increment it used.

Monte Carlo: slip pulses are drawn once per seed and held fixed; stroke
gains, faults and all noise are redrawn every trial.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import measure_performance, averaged_nees_bounds, nees  # noqa: E402
from inchworm_two_pad_ekf import (STATE_DIM, B, F, TH, rot, drot, trapezoid, back_is_anchored,  # noqa: E402
                                  draw_slip_episodes, slip_speed, update_closure, state_error,
                                  window_flags)

VARIANTS = ("random_walk", "encoder_model", "moving_sensor", "both", "cross_check")
DETECTORS = ("threshold", "cross_check")
PADS = ("back", "front")
NO_FAULT, SPIKE, DROPOUT = 0, 1, 2


def draw_stroke_gains(n_phases, stroke_err_std, rng, per_cycle=False):
    """Per-phase gain on the commanded stroke rate: 1 + N(0, stroke_err_std^2).
    With per_cycle, each contract phase reuses the gain of the extend before
    it, so the body returns to its starting length every cycle (same draws,
    every other one unused)."""
    gains = 1.0 + stroke_err_std * rng.normal(size=n_phases)
    if per_cycle:
        gains[1::2] = gains[0::2][:len(gains[1::2])]
    return gains


def draw_faults(n_phases, t_phase, t_fault, p_fault, rng, fault_type="both", v_spike=0.08,
                random_spike_speed=False):
    """Draws sensor-fault episodes: for each phase and each pad sensor, one
    episode with probability p_fault, starting uniformly so it fits inside the
    phase. With fault_type "both" it is a spike or a dropout with equal
    probability; "spike" or "dropout" forces one kind. A spike's spurious
    velocity is backward along the body at v_spike, or, with
    random_spike_speed, has a speed drawn uniformly in [0.5, 1.5] * v_spike
    and a uniformly random direction. Always draws three numbers per phase and
    pad so the rng stream depends on neither p_fault nor fault_type; with
    random_spike_speed, each spike draws two more (the stream then depends on
    p_fault).
    Returns:
        list of (pad, kind, t_start, t_end, v), pad in PADS, kind SPIKE or DROPOUT, v the spike's
        spurious velocity in the body frame (2,) (m/s), None for a dropout
    """
    faults = []
    for j in range(n_phases):
        for pad in PADS:
            has, kind, offset = rng.random(), rng.random(), rng.random()
            if has < p_fault:
                start = j * t_phase + offset * (t_phase - t_fault)
                spike = kind < 0.5 if fault_type == "both" else fault_type == "spike"
                v = None
                if spike:
                    v = np.array([-v_spike, 0.0])
                    if random_spike_speed:
                        speed, ang = v_spike * (0.5 + rng.random()), 2.0 * np.pi * rng.random()
                        v = speed * np.array([np.cos(ang), np.sin(ang)])
                faults.append((pad, SPIKE if spike else DROPOUT, start, start + t_fault, v))
    return faults


def fault_at(t, faults, pad):
    """Fault on `pad`'s sensor at time t: (kind, spike velocity), (NO_FAULT, None) if none."""
    for p, kind, a, b, v in faults:
        if p == pad and a <= t < b:
            return kind, v
    return NO_FAULT, None


def simulate(n_steps, dt, t_phase, t_ramp, v_stroke, l_min, steer, episodes, v_slip, gains, faults,
             micro_slip_std, gyro_noise_std, sensor_noise_std, moving_sensor_noise_std, closure_noise_std,
             closure_lat_noise_std, rng):
    """Builds one true trajectory and its sensor readings, as
    inchworm_two_pad_ekf.simulate plus the stroke gains, the moving pad's
    sensor and the sensor faults.
    Arguments:
        n_steps, dt, t_phase, t_ramp, v_stroke, l_min, steer, episodes, v_slip: as in
            inchworm_two_pad_ekf.simulate
        gains: (n_phases,) stroke-rate gain per phase
        faults: sensor-fault episodes from draw_faults
        micro_slip_std, gyro_noise_std: as in inchworm_two_pad_ekf.simulate
        sensor_noise_std: noise of a pad sensor while its pad is anchored (m per reading and axis)
        moving_sensor_noise_std: noise of a pad sensor while its pad moves (m per reading and axis)
        closure_noise_std, closure_lat_noise_std: closure noise along / across the body (m)
        rng: numpy random number generator
    Returns:
        x_true: (N, 5) true [b, f, theta]
        gyro: (n_steps,) turn-rate readings
        d_a, d_m: (n_steps, 2) anchored / moving pad readings, body frame
        z: (N, 2) closure readings, body frame
        back_anchored: (n_steps,) bool
        e_cmd: (n_steps, 2) commanded stroke [dl, dtheta] per tick (dtheta 0 in contract)
        fault_a, fault_m: (n_steps,) fault kind on the anchored / moving pad's reading
    """
    x = np.zeros((n_steps + 1, STATE_DIM))
    x[0, F] = [l_min, 0.0]
    length = l_min
    gyro = np.zeros(n_steps)
    d_a, d_m = np.zeros((n_steps, 2)), np.zeros((n_steps, 2))
    e_cmd = np.zeros((n_steps, 2))
    back_anchored = np.zeros(n_steps, dtype=bool)
    fault_a, fault_m = np.zeros(n_steps, dtype=int), np.zeros(n_steps, dtype=int)
    for k in range(n_steps):
        t0 = k * dt
        phase = int(np.floor(t0 / t_phase + 1e-9))
        tau0, tau1 = t0 - phase * t_phase, t0 + dt - phase * t_phase
        back = back_is_anchored(phase)
        sign = 1.0 if back else -1.0
        dl_cmd = sign * 0.5 * (trapezoid(tau0, t_phase, t_ramp, v_stroke)
                               + trapezoid(tau1, t_phase, t_ramp, v_stroke)) * dt
        dth_cmd = dl_cmd * np.tan(steer) / l_min if back else 0.0
        dl, dtheta = gains[phase] * dl_cmd, gains[phase] * dth_cmd
        ds = 0.5 * (slip_speed(t0, episodes, t_ramp, v_slip)
                    + slip_speed(t0 + dt, episodes, t_ramp, v_slip)) * dt

        theta = x[k, TH]
        anch, mov = (B, F) if back else (F, B)
        step = -ds * rot(theta)[:, 0] + rng.normal(0.0, micro_slip_std * dt, 2)
        a_new = x[k, anch] + step
        length += dl
        theta_new = theta + dtheta
        axis = length * rot(theta_new)[:, 0]
        x[k + 1, TH] = theta_new
        x[k + 1, anch] = a_new
        x[k + 1, mov] = a_new + axis if back else a_new - axis

        back_anchored[k] = back
        e_cmd[k] = [dl_cmd, dth_cmd]
        gyro[k] = dtheta / dt + rng.normal(0.0, gyro_noise_std)
        reading = {}
        for pad, sl, std in (("back", B, sensor_noise_std if back else moving_sensor_noise_std),
                             ("front", F, moving_sensor_noise_std if back else sensor_noise_std)):
            r = rot(theta).T @ (x[k + 1, sl] - x[k, sl]) + rng.normal(0.0, std, 2)
            kind, v = fault_at(t0 + 0.5 * dt, faults, pad)
            if kind == SPIKE:
                r = r + v * dt
            elif kind == DROPOUT:
                r = np.zeros(2)
            reading[pad] = (r, kind)
        (d_a[k], fault_a[k]), (d_m[k], fault_m[k]) = ((reading["back"], reading["front"]) if back
                                                     else (reading["front"], reading["back"]))

    rel = x[:, F] - x[:, B]
    z = np.einsum("kji,kj->ki", np.array([rot(th) for th in x[:, TH]]), rel)
    z = z + rng.normal(0.0, 1.0, (n_steps + 1, 2)) * [closure_noise_std, closure_lat_noise_std]
    return x, gyro, d_a, d_m, z, back_anchored, e_cmd, fault_a, fault_m


def commanded_change(e_cmd_k, length):
    """Commanded change of the front-minus-back vector over one tick, in the
    body frame at the tick's start: [(l + dl) cos(dth) - l, (l + dl) sin(dth)]."""
    dl, dth = e_cmd_k
    return np.array([(length + dl) * np.cos(dth) - length, (length + dl) * np.sin(dth)])


def increments(variant, d_a, d_m, e, s, noise):
    """Body-frame increments of the anchored and moving pads for one tick, and
    their covariances.
    Arguments:
        variant: one of VARIANTS
        d_a, d_m: anchored / moving pad readings (2,), body frame
        e: commanded change of front-minus-back (2,), body frame
        s: +1 if the front pad moves, -1 if the back pad moves
        noise: dict of filter noise std-devs (see build_setup)
    Returns:
        u_a, u_m: increments (2,), body frame (u_m is None for random_walk)
        C_a, C_m, C_am: (2,2) covariances of u_a, u_m and their cross-covariance, body frame
        fault: None, "anchored" or "moving" (cross_check's verdict on this tick)
    """
    S_a = noise["sensor"] ** 2 * np.eye(2)
    S_m = noise["moving_sensor"] ** 2 * np.eye(2)
    S_e = np.diag((noise["stroke_err"] * np.abs(e)) ** 2)
    zero = np.zeros((2, 2))
    enc = d_a + s * e
    if variant == "random_walk":
        return d_a, None, S_a, noise["move"] ** 2 * np.eye(2), zero, None
    if variant == "encoder_model":
        return d_a, enc, S_a, S_a + S_e, S_a, None
    if variant == "moving_sensor":
        return d_a, d_m, S_a, S_m, zero, None
    if variant not in ("both", "cross_check"):
        raise ValueError(f"unknown variant: {variant!r}")
    if variant == "cross_check":
        r = d_m - enc
        if r @ np.linalg.solve(S_m + S_a + S_e, r) > noise["gate"] ** 2:
            m_a = d_m - s * e
            # Both pads' increments differ between the two hypotheses by exactly r
            D = np.outer(r, r) if noise["inflate"] else zero
            if d_a @ np.linalg.solve(S_a, d_a) > m_a @ np.linalg.solve(S_m + S_e, m_a):
                return m_a, d_m, S_m + S_e + D, S_m + D, S_m + D, "anchored"
            return d_a, enc, S_a + D, S_a + S_e + D, S_a + D, "moving"
    W = (S_a + S_e) @ np.linalg.inv(S_m + S_a + S_e)
    I_W = np.eye(2) - W
    u_m = W @ d_m + I_W @ enc
    C_m = W @ S_m @ W.T + I_W @ (S_a + S_e) @ I_W.T
    return d_a, u_m, S_a, C_m, S_a @ I_W.T, None


def predict(x, P, dt, omega, u_a, u_m, C_a, C_m, C_am, back_anchored, noise):
    """Predict step: each pad moves by its body-frame increment rotated to
    the world frame (the moving pad gets a loose random walk if u_m is None);
    theta integrates the gyro.
    Returns:
        x_new, P_new
    """
    anch, mov = (B, F) if back_anchored else (F, B)
    R = rot(x[TH])
    x_new = x.copy()
    Fm = np.eye(STATE_DIM)
    Q = np.zeros((STATE_DIM, STATE_DIM))
    x_new[anch] = x[anch] + R @ u_a
    Fm[anch, TH] = drot(x[TH]) @ u_a
    Q[anch, anch] = R @ C_a @ R.T
    if u_m is None:
        Q[mov, mov] = C_m
    else:
        x_new[mov] = x[mov] + R @ u_m
        Fm[mov, TH] = drot(x[TH]) @ u_m
        Q[mov, mov] = R @ C_m @ R.T
        Q[anch, mov] = R @ C_am @ R.T
        Q[mov, anch] = Q[anch, mov].T
    x_new[TH] = x[TH] + omega * dt
    Q[TH, TH] = (noise["gyro"] * dt) ** 2
    return x_new, Fm @ P @ Fm.T + Q


def run_ekf(x_init, P_init, gyro, d_a, d_m, z, back_anchored, e_cmd, dt, variant, noise):
    """Recursive EKF over one run: predict, then the closure update.
    Returns:
        x_list: (N, 5) estimates
        P_list: list of (5,5) covariances
        u_a_list: (n_steps, 2) anchored-pad increments used (body frame)
        verdicts: list of None / "anchored" / "moving", one per tick
    """
    R_cl = np.diag([noise["closure"] ** 2, noise["closure_lat"] ** 2])
    x, P = x_init.copy(), P_init.copy()
    x_list, P_list, u_a_list, verdicts = [x], [P], [], []
    for k in range(len(gyro)):
        s = 1.0 if back_anchored[k] else -1.0
        e = commanded_change(e_cmd[k], z[k][0])
        u_a, u_m, C_a, C_m, C_am, verdict = increments(variant, d_a[k], d_m[k], e, s, noise)
        x, P = predict(x, P, dt, gyro[k], u_a, u_m, C_a, C_m, C_am, back_anchored[k], noise)
        x, P = update_closure(x, P, z[k + 1], R_cl)
        x_list.append(x)
        P_list.append(P)
        u_a_list.append(u_a)
        verdicts.append(verdict)
    return np.array(x_list), P_list, np.array(u_a_list), verdicts


def slip_flags(detector, d_a, u_a, d_thr):
    """Per-tick slip flags (n_steps,): `threshold` on the anchored reading,
    `cross_check` on the anchored increment the cross_check filter used."""
    v = d_a if detector == "threshold" else u_a
    return np.linalg.norm(v, axis=1) > d_thr


def true_anchor_steps(x_true, back_anchored):
    """(n_steps,) true displacement of the anchored pad over each tick (m)."""
    step_b = np.linalg.norm(np.diff(x_true[:, B], axis=0), axis=1)
    step_f = np.linalg.norm(np.diff(x_true[:, F], axis=0), axis=1)
    return np.where(back_anchored, step_b, step_f)


def detection_counts(flags, tick_mid, episodes, fault_a, true_step, d_thr):
    """Scores one run's slip flags.
    Returns:
        detected: number of slip pulses with at least one flagged tick
        slip_ticks, missed: ticks whose true anchored-pad step exceeds d_thr, and how many of them are not flagged
        false_fault: flagged ticks outside pulses while the anchored reading is faulty
        false_other: flagged ticks outside pulses otherwise
    """
    in_pulse = np.array([any(a <= t <= b for _, a, b in episodes) for t in tick_mid])
    detected = sum(bool(np.any(flags & (tick_mid >= a) & (tick_mid <= b))) for _, a, b in episodes)
    big = true_step > d_thr
    outside = flags & ~in_pulse
    return (detected, int(np.sum(big)), int(np.sum(big & ~flags)),
            int(np.sum(outside & (fault_a != NO_FAULT))), int(np.sum(outside & (fault_a == NO_FAULT))))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--duration", type=float, default=20.0, help="Simulation length (s); 20 s = 10 cycles")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/sensor step interval (s)")
    parser.add_argument("--t-phase", type=float, default=1.0, help="Duration of each extend or contract phase (s)")
    parser.add_argument("--t-ramp", type=float, default=0.2,
                        help="Ramp at each end of a stroke and of a slip pulse (s)")
    parser.add_argument("--v-stroke", type=float, default=0.1, help="Peak commanded stroke rate dl/dt (m/s)")
    parser.add_argument("--l-min", type=float, default=0.15,
                        help="Body length at the start of extend, also the steering wheelbase (m)")
    parser.add_argument("--steer", type=float, default=0.3, help="Steering angle during extend (rad)")
    parser.add_argument("--stroke-err-std", type=float, default=0.05,
                        help="Per-phase stroke-rate gain error std-dev, true and assumed by the filter (fraction)")
    parser.add_argument("--stroke-err-per-cycle", action="store_true",
                        help="One stroke gain per cycle (extend and the following contract) instead of per phase")

    parser.add_argument("--p-slip", type=float, default=0.5,
                        help="Probability that a phase contains a slip pulse; 0 = never slips")
    parser.add_argument("--v-slip", type=float, default=0.08, help="Peak backward slip speed (m/s)")
    parser.add_argument("--t-slip", type=float, default=0.6,
                        help="Slip pulse duration including both ramps (s); 2*t-ramp <= t-slip <= t-phase")
    parser.add_argument("--micro-slip-std", type=float, default=0.01,
                        help="White creep of the anchored pad (m/s per axis)")

    parser.add_argument("--p-fault", type=float, default=0.2,
                        help="Probability of a fault episode per phase and pad sensor; 0 = no faults")
    parser.add_argument("--t-fault", type=float, default=0.3, help="Fault episode duration (s); at most t-phase")
    parser.add_argument("--fault-type", choices=("both", "spike", "dropout"), default="both",
                        help="Fault kinds drawn: both (equal probability), spikes only, or dropouts only")
    parser.add_argument("--v-spike", type=float, default=0.08,
                        help="Speed of the spurious backward reading during a spike (m/s)")
    parser.add_argument("--random-spike-speed", action="store_true",
                        help="Each spike gets a speed uniform in [0.5, 1.5] * v-spike and a random direction")

    parser.add_argument("--gyro-noise-std", type=float, default=0.02, help="Gyro noise std-dev (rad/s)")
    parser.add_argument("--sensor-noise-std", type=float, default=0.0005,
                        help="Pad sensor noise while the pad is anchored, per reading and axis (m)")
    parser.add_argument("--moving-sensor-noise-std", type=float, default=0.0005,
                        help="Pad sensor noise while the pad moves, per reading and axis (m); "
                             "true and assumed by the filter")
    parser.add_argument("--closure-noise-std", type=float, default=0.001,
                        help="Kinematic closure noise along the body axis (m)")
    parser.add_argument("--closure-lat-noise-std", type=float, default=0.001,
                        help="Kinematic closure noise across the body axis (m)")
    parser.add_argument("--no-cross-check-inflate", dest="cross_check_inflate", action="store_false",
                        help="On a disagreement, trust the chosen sensor fully instead of adding r r^T to the "
                             "increments' covariance")
    parser.add_argument("--gate-sigmas", type=float, default=3.0,
                        help="Slip threshold in std-devs of a still pad's reading, and the cross_check "
                             "disagreement gate (Mahalanobis distance)")
    parser.add_argument("--move-noise-std", type=float, default=0.02,
                        help="Moving-pad random walk per tick for random_walk (m)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.005, help="Initial pad position error (m)")
    parser.add_argument("--init-heading-noise-std", type=float, default=0.02, help="Initial heading error (rad)")

    parser.add_argument("--n-mc-trials", type=int, default=300, help="Number of Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    return parser


def build_setup(args, rng):
    """Draws the slip pulses (fixed for the seed) and collects the filter settings.
    Returns:
        dict with n_steps, n_phases, episodes, noise, init_std, P_init
    """
    if not 2.0 * args.t_ramp <= args.t_slip <= args.t_phase:
        raise SystemExit("need 2*t-ramp <= t-slip <= t-phase")
    if not 0.0 < args.t_fault <= args.t_phase:
        raise SystemExit("need 0 < t-fault <= t-phase")
    n_steps = int(round(args.duration / args.dt))
    n_phases = int(np.ceil(args.duration / args.t_phase))
    episodes = draw_slip_episodes(n_phases, args.t_phase, args.t_slip, args.p_slip, rng)
    still_std = np.hypot(args.sensor_noise_std, args.micro_slip_std * args.dt)
    noise = dict(sensor=args.sensor_noise_std, moving_sensor=args.moving_sensor_noise_std,
                 stroke_err=args.stroke_err_std, move=args.move_noise_std, gyro=args.gyro_noise_std,
                 closure=args.closure_noise_std, closure_lat=args.closure_lat_noise_std,
                 d_thr=args.gate_sigmas * still_std, gate=args.gate_sigmas, inflate=args.cross_check_inflate)
    init_std = np.array([args.init_pos_noise_std] * 4 + [args.init_heading_noise_std])
    return dict(n_steps=n_steps, n_phases=n_phases, episodes=episodes, noise=noise, init_std=init_std,
                P_init=np.diag(init_std ** 2))


def simulate_trial(args, setup, rng):
    """One trial: fresh stroke gains, faults, noise and initial error.
    Returns:
        dict with x_true, x_init, gyro, d_a, d_m, z, back, e_cmd, fault_a, fault_m, faults, path
    """
    gains = draw_stroke_gains(setup["n_phases"], args.stroke_err_std, rng, args.stroke_err_per_cycle)
    faults = draw_faults(setup["n_phases"], args.t_phase, args.t_fault, args.p_fault, rng, args.fault_type,
                         args.v_spike, args.random_spike_speed)
    x_true, gyro, d_a, d_m, z, back, e_cmd, fault_a, fault_m = simulate(
        setup["n_steps"], args.dt, args.t_phase, args.t_ramp, args.v_stroke, args.l_min, args.steer,
        setup["episodes"], args.v_slip, gains, faults, args.micro_slip_std, args.gyro_noise_std,
        args.sensor_noise_std, args.moving_sensor_noise_std, args.closure_noise_std,
        args.closure_lat_noise_std, rng)
    x_init = x_true[0] + rng.normal(0.0, 1.0, STATE_DIM) * setup["init_std"]
    mid = 0.5 * (x_true[:, B] + x_true[:, F])
    path = np.sum(np.linalg.norm(np.diff(mid, axis=0), axis=1))
    return dict(x_true=x_true, x_init=x_init, gyro=gyro, d_a=d_a, d_m=d_m, z=z, back=back, e_cmd=e_cmd,
                fault_a=fault_a, fault_m=fault_m, faults=faults, path=path)


def run_trial(trial, args, setup, variant):
    """Runs one variant's EKF on a trial from simulate_trial."""
    return run_ekf(trial["x_init"], setup["P_init"], trial["gyro"], trial["d_a"], trial["d_m"], trial["z"],
                   trial["back"], trial["e_cmd"], args.dt, variant, setup["noise"])


def midpoint_error(x_true, x_est):
    """(N,) distance between the true and estimated body midpoints."""
    return np.linalg.norm(0.5 * (x_true[:, B] + x_true[:, F]) - 0.5 * (x_est[:, B] + x_est[:, F]), axis=1)


def run_monte_carlo(args, setup, n_trials, rng, variants=VARIANTS):
    """Repeats simulate_trial n_trials times and runs each variant on it.
    Returns:
        stats: dict variant -> dict with nees (N,) per-tick average NEES, pos_rmse (N,) per-tick midpoint
            RMSE over trials (m), final_pct (mean final midpoint error, % of the path), heading_rmse (rad,
            over trials and ticks)
        det: dict detector -> dict with detected, n_pulses, slip_ticks, missed, false_fault, false_other
            (totals over trials),
            plus "faults" -> dict kind -> [episodes caught by cross_check's disagreement test, total]
    """
    n_pts = setup["n_steps"] + 1
    stats = {v: dict(nees=np.zeros(n_pts), pos_sq=np.zeros(n_pts), final_pct=0.0, heading_sq=0.0) for v in variants}
    det = {d: dict(detected=0, n_pulses=0, slip_ticks=0, missed=0, false_fault=0, false_other=0)
           for d in DETECTORS}
    caught = {SPIKE: [0, 0], DROPOUT: [0, 0]}
    tick_mid = (np.arange(setup["n_steps"]) + 0.5) * args.dt
    for _ in range(n_trials):
        tr = simulate_trial(args, setup, rng)
        for v in variants:
            xe, Pe, u_a, verdicts = run_trial(tr, args, setup, v)
            err = state_error(tr["x_true"], xe)
            o = stats[v]
            o["nees"] += [nees(err[k], Pe[k]) for k in range(n_pts)]
            mid_err = midpoint_error(tr["x_true"], xe)
            o["pos_sq"] += mid_err ** 2
            o["final_pct"] += 100.0 * mid_err[-1] / tr["path"]
            o["heading_sq"] += np.mean(err[:, TH] ** 2)
            if v == "cross_check":
                d_thr = setup["noise"]["d_thr"]
                true_step = true_anchor_steps(tr["x_true"], tr["back"])
                for name in DETECTORS:
                    flags = slip_flags(name, tr["d_a"], u_a, d_thr)
                    counts = detection_counts(flags, tick_mid, setup["episodes"], tr["fault_a"], true_step, d_thr)
                    r = det[name]
                    r["n_pulses"] += len(setup["episodes"])
                    for key, c in zip(("detected", "slip_ticks", "missed", "false_fault", "false_other"), counts):
                        r[key] += c
                flagged = np.array([vd is not None for vd in verdicts])
                for _, kind, a, b, _ in tr["faults"]:
                    caught[kind][1] += 1
                    caught[kind][0] += int(np.any(flagged & (tick_mid >= a) & (tick_mid < b)))
    for o in stats.values():
        o["nees"] /= n_trials
        o["final_pct"] /= n_trials
        o["heading_rmse"] = np.sqrt(o.pop("heading_sq") / n_trials)
        o["pos_rmse"] = np.sqrt(o.pop("pos_sq") / n_trials)
    det["faults"] = caught
    return stats, det


def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    setup = build_setup(args, rng)
    n_steps, episodes, noise = setup["n_steps"], setup["episodes"], setup["noise"]
    is_slip, is_clean = window_flags(n_steps, args.dt, args.t_phase, episodes)

    trial = simulate_trial(args, setup, rng)
    n_spike = sum(f[1] == SPIKE for f in trial["faults"])
    print(f"Slip pulses: {len(episodes)} of {setup['n_phases']} phases; sensor faults in this run: "
          f"{n_spike} spikes, {len(trial['faults']) - n_spike} dropouts; slip threshold "
          f"{noise['d_thr'] * 1e3:.2f} mm per tick")

    runs = {}
    for v in VARIANTS:
        (x_est, P_est, u_a, verdicts), t_step, mem = measure_performance(
            run_trial, trial, args, setup, v, n_steps=n_steps)
        runs[v] = (x_est, P_est, u_a, verdicts, t_step, mem)

    print(f"\nSingle run (path length of the body midpoint {trial['path']:.3f} m):")
    for v in VARIANTS:
        pos_err = midpoint_error(trial["x_true"], runs[v][0])
        th_err = state_error(trial["x_true"], runs[v][0])[:, TH]
        print(f"  {v:<14s} final midpoint error={pos_err[-1]:7.4f} m ({100 * pos_err[-1] / trial['path']:5.1f}% "
              f"of path), heading RMSE={np.sqrt(np.mean(th_err ** 2)):6.4f} rad")

    print(f"\nRunning Monte Carlo ({args.n_mc_trials} trials)...")
    stats, det = run_monte_carlo(args, setup, args.n_mc_trials, rng)
    lo, hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)
    print(f"\nMonte Carlo (consistent {STATE_DIM}-DoF NEES ~{STATE_DIM}; 95% bounds for a "
          f"{args.n_mc_trials}-trial average: [{lo:.2f}, {hi:.2f}]):")
    windows = [("overall", np.ones_like(is_slip)), ("clean", is_clean), ("slip", is_slip)]
    print("  " + " " * 14 + "".join(f"{name + ' NEES':>14s}" for name, _ in windows)
          + f"{'final err %':>13s}{'heading RMSE':>14s}")
    for v in VARIANTS:
        n = stats[v]["nees"]
        cells = "".join(f"{np.mean(n[w]) if w.any() else float('nan'):14.2f}" for _, w in windows)
        print(f"  {v:<14s}{cells}{stats[v]['final_pct']:13.1f}{stats[v]['heading_rmse']:14.4f}")

    print("\nMonte Carlo slip detection (totals): pulses detected / slip ticks missed / false-alarm ticks "
          "outside pulses (during an anchored-sensor fault + otherwise)")
    for d in DETECTORS:
        r = det[d]
        print(f"  {d:<12s} {r['detected']} of {r['n_pulses']} / {r['missed']} of {r['slip_ticks']} / "
              f"{r['false_fault']} + {r['false_other']}")
    for kind, name in ((SPIKE, "spikes"), (DROPOUT, "dropouts")):
        c, n = det["faults"][kind]
        print(f"  cross_check disagreement test caught {c} of {n} {name}")

    print("\nAverage time (per-step) and peak memory (whole-run) per variant, empirical:")
    for v in VARIANTS:
        print(f"  {v:<14s} avg time={runs[v][4] * 1e6:9.2f} us/step | peak mem={runs[v][5] / 1024.0:9.3f} KB")

    t_hist = np.arange(n_steps + 1) * args.dt
    colors = {"random_walk": "tab:gray", "encoder_model": "tab:blue", "moving_sensor": "tab:orange",
              "both": "tab:purple", "cross_check": "tab:green"}
    fig = plt.figure(figsize=(12, 9))
    ax_top = fig.add_subplot(1, 2, 1)
    ax_d = fig.add_subplot(3, 2, 2)
    ax_err = fig.add_subplot(3, 2, 4, sharex=ax_d)
    ax_nees = fig.add_subplot(3, 2, 6, sharex=ax_d)

    mid_true = 0.5 * (trial["x_true"][:, B] + trial["x_true"][:, F])
    ax_top.plot(mid_true[:, 0], mid_true[:, 1], color="black", linestyle="--", linewidth=2,
                label="Ground truth (body midpoint)")
    for v in VARIANTS:
        mid_est = 0.5 * (runs[v][0][:, B] + runs[v][0][:, F])
        ax_top.plot(mid_est[:, 0], mid_est[:, 1], color=colors[v], label=v)
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_aspect("equal")
    ax_top.set_title("Two-pad inchworm with both pad sensors, top view")
    ax_top.legend(fontsize=8)

    s = np.where(trial["back"], 1.0, -1.0)[:, None]
    stroke = np.array([commanded_change(trial["e_cmd"][k], trial["z"][k][0]) for k in range(n_steps)])
    ax_d.plot(t_hist[1:], np.linalg.norm(trial["d_a"], axis=1) * 1e3, color="tab:orange", linewidth=1,
              label="|anchored reading|")
    ax_d.plot(t_hist[1:], np.linalg.norm(trial["d_m"] - s * stroke, axis=1) * 1e3, color="tab:blue",
              linewidth=1, label="|moving reading - commanded stroke|")
    ax_d.axhline(noise["d_thr"] * 1e3, color="tab:red", linestyle="--", linewidth=1, label="slip threshold")
    for pad, kind, a, b, _ in trial["faults"]:
        ax_d.axvspan(a, b, ymin=0.9, ymax=1.0, color="tab:red" if kind == SPIKE else "tab:gray", linewidth=0)
    ax_d.set_ylabel("Per tick (mm)")
    ax_d.legend(fontsize=7)

    for v in VARIANTS:
        ax_err.plot(t_hist, midpoint_error(trial["x_true"], runs[v][0]) * 1e3, color=colors[v], label=v)
    ax_err.set_ylabel("Midpoint error (mm)")

    for v in VARIANTS:
        ax_nees.plot(t_hist, stats[v]["nees"], color=colors[v], label=v)
    ax_nees.axhline(STATE_DIM, color="black", linewidth=1, label=f"Expected NEES ({STATE_DIM} DoF)")
    ax_nees.axhline(lo, color="black", linestyle="--", linewidth=1)
    ax_nees.axhline(hi, color="black", linestyle="--", linewidth=1)
    ax_nees.set_yscale("log")
    ax_nees.set_xlabel("Time (s)")
    ax_nees.set_ylabel(f"MC avg. NEES ({args.n_mc_trials})")
    for ax in (ax_d, ax_err, ax_nees):
        for _, a, b in episodes:
            ax.axvspan(a, b, color="gold", alpha=0.25, linewidth=0)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
