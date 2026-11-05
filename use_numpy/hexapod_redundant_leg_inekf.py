'''
hexapod_joint_kinematics_inekf.py on a hexapod with four joints per leg: the
CTU HAntR (Cizek, Zoula & Faigl 2021, IEEE Access 9:17866-17881, CC-BY 4.0).

A foot position has 3 coordinates and the leg has 4 joints, so the leg
Jacobian J = dFK/dq is 3x4 and has a null space: at any pose, one combination
of the leg's joint angles (its direction n, with J n = 0) leaves the foot where
it is. An encoder offset along n moves no foot reading at that pose; only a
change of pose can reveal it.

Geometry (the paper's Table 2 and Eq. 3): Denavit-Hartenberg links
(alpha, a, d, theta_off) coxa (pi/2, 0, 0, 0), trochanter (-pi/2, 25.4 mm,
-18.5 mm, pi/2), femur (0, 81.6 mm, 0, -0.52), tibia (0, 205.5 mm, 0, -1.06),
a roll-yaw-pitch-pitch leg; legs at mount angles beta = 0, 60, ..., 300 deg,
102 mm from the body centre and 38.5 mm above the body frame, coxa axis
radial. Default footholds (Section IV): leg span 240 mm, read here as the
foothold radius from the body centre (Table 1 gives a 0.5 x 0.5 m outline),
and body ground clearance 100 mm, read here as the body frame's height above
the ground.

Redundancy: the truth's joint angles follow the paper's inverse kinematics,
which reaches the foot while keeping the tibia (O3 -> foot) as close as
possible to the gravity direction. Solved here numerically: foot position
plus the condition that the alignment cost has zero slope along the null
space.

Encoders: Dynamixel AX-12A, resolution 0.29 deg (Robotis e-manual).
Quantization to that step has std 0.29 / sqrt(12) = 0.084 deg = 1.46 mrad,
modeled here as white noise of that std. Joint zero offsets are assumed and
swept, as in hexapod_joint_kinematics_inekf.py.

Legs in LEG_NAMES slots (for the tripod groups) by mount angle:
L1 60, L2 120, L3 180, R1 0, R2 300, R3 240 deg, so each tripod group is every
other leg around the body, as in the paper's G = {{1, 3, 5}, {2, 4, 6}}.

Experiments:
  1. White encoder noise at 1 and 3.5 quantization stds (1.46 and 5.06 mrad):
     iso / iso_max / joint models (a check, as in the 3-joint script).
  2. Joint zero offsets (sweep), white noise 1.46 mrad: joint /
     joint_inflated / joint_offsets (24 offset states). For joint_offsets,
     each leg's final offset error is split into its part along the nominal
     null direction n and the rest, and each is compared with the std the
     filter reports for it.
'''

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import averaged_nees_bounds  # noqa: E402
from state_estimation_simulations.use_numpy.hexapod_contact_inekf import Config  # noqa: E402
from state_estimation_simulations.use_numpy.hexapod_gait_ekf import LEG_NAMES  # noqa: E402
from state_estimation_simulations.use_numpy.hexapod_joint_kinematics_inekf import (  # noqa: E402
    LegModel, WHITE_KINDS, OFFSET_KINDS, iso_variances, nominal_joint_cov, run_filter, run_monte_carlo,
    simulate_trial, summarize)

# (alpha (rad), a (m), d (m), theta_off (rad)): coxa, trochanter, femur, tibia
DH = [(np.pi / 2.0, 0.0, 0.0, 0.0), (-np.pi / 2.0, 0.0254, -0.0185, np.pi / 2.0),
      (0.0, 0.0816, 0.0, -0.52), (0.0, 0.2055, 0.0, -1.06)]
BODY_RADIUS = 0.102            # l_r: coxa frame from the body centre (m)
COXA_HEIGHT = 0.0385           # l_z: coxa frame above the body frame (m)
LEG_SPAN = 0.24                # d_def: default foothold radius (m)
GROUND_CLEARANCE = 0.10        # h_def: body frame above the ground (m)
MOUNT_DEG = np.array([60.0, 120.0, 180.0, 0.0, 300.0, 240.0])   # LEG_NAMES slots L1 L2 L3 R1 R2 R3
AX12_STEP = np.radians(0.29)   # AX-12A position resolution
SIGMA_Q_QUANT = AX12_STEP / np.sqrt(12.0)
LABELS = {"iso": "isotropic, same total variance", "iso_max": "isotropic, largest axis",
          "joint": "joint model σq² J Jᵀ", "joint_inflated": "joint model, offsets as extra white noise",
          "joint_offsets": "joint model + 24 offset states"}


def _dh(theta, alpha, a, d):
    c, s, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s * ca, s * sa, a * c], [s, c * ca, -c * sa, a * s], [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


class HAntRLeg(LegModel):
    """HAntR's 4-joint legs (see the module docstring)."""
    name = "HAntR"
    n_joints = 4
    body_height = GROUND_CLEARANCE

    def frames(self, q, leg):
        """Joint origins (4, 3), joint axes (4, 3) (each the z axis of the
        previous DH frame) and the foot (3,), body frame."""
        b = np.radians(MOUNT_DEG[leg])
        cb, sb = np.cos(b), np.sin(b)
        T = np.eye(4)
        T[0:3, 0:3] = [[-sb, 0.0, cb], [cb, 0.0, sb], [0.0, 1.0, 0.0]]
        T[0:3, 3] = [BODY_RADIUS * cb, BODY_RADIUS * sb, COXA_HEIGHT]
        origins, axes = np.zeros((4, 3)), np.zeros((4, 3))
        for i, (alpha, a, d, off) in enumerate(DH):
            origins[i], axes[i] = T[0:3, 3], T[0:3, 2]
            T = T @ _dh(q[i] + off, alpha, a, d)
        return origins, axes, T[0:3, 3]

    def tibia(self, q, leg):
        """Unit tibia direction, joint 4's origin to the foot."""
        origins, _, foot = self.frames(q, leg)
        u = foot - origins[3]
        return u / np.linalg.norm(u)

    def null_direction(self, q, leg):
        """Unit n with J(q) n = 0 (sign: largest entry positive)."""
        n = np.linalg.svd(self.jacobian(q, leg))[2][-1]
        return n if n[np.argmax(np.abs(n))] > 0 else -n

    def _ik_residual(self, q, leg, target, o):
        """Foot error (3) and the slope of the alignment cost -u.o along the
        null direction (du/dq_i = axis_i x u for every joint)."""
        origins, axes, foot = self.frames(q, leg)
        u = (foot - origins[3]) / np.linalg.norm(foot - origins[3])
        grad = np.array([-np.cross(axes[i], u) @ o for i in range(4)])
        return np.concatenate([foot - target, [self.null_direction(q, leg) @ grad]])

    def ik(self, target, leg, q0, R=None, tol=1e-12, max_iter=50):
        """Joint angles reaching target (body frame) with the tibia as close as
        possible to gravity, R^T (0, 0, -1) in the body frame (R: body to world;
        None = level). Newton's method from q0 on the 4 conditions of
        _ik_residual (finite-difference matrix). Raises if it does not converge."""
        o = np.array([0.0, 0.0, -1.0]) if R is None else R.T @ np.array([0.0, 0.0, -1.0])
        q = np.array(q0, dtype=float)
        h = 1e-7
        for _ in range(max_iter):
            r = self._ik_residual(q, leg, target, o)
            if r[:3] @ r[:3] < tol and abs(r[3]) < 1e-9:
                return q
            G = np.column_stack([(self._ik_residual(q + e, leg, target, o) - r) / h for e in np.eye(4) * h])
            q = q - np.linalg.lstsq(G, r, rcond=None)[0]
        raise RuntimeError(f"HAntR ik did not converge for leg {LEG_NAMES[leg]}: residual {np.abs(r).max():.2e}")

    def stance_foot(self, leg):
        """Default foothold (body frame): LEG_SPAN along the mount direction, GROUND_CLEARANCE below."""
        b = np.radians(MOUNT_DEG[leg])
        return np.array([LEG_SPAN * np.cos(b), LEG_SPAN * np.sin(b), -GROUND_CLEARANCE])

    def stance_ik(self, leg):
        """Joint angles at the default foothold, body level."""
        return self.ik(self.stance_foot(leg), leg, np.zeros(4))


HANTR = HAntRLeg()


def null_split(b_err, b_cov, model=HANTR):
    """Per leg, the offset error (n,) split along the nominal null direction n
    and the rest: returns (null error (6,), null std (6,), rest squared error
    (6,), rest variance (6,)) from the final error and offset covariance."""
    nj = model.n_joints
    out = np.zeros((4, 6))
    for leg in range(6):
        n = model.null_direction(model.stance_ik(leg), leg)
        e = b_err[nj * leg:nj * (leg + 1)]
        C = b_cov[nj * leg:nj * (leg + 1), nj * leg:nj * (leg + 1)]
        en = n @ e
        out[:, leg] = [en, np.sqrt(n @ C @ n), e @ e - en ** 2, np.trace(C) - n @ C @ n]
    return out


def run_offsets(cfg, n_trials, gait, sigma_q, sigma_b, rng, model=HANTR):
    """Experiment 2 at one offset std: run_monte_carlo's averages for
    OFFSET_KINDS, plus for joint_offsets the final null / rest split over all
    legs and trials (RMSE vs root-mean reported variance), and the per-tick
    reported null std (root mean square over legs), averaged over trials;
    per joint, the final offset RMSE and root-mean reported std; and the
    direction (n,) along which the final per-leg errors are largest (top
    eigenvector of their covariance, pooled over legs and trials), with its
    RMSE and the root-mean reported std along it."""
    n_pts = cfg.n_steps + 1
    iso_var = iso_variances(sigma_q, model)
    acc = {kd: dict(nees_vel=np.zeros(n_pts), nees_tilt=np.zeros(n_pts), mse_vel=np.zeros((n_pts, 3)),
                    mse_tilt=np.zeros((n_pts, 2)), mse_b=np.zeros(n_pts), var_b=np.zeros(n_pts))
           for kd in OFFSET_KINDS}
    nulls = np.array([model.null_direction(model.stance_ik(leg), leg) for leg in range(6)])
    split, null_std_t, errs, covs = [], np.zeros(n_pts), [], []
    for _ in range(n_trials):
        trial, q_meas, b = simulate_trial(cfg, [gait], rng, sigma_q, sigma_b, model)
        for kind, a in acc.items():
            o = run_filter(kind, trial, q_meas, gait, cfg, sigma_q, sigma_b, iso_var, b, model,
                           keep_offset_cov=(kind == "joint_offsets"), offset_dirs=nulls)
            a["nees_vel"] += o["nees_vel"]
            a["nees_tilt"] += np.sum(o["tilt_err"] ** 2 / o["tilt_var"], axis=1)
            a["mse_vel"] += o["vel_err"] ** 2
            a["mse_tilt"] += o["tilt_err"] ** 2
            a["mse_b"] += np.mean(o["b_err"] ** 2, axis=1)
            a["var_b"] += np.mean(o["b_std"] ** 2, axis=1)
            if kind == "joint_offsets" and sigma_b > 0:
                split.append(null_split(o["b_err"][-1], o["b_cov"], model))
                nj = model.n_joints
                errs.append(o["b_err"][-1].reshape(6, nj))
                covs.extend(o["b_cov"][nj * leg:nj * (leg + 1), nj * leg:nj * (leg + 1)] for leg in range(6))
                null_std_t += np.sqrt(np.mean(o["dir_std"] ** 2, axis=1))
    for a in acc.values():
        for name in a:
            a[name] /= n_trials
    res = dict(acc=acc)
    if split:
        s, nj = np.array(split), model.n_joints   # s: (trials, 4, 6)
        res["null_rmse"] = np.sqrt(np.mean(s[:, 0] ** 2))
        res["null_std"] = np.sqrt(np.mean(s[:, 1] ** 2))
        res["rest_rmse"] = np.sqrt(np.mean(s[:, 2]) / (nj - 1))
        res["rest_std"] = np.sqrt(np.mean(s[:, 3]) / (nj - 1))
        res["null_nees"] = np.mean(s[:, 0] ** 2 / s[:, 1] ** 2)
        res["null_std_t"] = null_std_t / n_trials
        E, C = np.concatenate(errs), np.array(covs)       # (trials x 6, n), (trials x 6, n, n)
        res["joint_rmse"] = np.sqrt(np.mean(E ** 2, axis=0))
        res["joint_std"] = np.sqrt(np.mean(np.diagonal(C, axis1=1, axis2=2), axis=0))
        w = np.linalg.eigh(E.T @ E / len(E))[1][:, -1]
        w = w if w[np.argmax(np.abs(w))] > 0 else -w
        res["worst_dir"] = w
        res["worst_rmse"] = np.sqrt(np.mean((E @ w) ** 2))
        res["worst_std"] = np.sqrt(np.mean(np.einsum("i,kij,j->k", w, C, w)))
        res["worst_foot"] = np.mean([np.linalg.norm(model.jacobian(model.stance_ik(leg), leg) @ w) for leg in range(6)])
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--duration", type=float, default=d.n_steps * d.dt, help="Simulation length (s)")
    parser.add_argument("--gait", type=str, default="tripod", choices=["tripod", "ripple", "wave"], help="Gait")
    parser.add_argument("--sigma-q", type=float, default=SIGMA_Q_QUANT,
                        help="Encoder white-noise std per joint (rad); default = AX-12A quantization std")
    parser.add_argument("--sweep-q", type=float, nargs="+", default=[SIGMA_Q_QUANT, AX12_STEP],
                        help="Experiment 1: encoder white-noise stds (rad)")
    parser.add_argument("--sweep-b-deg", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0],
                        help="Experiment 2: joint zero-offset stds (deg)")
    parser.add_argument("--n-mc-trials", type=int, default=30, help="Monte Carlo trials per setting")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    args = parser.parse_args()

    cfg = HANTR.config(replace(d, n_steps=int(round(args.duration / d.dt))))
    skip = int(round(1.0 / cfg.dt))
    lo3, hi3 = averaged_nees_bounds(3, args.n_mc_trials)
    lo2, hi2 = averaged_nees_bounds(2, args.n_mc_trials)
    print(f"HAntR legs at the default footholds; foot noise std along the principal axes for "
          f"{args.sigma_q * 1e3:.2f} mrad encoder noise (mm); null direction n (coxa, trochanter, femur, tibia):")
    for leg in range(6):
        q = HANTR.stance_ik(leg)
        sd = np.sqrt(np.linalg.eigvalsh(nominal_joint_cov(leg, args.sigma_q, HANTR)))[::-1] * 1e3
        tilt = np.degrees(np.arccos(-HANTR.tibia(q, leg)[2]))
        print(f"  {LEG_NAMES[leg]} (mount {MOUNT_DEG[leg]:3.0f} deg): q = {np.round(np.degrees(q), 1)} deg, "
              f"tibia {tilt:.1f} deg from vertical -> {np.round(sd, 3)}; n = {np.round(HANTR.null_direction(q, leg), 3)}")
    print(f"Consistent run averages ({args.n_mc_trials} trials): velocity NEES in [{lo3:.2f}, {hi3:.2f}], "
          f"tilt NEES in [{lo2:.2f}, {hi2:.2f}]; averages leave out the first second")

    print(f"\nExperiment 1: white encoder noise ({args.gait})")
    for i, s in enumerate(args.sweep_q):
        res = run_monte_carlo(cfg, args.n_mc_trials, args.gait, s, 0.0, WHITE_KINDS,
                              np.random.default_rng(args.seed + i), HANTR)
        for kind, a in res.items():
            nv, nt, rv, rt = summarize(a, skip)
            print(f"  sigma_q {s * 1e3:5.2f} mrad  {kind:<8s} vel NEES {nv:6.2f}  tilt NEES {nt:6.2f} | "
                  f"vel RMSE {rv * 1e3:6.2f} mm/s  tilt RMSE {rt:.4f} deg")

    exp2 = {}
    print(f"\nExperiment 2: joint zero offsets, encoder noise {args.sigma_q * 1e3:.2f} mrad ({args.gait})")
    for i, sb in enumerate(args.sweep_b_deg):
        exp2[sb] = run_offsets(cfg, args.n_mc_trials, args.gait, args.sigma_q, np.radians(sb),
                               np.random.default_rng(args.seed + 100 + i))
        for kind, a in exp2[sb]["acc"].items():
            nv, nt, rv, rt = summarize(a, skip)
            print(f"  offsets {sb:3.1f} deg  {kind:<15s} vel NEES {nv:6.2f}  tilt NEES {nt:6.2f} | "
                  f"vel RMSE {rv * 1e3:6.2f} mm/s  tilt RMSE {rt:.4f} deg")
        if sb > 0:
            r = exp2[sb]
            print(f"    offset error at the end, per leg: null direction RMSE {np.degrees(r['null_rmse']):.3f} deg vs "
                  f"reported {np.degrees(r['null_std']):.3f} (NEES {r['null_nees']:.2f}); other directions RMSE "
                  f"{np.degrees(r['rest_rmse']):.3f} deg vs reported {np.degrees(r['rest_std']):.3f} (prior {sb:g})")
            print(f"    per joint (coxa, trochanter, femur, tibia): RMSE {np.round(np.degrees(r['joint_rmse']), 3)} deg "
                  f"vs reported {np.round(np.degrees(r['joint_std']), 3)}")
            print(f"    largest-error direction {np.round(r['worst_dir'], 2)}: RMSE {np.degrees(r['worst_rmse']):.3f} deg "
                  f"vs reported {np.degrees(r['worst_std']):.3f}; moves the foot {1e3 * r['worst_foot']:.1f} mm per rad "
                  f"at the default foothold")

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    axes = axs.ravel()
    colors = {"joint": "tab:blue", "joint_inflated": "tab:purple", "joint_offsets": "tab:red"}
    for kind in OFFSET_KINDS:
        stats = np.array([summarize(exp2[sb]["acc"][kind], skip) for sb in args.sweep_b_deg])
        axes[0].plot(args.sweep_b_deg, stats[:, 0], "o-", color=colors[kind], label=LABELS[kind])
        axes[1].plot(args.sweep_b_deg, stats[:, 2] * 1e3, "o-", color=colors[kind], label=LABELS[kind])
    axes[0].axhspan(lo3, hi3, color="0.85", label=f"95% band, consistent ({args.n_mc_trials} trials)")
    axes[0].set(xlabel="joint zero-offset std σb (deg)", ylabel="velocity NEES (run average, 3 DoF)",
                title="Velocity consistency")
    axes[1].set(xlabel="joint zero-offset std σb (deg)", ylabel="velocity RMSE (mm/s)", title="Velocity error")
    axes[0].legend(fontsize=8)
    t = np.arange(cfg.n_steps + 1) * cfg.dt
    for sb, c in zip([s for s in args.sweep_b_deg if s > 0], ["tab:olive", "tab:orange", "tab:red", "tab:brown"]):
        r = exp2[sb]
        axes[2].plot(t, np.degrees(r["null_std_t"]) / sb, color=c, label=f"σb = {sb:g}°: reported, null direction")
        axes[2].axhline(np.degrees(r["null_rmse"]) / sb, color=c, ls="--", lw=1)
    axes[2].plot([], [], color="black", ls="--", lw=1, label="actual RMSE at the end (null direction)")
    axes[2].set(xlabel="time (s)", ylabel="offset std / prior std", ylim=(0, None),
                title="Offset along each leg's null direction")
    axes[2].legend(fontsize=8)
    sb = max(args.sweep_b_deg)
    r, x = exp2[sb], np.arange(HANTR.n_joints)
    axes[3].bar(x - 0.2, np.degrees(r["joint_rmse"]), 0.4, color="tab:red", label="actual RMSE at the end")
    axes[3].bar(x + 0.2, np.degrees(r["joint_std"]), 0.4, color="0.6", label="reported std at the end")
    axes[3].axhline(sb, color="black", ls=":", lw=1, label=f"prior std ({sb:g}°)")
    axes[3].set_xticks(x, ["coxa", "trochanter", "femur", "tibia"])
    axes[3].set(ylabel="joint offset error (deg)", title=f"Offset states per joint, σb = {sb:g}°")
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
