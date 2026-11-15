# IMM: estimating "anchored or slipping" instead of thresholding it

## Intuition

When the anchor may slip, the filter has to decide whether "velocity = 0" is true right now. [`inchworm_zupt_ekf.md` §6](inchworm_zupt_ekf.md#6-side-extension-anchor-slip-and-a-measured-gate) decides with a hard rule: the schedule, a threshold gate on the pad reading, or no decision at all (use the reading as a velocity). An **interacting multiple model (IMM)** filter decides softly. It runs two Kalman filters side by side, one that assumes the body is **still** and one that assumes it is **moving**, and keeps a probability for each. Each tick, the mode that predicted the new measurements better gains probability, and the output is the probability-weighted blend of the two.

Worked number. A pad reading of $`-0.05`$ m/s arrives on an anchored tick (a mid-slip speed). If the body were still, the filter expects 0 with a spread of $`\sqrt{0.01^2 + 0.01^2} = 0.014`$ m/s (the still mode's own velocity spread plus the reading noise, both 0.01 m/s). The reading is $`0.05 / 0.014 = 3.5`$ spreads away, which scales the still mode's likelihood by $`e^{-3.5^2/2} \approx 0.002`$ against its best case. The moving mode does not expect 0, so it is not surprised, and the probability moves to it. At the defaults this takes 0.08 s on average (table in [Slip detection](#slip-detection)).

Question: does this beat the hard rules, and in which regime? Measured answer, in the order of the sections below:

- **Pad reading available:** it matches `sensor_fused`'s consistency in the slip window and lowers the velocity error, but only on still ticks. As the pad gets noisier, `sensor_fused` stays consistent in the slip window and the IMM does not.
- **No pad reading:** position noise (20 mm) hides a slip of 4 mm per tick, so the IMM cannot detect it. The schedule-aware IMM improves on `schedule`'s **NEES** in the slip window (9.34 vs. 47.27, [glossary](../glossary.md#consistency-and-nees)) at the same accuracy, but stays far above `never` (2.58).
- **A transition rule that looks better than it is:** the first schedule rule considered (the matrix depends only on the next tick's scheduled phase) seems to detect slip without the pad. It does not; see [the artifact](#the-next-tick-only-rule-looks-like-detection-and-is-not).

## Model

Everything is reused from §6 of the parent doc (a real inchworm with a pad sensor is its motivation, reference 2): the 1D point mass with state $`x = [p, v]`$ (position $`p`$ in m, velocity $`v`$ in m/s), the gait (1 s anchor, 1 s extend with 0.2 s ramps, cruise speed 0.1 m/s), the backward slip pulses (probability 0.5 per anchor phase, peak 0.08 m/s, 0.6 s long, drawn once per seed), the position measurement $`z_k`$ every tick (noise std 0.02 m), the pad reading $`d_k = \Delta t\, v_k + \epsilon_k`$ (displacement per tick, $`\Delta t = 0.05`$ s, noise std $`\sigma_d = 0.5`$ mm, so $`d_k / \Delta t`$ is a velocity with std 0.01 m/s), the process noise 0.5 m/s², 500 Monte Carlo trials and the same random-number order, so seed 0 has the same two pulses (24 slip ticks). The **NEES** windows (**clean anchor**, **slip**, **cruise**, **overall**) are the parent's. The parent's five policies run unchanged (`never`, `schedule`, `sensor_gate`, `schedule_and_sensor`, `sensor_fused`).

What is new:

| Part | Definition |
| --- | --- |
| **Still mode (S)** | $`v' = 0 + w`$ and $`p' = p`$ (a prime is the value at the next tick), with $`w`$ noise of std $`\sigma_S = 0.01`$ m/s, the **ZUPT** noise ([glossary](../glossary.md#zupt-and-zaru)). The belief "v = 0" is the dynamics, not a pseudo-measurement. |
| **Moving mode (M)** | The parent's constant-velocity predict with acceleration noise 0.5 m/s². |
| **Measurement vector** | $`y_k = [z_k]`$ (position only) or $`y_k = [z_k,\ d_k / \Delta t]^\top`$ (position and pad reading). Both modes receive the same $`y_k`$, so their likelihoods are comparable. |
| **Variants** | `imm` and `imm_schedule` use the pad; `imm_pos` and `imm_schedule_pos` use position only. `imm` and `imm_pos` use a constant transition matrix; `imm_schedule` and `imm_schedule_pos` follow the schedule (see [below](#transition-matrix-a-design-choice)). |

Where the truth and the model differ (the first two numbers are printed by the script): the extend ramps accelerate at 0.50 m/s² and the slip ramps at 0.40 m/s², within mode M's process noise of 0.50 m/s²; the still mode's $`v = 0`$ is exact on clean anchors; and the truth has no random disturbance at all. The last one means a NEES below 2 in a window (e.g. clean anchor, 0.79-1.33 for the IMMs) shows a covariance larger than the actual error, not an exactly calibrated filter.

## The IMM in one table

Five steps per tick: the four steps of Blom & Bar-Shalom (reference 1, §III), with their step 3 split into likelihood and mode probabilities. Each mode $`j`$ has its own estimate $`\hat x_j`$ and covariance $`P_j`$ after the update, and the filter holds the mode probabilities $`\mu_j`$.

| Step | What it does | Paper equations |
| --- | --- | --- |
| 1. **Mix** | Predicts how likely each mode is now from the previous probabilities and the transition matrix, then builds each mode's starting point as a blend of both previous estimates. | (12)-(14) |
| 2. **Predict and update** | Each mode runs its own Kalman filter from its mixed starting point, on the same $`y_k`$. | text of step 2 |
| 3. **Likelihood** | Each mode scores how well it predicted $`y_k`$: the Gaussian density of its innovation (measurement minus predicted measurement) under its innovation covariance. | (15)-(17) |
| 4. **Mode probabilities** | Likelihood times predicted probability, normalized to sum to 1. | (15) |
| 5. **Combine** | The output is the probability-weighted mean, and a covariance that adds the spread of the two means. | (18)-(19) |

Mix (step 1), with $`\Pi_{ij}`$ the probability of moving from mode $`i`$ at the previous tick to mode $`j`$ now:

```math
\bar\mu_j = \sum_i \Pi_{ij}\,\mu_i, \qquad
\hat x^{0j} = \sum_i \frac{\Pi_{ij}\,\mu_i}{\bar\mu_j}\,\hat x_i, \qquad
P^{0j} = \sum_i \frac{\Pi_{ij}\,\mu_i}{\bar\mu_j}\Big[P_i + (\hat x_i - \hat x^{0j})(\hat x_i - \hat x^{0j})^\top\Big]
```

Likelihood and mode probabilities (steps 3 and 4), with innovation $`s_j = y - H\hat x^{-}_j`$ and innovation covariance $`S_j = H P^{-}_j H^\top + R`$:

```math
\Lambda_j = \frac{1}{\sqrt{(2\pi)^m \det S_j}}\exp\!\Big(-\tfrac12\, s_j^\top S_j^{-1} s_j\Big), \qquad
\mu_j = \frac{\Lambda_j\,\bar\mu_j}{\sum_i \Lambda_i\,\bar\mu_i}
```

Combine (step 5):

```math
\hat x = \sum_j \mu_j\,\hat x_j, \qquad
P = \sum_j \mu_j\Big[P_j + (\hat x_j - \hat x)(\hat x_j - \hat x)^\top\Big]
```

| Symbol | Meaning |
| --- | --- |
| $`\Lambda_j`$ | Likelihood of mode $`j`$: the density of its innovation. |
| $`\mu_j`$, $`\bar\mu_j`$ | Probability of mode $`j`$ after step 4, and its predicted value from step 1 (0 to 1, each set sums to 1). $`\mu_M`$ is the P(M) plotted below. |
| $`\Pi`$ | $`2\times2`$ matrix of mode-switch probabilities per tick, one row per previous mode (rows sum to 1). |
| $`\hat x^{0j}`$, $`P^{0j}`$ | Mixed starting mean and covariance of mode $`j`$. |
| $`\hat x^{-}_j`$, $`P^{-}_j`$ | Mode $`j`$'s predicted mean and covariance (after its predict step, before the update). |
| $`H`$, $`R`$ | Measurement matrix and measurement-noise covariance of $`y`$: $`H = [1\ 0]`$, $`R = 0.02^2`$ for position only; $`H = I`$, $`R = \mathrm{diag}(0.02^2,\ 0.01^2)`$ with the pad. |
| $`s_j`$, $`S_j`$ | Mode $`j`$'s innovation and its covariance (defined in the likelihood equation above). |
| $`m`$ | Length of $`y`$ (1 or 2). |

- **Convention:** the paper's matrix $`H_{ij}`$ is $`P(\text{mode}_t = i \mid \text{mode}_{t-1} = j)`$, so its columns sum to 1. The code uses its transpose $`\Pi`$ (row = previous mode), as the sums above are written. A test compares the mixing to the paper's equations (12)-(14) written literally with an asymmetric matrix.
- **Likelihood factor:** the paper's (15) drops the $`(2\pi)^{-m/2}`$ factor. It is the same for both modes here (same $`y`$), so it cancels in step 4; the code keeps it and works with logs, so a very surprising measurement does not underflow.
- **Pruning:** a mode with $`\bar\mu_j = 0`$ is pruned in the paper. The code keeps it with its probability at 0 (a mode that can never be entered, as in the "always moving" test).
- **Which switching fits:** the paper reports that the IMM works well when the dynamics matrices or noise gains switch between modes (§IV), which is the case here ($`F`$ and $`Q`$ differ between S and M; $`H`$ does not). The paper's model, eq. (1)-(2), lets the dynamics, the noise gains and the measurement matrix switch with the mode.
- **Checks:** with the transition matrix forced to "always moving" the IMM equals the parent's `sensor_fused` (pad) and `never` (position only) to $`10^{-9}`$ (test), and a plain-loop reference written from the paper's equations matches `run_imm` to $`10^{-10}`$.

## Transition matrix: a design choice

$`\Pi`$ is not measured: it is a **filter design choice**, derived here from the simulation's own parameters (all printed by the script). Two rules:

Symbols: $`t_{\text{anchor}} = t_{\text{extend}} = 1`$ s are the anchor and extend durations per cycle, $`t_{\text{slip}} = 0.6`$ s the slip pulse duration and $`p_{\text{slip}} = 0.5`$ the probability that an anchor phase contains a pulse. "Still to moving" is the probability of switching from S to M within one tick.

| Rule and tick | Entries (defaults) | Derivation |
| --- | --- | --- |
| **Constant** (`imm`, `imm_pos`) | still to moving 0.1071, moving to still 0.0577 | Mean sojourn time of each mode over one gait cycle: still $`T_S = (t_{\text{anchor}} - p_{\text{slip}}\, t_{\text{slip}}) / (1 + p_{\text{slip}}) = 0.7 / 1.5 = 0.467`$ s, moving $`T_M = (t_{\text{extend}} + p_{\text{slip}}\, t_{\text{slip}}) / (1 + p_{\text{slip}}) = 1.3 / 1.5 = 0.867`$ s. Still is entered at each anchor start and after each slip pulse, moving at each extend start and each slip start, so each is entered $`1 + p_{\text{slip}}`$ times per cycle. Switch probability per tick $`= \Delta t / T`$. |
| **Schedule-aware** (`imm_schedule`, `imm_schedule_pos`): scheduled extend tick | both rows $`[0.05,\ 0.95]`$ | $`\Delta t / t_{\text{extend}} = 0.05`$ is the share of extend ticks with true speed exactly 0 (the first one), so the next mode is moving with probability 0.95. |
| Schedule-aware: scheduled anchor tick | still to moving 0.0357, moving to still 0.0833 | A slip starts at rate $`p_{\text{slip}} / (t_{\text{anchor}} - p_{\text{slip}}\, t_{\text{slip}}) = 0.714`$ per still second, times $`\Delta t`$. A slip lasts $`t_{\text{slip}}`$, so it ends with probability $`\Delta t / t_{\text{slip}}`$ per tick. |
| Schedule-aware: first scheduled anchor tick after an extend tick | both rows $`[1,\ 0]`$ | The gait ends the extend ramp at $`v = 0`$ and a slip pulse starts from $`v = 0`$, so the body is exactly still on that tick. |

Initial mode probabilities are the stationary distribution of the matrix used at tick 1: still 0.35 (constant) and 0.7 (schedule-aware). A test checks the sojourn formula against the simulated truth (within 12%). Both modes start from the same initial state and covariance.

The last row is part of the default. The first version of the schedule-aware rule used the scheduled phase of the next tick only, so that tick had the anchor matrix too. It is kept as the labeled variant `imm_schedule_next`.

## Results

All tables: seed 0, 500 trials, NEES by window against the 95% interval $`[1.83, 2.18]`$ for the average of a consistent 2-DoF filter (the expected value is 2). Single-run RMSE is one noise realization (RMSE vel in m/s). Command that prints every table of §A, §B, §C, the sensitivity table and the detection tables (about two minutes): `uv run python use_numpy/inchworm_slip_imm.py --out assets/inchworm_slip_imm.png`. The other tables name their command.

### A: with the pad sensor

| Method | RMSE vel | Clean anchor | Slip | Cruise | Overall |
| --- | --- | --- | --- | --- | --- |
| `never` | 0.0496 | 1.42 | 2.58 | 1.64 | 1.66 |
| `schedule` | 0.0404 | 0.94 | 47.27 | 1.85 | 4.60 |
| `sensor_gate` | 0.0403 | 1.05 | 3.57 | 2.67 | 2.20 |
| `schedule_and_sensor` | 0.0392 | 0.94 | 3.56 | 1.70 | 1.54 |
| `sensor_fused` | 0.0090 | 1.69 | 1.95 | 1.77 | 1.74 |
| `imm` | 0.0081 | 1.33 | 1.83 | 1.99 | 1.69 |
| `imm_schedule` | 0.0078 | 1.31 | 1.94 | 1.99 | 1.65 |

![Three panels from inchworm_slip_imm.py: the velocity of the truth, schedule_and_sensor, sensor_fused and imm_schedule; the probability of the moving mode of imm_schedule, imm_schedule_pos and imm_pos; and the Monte Carlo NEES of seven methods on a log scale](../../assets/inchworm_slip_imm.png)

*Figure: `use_numpy/inchworm_slip_imm.py` at its defaults, seed 0, plotted by `uv run python use_numpy/inchworm_slip_imm.py --out assets/inchworm_slip_imm.png`. Gold = slip pulses, light blue = scheduled anchor phases. P(M) is the probability of the moving mode.*

- **Slip window:** both IMMs are as consistent as `sensor_fused` (1.83 and 1.94 vs. 1.95; the interval's upper end is 2.18) and far below `schedule` (47.27).
- **Velocity error:** both IMMs are lower than `sensor_fused` in the single run (0.0081 and 0.0078 vs. 0.0090). Over the 500 trials, split by window, the gain is only on still ticks:

| RMS velocity error over the trials (m/s) | Clean anchor | Slip | Cruise | Overall |
| --- | --- | --- | --- | --- |
| `schedule` | 0.0040 | 0.0596 | 0.0553 | 0.0410 |
| `sensor_fused` | 0.0097 | 0.0091 | 0.0089 | 0.0093 |
| `imm_schedule` | 0.0066 | 0.0098 | 0.0089 | 0.0081 |

On clean anchors the still mode adds its "v = 0" belief to the pad reading, which `sensor_fused` lacks. It does not match the plain ZUPT of `schedule` there (0.0066 vs. 0.0040). In the slip window the IMM is slightly worse than `sensor_fused` (0.0098 vs. 0.0091) and in cruise they are equal.

- **Clean anchor and cruise NEES:** 1.31 and 1.99 for `imm_schedule` against 1.69 and 1.77 for `sensor_fused`. The truth has no random disturbance (see [Model](#model)), so values below 2 mean a conservative covariance.
- **Cost:** roughly 3-5 times a parent EKF per tick (105 µs per step for `imm_schedule` vs. 21-35 µs for the five policies in one run on a shared laptop; the IMM is run as a batch of one trial).

Seeds 0-3 (each seed draws different slip pulses: 2 in seed 0 with 24 slip ticks, 6 in seeds 1-3 with 72). Command: `uv run python use_numpy/inchworm_slip_imm.py --seeds-table 0 1 2 3`.

| Seed | Slip ticks | Slip NEES: `schedule` | `sensor_fused` | `imm_schedule` | RMSE vel: `sensor_fused` | `imm_schedule` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 24 | 47.27 | 1.95 | 1.94 | 0.0090 | 0.0078 |
| 1 | 72 | 48.29 | 1.87 | 1.94 | 0.0094 | 0.0085 |
| 2 | 72 | 47.58 | 1.90 | 1.93 | 0.0088 | 0.0079 |
| 3 | 72 | 47.65 | 1.90 | 1.93 | 0.0092 | 0.0084 |

The slip-window NEES of `imm_schedule` is within 0.07 of `sensor_fused`'s in every seed, and its single-run velocity RMSE is lower in all four (9-13%). This is one noise realization per seed, so the RMSE gap is consistent in sign but is not a significance test.

### B: no pad sensor (position only)

| Method | RMSE pos | RMSE vel | Clean anchor | Slip | Cruise | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| `never` | 0.0101 | 0.0496 | 1.42 | 2.58 | 1.64 | 1.66 |
| `schedule` | 0.0089 | 0.0404 | 0.94 | 47.27 | 1.85 | 4.60 |
| `imm_pos` | 0.0097 | 0.0457 | 1.03 | 2.92 | 2.83 | 1.86 |
| `imm_schedule_pos` | 0.0087 | 0.0401 | 0.79 | 9.34 | 1.92 | 1.85 |

- **`imm_schedule_pos`** keeps the schedule's accuracy (RMSE pos 0.0087 vs. 0.0089, vel 0.0401 vs. 0.0404; both better than `never`) and cuts the slip-window NEES from 47.27 to 9.34. It is still far above `never` (2.58) and the interval. Across seeds 1-3 its slip NEES is 6.40-6.96 (`schedule`: 47.58-48.29, `never`: 2.00-2.13).
- **`imm_pos`** (no schedule) stays near `never` (RMSE vel 0.0457 vs. 0.0496, slip 2.92 vs. 2.58) with a worse cruise NEES (2.83 vs. 1.64). It cannot tell the modes apart: P(M) above 0.5 on 99.2% of clean-anchor ticks, and its mean P(M) is 0.60 on clean-anchor ticks and 0.63 on slip ticks ([Slip detection](#slip-detection)).
- **Why position alone does not work:** the largest slip displacement per tick is $`0.08 \times 0.05 = 4`$ mm against a position noise of 20 mm.

### C: pad noise sweep

Gate threshold stays $`3\sigma_d`$ as in the parent. Command: the same default command (it prints three "Regime C" blocks; the extra levels are the defaults of `--noise-sweep`, 0.002 and 0.005 m). `imm_pos` ignores the pad, so its NEES is the same at every level (checked: maximum difference 0).

| Pad noise $`\sigma_d`$ (as a speed) | Method | RMSE vel | Clean anchor | Slip | Cruise |
| --- | --- | --- | --- | --- | --- |
| 0.5 mm (0.010 m/s) | `schedule_and_sensor` | 0.0392 | 0.94 | 3.56 | 1.70 |
| | `sensor_fused` | 0.0090 | 1.69 | 1.95 | 1.77 |
| | `imm_schedule` | 0.0078 | 1.31 | 1.94 | 1.99 |
| 2 mm (0.040 m/s) | `schedule_and_sensor` | 0.0404 | 0.94 | 42.30 | 1.84 |
| | `sensor_fused` | 0.0231 | 1.64 | 2.02 | 1.72 |
| | `imm_schedule` | 0.0195 | 1.05 | 4.27 | 1.82 |
| 5 mm (0.100 m/s) | `schedule_and_sensor` | 0.0404 | 0.94 | 46.80 | 1.85 |
| | `sensor_fused` | 0.0353 | 1.53 | 2.27 | 1.68 |
| | `imm_schedule` | 0.0302 | 0.91 | 8.26 | 1.92 |

- **The IMM keeps the lower velocity error** (0.0195 vs. 0.0231 and 0.0302 vs. 0.0353) and a lower clean-anchor NEES (0.91-1.05 vs. 1.53-1.64; both are below 2, see [Model](#model)).
- **`sensor_fused` keeps the slip window consistent** (2.02 and 2.27) and the IMM does not (4.27 and 8.26). The IMM's mean P(M) on slip ticks falls to 0.58 and 0.31 ([Slip detection](#slip-detection)), so on the undetected ticks the still mode, which holds $`v = 0 \pm 0.01`$ m/s while the body slides at up to 0.08 m/s, keeps a large weight. This is the likely cause; no run isolates it.
- **The gate collapses first:** at 2 mm the gate threshold ($`3\sigma_d = 6`$ mm per tick) is above the largest slip displacement (4 mm per tick), so `schedule_and_sensor` rarely vetoes the ZUPT (only when noise pushes a reading past the threshold) and nearly returns to `schedule`'s level (42.30 vs. 47.27).

### Switch-rate sensitivity

All switch probabilities of the schedule-aware matrices (except the exact $`[1, 0]`$ row) are multiplied by a factor, and clipped to 1. The two bottom rows scale only one of the two matrices, to isolate where the failure at $`\times 10`$ comes from.

| Variant | Factor | Clean anchor | Slip | Cruise | Overall | RMSE vel |
| --- | --- | --- | --- | --- | --- | --- |
| `imm_schedule` | 0.1 | 1.31 | 2.18 | 1.98 | 1.67 | 0.0079 |
| | 1 | 1.31 | 1.94 | 1.99 | 1.65 | 0.0078 |
| | 10 | 1.48 | 1.95 | 2.14 | 1.83 | 0.0081 |
| `imm_schedule_pos` | 0.1 | 0.90 | 24.00 | 1.74 | 2.86 | 0.0405 |
| | 1 | 0.79 | 9.34 | 1.92 | 1.85 | 0.0401 |
| | 10 | 18.72 | 21.39 | 40.80 | 27.39 | 0.0585 |
| | 10, anchor matrix only | 0.90 | 18.32 | 2.11 | 2.96 | 0.0402 |
| | 10, extend matrix only | 10.91 | 17.28 | 25.56 | 17.67 | 0.0606 |

- **With the pad** the IMM changes little over two decades of switch rates (overall NEES 1.65-1.83), likely because the pad reading outweighs the prior.
- **Position only** it depends on them, because the data cannot overrule the prior. Slowing the switches (0.1) pushes it toward `schedule` (slip 24.00); speeding them up 10 times collapses it. The cruise NEES shows which matrix: 25.56 when only the extend matrix is scaled (at $`\times 10`$ it puts probability 0.5 on "still" at every extend tick, instead of 0.05), 2.11 when only the anchor matrix is.

### Ablations

Command: `uv run python use_numpy/inchworm_slip_imm.py --p-slip 0 --noise-sweep`.

**No slip (`--p-slip 0`).** With nothing to detect, what does the IMM lose against `schedule`?

| Method | RMSE pos | RMSE vel | Clean anchor | Cruise | Overall |
| --- | --- | --- | --- | --- | --- |
| `schedule` | 0.0080 | 0.0360 | 0.93 | 1.66 | 1.32 |
| `imm_schedule_pos` | 0.0080 | 0.0356 | 0.93 | 1.98 | 1.39 |
| `sensor_fused` | 0.0038 | 0.0090 | 1.69 | 1.76 | 1.72 |
| `imm_schedule` | 0.0038 | 0.0074 | 1.32 | 2.08 | 1.66 |

It loses nothing in accuracy, raises no false alarms on clean anchors (0.000 for both schedule-aware IMMs), and loses a little cruise consistency (1.98 vs. 1.66 without the pad; 2.08 vs. 1.76 for `sensor_fused` with it).

**Still-mode position dynamics** (`--still-model integrate --noise-sweep`, $`p' = p + v\,\Delta t`$ with the carried velocity instead of $`p' = p`$). Neither is clearly better, so $`p' = p`$ stays the default:

| `imm_schedule` | RMSE pos | Clean anchor | Slip | Cruise |
| --- | --- | --- | --- | --- |
| hold, $`p' = p`$ (default) | 0.0038 | 1.31 | 1.94 | 1.99 |
| integrate, $`p' = p + v\,\Delta t`$ | 0.0035 | 1.35 | 2.24 | 1.78 |

## Slip detection

The same default run, animated: the body on its track (colored by `imm_schedule`'s P(M)), the velocity estimates, P(M) of the two schedule-aware IMMs, and the Monte Carlo NEES. Slip pulses are shaded gold.

![Animation of the 1D inchworm over ten cycles with two slip pulses: the body colored by the probability of the moving mode, velocity estimates of schedule_and_sensor, sensor_fused and imm_schedule, the probability of the moving mode of imm_schedule and imm_schedule_pos, and the Monte Carlo NEES of four methods on a log scale against the expected value 2](../../assets/inchworm_slip_imm.gif)

*Animation: `generator/animate_inchworm_slip_imm.py` at its defaults (seed 0, 500 trials), rendered by `uv run python generator/animate_inchworm_slip_imm.py --out assets/inchworm_slip_imm.gif`.*

A method "detects slip" on a tick when its P(M) is above 0.5. The table counts, over the 500 trials: the share of slip ticks flagged, the share of clean-anchor ticks flagged (**false alarms**), the mean P(M) on those two windows, the share of (trial, pulse) pairs already flagged on the pulse's first tick, and the **delay** from a pulse's first tick to its first flagged tick (over the pairs not already flagged and flagged within the pulse). The reference row flags a tick when $`\lvert d_k \rvert`$ is above the gate threshold ($`3\sigma_d = 1.5`$ mm).

| Method | Slip ticks flagged | Clean false alarms | Mean P(M) slip / clean | Delay (s) | Pulses never flagged (of those not pre-flagged) |
| --- | --- | --- | --- | --- | --- |
| `imm` | 0.949 | 0.112 | 0.918 / 0.247 | 0.060 | 0.000 |
| `imm_schedule` | 0.857 | 0.000 | 0.837 / 0.047 | 0.081 | 0.000 |
| `imm_pos` | 0.995 | 0.992 | 0.631 / 0.602 | 0.109 | 0.000 |
| `imm_schedule_pos` | 0.079 | 0.005 | 0.233 / 0.184 | 0.484 | 0.538 |
| sensor gate, $`\lvert d_k \rvert > 1.5`$ mm | 0.752 | 0.003 | not applicable | 0.066 | 0.000 |

At pad noise 2 mm and 5 mm (the Regime C blocks of the same command), `imm_schedule` flags 0.592 and 0.209 of the slip ticks, with false alarms 0.009 and 0.009, while the gate flags 0.078 and 0.009.

- **`imm_schedule` flags more slip ticks than the gate** (0.857 vs. 0.752) with no false alarms on clean anchors (0.000 vs. 0.003), at a delay of 0.081 s against 0.066 s.
- **`imm` (no schedule) has false alarms** on 11.2% of clean-anchor ticks; the schedule removes them.
- **`imm_schedule_pos` misses most of the slip** (7.9% of slip ticks flagged, 53.8% of the not-already-flagged pulses never flagged).

### The next-tick-only rule looks like detection and is not

With the first rule (`imm_schedule_next`: the matrix depends only on the scheduled phase of the next tick), the first anchor tick after an extend phase uses the anchor matrix, whose moving-to-still entry is only 0.0833 per tick. The moving mode left over from the extend phase therefore lingers into the anchor phase. Both of seed 0's pulses begin within 0.2 s of the anchor phase's start (0.05 s and 0.20 s after it began).

| Method (seed 0) | Slip ticks flagged | Clean false alarms | Pairs pre-flagged | Slip NEES |
| --- | --- | --- | --- | --- |
| `imm_schedule_next` (pad) | 0.923 | 0.072 | 0.384 | 1.89 |
| `imm_schedule` (pad) | 0.857 | 0.000 | 0.003 | 1.94 |
| `imm_schedule_next_pos` | 0.407 | 0.264 | 0.902 | 3.49 |
| `imm_schedule_pos` | 0.079 | 0.005 | 0.000 | 9.34 |

Without the pad, `imm_schedule_next_pos` flags 40.7% of slip ticks and has a slip NEES of 3.49, which looks like detection. The two rules differ only on the first anchor tick after each extend phase, and the table shows what that changes: 90.2% of its (trial, pulse) pairs are already flagged on the pulse's first tick (0.0% for `imm_schedule_pos`) and 26.4% of clean-anchor ticks are false alarms (0.5%). So these flags are leftover motion, not evidence. With the first tick forced to "still", the flagged share falls to 7.9% and the slip NEES rises to 9.34 (likely because the lingering moving mode had been keeping it down). With the pad, the evidence outweighs the prior and the two rules differ little (slip NEES 1.89 vs. 1.94). Seed 0 has both pulses close to the boundary, which suits the artifact; seeds 1-3 were not run with the next-tick-only rule.

## What this toy does not show

- **A pad sensor that sees only the anchored pad.** The reading here exists on every tick, as in §6 of the parent doc. A real pad-mounted sensor reads only while its pad is on the ground.
- **Transition probabilities from data.** They are derived from this toy's own gait and slip statistics, which a real robot does not give. The sweep shows the pad-aided IMM changes little over two decades of switch rates, and the position-only IMM does not.
- **Only two modes and one slip shape.** One peak speed, one duration, no dropout or quantization of the sensor, noise levels not any robot's.
- **Few pulses per seed.** The sweeps use seed 0 (2 pulses; both near the extend phase). Seeds 0-3 are shown for Regimes A and B only; Regime C and the sweeps are seed 0 only.
- **Single-run RMSE** is one noise realization (the windowed RMS over trials is given for velocity in A only).
- **Truth without random disturbance**, so NEES below 2 measures a conservative covariance, not exact calibration. Process noise is matched (0.5 m/s²) to the ramps (0.5 and 0.4 m/s²), as in §6.

## Takeaway

- **With a pad reading,** the IMM matches `sensor_fused`'s slip-window NEES and lowers the velocity error on still ticks, with fewer false alarms and more slip detected than the gate. With a noisy pad it keeps the lower velocity error but loses slip-window consistency (4.27 and 8.26 at 2 and 5 mm, against 2.02 and 2.27 for `sensor_fused`).
- **Without a pad reading,** position noise hides the slip. Weighing two hypotheses cannot add information the measurements lack, and the schedule-aware IMM only improves on `schedule`'s overconfidence (47.27 to 9.34), not on `never`.
- **The transition matrix matters most where the data is weakest.** In this toy, pinning the first anchor tick to "still" is what exposes the missing detection, and a "detection" result should be checked against the share of pulses already flagged on their first tick.

## References

1. Blom, H. A. P., & Bar-Shalom, Y. (1988). *The Interacting Multiple Model Algorithm for Systems with Markovian Switching Coefficients*. IEEE Transactions on Automatic Control, 33(8), 780-783. https://doi.org/10.1109/9.1299 - the IMM algorithm: model (1)-(2), the cycle in §III with mixing (12)-(14), mode probabilities (15)-(17) and combination (18)-(19); §IV for the comparison of which switching the IMM handles well. The transition-matrix convention differs by a transpose (see [the table above](#the-imm-in-one-table)).
2. Kubík, J. & Faigl, J. (2026). *Adaptive Friction-based Inchworm-like Locomotion*. 2026 IEEE International Conference on Robotics and Automation (ICRA), 9951-9958. https://doi.org/10.1109/ICRA57385.2026.11696680 - an inchworm robot with a laser-mouse displacement sensor in each pad; the motivation for the parent doc's §6, whose toy this doc extends. This doc is a 1D toy, not a model of that robot.
