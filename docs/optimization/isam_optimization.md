# iSAM / incremental optimization

Instead of solving the entire factor graph from scratch every time a new measurement arrives, update the existing solution intelligently.

---

## 1. The problem with ordinary batch optimization

Imagine a robot moving:

```text
t0       t1       t2       t3
X0 ───── X1 ───── X2 ───── X3
```

At every timestep, you receive new measurements.

A traditional **batch** approach might do:

```text
Measurement at t0
       ↓
Optimize X0

Measurement at t1
       ↓
Optimize X0, X1

Measurement at t2
       ↓
Optimize X0, X1, X2

Measurement at t3
       ↓
Optimize X0, X1, X2, X3
```

So every time a new measurement arrives, you potentially solve the **whole problem again**.

For a large SLAM system:

```text
X0 X1 X2 ... X1000
```

re-solving everything repeatedly becomes expensive.

---

## 2. Incremental optimization says:

Instead:

```text
Old solution
     +
New measurement
     ↓
Update only what needs updating
```

Conceptually:

```text
Before:

X0 ─── X1 ─── X2 ─── X3
              ↑
        already solved


New measurement:

X3 ─── X4
       ↑
      new


Don't throw everything away!

Just update the existing solution.
```

This is the fundamental idea behind **iSAM (incremental Smoothing and Mapping)**.

---

## 3. Why is this possible?

This is where things get interesting.

Suppose we have a [factor graph](factor_graph.md):

```text
X0 ─── X1 ─── X2 ─── X3
              │
              L1
```

After linearization, the nonlinear optimization becomes approximately:

$$A\Delta x=b$$

Instead of directly solving this system every time, we can factorize it:

$$A \approx QR$$

or equivalently work with a related factorization of the information/Hessian system.

The important point is:

> **When a new factor is added, much of the previous factorization is still useful.**

We don't necessarily need to throw away the previous computation.

---

## 4. An analogy: editing a spreadsheet

Imagine you have a huge spreadsheet:

```text
100,000 calculations
```

You change one cell.

A stupid approach would be:

> "Let's recalculate absolutely everything from scratch."

An incremental approach says:

> "Which calculations actually depend on this cell?"

Then update only those.

iSAM applies a similar philosophy to the optimization problem.

---

## 5. The SLAM example

Suppose your robot has estimated:

```text
X0 ── X1 ── X2 ── X3 ── X4
```

and each edge represents odometry.

Then you get a new camera observation at X4.

Maybe it observes landmark L0:

```text
                L0
                ●
               /
              /
X0 ●──X1──X2──X3──X4
                  ↑
             new camera
```

A batch optimizer says:

> Rebuild and solve the entire nonlinear problem.

An incremental optimizer says roughly:

> "The new factor primarily affects X4 and the variables connected to it. Update the existing solution accordingly."

---

## 6. But what about loop closure?

This is where incremental SLAM becomes especially interesting.

Suppose we have:

```text
X0 ── X1 ── X2 ── X3 ── X4
│                         │
└─────────────────────────┘
           loop closure
```

The loop closure can affect **many previous poses**.

So incremental optimization cannot simply update X4.

It needs to determine:

> **Which variables are affected by this new constraint?**

Then update the relevant part of the solution.

This is why iSAM maintains and updates a structured factorization of the problem.

---

## 7. The key concept: Bayes tree

Modern incremental smoothing approaches, particularly **iSAM2**, use a structure called a **Bayes tree**.

You can think of it as a clever organization of the factor graph that tells the optimizer:

> **"If this new measurement arrives, these are the parts of the solution that need to be reconsidered."**

Very roughly:

```text
Factor graph
     ↓
Eliminate variables
     ↓
Bayes tree

        X0
        │
       X1
      /  \
    X2    X3
          │
          X4
```

When a new factor connects X4 to X0:

```text
X0 ───────────── X4
```

the algorithm identifies the affected portion of the Bayes tree and **relinearizes/recalculates that portion**, rather than rebuilding everything.

That's the really powerful idea behind iSAM2.

---

## 8. Batch vs incremental

Here's the simplest comparison:

| Batch optimization              | Incremental optimization        |
| ------------------------------- | ------------------------------- |
| Add measurements                | Add measurements                |
| Rebuild/solve the whole problem | Reuse previous computation      |
| Potentially expensive           | Much more efficient             |
| Good for offline SLAM           | Good for online SLAM            |
| Simple conceptual model         | More complicated implementation |
| Example: standard Gauss-Newton  | Example: iSAM/iSAM2             |

Think:

```text
BATCH

new measurement
      ↓
┌───────────────────┐
│ Optimize ALL      │
│ variables again   │
└───────────────────┘
```

versus:

```text
INCREMENTAL

new measurement
      ↓
┌───────────────────┐
│ Find affected     │
│ variables         │
└─────────┬─────────┘
          ↓
   Update those parts
```

---

## 9. iSAM doesn't mean "never touch old variables"

This is an important misconception.

You might think:

> "Incremental means only optimize the newest pose."

**No.**

Old poses can absolutely be updated.

For example:

```text
Before loop closure:

X0 ── X1 ── X2 ── X3 ── X4
↑
slightly wrong


Loop closure arrives:

X4 ───────── X0
```

Now the loop closure tells us the trajectory is inconsistent.

The optimizer may change:

```text
X0
X1
X2
X3
X4
```

But it does so **selectively and efficiently**, using the existing structure.

That's why the term **smoothing** is important.

The system is not merely estimating:

$$X_t$$

It is continually refining:

$$X_0,\ldots,X_t$$

using all available information.

---

## 10. Filtering vs iSAM

This connects directly to **[filtering vs optimization/smoothing](../filtering_smoothing.md)**.

### EKF-style filtering

Conceptually:

```text
X0 → X1 → X2 → X3 → X4
                 ↑
            current state
```

Once you've processed X0, you largely summarize its information and move forward.

You primarily care about:

$$P(X_t|Z_{1:t})$$

---

### iSAM / smoothing

Instead:

```text
X0 ── X1 ── X2 ── X3 ── X4
│     │     │     │     │
└─────┴─────┴─────┴─────┘
       all history
```

You maintain a representation of the entire trajectory:

$$X_{0:t}$$

and continuously refine it.

So:

> **Filtering:** "What is my best estimate of the robot NOW?"

> **Smoothing:** "Given everything I've seen, what were the best estimates of the robot's states throughout the entire trajectory?"

---

## 11. Where iSAM fits into SLAM mental map

You can now connect your previous topics like this:

```text
                    SLAM
                     │
             State estimation
                     │
          ┌──────────┴──────────┐
          │                     │
      Filtering             Smoothing
          │                     │
       EKF / KF           Factor graph
                                │
                       Nonlinear optimization
                                │
                    ┌───────────┴──────────┐
                    │                      │
                  Batch              Incremental
                    │                      │
             Gauss-Newton            iSAM / iSAM2
             Levenberg-Marquardt
```

And underneath the pose optimization:

```text
SE(3)
 ↓
Lie algebra
 ↓
Perturbation
 ↓
Jacobian
 ↓
Linearization
 ↓
Gauss-Newton
 ↓
Factorization
 ↓
iSAM / iSAM2
```

---

## 12. The most intuitive way to remember iSAM

Imagine you're drawing a map while walking.

### Batch approach

Every time you take one more step:

> "Let me redraw the entire map from scratch."

### Incremental approach

Every time you take one more step:

> "I already have a pretty good map. I'll incorporate this new information into it."

And when you recognize a place you've visited before:

> "Oh! This new observation conflicts with my old map. I need to adjust the affected parts."

That's **iSAM**.

## 13. One-sentence summary

> **iSAM is an incremental factor-graph smoothing algorithm that continuously updates the SLAM solution as new measurements arrive, reusing previous computations instead of repeatedly solving the entire problem from scratch.**

And **iSAM2** takes this further by using a **Bayes tree** to efficiently identify and update the parts of the solution affected by new measurements.

---

## 14. References

1. Dellaert, F., & Kaess, M. (2006). *Square Root SAM: Simultaneous Localization and Mapping via Square Root Information Smoothing*. International Journal of Robotics Research, 25(12), 1181–1203. https://doi.org/10.1177/0278364906072768 - the sparse QR/square-root-information factorization behind §3's $A \approx QR$ and the claim that most of the previous factorization stays reusable.
2. Kaess, M., Ranganathan, A., & Dellaert, F. (2008). *iSAM: Incremental Smoothing and Mapping*. IEEE Transactions on Robotics, 24(6), 1365–1378. https://doi.org/10.1109/TRO.2008.2006706 - the original iSAM algorithm (incremental QR updates via Givens rotations, with periodic variable reordering) behind §1, §2, §4–§6, and §9.
3. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). *iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree*. International Journal of Robotics Research, 31(2), 216–235. https://doi.org/10.1177/0278364911430419 - the Bayes-tree data structure and fluid relinearization behind §7, §11, and §13's iSAM2 description.
4. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - general reference for the factor-graph formulation underlying §3 and §11.
