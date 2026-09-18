# 04 — Inverse Problems

The forward simulator answers *"given actions, what happens?"* This layer answers:

1. **Estimation:** *"given what I observe, what is the state (including hidden parts)?"*
2. **Identification:** *"given interaction data, what are the material parameters?"*
3. **Envelopes:** *"what range of actions leaves the object's irreversible state unchanged while achieving the task?"*
4. **Inverse manipulation:** *"what actions take the object from its current state to a target state?"*

## 0. A necessary caveat on "know the forces → know the manipulation"

The intuition is right, but four facts shape the design:

- **Irreversibility & path dependence.** Plastic deformation depends on the *history* of loading, not only on the final load. Many target states are **unreachable** from the current one (you cannot perfectly un-dent a can by pushing it back). So the inverse map is not a function, and existence has to be checked, not assumed.
- **Non-uniqueness.** Where a solution exists, many action sequences reach it. We need a cost (effort, time, margin to damage) to pick one.
- **Ill-conditioning near instabilities.** Near buckling or snap-through, tiny changes in force produce large changes in outcome. Sensitivities (gradients) blow up. Plans in that regime must keep margin from the instability instead of balancing on it.
- **Partial observability.** Cameras see geometry, not stiffness, pressure, wall thickness or fill level. These must be inferred, often by *touching* the object.

The system therefore always works with a **belief** (a distribution over state and parameters) and produces **robust, closed-loop** plans. It does not claim exact knowledge.

## 1. State estimation (observation → belief)

Inputs: RGB-D sequences, tactile images, F/T, joint torques, robot proprioception. Output: posterior over `(geometry, history field, lumped state, material params)`.

Approach, layered from fast to rigorous:
1. **Learned prior / amortized inference.** A network trained on T2 simulations maps observations to an initial belief: object class, geometry, a coarse dent map, open/sealed probability, fill-level distribution.
2. **Analysis-by-synthesis refinement.** Differentiable rendering + differentiable simulation. Parameters are adjusted to minimize the mismatch between simulated and observed sensor data (depth, silhouettes, tactile, forces). The rest shape (including existing dents) is recovered as a plastic-strain field or directly as a perturbed rest mesh.
3. **Sequential filtering** during manipulation. An ensemble Kalman filter or particle filter runs over a *reduced* state (T1 coordinates + lumped variables + a few material params). The ensemble members are simulations, which gives uncertainty for free.

### Active probing (interactive perception)
When the belief is too broad for a safe decision (for example "sealed full" vs "open empty"), the planner picks a **probing action** that maximizes expected information gain under a strict safety limit. Examples: a light squeeze reading force vs displacement (stiffness shows pressure), a small lift reading the load (shows mass and fill), a gentle tilt (slosh dynamics show fill level). The probe is itself planned with the envelope machinery (§3), so it never damages the object.

## 2. System identification (interaction data → material parameters)

- **Offline** (building the material DB): fit constitutive parameters to lab tests (tensile, Lankford, indentation, can crush tests), minimizing a weighted misfit via adjoint gradients + Levenberg–Marquardt. Report the posterior covariance (Laplace approximation or MCMC on T1 surrogates).
- **Online** (per object instance): update the few parameters that actually vary per instance (wall thickness scaling, friction, pressure) during the episode.
- **Identifiability checks:** compute the Fisher information of the planned experiment before running it. Parameters that are unidentifiable from the available data are flagged, not silently "fit".

## 3. Grasp/manipulation envelopes (the "don't damage it" query)

```python
env = sim.query.grasp_envelope(
    obj="can", gripper="parallel_2f_silicone",
    contact_region=region,          # where on the object (a surface patch or set of candidates)
    motion=lift_trajectory,         # accelerations matter for F_min
    belief=belief,                  # distribution over mass, μ, p, thickness...
    invariant="can_intact",
    risk=1e-3)                      # max acceptable probability of violating either bound
# → Envelope(F_min=..., F_max=..., margin=..., binding_constraints=[...], per_region_map=...)
```

Computation:
- `F_min(θ)`: the smallest grip force with no slip over the trajectory, found by bisection on force with a slip detector (T1/T2), seeded from the analytic bound in [02](02-rigid-body-and-contact.md) §3
- `F_max(θ)`: the largest grip force for which the invariant holds, found by a continuation (load stepping) method on T2 with a residual-displacement check at the end. The load–displacement path is traced, not just its end point, so buckling is detected as a limit point.
- Uncertainty: evaluate over samples of `θ` from the belief (with T1 surrogates plus importance sampling for the tails). The envelope becomes `[quantile_{1−δ}(F_min), quantile_δ(F_max)]`.
- **Empty envelope** (`F_min > F_max`) is a first-class result. It means "this grasp cannot work safely", and the planner must change the contact region, the pad or the motion (slower lift → lower `F_min`), or probe to shrink the uncertainty.
- The output also includes a **per-region map** over the object surface, because stiffness varies (near rims vs mid-wall).

Precomputation: envelopes for canonical (object class, gripper, region) triples are tabulated offline and interpolated at runtime. The live query is only used for novel situations.

## 4. Inverse manipulation (current state → target state)

Formulated as chance-constrained optimal control:

```
min_{u_0..T}   E_θ[ ℓ_T(s_T, s*) ] + Σ_t c(u_t)
s.t.           s_{t+1} = step(s_t, u_t, θ),     θ ~ belief
               P( Invariant violated ) ≤ δ          # e.g. "don't damage"  (or, for shaping tasks,
               robot limits (torque, force, workspace)  #  the target *is* a ledger state)
```

- For "don't alter" tasks, the target is *pose*, and the invariant keeps the ledger constant.
- For **deliberate shaping** tasks (crush a can to a given height for recycling, bend a wire, fold a box), the target *is* a ledger state such as a target residual shape. Reachability is checked first, with a reachable-set approximation built from the plastic-mode basis.

Solvers, combined:
- **Gradient-based** (differentiable sim, implicit adjoints; see [01](01-architecture.md) §3) for smooth phases, such as refining a squeeze profile
- **Sampling-based** (CEM / MPPI) for contact-mode changes and non-smooth landscapes, with gradients used to refine the best samples
- **Receding-horizon execution:** replan at 10–50 Hz on T1 while the estimator (§1) updates the belief from tactile and force feedback. Open-loop plans from any model will fail on real objects. Closing the loop on force is what makes the envelope usable.

## 5. Differentiability notes

- Implicit differentiation through each Newton solve (a single extra linear solve per step with the existing factorization)
- Contact: the IPC barrier is smooth for `d > 0`, and friction is smoothed with the `ε_v` transition. Gradients exist but can be stiff. A *randomized smoothing* option perturbs inputs and averages, which gives better-behaved gradients for planning.
- Plasticity: the return map is non-smooth at yield onset. A smoothed yield function (a regularized max) is used for gradients only, with the exact version in the forward pass.
- Memory: checkpoint every k steps and recompute during the backward pass

## 6. Outputs for downstream users

Every inverse query returns, besides the answer: **uncertainty**, **which constraint binds**, **sensitivity** to each parameter, and a **validity flag** saying whether the query stayed within the calibrated range of the material models. Downstream users (planners, VLA data generators) must be able to tell "safe with margin" from "safe according to a model we have not validated for this case".
