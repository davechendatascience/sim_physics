# 01 — Architecture

## 1. Layered view

```
┌──────────────────────────────────────────────────────────────────────┐
│  L5  Learning & Data     VLA env API · dataset export · task specs     │
├──────────────────────────────────────────────────────────────────────┤
│  L4  Inverse / Query     estimate_state · identify · grasp_envelope    │
│                          plan_to_state · robust (chance-constrained)   │
├──────────────────────────────────────────────────────────────────────┤
│  L3  Sensors & Render    RGB-D · segmentation · tactile · F/T · proprio│
├──────────────────────────────────────────────────────────────────────┤
│  L2  Scene & State       bodies · materials · robots · latent state    │
│                          irreversible-state ledger · invariants        │
├──────────────────────────────────────────────────────────────────────┤
│  L1  Solver              incremental-potential step · contact (IPC)    │
│                          constitutive updates · implicit adjoints      │
├──────────────────────────────────────────────────────────────────────┤
│  L0  Compute             GPU kernels · sparse linear algebra · autodiff│
└──────────────────────────────────────────────────────────────────────┘
```

Each layer depends only on the layers below it. L4 treats L1–L3 as a differentiable (or at least sampleable) function `s_{t+1} = step(s_t, u_t, θ)`.

## 2. Data model

### Scene
- **Body**: a geometry plus a discretization plus a material assignment. Kinds:
  - `Rigid`: affine body (12 DOF, stiffly constrained to rotation), see [02](02-rigid-body-and-contact.md)
  - `Solid`: tetrahedral FEM (volumetric: rubber, foam, food, fingertip pads)
  - `Shell`: triangle shell (thin walls: cans, bottles, sheet metal, packaging)
  - `Rod`: discrete elastic rod (cables, wires)
  - `Cloth`: codimensional membrane/shell with low bending stiffness
  - `Particles`: MPM/SPH (granular media, liquids, fracture-heavy materials). Later phase.
- **Material**: a constitutive model plus parameter *distributions*, see [03](03-deformables-and-materials.md)
- **Robot**: articulated rigid bodies, actuators with torque, impedance or force-control modes, and compliant fingertip pads (`Solid` bodies)
- **Lumped subsystems**: low-dimensional states attached to a body. Examples: an enclosed gas volume (pressure), liquid fill (mass + slosh model), a hinge with a detent.

### State
```
SimState
  t
  q, v                      # generalized positions/velocities (all DOFs, stacked)
  history[body][quad_pt]    # irreversible: F_p (plastic def.), α (hardening),
                            #   d (damage), buckled flags, fracture surfaces
  lumped[body]              # pressure p, fill volume, slosh modes, temperature
  contacts                  # active pairs, lagged friction frames, stick/slip
  latent_belief             # (L4) distribution over unobserved params/state
```

The **irreversible-state ledger** is a view over `history` and `lumped`. It answers "has anything irreversible changed since t0?" It exposes scalar summaries such as `max_plastic_strain(body)`, `dent_depth(body)`, `has_buckled(body)`, `leaked(body)` and `spilled_volume(body)`. These are what task success and safety constraints refer to.

### Invariants
An `Invariant` is a predicate over the ledger that must hold for the whole episode:
```python
Invariant("can_intact",
          lambda L: L.max_plastic_strain("can") < 1e-3
                    and not L.has_buckled("can")
                    and L.leaked("can") == 0)
```
Invariants can be used as success criteria, as constraints in inverse queries (L4), or as terminal/penalty signals during training (L5).

## 3. Time stepping: incremental potential

Every step solves

```
q_{n+1} = argmin_q  E(q) =  ½‖q − q̃‖²_M / h²              (inertia; q̃ = q_n + h v_n + h² M⁻¹ f_ext)
                          + Ψ_elastic(q; history_n)          (hyperelastic energy of elastic part)
                          + Φ_plastic(q; history_n)          (variational constitutive update)
                          + B_contact(q)                     (IPC log-barrier, guarantees no penetration)
                          + D_friction(q; lagged normals)    (smoothed Coulomb friction potential)
                          + Σ penalty/augmented-Lagrangian terms for joints & actuators
```

and then updates `history` with the constitutive return map and `v_{n+1} = (q_{n+1} − q_n)/h`.

Why this formulation:
- **Unified coupling.** Rigid, shell, solid and robot DOFs are all just `q`. Contact between any pair is the same barrier term.
- **Robustness.** Newton with line search, filtered by continuous collision detection (CCD), never produces penetration or tunneling. This matters because thin shells (0.1 mm walls) break penalty-based contact.
- **Stability at large time steps** (implicit, dissipative), which throughput needs.
- **Differentiability.** At a converged minimum, `∇E = 0`, so the implicit function theorem gives `∂q_{n+1}/∂(q_n, v_n, u, θ) = −H⁻¹ ∂²E/∂q∂(·)`. Only one extra linear solve with the Hessian already factorized is needed. Adjoints chain across steps.
- **Conservation, stated honestly.** Implicit Euler on this potential conserves linear momentum exactly, because internal and contact forces cancel in pairs. It does *not* conserve angular momentum or energy exactly: both drift at first order in h, and the energy error is always a loss, never a gain. That dissipation is what makes large steps stable. The design-validation suite checks all three properties.
- **Plasticity fits in.** Variational constitutive updates (Ortiz & Stainier 1999) turn associative J2 plasticity into a minimization, so it joins the same potential.

Solver details:
- Newton + projected Hessian (SPD per element) + backtracking line search with CCD step bound
- Linear solves: GPU preconditioned CG for large systems, sparse Cholesky for small/medium, with reuse of the symbolic factorization
- Adaptive time step: shrink on non-convergence or on contact events. Substepping around buckling onset.
- Lumped subsystems are coupled monolithically when stiff (gas pressure ↔ shell volume) or by staggering when weak (slosh)

## 4. Fidelity tiers

The same scene description can run at three fidelities:

| Tier | Purpose | Deformables | Contact | Target speed |
|---|---|---|---|---|
| **T2: Reference** | Ground truth, validation, inverse queries offline | Full FEM shells/solids, elastoplastic, adaptive mesh under contact | IPC + friction | 0.01–1× real time per env |
| **T1: Reduced** | Planning, data generation, closed-loop MPC | Reduced-order (modal + plastic-mode basis, or neural latent dynamics trained on T2) | IPC on reduced coords | 10–100× real time per env |
| **T0: Fast** | Large-scale VLA RL/rollouts, 1000s of envs on GPU | Rigid + learned *damage surrogates* (predict ledger quantities from contact wrench history) | Compliant (MuJoCo-style) contact | 1000s of env-steps per second per GPU |

**Consistency contract:** every T1/T0 model ships with a validation report against T2 on a held-out scenario set. The report gives error in trajectories, contact forces and ledger quantities. T0 must be *conservative* on invariants: its damage surrogate may flag false alarms but must have a bounded miss rate. During data generation, a small fraction of episodes (for example 1%) is re-simulated at T2 to monitor drift.

## 5. Proposed tech stack

- **Kernels:** NVIDIA Warp (Python-authored CUDA, built-in autodiff) for the first version. Move hot paths to C++/CUDA if profiling shows it is worth it. Alternative: JAX, which is cleaner for autodiff but harder for sparse Newton and CCD.
- **Linear algebra:** cuSPARSE/cuDSS, AMGX as a preconditioner option
- **Geometry:** a BVH on GPU for broad phase. CCD adapted from IPC Toolkit (MIT-licensed) as the reference.
- **Front end:** Python package `simphys` with a typed scene API. Scenes serialize to a USD/MJCF-compatible superset with a `material` schema.
- **Rendering:** a rasterizer for speed (T0/T1) and path tracing for photoreal data. Deformation is passed to the renderer each frame.
- **Units:** SI everywhere, checked in the scene loader (a thin-wall mistake of mm vs m is fatal).

## 6. Package layout (proposed)

```
simphys/
  core/        # math, sparse LA, autodiff glue, units
  geometry/    # meshes, BVH, CCD, remeshing
  solver/      # incremental potential, Newton, contact barrier, friction
  bodies/      # rigid (ABD), shell, solid, rod, cloth, particles
  materials/   # constitutive models, material DB, parameter distributions
  lumped/      # gas volume, liquid fill/slosh, joints
  robots/      # articulation, actuators, grippers, fingertip pads
  sensors/     # cameras, tactile, F/T
  ledger/      # irreversible-state tracking, invariants
  inverse/     # estimation, system ID, envelopes, planning
  surrogates/  # T1/T0 model training + validation harness
  envs/        # VLA/RL environment wrappers, dataset exporters
  validation/  # analytic benchmarks, reference comparisons, physical test data
```
