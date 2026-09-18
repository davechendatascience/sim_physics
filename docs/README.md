# sim_physics — Design Docs

A physics simulator for robot manipulation that models **what happens inside the objects a robot touches**, not just how they move. It covers rigid bodies, deformable solids, and the material science that decides whether an object bends back or stays bent. On top of that it answers inverse questions like *"what is the range of grip force that lifts this can without denting it?"*

The main customer is **VLA (vision-language-action) training**. The simulator produces demonstrations, labels and success checks that are grounded in real material behavior.

## Why this doesn't exist yet

| Simulator family | Strong at | Missing for this goal |
|---|---|---|
| MuJoCo / MJX, PhysX / Isaac, Bullet | Fast rigid-body contact, GPU batching | Deformables are elastic-only or approximate. No plasticity, no damage, no internal state such as pressure. |
| Drake | Accurate contact (hydroelastic), rigorous API | Limited nonlinear materials. Not built for batched learning. |
| Genesis, DiffTaichi, PlasticineLab (MPM/FEM) | Many solvers, differentiable | Simplified material models. Not validated against engineering data. |
| Abaqus, LS-DYNA, Ansys | Engineering-grade constitutive models and validated results | Slow, not differentiable, not built for robots or learning. No batched rollouts. |

The gap: nothing combines **engineering-grade constitutive models**, **tracking of irreversible state** (plastic strain, damage, buckling), **inverse and safety queries**, and **robot-learning throughput**. This design aims at that combination.

## Documents

1. [Architecture](01-architecture.md): layers, data model, time stepping, fidelity tiers
2. [Rigid bodies & contact](02-rigid-body-and-contact.md): dynamics, friction, grasp mechanics
3. [Deformables & materials](03-deformables-and-materials.md): continuum mechanics, constitutive models, material library
4. [Inverse problems](04-inverse-problems.md): state estimation, system ID, inverse manipulation, safety envelopes
5. [Case study: soda can](05-case-study-soda-can.md): the running example worked end to end
6. [VLA integration](06-vla-integration.md): APIs, data generation, labels, sim-to-real
7. [Roadmap & validation](07-roadmap-and-validation.md): phases, benchmarks, risks, open questions
8. [Design validation](08-design-validation.md): how the design is checked against classical physics before implementation
9. [CPU simulator](09-simulator.md): the first implementation (3D, CPU-only), its components, and the closed-form checks it must pass
10. [Performance](10-performance.md): fast on CPU, built for GPU, without changing the physics

## Core ideas in one page

1. **One solver formulation for everything.** Each time step minimizes one *incremental potential*: inertia + elastic energy + plastic dissipation + contact barrier + friction. Rigid bodies, shells, solids and cloth all fit this form, so coupling them needs no special-case code, and the converged step can be differentiated implicitly.
2. **Irreversible state is a first-class citizen.** Every material point carries history variables (plastic deformation, hardening, damage). An object's "state" means its *irreversible* state. Elastic deformation that springs back is allowed. Plastic strain, buckling, fracture and leakage are not. "Don't alter the can" becomes a checkable constraint.
3. **Latent state that cameras can't see is modeled explicitly.** Examples are internal pressure, fill level, material parameters and pre-existing dents. The simulator keeps a probability distribution over these values rather than a single point estimate, because a sealed full can and an opened half-empty can look the same but need very different grasps.
4. **Inverse queries are part of the API.** `grasp_envelope()`, `estimate_state()`, `plan_to_state()` are built-in operations backed by differentiable simulation plus sampling. They are not research scripts bolted on later.
5. **Fidelity tiers with a consistency contract.** Full FEM is the ground truth. Reduced and learned surrogates are trained from it and have measured error bounds. VLA training runs on the fast tiers, and results are spot-checked against ground truth.
6. **Honest about limits.** Plasticity depends on the load path and can't be undone. Buckling is sensitive to tiny imperfections. Material parameters vary from one object to the next. So "knowing all the forces" gives a *posterior* over outcomes, not certainty. The system is built for robust, closed-loop control under uncertainty.
