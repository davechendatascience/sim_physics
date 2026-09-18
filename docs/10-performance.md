# 10 — Performance: Fast on CPU Today, Built for GPU

The simulator should run as fast as the hardware allows **without changing the physics it computes**. Every speed-up here is one of two kinds:

- **Exact reorganisations** that produce the same answer to round-off: better orderings, fewer copies, batching, a different linear solver.
- **Approximations whose error is proven, by a design check, to be far below the physical resolution** that matters: the plastic-strain threshold, the contact gap, the 1 µm displacement resolution of the dent verdict.

Anything else is out. In particular, the IPC barrier's singular `ln(d/d̂)` is never approximated. Its blow-up at contact is what guarantees non-penetration ([02](02-rigid-body-and-contact.md) §2).

All demos and scenes run on one physics library, `simphys`. The reference models in `design/oracles/` stay separate on purpose: they are the independent implementation that `simphys` is differentially tested against.

## 1. Where the time goes (measured, open-can squeeze, 3 618 DOF, CPU)

| Cost | Before | Fix | After |
|---|---|---|---|
| Sparse solve per Newton iteration | 0.44 s (default ordering) | SPD mode, minimum-degree ordering on A + Aᵀ | 0.083 s, same answer to 3×10⁻¹¹ |
| Newton iterations per step (pads in contact) | 9–64, often at the 120 cap | the squeeze protocol of [09](09-simulator.md) §5: supported can, pads on a prescribed path | 4–5 in the first steps of the squeeze; adaptive stiffness (§3) not yet built |
| Everything else per Newton iteration (assembly, projection, broad phase, copies) | ~0.4–0.6 s | fused kernels, candidate reuse (§2) | whole iteration now 0.20 s |
| Compilation at the start of every run | ~24 s | persistent compilation cache (§2) | ~4 s with a warm cache |

Friction is not the iteration driver: with friction switched off, steps still took 9–39 iterations. Projecting negative Hessian eigenvalues to their absolute value instead of zero was also measured. It needed *more* iterations (for example 26 instead of 19 per step) for the same answer, so the design keeps clamping to zero.

## 2. One code path for CPU and GPU

Every energy is already a JAX function of a small stencil, differentiated by JAX and batched with `vmap`. JAX compiles the same code for CPU and GPU. The rest of the pipeline follows the same rule:

- **State lives on the device.** Positions, velocities and plastic state stay as device arrays between steps. Host copies happen only for rendering and ledgers.
- **Fixed-capacity buffers.** Contact pair lists and stencil batches are padded to power-of-two capacities, so each size compiles once.
- **Persistent compilation cache.** Compiled kernels are stored on disk, so compilation is paid once per machine, not once per run.
- **Fused kernels.** One compiled call per energy type returns the energy, gradient and PSD-projected Hessian of every stencil, so each is evaluated once and copied off the device once.
- **Candidate pairs are reused.** The broad phase runs with a safety margin, and its pair list stays valid until some vertex has moved more than half that margin from where the list was built. Two primitives that were farther apart than the search radius plus the margin cannot then come within the search radius, so no contact can be missed.
- **Linear solver behind one interface:**
  - *CPU:* sparse direct factorisation (SPD mode, minimum-degree ordering).
  - *GPU:* matrix-free preconditioned conjugate gradients. `H v` is computed stencil by stencil and scattered with a segment sum, so the global matrix is never assembled. The preconditioner is block-Jacobi with 3×3 blocks per vertex and 6×6 per rigid body.
- **Broad phase by spatial hashing:** sort primitives by grid cell, then compare neighbouring cells. This is data-parallel and GPU-friendly; the KD-tree stays for the CPU path.
- **Batched environments.** `vmap` over whole scenes runs many environments per device, which is what VLA data generation needs ([06](06-vla-integration.md) §8).

## 3. Newton: fewer iterations without changing the solution

- **Tolerance in physical units.** Newton stops when no vertex would move more than a set distance in one more iteration. Validation scenes use 10⁻⁵ m/s × h, where it measurably matters: the long-can compliance moved 5% between 10⁻³ and 10⁻⁵ m/s. Demo scenes use 1 µm per step.
- **Adaptive barrier stiffness** (IPC's rule): start κ small enough that the barrier's stiffness is comparable to the elastic stiffness of the bodies in contact, and raise it only while a contact gap keeps shrinking toward zero. The barrier is still infinite at contact, so non-penetration is unchanged.
- **Warm start** from the inertial prediction, cut back by CCD, instead of from the previous state.

## 4. Precision on GPU

Consumer GPUs run float64 at 1/32 to 1/64 of float32 speed, so the GPU path computes stencil energies in float32 and accumulates in float64. Thin shells make this delicate: plainly computed in float32, the shell gradient of a can wall carries a relative error of 6×10⁻⁴ at 1 µm deformations. That comes from cancellation in `a − ā`, the current minus the rest metric, and it would stall Newton near convergence.

Two changes remove it. Kernels take **displacements** `u = x − X` and compute `a − ā = E·Dᵀ + D·Eᵀ + D·Dᵀ`, where E are rest edges and D displacement differences. They also work in a **local origin per stencil**. The float32 gradient error then drops to 2×10⁻⁷, uniformly from 1 µm to 0.1 mm deformations. Solves, residuals and contact distances stay in float64.

## 5. Approximations in the style of Ramanujan

Ramanujan's closed forms are valuable where a cheap formula has an error provably below anything physical. One such place is a fast **rim hoop-strain check** for the ledger and for state estimation from images ([04](04-inverse-problems.md) §1). The rim of an ovalized can is close to an ellipse with semi-axes a and b, and its perimeter by Ramanujan's second approximation is

```
P ≈ π (a + b) (1 + 3h / (10 + √(4 − 3h))),   h = ((a − b)/(a + b))²
```

The hoop strain is `P / (2πR) − 1`. Against the exact elliptic integral, the formula's relative error is below 10⁻¹⁵ up to 5% ovalization and 3.3×10⁻⁸ at 50%. That is at least 6×10⁴ times smaller than the 2×10⁻³ plastic-strain threshold, so the fast check can never flip a dent verdict by itself. A rim that measures longer than 2πR by more than the threshold has been stretched plastically, with no simulation needed.

Where no such bound exists, there is no approximation. The barrier logarithm, the return map, and the material laws stay exact.

**Assessment of the reference brief** (`docs/reference/technical_brief_ramanujan_s_mathematics_in_classical_elasticity_and_shell_mechanics.md`) for speeding up shells and soft solids:

| Brief section | Applies to simphys? | Why |
|---|---|---|
| §2.2 Elliptic-integral metrics instead of quadrature | No speed-up | Discrete Kirchhoff–Love faces are flat: their metric is exact with no quadrature. The through-thickness rule is already exact (composite Simpson). |
| §2.4 Continued-fraction shape functions | No | These are for p-FEM. A rational approximation of the barrier's log would lose its blow-up at contact, which is the non-penetration guarantee. |
| §2.3 Ramanujan sums, modular lattice sums | Not in scope | These need periodic metamaterial shells; none are modeled. |
| Neo-Hookean solids ("jelly") | No | Their per-tet energies are already cheap closed forms; the time goes to Newton, assembly and the solve (§1). |
| §2.1 Elastica in elliptic functions | **Yes, but for validation and one fast regime, not the general solver** | A can squeezed flat along its whole length is an elastica at large deflection. The closed form (K(k), E(k), which cost nanoseconds) checks the 3D shell beyond the small-deflection ring test. It can also serve as an instant surrogate for that one loading case in the fast tier. |
| Perimeter of an ellipse | **Yes** | The rim hoop-strain check above. |

Some of the brief's claims (for example, reducing O(N³) to O(log N)) do not transfer to a contact-driven 3D solver. The speed-ups in §1–§4 do.

## 6. What is measured before a speed-up ships

Every fast path must reproduce the reference path on the validation scenes: free fall, momentum through a collision, energy, incline friction, tunnelling, rolled-strip bending, long-can compliance and sealed-can swelling. It must match to the same tolerances those tests use. The speed-up is recorded as a before/after measurement in §1.
