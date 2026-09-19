# 12 — Differentiable Simulation Without Changing the Physics

The inverse layer ([04](04-inverse-problems.md)) needs derivatives: how a grasp force, a dent or a final pose changes with a pad path, a material parameter or an initial state. This document fixes how `simphys` provides them. The rule is that **the forward simulation is untouched**. Derivatives are computed *about* the physics, never by replacing it with a smoother one.

## 1. Where derivatives come from

Each time step minimises an incremental potential E(x; θ) (docs/01 §3), where θ collects everything a derivative might be taken with respect to: previous state, pad path, material parameters. At a converged step the net force vanishes, ∇ₓE(x*; θ) = 0. Differentiating that condition gives the **implicit derivative**

```
dx*/dθ = −H⁻¹ ∂²E/∂x∂θ,     H = ∇ₓ²E(x*; θ)
```

This costs one extra linear solve per step, with the same sparse structure as a Newton step. Chaining it backwards through the steps (an adjoint) gives derivatives of any later quantity, and checkpointing keeps memory bounded. Nothing in this changes x*, so every law the forward simulation obeys (momentum, dissipation, non-penetration, Coulomb friction, yield) still holds.

The collision check and the line search only decide *how* Newton reaches x*, not *where* x* is, so they are not differentiated.

## 2. Three conditions for correct derivatives

1. **Use the exact Hessian in the adjoint.** Newton projects each stencil's Hessian to PSD, which helps convergence but changes the matrix. The sum of projected stencil Hessians is not the Hessian of the energy. A stencil can have negative curvature at a perfectly stable equilibrium, for example a spring under compression resisting sideways motion. There the projected matrix gives a wrong derivative even though x* is right. The adjoint therefore uses the unprojected Hessian, which the JAX kernels already produce. On the reference model (two masses and a compressed spring, stable at x*), the derivative of the step with respect to the previous state matches finite differences to 10⁻⁹ with the exact Hessian but is 20% wrong with the projected one. That derivative is exactly what an adjoint multiplies step after step. The derivative with respect to the spring stiffness happens to be unaffected, because that parameter pushes only along the spring while the negative curvature is sideways.
2. **Converge before differentiating.** The implicit derivative assumes ∇ₓE = 0. Any leftover residual turns into derivative error, so steps that feed a derivative are solved to a tighter tolerance than the 1 µm the demos use.
3. **Carry the plastic history.** Plasticity is applied after each step (docs/09 §2). The derivative of the next step with respect to the current one therefore passes through the return map, and its *consistent tangent* is part of the chain. For the associative J2 model used here, that tangent is symmetric, as the energy-based stiffness is. Only non-associative plasticity would make it unsymmetric.

## 3. Events: exact forward, smoothed backward only

The response has corners: contact switching on, stick turning to slip, first yield, line contact becoming flat contact. At a corner the derivative jumps. That is correct physics, but a poor search direction for an optimiser. The rule is:

- The forward pass is always exact, with the sharp yield surface and the real contact and friction laws.
- The backward pass may replace the jump by a blend of the one-sided derivatives over a stated width. It may never change the forward result. As the width goes to zero the blended derivative must reduce to the exact one-sided derivatives away from the corner.
- The friction smoothing ε_v is different: it is part of the forward model already, as a bounded creep speed (docs/02 §2). Differentiating it is exact.

## 4. Where derivatives are not useful

Near a buckling or snap-through point the true sensitivity is very large. The short-pad runs show a force drop near 1.6 mm of travel ([11](11-analytic-squeeze.md) §4). Derivatives there are correct but useless for control. A planner keeps a margin from such points and treats a rapidly growing sensitivity as a signal that it is approaching one ([04](04-inverse-problems.md) §0).

## 5. What is already differentiable

The analytic squeeze tier has exact derivatives: `dP/dg = −2P/g` in flat contact and the derivative of the fitted branch in line contact ([11](11-analytic-squeeze.md) §1).

## 6. How it is checked

- **Implicit derivative = finite differences** for a converged step, on the reference integrator.
- **The projected Hessian gives a wrong derivative** where a stencil has negative curvature at a stable equilibrium, and the exact Hessian gives the right one.
- **Convergence matters:** the derivative error falls as the step's residual falls.
- **The J2 consistent tangent is symmetric** for associative plasticity.
- **Backward smoothing** leaves the forward result unchanged and reduces to the exact one-sided derivatives as its width goes to zero.

## 7. Implementation (simphys v1)

`simphys/adjoint.py` records the forward run and propagates an adjoint backwards through it.

- **Per step:** solve `H λ = a` with the *unprojected* sparse Hessian at the converged state (§2.1), restricted to free DOFs. λ gives the parameter gradient `−λᵀ ∂g/∂θ` and the adjoint passed to earlier states.
- **Through time:** implicit Euler makes the inertial target `x̃ₙ = 2xₙ − xₙ₋₁` for n ≥ 1 and `x̃₀ = x₀ + h v₀`. The adjoint therefore flows to the two previous states, and at the first step to the initial velocity. Rigid rotations follow the same rule on `Q̃ = 2Qₙ − Qₙ₋₁`, mapped to rotation increments.
- **Parameters:** `∂g/∂θ` for a material parameter (for example Young's modulus of a body) is a central difference of the gradient in θ, at relative step 10⁻⁶. That is accurate to about 10⁻⁹, which is far below any physical resolution. State dependence is exact.
- **Scope of v1:**
  - Plastic flow is not differentiated yet. If any fiber yielded during the recorded run, the gradient call raises an error instead of returning a derivative that ignores the return map (§2.3).
  - Friction's lagged normal force and tangent basis are held fixed within each step, as differentiable IPC implementations commonly do. Derivatives through frictional contact are therefore approximate. Frictionless and sticking-free scenes are exact.
  - Steps must be converged tightly (§2.2). The recorder stores each step's final residual, and the gradient call warns if it is large.

**What validating it found in the forward solver.** Checking gradients against finite differences of the full simulation exposed three forward-solver problems, which are now fixed or recorded:

1. **Parallel edges.** Edge–edge contact was not mollified, so a block landing flat oscillated in Newton without converging. It is now mollified ([02](02-rigid-body-and-contact.md) §2).
2. **Stopping at the floating-point floor.** Once the predicted decrease fell below the energy's floating-point resolution, Newton kept accepting ever-tinier steps until its iteration cap. It now stops there, as converged.
3. **Degenerate contact geometry, still open.** When a body's vertices lie exactly on another surface's mesh edges (a block centred on the ground's diagonal), the forward result jumps under perturbations of 10⁻³ m/s. A lateral shift then moved the block 18 times its kinematic amount, and 6–9 µm jumps appeared in the landing height. The derivative there is not meaningful. The tests use general position, and robust handling of coincident features is future work.

Validated (tests/sim/test_adjoint.py), against finite differences of the full forward run:

- a soft block landing on the ground, with respect to Young's modulus and all three components of the initial velocity (agreement 10⁻⁴ to 10⁻³);
- a spinning rigid box landing, with respect to its initial linear and angular velocity;
- a shell pressed by a pad, with respect to the wall's modulus.

A run with plastic flow is refused, and recording leaves the forward run bit-identical.
