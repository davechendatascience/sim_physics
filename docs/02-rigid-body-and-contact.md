# 02 — Rigid Bodies & Contact

## 1. Rigid bodies as affine bodies

Classic rigid-body engines use reduced coordinates (position + quaternion) with constraint impulses. To share one solver with deformables, rigid bodies here are **affine bodies** (Affine Body Dynamics, Lan et al. 2022):

- DOFs: `x(X) = A X + p`, with 12 DOFs per body
- An orthogonality potential `κ·V·‖AᵀA − I‖²_F` keeps `A` close to a rotation. Stiffness `κ` is large (for example 10⁸ Pa-scaled) but finite and handled implicitly.
- Mass matrix is constant (no gyroscopic terms to integrate explicitly). Contact with deformables uses the same barrier as everything else.
- For output, `A` is projected to the nearest rotation (polar decomposition) to report pose, angular velocity and so on.

A body that is "rigid but can break or yield" (a plastic cup, a thin bracket) is authored as a `Solid` or `Shell` with stiff material, not as `Rigid`. `Rigid` is an explicit approximation declaration.

### Articulated robots
- Joints are stiff penalty/augmented-Lagrangian terms in the potential for T2/T1. Reduced-coordinate Featherstone dynamics is used in T0 for speed.
- Actuators: position, velocity, torque and **impedance** (`τ = K(q_d − q) + D(q̇_d − q̇) + τ_ff`) modes. Gripper **force-control** mode: a target normal force with a bandwidth limit and a force-sensor noise model.
- Models of transmission compliance and backlash are optional but matter for force-accurate grasping.

## 2. Contact model

### Normal contact: IPC barrier
For every close primitive pair (point–triangle, edge–edge) at distance `d`:
```
b(d) = −κ_b (d − d̂)² ln(d/d̂)   for 0 < d < d̂,   else 0
```
- Guarantees no interpenetration and no tunneling, which is essential for 0.1 mm walls
- `d̂` (activation distance) is set per material pair, around 10⁻⁵–10⁻⁴ m. It is small enough not to create visible gaps and large enough to keep conditioning sane.
- Contact force = `−∂b/∂d`. We get **distributed pressure fields over contact patches** rather than a single point force. The patch field is exactly what tactile sensors and dent predictions need.

### Friction
- Lagged, smoothed Coulomb friction potential (IPC-style): the normal force and tangent basis are fixed from the previous Newton iterate or step, and the friction potential is `μ λ_n f₀(‖u_T‖)` with a C¹ transition at slip velocity `ε_v`
- Distinct static and kinetic `μ` via a velocity-dependent `μ(v)` (Stribeck-like curve), needed for stick-slip onset at the start of a lift
- **Pressure-dependent friction** for elastomer pads: `μ = μ₀ (p/p₀)^(n−1)` with n about 0.8–0.95. Rubber's effective friction coefficient falls as contact pressure rises, which changes the optimal grip force.
- **Surface condition as material-pair state**: dry, wet (condensation on a cold can!), oily. Each has its own `μ` distribution.

### Contact patch torsion (soft-finger)
Because contact is a distributed patch, torsional friction (resisting rotation about the contact normal) emerges automatically. For T0 (point contacts), an explicit **soft-finger model** is used instead: a friction ellipsoid `f_t²/μ² + τ_n²/(μ e)² ≤ f_n²`, where `e` is the effective patch radius. This matters when a can is grasped away from its center of mass: gravity torque must be resisted by patch torsion or the can swings in the grip.

## 3. Grasp mechanics: minimum hold force

For an object of mass `m` held by `k` contacts with normal forces `N_i`, the quasi-static requirement (translational part) is

```
Σ μ_i N_i  ≥  m ‖g + a‖ · SF
```

- `a` is the planned peak acceleration of the object (from the trajectory, not only the static case)
- `SF` is a safety factor that the robust layer computes from uncertainty in `m`, `μ` and `a`, instead of a hand-picked value, see [04](04-inverse-problems.md)
- Moments: with COM offset `r`, the patch torsion capacity must satisfy `Σ μ_i e_i N_i ≥ m ‖g + a‖ ‖r_⊥‖`. With liquid inside, `r` moves over time (slosh), see [03](03-deformables-and-materials.md) §5.

The simulator computes this from full dynamics. The formula is used for **sanity checks** and as the **T0 fallback**.

## 4. Grasp mechanics: maximum safe force

The upper bound is *not* a rigid-body quantity. It comes from the deformable's material model: the largest contact pressure field for which the irreversible ledger stays unchanged. The two bounds together form the **grasp force envelope** `[F_min, F_max]`. It is defined per contact location, pad geometry and motion, and it is the central quantity of this project. The worked example is in [05](05-case-study-soda-can.md).

## 5. T0 compliant contact

For the fast tier, the IPC barrier is replaced by MuJoCo-style soft constraints or hydroelastic pressure fields (Drake-style). They are calibrated per material pair so that force–penetration curves match T2 over the working range. The **contact wrench history** of each body in T0 is logged and fed to the damage surrogate, which predicts ledger quantities (see [01](01-architecture.md) §4).
