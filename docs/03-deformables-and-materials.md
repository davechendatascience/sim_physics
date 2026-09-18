# 03 — Deformables & Material Science

The simulator's claim to be different rests on this layer. Robotics simulators usually treat "soft" as "elastic with a stiffness knob". Real objects **yield, harden, buckle, crack, creep and leak**, and those irreversible changes are what "altering the state" means.

## 1. Kinematics

- Deformation gradient `F = ∂x/∂X`. Large-deformation (finite strain) throughout.
- Multiplicative split for elastoplasticity: `F = F_e F_p`, where `F_p` is stored per quadrature point as history
- Shells: Kirchhoff–Love discrete shells (membrane + bending from mid-surface) for thin walls with `t/R ≪ 1`. Reissner–Mindlin/solid-shell elements are an option for thicker walls. Through-thickness integration (5–7 Gauss points) is used so plasticity can start at the surface fibers and spread inward. Bending-driven denting needs this.
- **Why shells matter:** a can wall has t/R ≈ 0.003. Tetrahedral meshes of that wall would need either absurd element counts or would lock badly. Getting this wrong is the most common way a "soft body sim" misses denting entirely.

## 2. Constitutive model library

Each model implements `energy(F_e, params)`, `stress`, `tangent` (projected SPD) and `return_map(F_trial, history) → (F_e, history')`. The last one is written in variational form wherever possible so it fits the incremental potential.

| Family | Models | Typical objects |
|---|---|---|
| Hyperelastic | Neo-Hookean, Mooney–Rivlin, Ogden, Arruda–Boyce | Rubber, silicone pads, soft toys, tissue |
| Metal plasticity | J2 (von Mises) with isotropic (Voce/Swift) + kinematic (Armstrong–Frederick) hardening. **Hill48 anisotropic yield** for rolled sheet. Johnson–Cook rate/temperature dependence. | Aluminum cans, steel, foil, wire |
| Polymer | Viscoelastic (Prony series), viscoplastic (Bergström–Boyce) | PET bottles, cups, packaging, films |
| Foam / crushable | Crushable foam (volumetric hardening), Ogden-Hill | Packaging foam, sponges |
| Soft solids / food | Viscoelastic + fracture (phase field) | Fruit, bread, cheese, gels |
| Granular / paste | Drucker–Prager, Cam-clay (MPM) | Sand, dough, rice (later phase) |
| Damage & fracture | Continuum damage (Lemaitre), ductile failure (Johnson–Cook damage), phase-field brittle fracture, cohesive zones | Tearing, cracking, puncture |

Friction is a **material-pair** property with its own library: Coulomb/Stribeck parameters, pressure-dependence exponent, and surface condition, as described in [02](02-rigid-body-and-contact.md).

## 3. Parameters are distributions, with provenance

```yaml
material: aluminum_3004_H19_can_body
model: j2_hill48_voce
params:
  E:        {dist: normal, mean: 69.0e9,  std: 1.0e9,  unit: Pa, src: "ASM Handbook"}
  nu:       {value: 0.33}
  sigma_y0: {dist: normal, mean: 285e6,   std: 12e6,   unit: Pa, src: "supplier cert; to be tested"}
  voce_Q:   {dist: uniform, low: 10e6, high: 40e6, unit: Pa, src: "fit TBD"}
  hill_r:   {r0: 0.6, r45: 0.8, r90: 0.7, src: "placeholder: needs Lankford test"}
  density:  {value: 2720, unit: kg/m^3}
geometry_priors:
  wall_thickness: {dist: normal, mean: 0.100e-3, std: 0.005e-3, unit: m}
  imperfection_field: {kind: gaussian_random_field, amplitude_rms: 0.01e-3, corr_len: 5e-3}
```

- **Every value has a source**: handbook, supplier, test, or explicitly `placeholder`. The material DB reports what fraction of each material has been validated.
- Distributions feed (a) domain randomization for learning, (b) priors for estimation, and (c) chance constraints for safety.
- **Geometric imperfections** are part of the material record. Thin-shell buckling loads are notoriously sensitive to them. Without imperfections, simulated buckling loads are optimistic by 2× or more.

## 4. Irreversibility: what the ledger tracks

| Quantity | Definition | Default "altered" threshold |
|---|---|---|
| Equivalent plastic strain `ε̄_p` | Accumulated from the return map | Configurable. For example 2×10⁻³ = "dent likely visible/tactile". |
| Residual displacement | Unload the body (quasi-statically), compare to reference | For example 0.1 mm |
| Buckling | Sign change / near-zero eigenvalue of the tangent stiffness along the load path, or snap-through energy release | Any |
| Damage `d` | Continuum damage variable | For example `d > 0.05` |
| Fracture / leak | Crack surface present, or enclosed volume's pressure boundary breached | Any |
| Spill | Liquid volume leaving the container (lumped model) | Any |

The **residual displacement check** is the ground-truth definition of "state changed": simulate unloading and see what stays. `ε̄_p` is the cheap per-step proxy.

## 5. Lumped subsystems for enclosed contents

Fully resolving fluids inside containers is too expensive for most uses and usually unnecessary. Lumped models:

- **Enclosed gas:** ideal gas, optionally with CO₂ dissolution (Henry's law, temperature-dependent). Pressure `p = p(V, T, n)` acts as a follower load on the inner surface of the shell. The volume `V(q)` is computed from the shell mesh, and coupling is monolithic (the pressure term's energy is `−∫p dV`, which is added to the potential). **This is what makes a sealed can stiff.**
- **Liquid fill:** mass, a free-surface height, and an equivalent-mechanical sloshing model (pendulum/spring-mass per mode, from linear slosh theory) that exerts time-varying force and moment on the container. Tuned against SPH/FLIP references offline.
- **Open vs sealed** is a discrete latent state. Opening changes the gas model from a closed volume to atmospheric.

## 6. Multiresolution for contact regions

Denting is local (a few mm under a fingertip), but the object is ~100 mm. Therefore:
- Coarse global shell mesh plus **adaptive refinement** under active contact patches, driven by a curvature/strain error indicator
- The history variables (`F_p`, `α`, `d`) are transferred on refinement (with an L²-projection or quadrature-point inheritance), and the transfer is checked to conserve dissipated energy within tolerance
- For T1, a precomputed library of local "dent modes" (plastic-mode basis) is stored per object and per region

## 7. Temperature (optional, phase 4)

Temperature affects gas pressure (a warm can is more pressurized), polymer stiffness, and friction (condensation). The first version holds a scalar temperature per body as a lumped state. Full thermo-mechanical coupling is out of scope until the core is validated.
