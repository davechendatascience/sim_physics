# 09 — The CPU Simulator (`simphys`)

The first implementation of the design. It is **3D** and runs on a laptop CPU. It renders either headless (PNG/GIF files) or in a live window. It implements the T2 formulation of [01](01-architecture.md) at small scale: one incremental potential per step for rigid bodies, thin shells and solids, with IPC contact and friction.

## 1. Components

`simphys` is the enclosing component (`CMP-sim` in `belief.yaml`). It contains one component per design layer, and each consumes the design through an `IFC-design__*` interface:

| Component | Design source | What it does |
|---|---|---|
| `CMP-sim.solver` | [01](01-architecture.md) §3 | Incremental-potential step: sparse Newton with per-stencil PSD projection, CCD-bounded line search, lagged friction |
| `CMP-sim.contact` | [02](02-rigid-body-and-contact.md) §2 | Point–triangle and edge–edge IPC barrier with shell-thickness offsets, conservative CCD, smoothed Coulomb friction |
| `CMP-sim.bodies` | [02](02-rigid-body-and-contact.md) §1, [03](03-deformables-and-materials.md) §1 | Rigid bodies (affine, 12 DOF), thin shells (discrete Kirchhoff–Love triangles), solids (linear tetrahedra) |
| `CMP-sim.materials` | [03](03-deformables-and-materials.md) §2, §5 | Neo-Hookean solids, plane-stress J2 through the shell thickness, enclosed gas |
| `CMP-sim.ledger` | [03](03-deformables-and-materials.md) §4 | Plastic strain, residual shape, "altered?" verdict |
| `CMP-sim.render` | — | Headless (PNG/GIF) or windowed (matplotlib 3D) output |

## 2. Models

- **Rigid bodies** are affine: x = A X + p with 12 DOF in 3D, plus the orthogonality potential of [02](02-rigid-body-and-contact.md) §1. The affine kinetic energy uses the body's second moment `J = ∫ρ X Xᵀ dV`. For a body spinning about any axis, it equals the rigid kinetic energy `½ ωᵀ I ω` exactly.
- **Thin shells** are discrete Kirchhoff–Love triangles. Membrane strain comes from the first fundamental form of each face. Bending comes from the second fundamental form built from **mid-edge normals**: each edge's normal is the average of its two faces' normals. Both are expressed in the face's rest frame, so the energy does not change under rigid motion.
- **Through-thickness integration** uses **composite Simpson split at the mid-surface (9 points)**. Fiber strain is `ε(z) = ε_m + z κ`, where ε_m and κ are 2×2 tensors. This rule is exact for the elastic bending stiffness, for first yield at the surface fiber, and for the fully plastic moment. Gauss–Lobatto rules underestimate the fully plastic moment: by 8.7% with 5 points, by 4.0% with 7.
- **Shell plasticity** is plane-stress J2 at every fiber, with the closed-form spectral return map of Simo & Taylor and linear isotropic hardening.
- **Solids** are linear tetrahedra with Neo-Hookean energy.
- **Enclosed gas** is isothermal, with energy `−nRT ln V`, where V is the volume enclosed by a closed shell.
- **Contact** is point–triangle plus edge–edge. The distance is measured between shell mid-surfaces minus half of each thickness, so a 0.1 mm wall behaves like a 0.1 mm wall. Shell self-contact is not modeled in this version, so scenarios that fold a shell onto itself are out of scope.
- **Plasticity is staggered.** Plastic flow is applied by a return map after each converged step, not inside the step's minimization as in the variational update of [01](01-architecture.md) §3. The return map only lowers stored energy, so dissipation stays non-negative, and the scheme converges to the variational update as h → 0. This is the one deliberate simplification.

## 3. Rendering

Rendering is matplotlib 3D: pure CPU, no OpenGL. Shell faces are coloured by equivalent plastic strain, so a dent is visible where it forms. Headless mode writes a GIF and a final PNG. Windowed mode opens a matplotlib window and draws while it simulates.

## 4. Closed-form checks: the long-cylinder limit

A can squeezed along its **whole length** by two long flat pads is a ring in plane strain. Per metre of length it must reproduce these results (can wall: `E = 69 GPa`, `ν = 0.33`, `σ_y = 285 MPa`, `t = 0.1 mm`, `R = 33 mm`):

| Quantity | Formula | Value |
|---|---|---|
| Plane-strain modulus | `E' = E/(1 − ν²)` | E' = 77.4 GPa |
| Diametral compliance | `δ = (π/4 − 2/π) F R³/(E' I)` | 8.29×10⁻⁴ m per N/m |
| First-yield fiber stress (axial strain held at zero) | `σ_1y = σ_y/√(1 − ν + ν²)` | 323 MPa |
| First-yield load | `F_y = π M_y / R` with `M_y = σ_1y t²/6` | 51.2 N/m |
| Fully plastic fiber stress | `σ_y' = 2σ_y/√3` | σ_y' = 329 MPa |
| Collapse load (four plastic hinges) | `F_c = 4 M_p / R` with `M_p = σ_y' t²/4` | 99.7 N/m |
| Ring buckling pressure | `p_cr = 3 E' I / R³` | 539 Pa |
| Ovalization stiffening at 2.5 bar | series over even modes n | 339× |

The last line is why a sealed can is hard to dent. Internal pressure is 464 times the ring's buckling pressure, so ovalization is stiffened by more than two orders of magnitude.

The real question, a **short** pad on a can, has no closed form. That is what the 3D simulator is for. The long-cylinder numbers are a lower bound on it, because the shell around a short patch stiffens it.

## 5. Running it

```bash
.venv/Scripts/python -m simphys list
.venv/Scripts/python -m simphys run can_squeeze --headless --out out/can_squeeze.gif
.venv/Scripts/python -m simphys run box_drop --window
```
