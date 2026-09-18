# sim_physics

A physics simulator for robot manipulation that models **what happens inside the objects a robot touches**: rigid bodies, thin shells, soft solids, and the material science that decides whether an object springs back or stays dented. It is built to answer questions like *"what grip force lifts this soda can without denting it?"* and to produce physically grounded data for VLA training.

Two things make it different from a typical robotics simulator:

1. **The design is a tested component.** Every physics claim in the design (`docs/`) is registered and checked against classical physics *before* anything is implemented. The checks cover dimensions, conservation laws, thermodynamic admissibility, stated numbers, regime of validity, and cross-document consistency.
2. **Evidence, not narration.** Test results are recorded by the component-belief MCP server (from the `Theoretically_Driven_LLM_Planning` project) as trial-level evidence. A claim such as "the solver conserves momentum" is only reported once a declared test has measured it.

## Features

| Feature | Status | Where |
|---|---|---|
| **Design validation**: 6 physics-law checks over ~250 registered design claims | ✅ supported (belief server) | [`design/`](design/), [`tests/design/`](tests/design/), [docs/08](docs/08-design-validation.md) |
| **3D CPU simulator** (`simphys`) using one incremental potential per step, sparse Newton, and JAX-derived stencil Hessians | ✅ | [`simphys/scene.py`](simphys/scene.py), [docs/09](docs/09-simulator.md) |
| Rigid bodies with **exact rotations** (Newton on SO(3)) | ✅ | [`simphys/bodies.py`](simphys/bodies.py) |
| **Thin shells**: discrete Kirchhoff–Love, mid-edge-normal bending, 9-point through-thickness integration | ✅ validated against ring theory | [`simphys/kernels.py`](simphys/kernels.py) |
| **Shell plasticity**: plane-stress J2 at every fiber, staggered return map | ✅ matches the design reference model | [`simphys/materials.py`](simphys/materials.py) |
| **Soft solids**: Neo-Hookean tetrahedra | ✅ | [`simphys/bodies.py`](simphys/bodies.py) |
| **Enclosed gas**: a sealed can's internal pressure | ✅ hoop strain within 5% of theory | [`simphys/scene.py`](simphys/scene.py) |
| **IPC contact**: point–triangle + edge–edge barrier, shell-thickness offsets, Lipschitz-sampled CCD | ✅ | [`simphys/collision.py`](simphys/collision.py) |
| **Friction**: lagged, smoothed Coulomb | ✅ matches incline stick/slide | [`simphys/kernels.py`](simphys/kernels.py) |
| **Irreversible-state ledger**: plastic strain + residual shape → "was it altered?" | ✅ | [`simphys/ledger.py`](simphys/ledger.py) |
| **Rendering**: headless (GIF + PNG) or live window, matplotlib only (no GPU) | ✅ | [`simphys/render.py`](simphys/render.py) |
| Built-in scenes: box drop, gelatin jelly drop (squash, bounce, rock), can squeeze (open/sealed), long-can validation | ✅ | [`simphys/scenes.py`](simphys/scenes.py) |
| **Analytic squeeze tier**: can pinched by long pads, ring elastica in closed form (lemniscate constant), force–gap curve + first yield; 35 µs per query (tabulated branch) | ✅ matches the quadrature reference to 1e-6 | [`simphys/analytic.py`](simphys/analytic.py), [docs/11](docs/11-analytic-squeeze.md) |
| Inverse queries (`grasp_envelope`, state estimation), VLA environment API | 🔜 designed | [docs/04](docs/04-inverse-problems.md), [docs/06](docs/06-vla-integration.md) |

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows; use .venv/bin/python elsewhere

.venv/Scripts/python -m simphys list
.venv/Scripts/python -m simphys run box_drop --headless            # writes out/box_drop.gif, .png, .json
.venv/Scripts/python -m simphys run jelly_drop --window            # draws live in a window
.venv/Scripts/python -m simphys run can_squeeze --headless --steps 40
.venv/Scripts/python -m simphys squeeze-curve                     # long-pad pinch force-gap curve, instantly
```

Headless runs print and save a **ledger** for every shell: its maximum plastic strain, its shape change after rigid alignment, and whether it was altered by the design's thresholds (docs/03 §4).

## What has been validated

| Check | Result |
|---|---|
| Free fall | Implicit Euler reproduced exactly |
| Frictional collision | Linear momentum conserved |
| Frictionless bouncing | Energy never increases |
| Block on an incline | Sticks below the friction angle; slides at g(sin θ − μ cos θ) |
| Fast box (60× plate thickness per step) | Never tunnels through a thin plate |
| Rolled strip | Stores the plate-theory bending energy D κ²/2 |
| Long can between long pads | Matches ring compliance (π/4 − 2/π) F R³/(E′I); error shrinks with refinement |
| Sealed can at 2.5 bar | Swells by pR/(tE)(1 − ν/2) |
| Shell return map | Agrees with the design's independent reference model |

The design checks found and fixed real errors before implementation. Examples: Hertz theory proposed for a soft pad, a contact distance equal to the can's wall thickness, Gauss–Lobatto integration under-predicting the plastic moment by 9%, and linear ring theory applied at deflections larger than the radius. See [docs/08](docs/08-design-validation.md) §3 and [docs/09](docs/09-simulator.md) §4.

## Repository layout

```
docs/          design documents (01–09); the design is the first component
design/        claims registry, reference models ("oracles"), recomputed quantities
tests/design/  the six physics-law checks on the design
simphys/       the simulator: scene/solver, bodies, kernels (JAX), collision, materials, ledger, render
tests/sim/     component tests for each part of the simulator, plus end-to-end
tools/         pytest → trial adapter for the belief server
belief.yaml    components, interfaces, contracts, tests (read from git HEAD)
.belief/       the evidence ledger
```

## Components (belief graph)

`CMP-design-process` feeds every implementation component through an `IFC-design__*` interface. `CMP-sim` encloses `CMP-sim.solver`, `.contact`, `.bodies`, `.materials`, `.ledger`, `.render` and `.analytic`, each with its own contract and test. Check the current state with the belief server: `status(view="diagnose")`, then `run_test(...)`, then `status(view="belief")`.

## Development rules

- **Design first.** Register a feature's physics in `docs/` + `design/claims.yaml` + `design/oracles/`, and make the design checks pass before implementing it (see [CLAUDE.md](CLAUDE.md)).
- **Every new feature goes in the Features table above.**
- CPU-only: demos must be small and render on a laptop.
