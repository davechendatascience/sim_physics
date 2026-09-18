# 07 — Roadmap, Validation, Risks

## 1. Phases

Each phase ends with a demo and a validation report. Scope is deliberately narrow early on: **get the soda can right before generalizing.**

### Phase 0: Foundations
- Core math, units, sparse linear algebra on GPU, autodiff glue
- Incremental-potential Newton solver with line search
- IPC contact barrier + CCD + smoothed friction
- Affine rigid bodies. Tet FEM with Neo-Hookean material.
- Validation suite skeleton: analytic benchmarks (below), regression tests, determinism tests
- **Exit:** rigid stacking, elastic ball drop and Hertz contact match analytic solutions within 2%

### Phase 1: Thin shells, plasticity and the can
- Kirchhoff–Love shells with through-thickness integration
- J2 + Voce + Hill48 variational plasticity. Ledger with residual-displacement check.
- Enclosed-gas lumped model. Liquid mass + slosh model.
- Silicone-pad fingertips, a parallel gripper with a force-control loop
- **Exit:** T2 matches reference FEM (Abaqus/LS-DYNA) for can indentation with and without pressure. The first physical test campaign ([05](05-case-study-soda-can.md) §7) is under way.

### Phase 2: Differentiable & inverse
- Implicit adjoints through the step. Checkpointing.
- System ID tool fit to lab data. Material DB v1 with provenance.
- `grasp_envelope()` with uncertainty. Belief-space probing.
- **Exit:** the envelope's predicted `F_max` distribution covers the measured one for cans A/B/D. Parameters of real cans are identified from gripper squeeze data.

### Phase 3: Fidelity tiers & VLA
- T1 reduced models (modal + plastic-mode basis, neural latent dynamics)
- T0 rigid + damage surrogate with a conservative miss-rate guarantee
- Vectorized env API, tactile/F/T simulation, renderer integration, dataset exporters
- Privileged-teacher demo generation → first VLA trained on the "pick up the can" task family
- **Exit:** a VLA picks up real cans (all four latent states) with a lower damage rate than a position-control baseline. Measured, not assumed.

### Phase 4: Breadth
- More object classes: PET bottles (polymer viscoplasticity), paper cups, eggs/fruit (fracture), cloth, cables
- MPM for granular materials and heavy fracture. Temperature coupling.
- Deliberate-shaping tasks (crush, bend, fold) using reachability analysis

## 2. Validation pyramid

| Level | Examples | Pass criterion |
|---|---|---|
| Unit | Stress/tangent consistency (finite-difference checks), return map consistency, energy conservation without dissipation | Machine-precision / convergence-rate tests |
| Analytic | Cantilever bending, Hertz contact, pressurized cylinder, Euler column buckling, cylindrical-shell buckling with imperfection knockdown, friction incline | Error < 1–2% (buckling: within the published knockdown band) |
| Reference code | Abaqus/LS-DYNA on can indentation, axial crush, PET bottle squeeze | Load–displacement curves within 5–10% |
| Physical | Instrumented gripper tests, 3D scans before and after | Predicted distribution covers measured |
| Task | Real robot outcomes vs sim-predicted outcomes | Gap tracked per object class |

Every PR that touches the solver or materials runs unit + analytic tests. Reference and physical comparisons run nightly or weekly on stored datasets.

## 3. Key risks

| Risk | Impact | Mitigation |
|---|---|---|
| T2 too slow for inverse queries | Envelopes are impractical online | Tabulate offline, use T1 surrogates online, adaptive refinement only under contact |
| Surrogates (T0/T1) miss damage events | VLA learns unsafe behavior | Conservative training objective, a miss-rate bound as the release gate, T2 spot checks |
| Buckling/dent scatter from imperfections | Predicted thresholds are optimistic | Imperfection random fields in the material record. Report distributions, not points. |
| Non-smooth gradients (contact, yield) | Inverse optimization stalls | Hybrid sampling + gradient, randomized smoothing, continuation |
| Material parameters unknown for most objects | Model is confidently wrong | Provenance tracking, validity flags on outputs, online system ID, active probing |
| Scope creep (every material at once) | Nothing validated | Phase gates. One object family validated end to end before the next. |
| Controller effects dominate (overshoot, latency) | Sim envelope is right but real grasps still dent | Model actuator and force-loop dynamics as carefully as materials |

## 4. Open questions for you

1. **Target hardware:** which grippers and tactile sensors should be modeled first (parallel jaw + DIGIT? dexterous hand)?
2. **VLA stack:** which model/dataset format should be supported first (e.g., OpenVLA/π-style with LeRobot, RLDS)? Can the action space include gripper force?
3. **Build vs. extend:** building from scratch as designed, or building the material/inverse layers on top of an existing engine (e.g., Genesis or Warp-based Newton for the solver core) to reach Phase 3 sooner?
4. **Physical test capacity:** is there access to a gripper with F/T sensing and a 3D scanner for the calibration campaign? Without physical data, the material-science claims can't be validated.
5. **Object priorities** after cans: bottles, cups, food, cloth?
