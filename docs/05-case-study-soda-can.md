# 05 — Case Study: Picking Up a Soda Can Without Denting It

This is the reference scenario that every layer is built against. Numbers below are **order-of-magnitude design estimates** from standard mechanics. Each one marked *[calibrate]* must be replaced by T2 FEM results validated against physical tests before it is trusted.

## 1. The object

Standard 355 mL (12 oz) two-piece aluminum beverage can:

| Property | Value | Source / status |
|---|---|---|
| Body alloy | AA3004-H19 (drawn & ironed) | Industry standard |
| Young's modulus E | ~69 GPa | Handbook |
| Yield strength σ_y | ~280–300 MPa, with anisotropy from rolling | Handbook *[calibrate]* |
| Sidewall thickness t | ~0.10 mm (thinnest mid-wall; thicker near neck and base) | Literature *[measure]* |
| Radius R | ~33 mm | Measured |
| Mass, empty | ~13–15 g | Measured |
| Mass, full | ~0.37–0.39 kg | Measured |
| Internal pressure, sealed | ~2–4 bar gauge at room temperature, rising with temperature | Literature *[measure per product]* |

`t/R ≈ 0.003`, so this is a thin shell. It must be modeled with shell elements and through-thickness integration ([03](03-deformables-and-materials.md) §1).

## 2. Four cans that look identical

| Latent state | Mass | Wall behavior under a pinch | Grasp risk |
|---|---|---|---|
| **A. Sealed, full** | ~0.38 kg | Pressure pre-tensions the wall: hoop stress `σ_θ = pR/t ≈ 0.25 MPa × 33 mm / 0.1 mm ≈ 80 MPa` at 2.5 bar. The wall is very stiff against inward denting. | Mainly **slip** (heavy). Large `F_max`. |
| **B. Open, full** | ~0.38 kg | No pressure stiffening. The wall ovalizes and dents much more easily. | Heavy **and** dent-prone. **Narrowest envelope.** Liquid can spill. |
| **C. Open, half** | ~0.2 kg | Same as B, plus slosh moves the COM and loads during motion | Slip from slosh moments. Spill. |
| **D. Open, empty** | ~0.014 kg | Same as B | Almost no force needed. Easy to crush if the controller overshoots. |

Note that pressure helps against denting but also uses up some yield margin (~80 MPa of ~290 MPa is already hoop pre-stress). The T2 model captures this interaction automatically. A rule of thumb would not.

**Design consequence:** vision alone cannot separate A from B. The system must either infer from context (a pull-tab that is open vs closed is visible at some angles) or **probe** ([04](04-inverse-problems.md) §1).

## 3. Lower bound: F_min (don't drop it)

Two-finger parallel grasp, silicone pads, `μ` for dry aluminum–silicone about 0.5–1.0, lower if the can is wet from condensation. Lift with peak upward acceleration `a = 5 m/s²`:

```
F_min per finger = m (g + a) / (2 μ)
```

| Case | m | μ = 0.8 (dry) | μ = 0.3 (wet) |
|---|---|---|---|
| Full (A/B) | 0.38 kg | 0.38·14.8 / 1.6 ≈ **3.5 N** | ≈ **9.4 N** |
| Empty (D) | 0.014 kg | ≈ **0.13 N** | ≈ **0.35 N** |

Condensation alone nearly triples the required force. That is why surface condition belongs in the belief. The robust layer then adds margin for uncertainty in `μ`, `m` and `a`, instead of a fixed safety factor.

## 4. Upper bound: F_max (don't dent it)

Local indentation of a thin cylindrical shell by a small pad is governed by local bending and membrane action within a characteristic length

```
ℓ ≈ √(R t) / [3(1−ν²)]^¼ ≈ √(33 × 0.1) / 1.28 mm ≈ 1.4 mm
```

so a fingertip pad (~10–20 mm) spans many `ℓ`. The pad's size and compliance strongly change the peak stress, which is one reason soft, wide pads are gentler. The plastic indentation load of thin cylindrical shells scales as

```
P_dent ~ C · σ_y · t^{3/2} · R^{1/2}      with   σ_y t^{3/2} R^{1/2} ≈ 290e6 · 1e-6 · 0.18 ≈ 50 N
```

where `C` depends on pad size, pad compliance, boundary conditions and imperfections, and must be found from FEM/tests. This gives a *first plausibility check* that the unpressurized dent threshold is **tens of newtons, not hundreds, and not a fraction of a newton** *[calibrate]*. The actual `F_max` comes from the T2 continuation analysis with the residual-displacement check, evaluated as a **map over the surface**. It is higher near the rim and base, where the wall is thicker and stiffened by geometry, and lowest at mid-wall.

## 5. The envelope and what the robot does

Illustrative (pending calibration):

```
Case B (open, full, dry):  F_min ≈ 3.5 N (+ margin) ... F_max ≈ tens of N at mid-wall   → window OK, but
                           a stiff position-controlled gripper closing on the can can blow through F_max easily.
Case B (open, full, wet):  F_min ≈ 9–12 N w/ margin ... F_max unchanged                → window narrows sharply.
Case D (open, empty):      F_min ≈ 0.2 N ... F_max ≈ tens of N                         → wide window; risk is controller overshoot.
Case A (sealed):           F_max much larger                                           → basically a slip problem.
```

Resulting policy, produced by the inverse layer and later distilled into a VLA:
1. Choose a grasp region with a large `F_max` (lower third, near the base where the wall is thicker) that is also near the COM height, so torsion stays low.
2. Close in **force/impedance mode**, never pure position mode. Target a force in the robust window.
3. If the belief over {A, B, C, D} is broad: do a light-squeeze probe (a few N, safe in all cases) and read stiffness. Do a small lift and read load. Update the belief.
4. Lift with limited acceleration (this lowers `F_min`). For a partially filled can, also limit jerk and tilt (slosh).
5. Monitor tactile shear for incipient slip. Increase force within the envelope if slip starts. If the envelope would be exceeded, re-grasp instead.

## 6. What the simulator must get right for this to work

- [ ] Thin-shell elastoplasticity with Hill48 anisotropy and through-thickness integration (first version with isotropic J2: [09](09-simulator.md))
- [ ] Geometric imperfection fields (dent threshold and buckling scatter)
- [ ] Enclosed gas pressure monolithically coupled to shell volume
- [ ] Liquid mass + slosh lumped model
- [ ] Pad hyperelasticity + pressure-dependent friction + wet/dry state
- [ ] Residual-displacement "was it altered?" check
- [ ] Gripper force-control loop with realistic bandwidth, overshoot and sensor noise (overshoot is what actually dents cans)

## 7. Validation plan for this case

1. **Analytic:** a pressurized cylinder's hoop/axial stress, and Hertz contact of a stiff spherical indenter on a thick elastic block, where Hertz's assumptions hold. The silicone pad on the can is *not* a Hertz problem: its contact half-width is about 0.14 R and comparable to the pad thickness (pads are 2–5 mm thick), while Hertz assumes a half-space many contact widths deep. That contact is validated against reference FEM and physical tests instead.
2. **Reference FEM:** compare T2 to Abaqus/LS-DYNA on (a) lateral pad indentation of an empty can, (b) the same with 2.5 bar pressure, (c) axial crush
3. **Physical:** an instrumented gripper (F/T sensor + calibrated pads) squeezes 30+ cans per condition to the onset of permanent deformation, measured with a 3D scanner before and after. This gives the empirical `F_max` distribution. Lift tests with wet/dry cans give `F_min`. Internal pressure of sealed cans is measured per product with a piercing pressure gauge.
4. **Acceptance:** T2's predicted `F_max` distribution covers the measured one (for example, the measured median falls within the predicted 10–90% band), and T0's surrogate misses fewer than 1% of dent events on held-out tests.
