# 11 — Fast Analytic Tier: The Can Pinched Along Its Length

The 3D simulator ([09](09-simulator.md)) is needed whenever a short pad dents a can locally. One loading case, though, has an exact closed form: a can pinched between two **long flat pads** that span its whole length. Its cross-section is then an inextensible elastic ring (an *elastica*) in plane strain. This tier answers that case in microseconds, following the user's brief (`docs/reference/engineering_brief_can_squeeze_mechanics_via_ramanujan_mathematics.md` §1.1), with the corrections noted there.

## 1. Model

Take a quarter of the ring, from the contact with the lower pad to the side point B. By up-down symmetry the internal force is purely the vertical pad load P/2 (per unit length), and the tangent angle θ(s) obeys

```
θ'' = λ² cos θ,    λ² = P / (2 E'I)
```

with `E' = E/(1 − ν²)` and `I = t³/12` per unit length. The wall is thin (R/t ≈ 330), so the ring bends without stretching: arc length is conserved.

**Phase 1: line contact.** The ring touches each pad along one line, where its curvature κ_A falls from 1/R as the load grows. Integrating once gives `θ'² = κ_A² + 2λ² sin θ`, and κ_A follows from the arc-length condition. One scalar root-find does it.

**Phase 2: flat contact.** Once κ_A reaches zero, the ring lies flat on each pad over a length that grows with the load. The free arc starts straight (θ = 0, θ' = 0), so `θ'² = 2λ² sin θ` exactly. Its length and height are then the lemniscate integrals ∫dθ/√(2 sin θ) = ϖ/√2 and ∫√(sin θ / 2) dθ = π/(√2 ϖ), where ϖ = Γ(¼)²/(2√(2π)) is the **lemniscate constant**. This is the classical mathematics of Gauss and Ramanujan. Everything follows in closed form:

| Quantity | Formula | Can wall |
|---|---|---|
| Lemniscate constant | `ϖ = Γ(¼)²/(2√(2π))` | ϖ = 2.62206 |
| Pad gap at the onset of flat contact | `g₀ = π² R / ϖ²` | 47.4 mm |
| Pad force per length at that onset | `P₀ = 4π² E'I / (ϖ² g₀²)` | 16.5 N/m |
| Force–gap law in flat contact | `P = 4π² E'I / (ϖ² g²)` | exact, for g ≤ g₀ |
| Flat contact length per pad | `c = π R − ϖ² g / π` | grows from zero at g₀ |
| Gap at first yield (at the sides) | `g_y = 2π / (ϖ (Δκ_y + 1/R))` | 21.1 mm |
| Force per length at first yield | `P_y = E'I (Δκ_y + 1/R)²` | 83.4 N/m |

Here `Δκ_y = 2 σ_1y (1 − ν²) / (E t)` is the curvature change that brings the surface fiber to first yield (83 /m for this wall, [09](09-simulator.md) §4). The tightest curvature is at the side point B, `κ_B = √2 λ`, so that is where yield starts.

For the long-pad pinch this replaces the Newton solve entirely. Phase 2 is algebraic, with an exact slope `dP/dg = −2P/g`.

**Phase 1 is tabulated once.** In the dimensionless variables g/R and P R²/(E'I), the line-contact branch contains no material constants. A degree-12 Chebyshev fit on g/R ∈ [1.4355, 2] reproduces the quadrature reference to 1.3×10⁻¹⁰ of the onset force, so every query costs about a dozen floating-point operations. The fit and its derivative join the flat-contact law with matching slope at g₀ (C¹), so a force-feedback loop sees no kink. This follows revision 2 of the user's brief (`docs/reference/engineering_brief_can_squeeze_speedup_via_ramanujan_mathematics.md` §4), with its first-yield rule corrected: in plane strain, first yield comes at σ₁₁ = σ_y/√(1 − ν + ν²), not σ_y.

**The deformed shape is a readout too** (brief revision 3, §7.1). Every point of the can, at arc length s around the cross-section and at any height z, maps to the elastica at s; the cross-section is the same all along the pads. On the free arc of the flat-contact phase the profile is elementary:

```
x(θ) = √(2 sin θ) / λ,     y(θ) = (1/(√2 λ)) ∫₀^θ √(sin ϑ) dϑ
```

so y(π/2) = π/(√2 ϖ λ) is exactly half the gap. In line contact the same integrals carry κ_A² + 2λ² sin θ under the root, with κ_A taken from a second material-free Chebyshev fit. The whole cross-section follows by symmetry. A readout costs one 1D profile per load state and an interpolation per point, with no field solve.

## 2. Regime

- **Long pads only.** The pads must span the whole can so that every cross-section deforms alike (plane strain). Short pads dent locally and need the 3D simulator. For them this tier gives a lower bound on the force.
- **Elastic, up to first yield** (g ≥ g_y). Past that the ring forms plastic hinges, which this tier does not model.
- **Thin and inextensible:** R/t ≈ 330. Membrane stretching is negligible next to bending.
- **Open can.** Internal pressure would add hoop tension and a follower load, which the model leaves out.

## 3. How it is checked

- **Reference model:** `design/oracles/elastica.py` integrates the elastica by quadrature and root-finding, with no closed forms. The closed forms above must match it.
- **Small-load limit:** phase 1 must reproduce the linear ring compliance `(π/4 − 2/π) F R³/(E'I)` of [09](09-simulator.md) §4 as the load goes to zero.
- **Phase continuity:** force, gap and slope must be continuous where line contact becomes flat contact.
- **Tabulated branch:** the degree-12 fit must stay within 10⁻⁹ of the quadrature reference.
- **Shape readout:** the closed-form profile must match the quadrature reference, reach exactly half the gap at the side point, and keep the perimeter at 2πR.
- **Inextensibility:** the arc length stays πR/2 per quarter in both phases.
- **The 3D simulator:** its long-can scene must agree with this tier.

## 4. Short pads: what the 3D runs show

Brief revision 3 (§7.2) proposed a Pogorelov mirror isometry as the short-pad readout. Three 3D squeezes were run to 6 mm of pad travel per side with `experiments/short_pad_squeeze.py`, and compared with `experiments/mirror_isometry_check.py`. Pads were 10 × 10, 15 × 20 and 30 × 30 mm, and the can was open.

**The mirror isometry does not describe flat pads on a can.** Under the pad the simulated wall lies on the pad plane; it cannot pass beyond it. So its departure from the mirror prediction equals the pad travel, at every depth, for every pad size. Flattening a strip of a cylinder is itself an isometry, because the cylinder is developable, but it is a different one. Pogorelov's inverted cap needs an indenter smaller than the dent. The force grows as d^1.2 to d^1.3, not as Pogorelov's d^0.5 (which is derived for spheres).

**First short-pad dent measurements** (provisional: the mesh, about 4 × 5 mm, is coarser than the 1.8 mm bending boundary layer, so these are not mesh-converged):

| Pad | First yield (travel, force) | Plastic strain above 2×10⁻³ | After release |
|---|---|---|---|
| 10 × 10 mm | 2.40 mm, 12.2 N | 3.76 mm, 21.3 N | yielded; 0.08 mm residual shape change |
| 15 × 20 mm | 2.48 mm, 14.2 N | 3.52 mm, 24.3 N | dented; 0.23 mm |
| 30 × 30 mm | 1.84 mm, 9.7 N | 2.96 mm, 30.3 N | dented; 4.9 mm |

These are consistent with the "tens of newtons" estimate of [05](05-case-study-soda-can.md) §4. They also show the short-pad regime yielding at far lower force per metre than the long-pad limit: 12–14 N on a 10–20 mm pad is 600–1 400 N/m, against 83.4 N/m. The short-pad regime therefore needs its own model. The next candidates are the long-pad cross-section applied under the pad with an axial transition zone, and the reduced-order model of brief §7.3, trained on runs like these after a mesh-convergence study.
