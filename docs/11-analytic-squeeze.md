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

For the long-pad pinch this replaces the Newton solve entirely. Phase 1 takes a few function evaluations and phase 2 is algebraic, instead of hours of 3D stepping.

## 2. Regime

- **Long pads only.** The pads must span the whole can so that every cross-section deforms alike (plane strain). Short pads dent locally and need the 3D simulator. For them this tier gives a lower bound on the force.
- **Elastic, up to first yield** (g ≥ g_y). Past that the ring forms plastic hinges, which this tier does not model.
- **Thin and inextensible:** R/t ≈ 330. Membrane stretching is negligible next to bending.
- **Open can.** Internal pressure would add hoop tension and a follower load, which the model leaves out.

## 3. How it is checked

- **Reference model:** `design/oracles/elastica.py` integrates the elastica by quadrature and root-finding, with no closed forms. The closed forms above must match it.
- **Small-load limit:** phase 1 must reproduce the linear ring compliance `(π/4 − 2/π) F R³/(E'I)` of [09](09-simulator.md) §4 as the load goes to zero.
- **Phase continuity:** force and gap must be continuous where line contact becomes flat contact.
- **Inextensibility:** the arc length stays πR/2 per quarter in both phases.
- **The 3D simulator:** its long-can scene must agree with this tier.
