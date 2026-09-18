# Engineering Brief: Fast Thin-Walled Cylinder Crushing (Can Squeeze) via Ramanujan Mathematics

*Revision 2 — 2026-09-19. Supersedes the earlier brief; folds in the design-gate corrections and re-checks every number.*

## Executive Summary

The slowness we are fighting is not in evaluating elliptic integrals. It is in running a 3D non-linear shell FEM with arc-length continuation on a problem that, for long pads and elastic response, is a one-parameter family of planar elasticas. The speed strategy is therefore layered:

| Tier | What changes | Realistic gain |
| :--- | :--- | :--- |
| 0 | 3D shell FEM + Riks arc-length (baseline) | 1× (seconds–minutes per load step) |
| 1 | Replace the 3D model with the 1D inextensional ring elastica | 10³–10⁶× (microseconds per load) |
| 2 | Closed form in the flat-contact phase (lemniscate constant); tabulated one-parameter branch in the point-contact phase | O(1), no root-find online |
| 3 | Ramanujan's theta/nome series and perimeter formula for the remaining elliptic evaluations | ~10× on an already microsecond inner loop |

Ramanujan's equations are the cherry, not the cake. Tiers 1–2 are where the orders of magnitude come from; Tier 3 removes the last quadrature loops and root-finds so the whole force–gap law is a handful of flops. This brief specifies all three tiers, states exactly where Ramanujan's machinery applies, and where it does not.

Scope: long rigid pads, elastic response up to first yield, open (unpressurised) cans. Short pads dent locally and stay on the 3D simulator (§7).

---

## 1. Where the Time Goes (Baseline)

For a beverage can, R = 33 mm, t = 0.10 mm, so **R/t ≈ 330**. A converged shell mesh needs elements no coarser than ~√(Rt) ≈ 1.8 mm to resolve the bending boundary layer, i.e. ~10⁴–10⁵ elements. Each load step then costs:

- Tangent stiffness assembly and a sparse factorisation, O(N^1.5) for 2D bandwidth — tens of ms to seconds.
- Arc-length (Riks) sub-steps near contact onset and near the flat-contact transition, typically 5–20 per converged step.
- Contact iterations against the pads, each requiring re-linearisation.

Net: seconds per load point, minutes per force–gap curve, and no derivative information for free. None of that is inherent to the physics of a pinched ring.

---

## 2. Radial Pinch: Inextensional Ring Elastica

### 2.1 Governing equation

Opposite pads at gap g press a ring of radius R, wall thickness t, per unit pad length. Because R/t ≈ 330, membrane stretching costs orders of magnitude more energy than bending, so the wall deforms inextensionally. Let θ(s) be the tangent angle along the circumference, E' = E/(1 − ν²) the plane-strain modulus, and I = t³/12 the second moment per unit length. Bending moment is M = E'I(θ' − 1/R).

Up–down symmetry between the two pads makes the horizontal internal force zero, so each quarter ring carries only the vertical force P/2 and

$$\theta'' = \lambda^2 \cos\theta, \qquad \lambda^2 = \frac{P}{2E'I}.$$

(Earlier brief wrote sin θ; the cos form follows from measuring θ from the pad-parallel direction with the vertical load.)

First integral, with κ₀ the total curvature at the pad contact point:

$$\theta'^2 = \kappa_0^2 + 2\lambda^2 \sin\theta.$$

### 2.2 Two phases

**Point-contact phase** (2R ≥ g > 1.4355 R). The ring touches each pad at a point; κ₀ > 0. Quarter-arc length and half-gap give two conditions in (κ₀, λ):

$$\frac{\pi R}{2} = \int_0^{\pi/2} \frac{d\theta}{\sqrt{\kappa_0^2 + 2\lambda^2 \sin\theta}}, \qquad
\frac{g}{2} = \int_0^{\pi/2} \frac{\sin\theta\, d\theta}{\sqrt{\kappa_0^2 + 2\lambda^2 \sin\theta}}.$$

With sin θ = 1 − 2 sin²φ these are incomplete elliptic integrals of modulus k² = 4λ²/(κ₀² + 2λ²) evaluated at amplitude π/4. Given g, this is a single scalar root-find; the curve (g/R) ↦ (P R²/E'I, κ₀R) is a **one-parameter family with no material constants in it**, so it is computed once and tabulated (§4).

**Flat-contact phase** (g ≤ 1.4355 R). Once κ₀ reaches zero the ring lies flat on the pad, the free arc leaves the pad straight, and everything closes in the **lemniscate constant**

$$\varpi = \frac{\Gamma(\tfrac14)^2}{2\sqrt{2\pi}} = 2.622057\ldots$$

because ∫₀^{π/2} dθ/√(sin θ) = ϖ and ∫₀^{π/2} √(sin θ) dθ = π/ϖ. Then:

- Free quarter-arc length: s_f = ϖ² g / (2π).
- Flat-contact onset: 4 s_f = 2πR ⇒ **g* = π²R/ϖ² = 1.4355 R** (pads at 72% of the diameter).
- Force–gap law: $$P = \frac{4\pi^2 E'I}{\varpi^2 g^2}.$$
- Peak curvature (at the sides): κ_max = 2π/(ϖ g).
- First yield, with Δκ_y = 2σ_y/(E' t): $$g_y = \frac{2\pi}{\varpi\,(\Delta\kappa_y + 1/R)}.$$

No elliptic integral is evaluated online. Cost: one division and one multiply per force query, with exact dP/dg = −2P/g for free.

### 2.3 Why this is Ramanujan territory

The lemniscate constant is the first singular value of the elliptic integrals, k = 1/√2, q = e^{−π}: K(1/√2) = ϖ/√2 = Γ(¼)²/(4√π), and θ₃(e^{−π}) = π^{1/4}/Γ(¾). These are the evaluations Ramanujan's class-invariant work is built around; the flat-contact elastica happens to live exactly at that point of the modular curve. That is a mathematical fact worth knowing, but note it does not change the flop count: ϖ is a constant.

### 2.4 Worked numbers (aluminium 3004-H19 can body)

E = 69 GPa, ν = 0.33 ⇒ E' = 77.4 GPa; t = 0.10 mm ⇒ E'I = 6.45 × 10⁻³ N·m per metre of pad; R = 33 mm; σ_y = 280 MPa ⇒ Δκ_y = 72.3 m⁻¹.

| g (mm) | phase | P (N per m of pad) |
| :--- | :--- | :--- |
| 66 | contact onset | 0 |
| 47.4 | flat-contact onset (1.4355 R) | from table |
| 40 | flat | 23.2 |
| 33 | flat | 34.0 |
| **23.4** | **first yield** | **68.0** |

A 100 mm long pad therefore starts to yield the wall at about 7 N. Beyond g_y the model is elastic-plastic and outside this tier.

---

## 3. Elliptic Evaluations: Ramanujan vs AGM

For the point-contact branch (and any parametric re-derivation), K(k) and E(k) are needed. Two O(1) routes, both far faster than quadrature:

**Gauss AGM.** K(k) = π / (2 · AGM(1, k')), quadratically convergent, 5–6 iterations to double precision. E follows from the Legendre relation on the same AGM sequence.

**Ramanujan nome/theta series.** With k' = √(1 − k²) and ε = (1 − √k')/(2(1 + √k')),

$$q = \varepsilon + 2\varepsilon^5 + 15\varepsilon^9 + 150\varepsilon^{13} + \cdots$$

then

$$K = \frac{\pi}{2}\,\theta_3(q)^2 = \frac{\pi}{2}\,(1 + 2q + 2q^4 + 2q^9 + \cdots)^2, \qquad
k = \frac{\theta_2(q)^2}{\theta_3(q)^2}.$$

For k ≤ 1/√2 the nome satisfies q ≤ e^{−π} = 0.0432, so q⁹ ≈ 5 × 10⁻¹³: **three theta terms give double precision**. For k > 1/√2 apply the Landen/modular transformation k → k' first. Because the whole branch is smooth in q, parametrise it by q rather than k: the derivatives Newton needs are polynomial, no inner iteration.

Verdict: both routes are ~10–50 flops. The q-series wins when you need the branch as an explicit power series (for tabulation, symbolic differentiation, or a symbolic-regression target); AGM wins for a one-off scalar. Neither is the bottleneck once Tier 1 is in place, which is why the earlier brief's claim that q-series "bypass Newton–Raphson" was oversold: the root-find is one-dimensional and cheap either way.

---

## 4. Tabulated Branch and Online Kernel

```
Offline (once, no material constants):
  for k in Chebyshev nodes on (0, 1):
      solve (κ0 R, λ R) from the two §2.2 integrals   # AGM or q-series
      record  g/R,  P R^2 / (E'I),  κ0 R
  fit  P̂(g/R), κ̂0(g/R)  on g/R ∈ [1.4355, 2]  (degree ~12 suffices)

Online (per force query):
  if g > 1.4355 R:   P = (E'I / R^2) · P̂(g/R)          # Chebyshev eval, ~12 flops
  else:              P = 4π² E'I / (ϖ² g²)              # closed form, 2 flops
  yield check:       κ_max R  vs  (Δκ_y + 1/R) R
```

Both branches are C¹ at g* (κ₀ → 0 continuously), so the haptic loop sees no kink.

---

## 5. Rim Ovalization Metric: Ramanujan's Perimeter Formula

A pinched ring is **not** an ellipse: it is flat on the pads and tighter at the sides. Do not enforce an elliptical perimeter as a constraint — the elastica is inextensional by construction and the constraint would impose the wrong shape.

The ellipse formula is still the right O(1) tool for **measuring** an ovalized rim (semi-axes a, b from the vision pipeline) with h = ((a − b)/(a + b))²:

$$P_{\text{rim}} \approx \pi (a + b)\left(1 + \frac{3h}{10 + \sqrt{4 - 3h}}\right).$$

Checked relative error against the exact 4aE(e):

| a/b | h | rel. error |
| :--- | :--- | :--- |
| 1.05 | 5.9 × 10⁻⁴ | 1 × 10⁻¹⁶ |
| 1.25 | 1.2 × 10⁻² | 7 × 10⁻¹⁵ |
| 1.5 | 4.0 × 10⁻² | 2.5 × 10⁻¹² |
| 2.0 | 1.1 × 10⁻¹ | 4.6 × 10⁻¹⁰ |
| → ∞ (flat) | 1 | 4.0 × 10⁻⁴ (worst case) |

Error scales as h⁵. Comparing measured rim perimeter to 2πR then gives a stretch estimate in one line; anything beyond a few parts in 10⁴ means the wall has yielded or the measurement is off.

---

## 6. Pipeline

```
          [ gap g  (or load P) per frame ]
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Phase switch  (g ≶ 1.4355 R)  │
        └───────┬───────────────┬───────┘
                ▼               ▼
   ┌────────────────────┐  ┌──────────────────────────┐
   │ Point-contact tier │  │ Flat-contact tier         │
   │ Chebyshev branch   │  │ P = 4π²E'I / (ϖ² g²)      │
   │ (built via AGM /   │  │ κ_max = 2π/(ϖ g)          │
   │  q-series offline) │  │ exact dP/dg               │
   └─────────┬──────────┘  └────────────┬─────────────┘
             └──────────────┬───────────┘
                            ▼
        ┌───────────────────────────────┐
        │  Yield gate  κ_max vs Δκ_y+1/R │──► beyond yield: 3D simulator
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  Rim metric (Ramanujan P_rim)  │──► vision-side stretch check
        └───────────────────────────────┘
```

---

## 7. What Ramanujan Does Not Buy

1. **Axial crush (Yoshimura pattern).** The earlier brief proposed Chowla–Selberg-type lattice-sum acceleration for the fold energy. The modular transformation is real mathematics, but the mapping from membrane energy to a Epstein zeta sum was never derived and the O(1) mode-selection claim is unverified. Out of scope until it passes the design gate.
2. **Short pads.** Local denting is a genuinely 2D shell problem with a boundary layer of width √(Rt); it stays on the 3D simulator.
3. **Plasticity.** Past g_y the moment–curvature law is no longer linear and the closed form ends. A tabulated elastic-plastic ring branch is possible but is a different brief.
4. **Internal pressure.** Sealed cans carry membrane pre-stress that stiffens the ring; the inextensional elastica applies to open cans only.

---

## 8. Corrections to the Earlier Brief

| Item | Earlier | Corrected |
| :--- | :--- | :--- |
| R/t for a beverage can | 1000–2000 | ≈ 330 (R = 33 mm, t = 0.10 mm) |
| Elastica equation | θ'' + λ² sin θ = 0 | θ'' = λ² cos θ, E' = E/(1 − ν²) |
| Load–gap inversion | q-series, "2–3 evaluations" | Closed form below g* = 1.4355 R; one scalar root-find above, tabulated offline |
| Perimeter formula worst-case error | 4 × 10⁻⁵ | 4 × 10⁻⁴ (degenerate ellipse); 2.5 × 10⁻¹² at a/b = 1.5 |
| Perimeter as constraint | enforced every step | not needed; use only as a rim metric |
| Yoshimura lattice sums | O(1) mode selection | unverified, out of scope |

---

## 9. Key Takeaways

1. The speed problem is solved by model reduction: a pinched can under long pads is a planar inextensional elastica, and that is microseconds, not minutes.
2. Below 72% of the diameter the whole force–gap law is P = 4π²E'I/(ϖ²g²); the lemniscate constant is Ramanujan's first singular value showing up in a can.
3. Above that, one dimensionless branch is tabulated once; Ramanujan's q-series or Gauss's AGM build it equally well.
4. Ramanujan's perimeter formula is the right one-line rim-ovalization metric, with h⁵ error — but never a constraint.
5. Everything outside long-pad, elastic, open-can pinching stays on the 3D simulator.

---

## Review against the design gate (2026-09-19)

**Verified and adopted** ([docs/11](../11-analytic-squeeze.md), `simphys/analytic.py`):

- §2.2: flat-contact closed forms, g* = π²R/ϖ², κ_max = 2π/(ϖ g), and the exact slope dP/dg = −2P/g.
- §4: tabulating the material-free line-contact branch. A degree-12 Chebyshev fit matches the quadrature reference to 1.3×10⁻¹⁰, and a force query drops from 74 ms (nested root-finding) to 35 µs.
- §4: the claim that both branches are C¹ at g*. The line-contact and flat-contact slopes agree at the onset.
- §5: the corrected perimeter worst case of 4×10⁻⁴, for a completely flat ellipse. The earlier review wrongly accepted 4×10⁻⁵ as the worst case.

**Corrected here:**

1. **First-yield rule (§2.2, §2.4).** Δκ_y = 2σ_y/(E't) treats the surface fiber as uniaxial. A long pinched can is in plane strain: with the axial strain held at zero, the elastic axial stress is νσ₁₁, and von Mises yield comes at σ₁₁ = σ_y/√(1 − ν + ν²), about 13% higher. The rule is therefore Δκ_y = 2σ_y/(√(1 − ν + ν²) E' t).
2. **Worked numbers (§2.4).** With that rule and the material record's σ_y = 285 MPa (docs/03 §3), first yield is at **g_y = 21.1 mm, P_y = 83.4 N/m**, not 23.4 mm and 68.0 N/m. A 100 mm pad starts to yield the wall at about 8.3 N, not 7 N. The table's flat-phase forces at 40 mm and 33 mm (23.2 and 34.0 N/m) are correct.
3. **Baseline (§1).** simphys does not use Riks continuation. Its 3D baseline is an IPC incremental-potential solver, measured in [docs/10](../10-performance.md) §1. The model-reduction argument is unchanged.
