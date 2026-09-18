# Engineering Brief: Fast Thin-Walled Cylinder Crushing (Can Squeeze) via Ramanujan Mathematics

*Revision 3 — 2026-09-19. Supersedes the earlier brief; folds in the design-gate corrections, re-checks every number, and adds the pointwise shape readout (§7) so no FEM solve sits in the per-point loop.*

## Executive Summary

The slowness we are fighting is not in evaluating elliptic integrals. It is in running a 3D non-linear shell FEM with arc-length continuation on a problem that, for long pads and elastic response, is a one-parameter family of planar elasticas. The speed strategy is therefore layered:

| Tier | What changes | Realistic gain |
| :--- | :--- | :--- |
| 0 | 3D shell FEM + Riks arc-length (baseline) | 1× (seconds–minutes per load step) |
| 1 | Replace the 3D model with the 1D inextensional ring elastica | 10³–10⁶× (microseconds per load) |
| 2 | Closed form in the flat-contact phase (lemniscate constant); tabulated one-parameter branch in the point-contact phase | O(1), no root-find online |
| 3 | Ramanujan's theta/nome series and perimeter formula for the remaining elliptic evaluations | ~10× on an already microsecond inner loop |
| 4 | Pointwise shape readout: surface position at any (s, z) from the 1D profile, a mirror isometry, or an r-term reduced basis — never a per-point FEM query | O(1)–O(r) per grid point |

Ramanujan's equations are the cherry, not the cake. Tiers 1–2 are where the orders of magnitude come from; Tier 3 removes the last quadrature loops and root-finds so the whole force–gap law is a handful of flops; Tier 4 makes the deformed geometry as cheap as the force. This brief specifies all four tiers, states exactly where Ramanujan's machinery applies, and where it does not.

The organizing fact behind Tiers 1 and 4 is that thin-shell deformation is nearly isometric: the geometry is fixed kinematically almost everywhere, and the expensive physics is confined to creases and ridges of width ~√(Rt) ≈ 1.8 mm. Anything that resolves the whole surface uniformly is paying for information the problem does not contain.

Scope of the closed forms: long rigid pads, elastic response up to first yield, open (unpressurised) cans. Short pads dent locally and use the mirror-isometry readout (§7.2) or the reduced-order model (§7.3), not a full 3D solve per state (§8).

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
        │  Shape readout (§7)            │──► any (s, z): profile / mirror / Φ·a
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  Rim metric (Ramanujan P_rim)  │──► vision-side stretch check
        └───────────────────────────────┘
```

---

## 7. Pointwise Shape Readout: No FEM in the Per-Point Loop

A field solve, when one is needed at all, is one global solve per load state; the displacement at a grid point is then a lookup. But for this problem the deformed shape is known in closed or near-closed form almost everywhere, so most load states need no field solve. Four readouts, cheapest first.

### 7.1 Long pads: 1D profile extruded along the axis

Every surface point (s, z) maps to the cross-section elastica at arc length s, independent of z. In the flat-contact phase the profile is the quadrature of θ'² = 2λ² sin θ from §2.2:

$$x(\theta) = \int_0^{\theta} \frac{\cos\vartheta\, d\vartheta}{\lambda\sqrt{2\sin\vartheta}} = \frac{\sqrt{2\sin\theta}}{\lambda}, \qquad
y(\theta) = \frac{1}{\lambda\sqrt{2}}\int_0^{\theta} \sqrt{\sin\vartheta}\, d\vartheta,$$

so x is elementary and y is an incomplete lemniscatic integral (complete value π/ϖ at θ = π/2). In the point-contact phase the same integrals carry κ₀² + 2λ² sin θ under the root. Implementation: per load state, sample θ at ~50 Chebyshev nodes on one quarter arc, form (x, y, s)(θ) by one Clenshaw–Curtis pass, and evaluate any point by 1D interpolation in s. The four quarters follow by symmetry, and z is a copy. Cost per grid point: a 1D interpolation, ~10 flops, with the profile rebuilt in ~10 μs per state.

### 7.2 Short pads and dents: Pogorelov mirror isometry plus a ridge

A localized dent in a thin shell is, to leading order, the isometry obtained by reflecting the portion of the surface beyond the pad plane back across that plane. Reflection is an exact isometry, so the reflected cap carries no membrane energy; all the bending energy sits in a ridge along the intersection curve, of width

$$\ell \sim \sqrt{Rt} \approx 1.8\ \text{mm}.$$

Readout: for a point outside the ridge, deformed position = original position if on the unreflected side, else its mirror image across the pad plane. O(1), no solve. Inside the ridge, the cross-ridge profile is a boundary-layer elastica with a similarity form in the coordinate (n/ℓ), computed once per (dent depth, pad geometry) and blended in. This is the two-dimensional analogue of §7.1: kinematics fixes the shape, an elastica fixes the crease. Pogorelov's energy scaling, U ∝ E t^{5/2} d^{3/2}/R for dent depth d, gives the force–depth law of the dent for the same price. Cylinders are less clean than spheres here (dents lock into rhombic patterns at larger depth), so this readout is rated for shallow dents; beyond that, §7.3.

### 7.3 General contact: reduced-order model from offline snapshots

Where a genuine 3D solve is unavoidable (deep dents, off-axis pads, multiple contacts), it runs offline. Sweep the parameters (gap, pad width, pad position), collect displacement snapshots u(x; μ), and take the POD/PCA basis Φ ∈ ℝ^{3N × r} with r ≈ 10–30. Online, the state is the coefficient vector a(μ) ∈ ℝ^r found by Galerkin projection of the equilibrium equations, with DEIM to keep the nonlinear residual evaluation at O(r) cost independent of N. Any grid point is then

$$u(x_i; \mu) = \Phi_i\, a(\mu),$$

a dot product of length r. A Chebyshev-in-μ or neural-operator surrogate fitted to the same snapshots is a drop-in alternative that removes the online projection entirely; the POD form keeps the physics residual and is preferable when the design gate requires a computable error bound.

### 7.4 Hybrid domain decomposition

When the ridge model of §7.2 is not accurate enough, keep the analytic outer solution (§7.1 or the mirror isometry) everywhere and mesh only a strip of width ~3ℓ around the crease or contact zone, coupled through Dirichlet data from the outer solution. The mesh shrinks from 10⁴–10⁵ elements to a few hundred, and the outer region costs nothing per point.

### 7.5 Which readout when

| Situation | Readout | Per-point cost | Per-state cost |
| :--- | :--- | :--- | :--- |
| Long pads, elastic | §7.1 profile | ~10 flops | ~10 μs |
| Short pad, shallow dent | §7.2 mirror + ridge | O(1) | one ridge ODE |
| Deep dent, off-axis, multi-contact | §7.3 ROM | O(r) | O(r²)–O(r³) |
| Design-gate accuracy needed in the crease | §7.4 hybrid | O(1) outside strip | few-hundred-element solve |

---

## 8. What Ramanujan Does Not Buy

1. **Axial crush (Yoshimura pattern).** The earlier brief proposed Chowla–Selberg-type lattice-sum acceleration for the fold energy. The modular transformation is real mathematics, but the mapping from membrane energy to a Epstein zeta sum was never derived and the O(1) mode-selection claim is unverified. Out of scope until it passes the design gate.
2. **Short pads.** Local denting is a genuinely 2D shell problem with a boundary layer of width √(Rt). The shape outside the ridge is a mirror isometry (§7.2); the ridge and any deep-dent regime use the reduced-order model or hybrid strip (§7.3–7.4), not a full 3D solve per state. Ramanujan's identities play no role in that tier.
3. **Plasticity.** Past g_y the moment–curvature law is no longer linear and the closed form ends. A tabulated elastic-plastic ring branch is possible but is a different brief.
4. **Internal pressure.** Sealed cans carry membrane pre-stress that stiffens the ring; the inextensional elastica applies to open cans only.

---

## 9. Corrections to the Earlier Brief

| Item | Earlier | Corrected |
| :--- | :--- | :--- |
| R/t for a beverage can | 1000–2000 | ≈ 330 (R = 33 mm, t = 0.10 mm) |
| Elastica equation | θ'' + λ² sin θ = 0 | θ'' = λ² cos θ, E' = E/(1 − ν²) |
| Load–gap inversion | q-series, "2–3 evaluations" | Closed form below g* = 1.4355 R; one scalar root-find above, tabulated offline |
| Perimeter formula worst-case error | 4 × 10⁻⁵ | 4 × 10⁻⁴ (degenerate ellipse); 2.5 × 10⁻¹² at a/b = 1.5 |
| Perimeter as constraint | enforced every step | not needed; use only as a rim metric |
| Yoshimura lattice sums | O(1) mode selection | unverified, out of scope |

---

## 10. Key Takeaways

1. The speed problem is solved by model reduction: a pinched can under long pads is a planar inextensional elastica, and that is microseconds, not minutes.
2. Below 72% of the diameter the whole force–gap law is P = 4π²E'I/(ϖ²g²); the lemniscate constant is Ramanujan's first singular value showing up in a can.
3. Above that, one dimensionless branch is tabulated once; Ramanujan's q-series or Gauss's AGM build it equally well.
4. Ramanujan's perimeter formula is the right one-line rim-ovalization metric, with h⁵ error — but never a constraint.
5. The deformed geometry is as cheap as the force: a 1D profile for long pads, a mirror isometry plus a √(Rt) ridge for dents, an r-term basis for the rest. A FEM solve belongs offline, never in the per-point loop.
6. Everything outside long-pad, elastic, open-can pinching is handled by the readouts of §7 or, past yield and pressure, by the 3D simulator run offline.

---

## Review against the design gate (2026-09-19, revision 3)

**Verified and adopted** ([docs/11](../11-analytic-squeeze.md), `simphys/analytic.py`):

- §2.2 closed forms, the exact slope, the C¹ join at g*, and the degree-12 tabulated branch (1.3×10⁻¹⁰ of the quadrature reference; a force query in 35 µs instead of 74 ms).
- **§7.1 shape readout.** x(θ) = √(2 sin θ)/λ matches quadrature to 12 digits, y(π/2) is exactly half the gap, and the perimeter stays 2πR. The contact curvature κ₀ in line contact comes from a second material-free Chebyshev fit, as §4 proposes. A full cross-section costs about 130 µs per load state, with no field solve.
- §5 perimeter worst case of 4×10⁻⁴.

**Still to correct:** §2.2 and §2.4 use Δκ_y = 2σ_y/(E' t). A long pinched can is in plane strain: the elastic axial stress is νσ₁₁, so von Mises yield comes at σ₁₁ = σ_y/√(1 − ν + ν²), about 13% higher. With σ_y = 285 MPa from the material record, first yield is at **g_y = 21.1 mm and 83.4 N/m, about 8.3 N on a 100 mm pad**, not 23.4 mm, 68.0 N/m and 7 N.

**Not adopted yet (needs validation against the 3D simulator first):**

- **§7.2 mirror isometry.** Pogorelov's construction and the U ∝ E t^{5/2} d^{3/2}/R scaling are derived for doubly curved (spherical) shells. A cylinder has one zero principal curvature, and the brief itself notes rhombic locking. The construction also describes the inverted dent after snap-through, not the elastic loading before it, and that loading is what sets the grasp limit F_max. It becomes a design claim only after the 3D simulator confirms it for the can.
- **§7.3 reduced-order model.** This agrees with the design's T1 tier ([docs/01](../01-architecture.md) §4), and the 3D simulator is the offline snapshot source. Plasticity makes the response path-dependent, which a plain POD basis on displacements does not capture: the plastic state needs to be part of the reduced state. That is a separate design item.
- **§7.4 hybrid strip.** Coupling a meshed strip to an analytic outer solution through Dirichlet data is only as accurate as the outer solution. For short pads there is no validated outer solution yet (see §7.2).

**Update, same day: §7.2 tested against three 3D short-pad squeezes** (see [docs/11](../11-analytic-squeeze.md) §4). For flat pads on a can the mirror isometry does not hold. The wall conforms to the pad plane instead of inverting beyond it (a developable flattening, a different isometry), and the force grows as d^1.2 to d^1.3 rather than d^0.5. The mirror readout is therefore not adopted for grasping with flat pads. It may still apply to small, rounded indenters, which is untested.
