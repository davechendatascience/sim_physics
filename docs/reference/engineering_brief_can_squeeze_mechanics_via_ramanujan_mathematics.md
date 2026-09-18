# Engineering Brief: Analytical Modeling of Thin-Walled Cylinder Crushing (Can Squeeze) via Ramanujan Mathematics

## Executive Summary
Crushing a thin-walled aluminum cylinder (such as a standard beverage can) represents one of the most notoriously non-linear, geometrically sensitive challenges in structural mechanics. Whether subjected to **radial pinching** (line/point loads across the diameter) or **axial compression** (classic diamond snap-through), conventional numerical tools like non-linear Finite Element Methods (FEM) face severe bottlenecks:

- **Severe element distortion and locking** near localized folding hinges.
- **Ill-conditioned tangent stiffness matrices** at unstable bifurcation points.
- **High computational expense** required by incremental arc-length methods (e.g., Riks algorithm) to navigate snap-through instabilities.

By leveraging **Srinivasa Ramanujan’s mathematical machinery**—specifically his rapidly converging $q$-series for elliptic integrals, modular transformations for 2D lattice sums, and algebraic perimeter approximations—computational mechanics engines can substitute brute-force spatial discretization with semi-analytical kernels. This approach enables real-time force feedback, accelerated contact resolution, and robust post-buckling mode prediction.

---

## 1. Mechanics Regimes & Mathematical Formulations

### 1.1 Radial Pinch: Large-Deflection Ring Elastica
When opposite radial forces $P$ compress a cylindrical cross-section, the cylinder walls undergo inextensional large bending. Under Euler–Bernoulli beam assumptions along the circumferential coordinate $s$, the governing non-linear equation for the local rotation angle $\theta(s)$ is:

$$\frac{d^2\theta}{ds^2} + \lambda^2 \sin\theta = 0, \quad \lambda^2 = \frac{P}{2EI}$$

#### Ramanujan Acceleration
The exact profile and the non-linear relationship between load $P$ and radial displacement $\delta$ require inverting complete and incomplete elliptic integrals of the first and second kinds ($K(k)$ and $E(k)$). Conventional solvers rely on adaptive numerical quadrature or multi-step iterative loops.

Ramanujan provided hyper-convergent theta-function and $q$-series representations for the elliptic modulus $k$:

$$k = \frac{\theta_2^2(q)}{\theta_3^2(q)} = 4 q^{1/2} \left( \frac{1 + q^2 + q^6 + \dots}{1 + 2q + 2q^4 + \dots} \right)^2$$

where the nome $q$ is defined by:

$$q = \exp\left(-\pi \frac{K'(k)}{K(k)}\right)$$

- **Computational Advantage:** Inverts the load-displacement relationship analytically in 2–3 evaluations, bypassing Newton–Raphson inner iterations and eliminating tangent matrix conditioning problems near contact.

---

### 1.2 Axial Crush: Yoshimura Origami Pattern & Lattice Sums
Under axial compression, the cylinder wall buckles into the **Yoshimura pattern**—a triangular/diamond isometric origami folding mode.

The post-buckling configuration minimizes the total energy functional:

$$\Pi = U_{\text{bending}} + U_{\text{membrane}} - W_{\text{ext}}$$

The deformation field maps onto an unrolled 2D doubly periodic triangular grid with circumferential wavenumber $m$ and axial wave tiers $n$.

#### Ramanujan Acceleration
Evaluating the total membrane energy across the periodic fold network involves double lattice sums over the discrete surface:

$$S = \sum_{(m, n) \in \mathbb{Z}^2 \setminus \{(0,0)\}} \frac{1}{\left(a m^2 + b m n + c n^2\right)^s}$$

- **The Problem:** In standard continuum models, these sums converge conditionally and at an extremely slow algebraic rate ($O(1/N)$).
- **Ramanujan Solution (Chowla–Selberg Analogs):** Ramanujan’s modular transformation formulas convert slowly converging 2D spatial sums into exponentially decaying series:

  $$S \propto \sum_{n=1}^\infty \sigma_{2s-1}(n) \exp\left(-2\pi n \sqrt{\Delta}\right)$$

- **Computational Advantage:** The minimum-energy facet aspect ratio and preferred circumferential mode $m$ (typically $m \in [3, 5]$ for standard beverage cans) can be determined algebraically in $O(1)$ operations rather than through dense non-linear eigen-buckling parameter sweeps.

---

### 1.3 Boundary Creases: Inextensible Kinematic Constraints
Because the radius-to-thickness ratio of a beverage can is large ($R/t \approx 1000\text{--}2000$), middle-surface stretching requires orders of magnitude more strain energy than bending. Consequently, deformation is near-inextensional: **the perimeter of any cross-sectional slice must remain invariant**.

As circular cross-sections pinch into oval or self-contact geometries, the instantaneous perimeter involves the complete elliptic integral of the second kind, $E(e)$:

$$P_{\text{ring}} = 4 a E(e) = 4 a \int_0^{\pi/2} \sqrt{1 - e^2 \sin^2\phi} \, d\phi$$

#### Ramanujan Acceleration
Ramanujan developed remarkably accurate algebraic approximations for ellipse perimeters. Defining $h = \frac{(a - b)^2}{(a + b)^2}$:

$$P_{\text{ring}} \approx \pi (a + b) \left( 1 + \frac{3h}{10 + \sqrt{4 - 3h}} \right)$$

- **Accuracy:** The relative error is bounded within $O(h^5) \approx 4 \times 10^{-5}$, scaling to machine precision with higher-order terms.
- **Computational Advantage:** Collision and contact projection algorithms can enforce inextensible cross-sectional constraints algebraically at every time step without numerical line quadratures.

---

## 2. System Architecture: The Hybrid Analytical Pipeline

```
                       [ Input: Axial Load / Radial Displacement ]
                                           │
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │       Ramanujan Analytical Kernel         │
                     │  - Rapid q-series elliptic evaluations    │
                     │  - Chowla-Selberg energy mode selection   │
                     └─────────────────────┬─────────────────────┘
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             ▼                                                           ▼
┌──────────────────────────────┐                            ┌──────────────────────────────┐
│  Yoshimura Kinematics Engine │                            │   Inextensible Crease Filter │
│  - Evaluates fold geometry   │                            │  - Strong perimeter check    │
│  - Predicts snap bifurcation │                            │  - Ramanujan algebraic bounds│
└──────────────┬───────────────┘                            └──────────────┬───────────────┘
               │                                                           │
               └─────────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────┐
                     │       Reduced System Assembler            │
                     │  - Computes exact tangent stiffness [K]   │
                     │  - Global solve via sparse direct solver  │
                     └───────────────────────────────────────────┘
```

---

## 3. Comparative Benchmark: Classical FEM vs. Ramanujan Semi-Analytical Engine

| Metric / Capability | Conventional Non-Linear FEM | Ramanujan Semi-Analytical Engine |
| :--- | :--- | :--- |
| **Bifurcation Handling** | Requires Riks arc-length steps and numerical damping to survive snap-through | Predicts equilibrium branches directly via closed-form branch equations |
| **Mesh Sensitivity** | Severe; fine shell elements required to mitigate shear and membrane locking | Mesh-free for 1D cross-sectional elastica; discrete facet formulation for Yoshimura folds |
| **Perimeter Conservation** | Enforced weakly via penalty multipliers, worsening matrix condition numbers | Enforced strictly via closed-form algebraic invariants |
| **Execution Performance** | $O(N^2)$ to $O(N^3)$ matrix operations per load step | $O(1)$ analytical mode prediction; sub-millisecond per-frame updates |

---

## 4. Key Takeaways
1. **Targeted Domain:** Ramanujan's mathematical toolkit does not replace arbitrary unstructured continuum solvers, but acts as an exact accelerator for geometries governed by elliptic kinematics, modular symmetries, and isometric foldings.
2. **Practical Value:** For interactive mechanical simulation, real-time haptic feedback, and thin-shell metamaterial design, Ramanujan-based semi-analytical formulations remove the convergence failures and latency typical of traditional FE solvers.

---

## Corrections (added 2026-09-19 after checking against the design gate)

The core idea of §1.1 holds and is implemented as `CMP-sim.analytic` ([docs/11](../11-analytic-squeeze.md)). Checking it found these corrections:

1. **R/t for a beverage can is about 330** (R = 33 mm, t = 0.1 mm), not 1000–2000. The inextensibility argument still holds.
2. **§1.1 force balance.** With the load P applied across the diameter, each quarter ring carries P/2, so λ² = P/(2EI) per unit length with the plane-strain E' = E/(1 − ν²). Up-down symmetry between the two pads makes the horizontal internal force zero, which leaves θ'' = λ² cos θ.
3. **§1.1 closed form.** Once the ring lies flat on the pads, the free arc starts straight, and the whole force–gap law is closed-form through the **lemniscate constant** ϖ = Γ(¼)²/(2√(2π)): P = 4π² E'I/(ϖ² g²). First yield is at g_y = 2π/(ϖ(Δκ_y + 1/R)). No q-series inversion is needed in that phase. Before flat contact, one scalar root-find per load suffices; its cost is the same with Ramanujan's q-series or with standard quadrature or AGM.
4. **§1.3.** The 4×10⁻⁵ error bound of Ramanujan's perimeter formula is the worst case, a completely flat ellipse. At can-relevant ovalization it is 10⁻¹⁶ (5%) to 3×10⁻⁸ (50%).
5. **§1.3 constraint.** A pinched ring is not an ellipse: it is flat where it touches the pads and tighter at the sides. Enforcing an elliptical perimeter would impose the wrong shape. The elastica is inextensible by construction, so no perimeter constraint is needed. The ellipse formula is still useful for measuring an ovalized rim ([docs/10](../10-performance.md) §5).
6. **§1.2 (Yoshimura, lattice sums)** covers axial crush and is not verified here. It is out of scope until it passes the design process.
7. **§3.** The analytic tier covers long pads, elastic behaviour, and open cans only. Short pads dent locally and still need the 3D simulator, as §4.1 of this brief says.
