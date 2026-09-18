# 08 — Design Validation: Checking the Physics Before Implementation

The design is a component (`CMP-design-process` in `belief.yaml`). Everything built afterwards consumes it, so its claims are checked against classical physics **before** any of them are implemented. A design that violates a physical law cannot be rescued by a good implementation.

Classical mechanics is trustworthy only within its regime. It fails at grain scale, near wave speeds, at molecular gaps, and outside the assumptions of each closed-form result. So the checks test two things: *does the design obey the laws*, and *is it using each law where the law holds*.

## 1. The executable design

Each physics statement in `docs/` is registered in [`design/claims.yaml`](../design/claims.yaml) with a verbatim quote of the doc text. Five kinds of entry exist:

| Kind | What is registered | How it is checked |
|---|---|---|
| Equation | Expression plus the physical dimension of every symbol | Dimensional homogeneity |
| Model | A model the design specifies, plus the laws its class must obey | A minimal **reference model** in `design/oracles/` is run against each law |
| Quantity | A number stated in the docs plus its inputs | Recomputed from the inputs |
| Regime | The validity condition of a classical model plus the parameter range the design uses | Evaluated over the whole range |
| Empirical | A value tagged *[calibrate]* or *[measure]* | Must name the test that will settle it |

The reference models are deliberately tiny: 2D, a few particles, dense Newton, CPU only, well under a second each. They realise the *design's* equations, not the production code, so a failure points at the design. Later they also serve as oracles for differential testing of the implementation.

## 2. The six checks

Each check is a declared test in `belief.yaml`, run through the component-belief server. Each test case is one design claim, identified by claim id, so re-running a deterministic check does not count as new evidence.

1. **Dimensional** (`tests/design/test_dimensional.py`). Both sides of every relation, and every term of every sum, have the same dimension. Arguments of ln/exp are dimensionless, and exponents on dimensioned bases are numeric. Negative controls confirm that the checker rejects known-bad equations.
2. **Balance laws** (`test_balance_laws.py`), on the time-step and friction models:
   - linear momentum is conserved exactly;
   - contact forces obey Newton's third law;
   - energy never increases;
   - the energy and angular-momentum errors are first order in h, as the design states for implicit Euler;
   - results are Galilean invariant;
   - nothing penetrates or tunnels at large steps;
   - friction matches the Coulomb bound, sticking below the friction angle, and the kinetic slide rate on an incline;
   - friction only dissipates;
   - the F_min grasp threshold holds.
3. **Admissibility** (`test_admissibility.py`), on the materials and contact models:
   - Neo-Hookean: frame indifference, a stress-free reference, stress that derives from the energy, agreement with Hooke's law at small strain, energy that blows up at zero volume, ellipticity in the working range, a symmetric Cauchy stress, and zero net work on closed paths;
   - J2: stress stays on the yield surface, dissipation is non-negative, plastic flow keeps volume, closed cycles absorb work (Drucker–Ilyushin), elastic cycles lose nothing, and pure shear saturates at σ_y/√3;
   - Hill48: convex yield surface, reducing to von Mises when isotropic;
   - gas: pressure is −∂Ψ/∂V, and a volume cycle returns all its work;
   - barrier: repulsive only, infinite at contact, and switching on smoothly;
   - pad friction: the friction force rises with pressure.
4. **Quantities** (`test_quantities.py`). Every stated number is recomputed. Numbers measured on a reference model, such as the round-fingertip slide rate, are re-measured.
5. **Regime of validity** (`test_regime.py`). A regime that *holds* passes. A *marginal* regime passes only if the doc states a mitigation. A *violated* regime passes only if the doc explicitly excludes that use.
6. **Traceability** (`test_traceability.py`):
   - every registered quote is present in its doc;
   - every *[calibrate]*/*[measure]* tag is registered and has a validation plan;
   - parameters stated in several docs agree;
   - every (model, law) pair has a test, and no test covers an unregistered law;
   - every cross-doc link resolves.

## 3. What the checks found in the first design pass

| Finding | Check | Design change |
|---|---|---|
| Hertz theory was proposed for the silicone pad on the can, but the pad is thinner than the contact is wide | Regime | Hertz is kept only for a sphere on a thick block. Pad contact is validated by FEM and tests ([05](05-case-study-soda-can.md) §7). |
| Barrier activation distance up to 10⁻⁴ m, equal to the can wall thickness | Regime | `d̂` is bounded by 0.1× the thinnest feature ([02](02-rigid-body-and-contact.md) §2) |
| Linear slosh theory was used at robot accelerations up to 5 m/s² | Regime | Valid up to about 1 m/s², with a fallback beyond ([03](03-deformables-and-materials.md) §5) |
| A 0.1 mm wall is only 5–20 grains thick | Regime | Parameters are calibrated on can-wall specimens ([03](03-deformables-and-materials.md) §8) |
| Aluminum rate sensitivity shifts flow stress up to ~11% | Regime | Johnson–Cook rate term when the shift exceeds 5% |
| Round fingertips shed a slipping can ~4× faster than friction predicts | Balance law, on the reference model | F_min formula restricted to flat or conforming pads ([02](02-rigid-body-and-contact.md) §3) |
| Implicit Euler does not conserve angular momentum or energy exactly | Balance law | Stated honestly as first-order and dissipative ([01](01-architecture.md) §3) |
| Wave crossing time stated as 10 µs; it is 13 µs | Quantity | Corrected |

## 4. How to use it

- **Changing the design:** edit the doc and `design/claims.yaml` together, then run the six tests through the belief server (`run_test`). Never loosen a tolerance to make a check pass. If a check is wrong, fix it and say why.
- **Adding an implementation component:** register its models (with laws), equations and regimes here first. In `belief.yaml`, the component consumes `CMP-design-process` through an `IFC-design__<component>` interface.
- **Running locally:** `.venv/Scripts/python -m pytest tests/design -q`. This is for development only. Evidence counts only through `run_test`.
