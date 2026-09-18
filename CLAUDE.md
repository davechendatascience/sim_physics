# sim_physics

Physics simulator (rigid + deformable + material science) for VLA training. Design in `docs/`.

## Rule: the design process gates all implementation

The design process is a component (`CMP-design-process` in `belief.yaml`) and it safeguards everything built after it.

1. Before implementing any model, solver, or feature, register its physics in the design first: doc text in `docs/`, claims in `design/claims.yaml`, and a minimal reference model in `design/oracles/`.
2. The design checks must pass for it: dimensional consistency, balance laws, thermodynamic admissibility, stated numbers, regime of validity, traceability. Run them through the component-belief MCP (`run_test`), not an ad-hoc shell.
3. A new implementation component in `belief.yaml` must consume the design via an interface `IFC-design__<component>` with CMP-design-process as producer.
4. When a check finds a physics problem, fix the design (doc + claim) first. Never loosen a tolerance to make a check pass.

Classical physics has limits. Every model states the regime where it holds (continuum scale, quasi-static, ideal gas, linear slosh, Hertz assumptions...), and the regime checks enforce those limits.

## Rule: CPU-only, renderable demos

This machine has no usable GPU for the simulator. Prioritise small demos that test the most physics per unit of compute: 2D/low-DOF oracles, static matplotlib PNGs or short GIFs. Do not build demos that need a GPU or heavy rendering to view.

## Environment

- System `python` (3.11) has numpy, scipy, sympy, matplotlib, pytest, PyYAML. No pint/hypothesis.
- `belief.yaml` changes take effect only after a human commits them.
