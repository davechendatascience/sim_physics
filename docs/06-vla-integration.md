# 06 — VLA Integration

## 1. What VLA training needs from a simulator

1. **Throughput:** millions of episodes, batched on GPU
2. **Visual realism and diversity:** for the vision encoder
3. **Language-grounded tasks** with *programmatic success checks*
4. **Actions a real robot can execute**, including force-aware ones
5. **Physically meaningful labels** beyond pixels and joint angles

Today's pipelines usually lack items 4–5 for contact-rich tasks. A VLA trained on position-only actions with "did it lift?" as the only success signal learns nothing about crushing. This design targets that gap.

## 2. Environment API

Gymnasium-compatible, vectorized:

```python
envs = simphys.make_vec("pick_container", num_envs=4096, tier="T0",
                        randomize=Randomization.from_material_db(), seed=0)
obs, info = envs.reset()
# obs: {"rgb": [N,V,H,W,3], "depth": ..., "proprio": ..., "tactile": [N,2,h,w,3], "ft": [N,6],
#       "instruction": ["pick up the open can gently", ...]}
obs, reward, terminated, truncated, info = envs.step(action)
# info["ledger"]: per-object irreversible quantities (plastic strain, dent depth, spill, ...)
# info["envelope"]: current grasp envelope (privileged, optional)
# info["contacts"]: contact wrenches/patch pressure fields (privileged, optional)
```

### Action spaces
- `ee_pose_delta + gripper_position` (compatible with existing datasets)
- `ee_pose_delta + gripper_force` (recommended for contact-rich tasks)
- `ee_pose_delta + impedance(K, D) + gripper_force` (full compliance control)

We **strongly recommend training VLAs with a force or impedance component** in the action space. The envelope result in [05](05-case-study-soda-can.md) shows that position-controlled grippers are the main cause of damage.

## 3. Task specification

Tasks are declared as language templates plus physical predicates over the scene and the ledger:

```yaml
task: pick_container_intact
instruction_templates:
  - "pick up the {object} without crushing it"
  - "lift the {state_adj} {object} carefully"
success:
  all:
    - lifted(obj, height>=0.10, hold_time>=1.0)
    - invariant(obj, "intact")              # ledger: no plastic strain / buckle / leak
    - spill(obj) < 1e-6                     # m^3
failure_early_termination:
  - invariant_violated(obj, "intact")
reward_shaping: [approach, contact_quality, envelope_margin]
```

Success is *physical*, not visual. A can that looks fine but has 0.2 mm of residual dent fails.

## 4. Data generation: privileged expert → VLA

The inverse layer ([04](04-inverse-problems.md)) acts as a **privileged teacher**. It sees the true latent state and material parameters and solves for robust, envelope-respecting trajectories. Pipeline:

1. **Scenario sampler:** object assets × material-DB distributions × latent states (sealed/open/fill/wet) × scene clutter × lighting/textures × instruction templates
2. **Expert solve** (T1, with T2 spot checks). It generates trajectories with force profiles. Probing behavior is included when the latent state is ambiguous *from observations alone*, which teaches the VLA to probe instead of guessing.
3. **Render** observations (photoreal renderer for a subset, rasterizer for the rest)
4. **Export** to standard formats (LeRobot, RLDS/Open-X-compatible), with extra fields:
   - `action.gripper_force`, `action.impedance`
   - `labels.envelope_min/max`, `labels.ledger.*`, `labels.latent_state` (for auxiliary losses)
   - `labels.contact_patches` (sparse), `labels.tactile`
5. **Distill** into the VLA via behavior cloning. Optionally RL fine-tune in T0 with ledger-based penalties. Auxiliary heads that predict latent state and envelope margin give the policy an internal model of fragility.

The teacher must *not* leak privileged information into actions that the student cannot justify from observations. Probe-then-act demos are generated through the belief-space planner, not the ground-truth planner.

## 5. Tactile & force sensing simulation

- Tactile (GelSight/DIGIT-style) images are generated from the **contact pressure field** on the pad (available from IPC) through an optical model. Marker displacement comes from the pad's FEM surface.
- F/T sensors are simulated with bias drift, noise and bandwidth limits.
- These signals are what the VLA needs to feel incipient slip and rising stiffness in closed loop.

## 6. Domain randomization, done with physics

Randomization is drawn from the **material DB distributions** ([03](03-deformables-and-materials.md) §3), not from arbitrary ranges. So a robust policy is robust to physically plausible variation, not to nonsense variation. Additional axes: actuator gains, latencies, sensor noise, camera intrinsics and extrinsics, textures and lighting.

## 7. Sim-to-real loop

1. Deploy the policy on a real robot with logging of tactile, F/T, video and outcomes (with post-hoc 3D scans of objects for damage)
2. Run system ID ([04](04-inverse-problems.md) §2) on real logs. Update the material DB posteriors.
3. Regenerate or reweight simulation data. Retrain.
4. Track a **sim-to-real gap dashboard**: predicted vs actual slip and damage rates per object class.

## 8. Throughput targets (initial)

| Use | Tier | Target |
|---|---|---|
| RL fine-tuning | T0 | ≥ 50k env-steps/s on one high-end GPU (rigid + damage surrogate) |
| Demo generation | T1 | ≥ 1k expert episodes/hour/GPU |
| Validation spot checks | T2 | 1% of generated episodes |
