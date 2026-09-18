"""Command line: python -m simphys list | run <scene> [--headless | --window] [...]"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="simphys", description="3D CPU physics simulator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list built-in scenes")
    run = sub.add_parser("run", help="run a scene")
    run.add_argument("scene")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="render off-screen to files (default)")
    mode.add_argument("--window", action="store_true", help="draw live in a matplotlib window")
    run.add_argument("--out", help="GIF path for headless mode (default out/<scene>.gif)")
    run.add_argument("--steps", type=int, help="override the scene's step count")
    run.add_argument("--every", type=int, default=2, help="draw every N steps")
    run.add_argument("--fps", type=int, default=20)
    run.add_argument("--no-render", action="store_true", help="simulate only; print the ledger")
    args = ap.parse_args(argv)

    from .scenes import SCENES
    if args.cmd == "list":
        for name, (_, desc) in SCENES.items():
            print(f"{name:22s} {desc}")
        return 0

    if args.scene not in SCENES:
        print(f"unknown scene {args.scene!r}; try: python -m simphys list", file=sys.stderr)
        return 2
    from . import ledger
    from .render import Renderer

    scene, info = SCENES[args.scene][0]()
    steps = args.steps or info["steps"]
    renderer = None
    if not args.no_render:
        renderer = Renderer(scene, "window" if args.window else "headless",
                            view=info.get("view", (20, -60)), title=args.scene)
        renderer.draw()
    t0 = time.time()
    for i in range(steps):
        scene.step()
        if renderer and (i + 1) % args.every == 0:
            renderer.draw()
        print(f"\rstep {i + 1}/{steps}  t={scene.t:.3f}s  newton={scene.stats['newton'][-1]}", end="", flush=True)
    print(f"\nsimulated {steps} steps in {time.time() - t0:.1f} s")

    summary = {"scene": args.scene, "steps": steps, "t": scene.t, "ledger": ledger.report(scene),
               "newton_iterations": sum(scene.stats["newton"]),
               "peak_drive_force_N": {k: float(max((abs(f[0]) for f in v), default=0.0))
                                      for k, v in scene.stats["drive_force"].items()}}
    for k, v in summary["peak_drive_force_N"].items():
        print(f"peak force on {k}: {v:.2f} N")
    for r in summary["ledger"]:
        print(f"ledger[{r['body']}]: max plastic strain {r['max_plastic_strain']:.2e}, "
              f"shape change {r['max_shape_change_mm']:.3f} mm, "
              f"altered={r['altered_by_plastic_strain'] or r['altered_by_shape']}")
    if renderer:
        if args.window:
            renderer.close()
        else:
            gif = Path(args.out or f"out/{args.scene}.gif")
            gif, png = renderer.save(gif, fps=args.fps)
            gif.with_suffix(".json").write_text(json.dumps(summary, indent=2))
            print(f"wrote {gif}, {png}, {gif.with_suffix('.json')}")
            renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
