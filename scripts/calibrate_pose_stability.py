import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

STEP_TAG = "POSE_STABILITY_STEP_JSON "
FRAME_TAG = "POSE_STABILITY_FRAME_JSON "
EP_RE = re.compile(r"Test episode: (\d+)")
RESULT_RE = re.compile(r"Test episode result: success=\[(.*?)\].*total_move_distance=([\[\]0-9eE+.,\-\s]+)")


def parse_float_list(text):
    text = text.strip().replace(",", " ")
    if not text:
        return []
    return [float(x) for x in text.replace("[", " ").replace("]", " ").split()]


def parse_log(path):
    episodes = []
    cur = None
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = EP_RE.search(line)
        if m:
            if cur is not None:
                episodes.append(cur)
            cur = {"id": int(m.group(1)), "steps": [], "frames": {}, "success": None}
            continue
        if cur is None:
            continue
        if STEP_TAG in line:
            cur["steps"].append(json.loads(line.split(STEP_TAG, 1)[1]))
            continue
        if FRAME_TAG in line:
            data = json.loads(line.split(FRAME_TAG, 1)[1])
            cur["frames"][int(data["step"])] = data
            continue
        m = RESULT_RE.search(line)
        if m:
            cur["success"] = parse_float_list(m.group(1))
            cur["distance"] = parse_float_list(m.group(2))
    if cur is not None:
        episodes.append(cur)
    return [ep for ep in episodes if ep["steps"]]


def as_bool(value, env_id=0):
    if isinstance(value, list):
        return bool(value[env_id])
    return bool(value)


def as_float(value, env_id=0, default=0.0):
    if value is None:
        return default
    if isinstance(value, list):
        if not value:
            return default
        return float(value[env_id])
    return float(value)


def simulate_episode(ep, trans, rot, min_views, stable_frames, max_step, env_id=0):
    stable_count = 0
    for rec in sorted(ep["steps"], key=lambda x: x.get("episode_step", 0)):
        step = int(rec.get("episode_step", rec.get("step", 0)))
        view_count = int(rec.get("view_count", step + 1))
        if view_count < min_views:
            stable_count = 0
            continue
        mask_ok = as_bool(rec.get("mask_quality", False), env_id)
        delta_t = as_float(rec.get("delta_t"), env_id, default=float("inf"))
        delta_r = as_float(rec.get("delta_r_deg"), env_id, default=float("inf"))
        stable = mask_ok and delta_t < trans and delta_r < rot
        stable_count = stable_count + 1 if stable else 0
        if stable_count >= stable_frames and step < max_step:
            return step
    return None


def evaluate(episodes, trans, rot, min_views, stable_frames, max_step, final_t_tol, final_r_tol):
    stops = []
    risky = 0
    full_success = []
    triggered_success = []
    for ep in episodes:
        success = bool(ep.get("success") and ep["success"][0] > 0.5)
        full_success.append(success)
        stop_step = simulate_episode(ep, trans, rot, min_views, stable_frames, max_step)
        if stop_step is None:
            stops.append(max_step)
            continue
        stops.append(stop_step)
        triggered_success.append(success)
        frame = ep["frames"].get(stop_step, {})
        dt_final = as_float(frame.get("delta_t_to_final"), 0, default=float("inf"))
        dr_final = as_float(frame.get("delta_r_to_final"), 0, default=float("inf"))
        if dt_final > final_t_tol or dr_final > final_r_tol:
            risky += 1
    n = len(episodes)
    triggered = sum(1 for s in stops if s < max_step)
    return {
        "translation_threshold": trans,
        "rotation_threshold_deg": rot,
        "min_views": min_views,
        "stable_frames": stable_frames,
        "episodes": n,
        "baseline_success_rate": float(np.mean(full_success)) if full_success else 0.0,
        "triggered": triggered,
        "trigger_rate": triggered / n if n else 0.0,
        "avg_exploration_steps": float(np.mean(stops)) if stops else 0.0,
        "saved_steps": max_step - float(np.mean(stops)) if stops else 0.0,
        "triggered_full_success_rate": float(np.mean(triggered_success)) if triggered_success else None,
        "risky_trigger_rate": risky / triggered if triggered else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate pose-stability early-stop thresholds from observe-mode logs.")
    parser.add_argument("logs", nargs="+", help="train.log files produced with pose_stability_early_stop.mode=observe")
    parser.add_argument("--max-step", type=int, default=4)
    parser.add_argument("--translation-grid", default="0.01,0.015,0.02,0.025,0.03,0.04,0.05")
    parser.add_argument("--rotation-grid", default="5,7.5,10,12.5,15")
    parser.add_argument("--min-views-grid", default="2,3,4")
    parser.add_argument("--stable-frames-grid", default="1,2")
    parser.add_argument("--final-t-tol", type=float, default=0.03)
    parser.add_argument("--final-r-tol", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    episodes = []
    for log in args.logs:
        episodes.extend(parse_log(log))
    if not episodes:
        raise SystemExit("No POSE_STABILITY_*_JSON records found. Run observe mode first.")

    t_grid = [float(x) for x in args.translation_grid.split(",") if x]
    r_grid = [float(x) for x in args.rotation_grid.split(",") if x]
    mv_grid = [int(x) for x in args.min_views_grid.split(",") if x]
    sf_grid = [int(x) for x in args.stable_frames_grid.split(",") if x]

    rows = []
    for trans, rot, min_views, stable_frames in itertools.product(t_grid, r_grid, mv_grid, sf_grid):
        rows.append(evaluate(episodes, trans, rot, min_views, stable_frames, args.max_step, args.final_t_tol, args.final_r_tol))

    rows.sort(key=lambda r: (r["risky_trigger_rate"], -r["saved_steps"], -r["trigger_rate"], r["translation_threshold"], r["rotation_threshold_deg"]))
    print(f"Loaded observe episodes: {len(episodes)}")
    print(f"Baseline success rate: {rows[0]['baseline_success_rate']:.3f}")
    print("| trans | rot | min_views | stable_frames | triggered | trig_rate | avg_steps | saved | trig_success | risky |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows[:args.top_k]:
        trig_success = row["triggered_full_success_rate"]
        trig_success_text = "-" if trig_success is None else f"{trig_success:.3f}"
        print(
            f"| {row['translation_threshold']:.3f} | {row['rotation_threshold_deg']:.1f} | "
            f"{row['min_views']} | {row['stable_frames']} | {row['triggered']}/{row['episodes']} | "
            f"{row['trigger_rate']:.3f} | {row['avg_exploration_steps']:.3f} | {row['saved_steps']:.3f} | "
            f"{trig_success_text} | {row['risky_trigger_rate']:.3f} |"
        )


if __name__ == "__main__":
    main()
