# evaluation/animate_compare.py
import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from matplotlib import patheffects
from matplotlib.transforms import Affine2D

from ml_models.env_gym import Simple2DEnv
from ml_models.fsm_policy import FsmPolicy
from ml_models.ppo_policy import PPOPolicy
from ml_models.ppo_shield_policy import PPOShieldPolicy


# -----------------------------
# Helpers
# -----------------------------
def _load_scenario_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unit_vec(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta)], dtype=float)


def _as_action(policy, obs):
    """
    Robustly handle policy.act that returns:
      - int
      - (int, info_dict)
    """
    out = policy.act(obs)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return int(out[0]), out[1]
    return int(out), {}


def _load_policy(name: str):
    name = name.lower()
    if name in ("fsm", "classical"):
        return FsmPolicy()
    if name in ("ppo",):
        return PPOPolicy("ml_models/ppo_policy.zip")
    if name in ("ppo_shield", "ppo+shield", "ppo-shield"):
        return PPOShieldPolicy("ml_models/ppo_policy.zip")
    raise ValueError(f"Unknown policy '{name}'")


def _obs_extract(obs):
    """
    Expected obs for this project:
      [v, front, left, right, goal_dist, goal_sin, goal_cos]  (7)
    but we tolerate 6 too.
    """
    o = np.array(obs, dtype=float).flatten()
    v = float(o[0]) if o.size >= 1 else 0.0
    front = float(o[1]) if o.size >= 2 else None
    left = float(o[2]) if o.size >= 3 else None
    right = float(o[3]) if o.size >= 4 else None
    goal_dist = float(o[4]) if o.size >= 5 else None
    goal_sin = float(o[5]) if o.size >= 6 else None
    goal_cos = float(o[6]) if o.size >= 7 else None
    return v, front, left, right, goal_dist, goal_sin, goal_cos


def _clamp_sensor(val, default_len):
    # treat 999 as "no hit"
    if val is None:
        return float(default_len)
    if val > 100:
        return float(default_len)
    return float(np.clip(val, 0.0, 5.0))


def _get_vehicle_obj(env):
    """
    Discover a vehicle-like object inside env by scanning attributes.

    We look for an object/dict that provides:
      - x, y
      - heading (or yaw/theta)
      - v (or speed)

    Supports:
      - direct attributes on env (env.x, env.y, ...)
      - nested objects in env.__dict__ (env.car, env.agent, env._vehicle, ...)
      - one-level-deeper nesting (env.sim.vehicle, env.state.vehicle, ...)
      - dict-like holders
      - optional env.get_state() / env.state_dict() returning a dict
    """
    # 0) callable state providers (common in custom envs)
    for fn_name in ("get_state", "state_dict", "get_vehicle_state"):
        if hasattr(env, fn_name) and callable(getattr(env, fn_name)):
            try:
                st = getattr(env, fn_name)()
                if isinstance(st, dict):
                    return st
            except Exception:
                pass

    def has_fields(obj) -> bool:
        if obj is None:
            return False
        if isinstance(obj, dict):
            keys = set(obj.keys())
            has_xy = ("x" in keys and "y" in keys) or ("pos" in keys)
            has_h = ("heading" in keys) or ("yaw" in keys) or ("theta" in keys)
            has_v = ("v" in keys) or ("speed" in keys)
            return has_xy and has_h and has_v
        has_xy = hasattr(obj, "x") and hasattr(obj, "y")
        has_h = hasattr(obj, "heading") or hasattr(obj, "yaw") or hasattr(obj, "theta")
        has_v = hasattr(obj, "v") or hasattr(obj, "speed")
        return has_xy and has_h and has_v

    # 1) env itself might store x/y directly
    if has_fields(env):
        return env

    # 2) scan env.__dict__ values
    d = vars(env) if hasattr(env, "__dict__") else {}
    for _, val in d.items():
        if has_fields(val):
            return val

    # 3) scan one level deeper (env.sim.vehicle, env.agent.state, etc.)
    for _, val in d.items():
        if val is None:
            continue
        if isinstance(val, dict):
            for _, v2 in val.items():
                if has_fields(v2):
                    return v2
            continue
        if hasattr(val, "__dict__"):
            for _, val2 in vars(val).items():
                if has_fields(val2):
                    return val2

    # 4) last resort: look for common attribute names explicitly
    for cand_name in ("car", "agent", "vehicle", "_vehicle", "ego", "robot"):
        if hasattr(env, cand_name):
            cand = getattr(env, cand_name)
            if has_fields(cand):
                return cand

    keys = list(d.keys())
    raise AttributeError(
        f"{type(env).__name__}: could not find vehicle state. "
        f"Tried env itself, env.__dict__ values, one-level nested scan, and common names. "
        f"Top-level env attributes: {keys}"
    )


def _read_state(vobj):
    """
    Reads x,y,heading,v from either an object with attributes or a dict.
    Tolerates common alternate layouts:
      - heading / yaw / theta
      - v / speed
      - dict with pos=[x,y]
    """
    if isinstance(vobj, dict):
        if "x" in vobj and "y" in vobj:
            x, y = vobj["x"], vobj["y"]
        elif "pos" in vobj and isinstance(vobj["pos"], (list, tuple)) and len(vobj["pos"]) >= 2:
            x, y = vobj["pos"][0], vobj["pos"][1]
        else:
            raise AttributeError(f"State dict missing x/y. Keys={list(vobj.keys())}")

        heading = vobj.get("heading", vobj.get("yaw", vobj.get("theta", 0.0)))
        v = vobj.get("v", vobj.get("speed", 0.0))
        return float(x), float(y), float(heading), float(v)

    if not (hasattr(vobj, "x") and hasattr(vobj, "y")):
        raise AttributeError(f"Vehicle object missing x/y attrs. type={type(vobj).__name__}")

    x = getattr(vobj, "x")
    y = getattr(vobj, "y")
    heading = getattr(vobj, "heading", getattr(vobj, "yaw", getattr(vobj, "theta", 0.0)))
    v = getattr(vobj, "v", getattr(vobj, "speed", 0.0))
    return float(x), float(y), float(heading), float(v)


# -----------------------------
# Cinematic compare animation
# -----------------------------
def animate_compare(
    scenario_path: str,
    out_path: str,
    fps: int = 20,
    max_steps: int = 600,
):
    # Policies
    pol_left = _load_policy("fsm")
    pol_right = _load_policy("ppo_shield")

    # Load scenario JSON (do not depend on env internals)
    sc = _load_scenario_json(scenario_path)
    vehicle0 = sc["vehicle"]
    goal = sc["goal"]
    obstacles = sc.get("obstacles", [])

    # Two env instances (same scenario)
    env_l = Simple2DEnv(scenario_path)
    env_r = Simple2DEnv(scenario_path)
    obs_l = env_l.reset()
    obs_r = env_r.reset()

    # Bounds (world box)
    gx, gy = float(goal["x"]), float(goal["y"])
    sx, sy = float(vehicle0["x"]), float(vehicle0["y"])
    ox = [float(o["x"]) for o in obstacles]
    oy = [float(o["y"]) for o in obstacles]

    minx_w = min([sx, gx] + (ox if ox else [sx])) - 2.0
    maxx_w = max([sx, gx] + (ox if ox else [sx])) + 2.0
    miny_w = min([sy, gy] + (oy if oy else [sy])) - 2.0
    maxy_w = max([sy, gy] + (oy if oy else [sy])) + 2.0

    world_w = maxx_w - minx_w
    world_h = maxy_w - miny_w

    # Cinematic camera window (fixed size, with a minimum)
    cam_w = max(8.0, min(0.75 * world_w, world_w))
    cam_h = max(6.0, min(0.75 * world_h, world_h))
    cam_alpha = 0.18  # smoothing; higher = snappier

    # Figure layout
    fig = plt.figure(figsize=(12.8, 5.2), dpi=120)
    gs = fig.add_gridspec(1, 2, wspace=0.06)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    for ax in (axL, axR):
        ax.set_xlim(minx_w, maxx_w)
        ax.set_ylim(miny_w, maxy_w)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("#0f111a")
        for spine in ax.spines.values():
            spine.set_alpha(0.25)
        ax.tick_params(colors="white", labelsize=9)

    axL.set_title("FSM (Deterministic)", color="white", fontsize=12, pad=10)
    axR.set_title("PPO + Safety Shield", color="white", fontsize=12, pad=10)

    # Static drawing
    def draw_static(ax):
        gpatch = Circle(
            (gx, gy),
            float(goal["radius"]),
            fill=False,
            linewidth=2.0,
            edgecolor="#ffd166",
        )
        ax.add_patch(gpatch)
        ax.text(
            gx,
            gy + float(goal["radius"]) + 0.25,
            "GOAL",
            color="#ffd166",
            fontsize=10,
            ha="center",
            va="bottom",
        )
        for o in obstacles:
            c = Circle(
                (float(o["x"]), float(o["y"])),
                float(o["radius"]),
                fill=False,
                linewidth=2.0,
                edgecolor="#56cfe1",
            )
            ax.add_patch(c)

    draw_static(axL)
    draw_static(axR)

    # Vehicle geometry (centered rectangle that we rotate via transform)
    veh_w, veh_h = 0.8, 0.45

    vehL = Rectangle((-veh_w / 2, -veh_h / 2), veh_w, veh_h, fill=False, linewidth=2.0, edgecolor="#ff4d6d")
    axL.add_patch(vehL)
    headL_line, = axL.plot([], [], linewidth=2.0, color="#ff4d6d")
    sensorsL = axL.plot([], [], linewidth=1.2, color="#ff9aa2", alpha=0.7)[0]
    trailL = LineCollection([], linewidths=2.0, colors=["#ff4d6d"], alpha=0.55)
    axL.add_collection(trailL)

    vehR = Rectangle((-veh_w / 2, -veh_h / 2), veh_w, veh_h, fill=False, linewidth=2.0, edgecolor="#2ecc71")
    axR.add_patch(vehR)
    headR_line, = axR.plot([], [], linewidth=2.0, color="#2ecc71")
    sensorsR = axR.plot([], [], linewidth=1.2, color="#9bffb8", alpha=0.7)[0]
    trailR = LineCollection([], linewidths=2.0, colors=["#2ecc71"], alpha=0.55)
    axR.add_collection(trailR)

    # Outcome markers (goal/collision)
    goal_hitL, = axL.plot([], [], marker="o", markersize=9, linestyle="None", color="#ffd166")
    coll_hitL, = axL.plot([], [], marker="x", markersize=11, linestyle="None", color="#ff4d6d")
    goal_hitR, = axR.plot([], [], marker="o", markersize=9, linestyle="None", color="#ffd166")
    coll_hitR, = axR.plot([], [], marker="x", markersize=11, linestyle="None", color="#2ecc71")

    # Text overlays
    txtL = axL.text(0.02, 0.98, "", transform=axL.transAxes, va="top", ha="left",
                    color="white", fontsize=10)
    txtR = axR.text(0.02, 0.98, "", transform=axR.transAxes, va="top", ha="left",
                    color="white", fontsize=10)

    banner = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=12, color="white")
    banner.set_path_effects([patheffects.withStroke(linewidth=3, foreground="black", alpha=0.7)])
    banner.set_alpha(0.0)

    ptsL = [(sx, sy)]
    ptsR = [(sx, sy)]
    step_times_ms = {"L": [], "R": []}

    # banner state
    flash_frames = 0
    flash_text = ""

    # action names (keep consistent with your project)
    ACTIONS = ["STOP", "GO_FORWARD", "TURN_LEFT", "TURN_RIGHT", "SLOW_DOWN"]

    def fmt_action(aid: int) -> str:
        if 0 <= aid < len(ACTIONS):
            return ACTIONS[aid]
        return str(aid)

    def set_banner(text: str, frames: int = 20):
        nonlocal flash_frames, flash_text
        flash_text = text
        flash_frames = int(frames)

    # camera centers (smoothed)
    cam_cx_L, cam_cy_L = sx, sy
    cam_cx_R, cam_cy_R = sx, sy

    def _update_camera(ax, x, y, which):
        nonlocal cam_cx_L, cam_cy_L, cam_cx_R, cam_cy_R

        if which == "L":
            cam_cx_L = (1.0 - cam_alpha) * cam_cx_L + cam_alpha * x
            cam_cy_L = (1.0 - cam_alpha) * cam_cy_L + cam_alpha * y
            cx, cy = cam_cx_L, cam_cy_L
        else:
            cam_cx_R = (1.0 - cam_alpha) * cam_cx_R + cam_alpha * x
            cam_cy_R = (1.0 - cam_alpha) * cam_cy_R + cam_alpha * y
            cx, cy = cam_cx_R, cam_cy_R

        # Clamp camera window to world bounds
        half_w, half_h = cam_w / 2.0, cam_h / 2.0
        cx = float(np.clip(cx, minx_w + half_w, maxx_w - half_w)) if world_w > cam_w else (minx_w + maxx_w) / 2.0
        cy = float(np.clip(cy, miny_w + half_h, maxy_w - half_h)) if world_h > cam_h else (miny_w + maxy_w) / 2.0

        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy - half_h, cy + half_h)

    def step_once(env, policy, obs):
        """
        Supports env.step returning:
          - (obs, done, info)
          - (obs, reward, done, info)
          - (obs, reward, terminated, truncated, info)  (gymnasium)
        """
        t0 = time.perf_counter()
        a, info_pol = _as_action(policy, obs)

        step_out = env.step(a)

        if isinstance(step_out, tuple) and len(step_out) == 3:
            obs2, done, info_env = step_out
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs2, _rew, done, info_env = step_out
        elif isinstance(step_out, tuple) and len(step_out) == 5:
            obs2, _rew, terminated, truncated, info_env = step_out
            done = bool(terminated) or bool(truncated)
        else:
            raise ValueError(f"Unexpected env.step() return format (len={len(step_out)}): {step_out}")

        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0

        if info_env is None:
            info_env = {}
        if not isinstance(info_env, dict):
            info_env = {"info": info_env}

        overridden = False
        if isinstance(info_pol, dict):
            overridden |= bool(info_pol.get("overridden", False))
            overridden |= bool(info_pol.get("shielded", False))
            overridden |= bool(info_pol.get("intervention", False))
        if isinstance(info_env, dict):
            overridden |= bool(info_env.get("overridden", False))
            overridden |= bool(info_env.get("shielded", False))
            overridden |= bool(info_env.get("intervention", False))

        done_reason = info_env.get("done_reason", "none")
        return obs2, a, bool(done), done_reason, float(ms), overridden

    def update_vehicle(ax, veh_patch, head_line, sensor_line, trail, pts, env, obs):
        vobj = _get_vehicle_obj(env)
        x, y, heading, v = _read_state(vobj)

        _v_obs, front, left, right, goal_dist, _gs, _gc = _obs_extract(obs)

        # ✅ rotate rectangle around its center using an affine transform
        trans = Affine2D().rotate(heading).translate(x, y) + ax.transData
        veh_patch.set_transform(trans)

        # heading arrow
        u = _unit_vec(heading)
        p0 = np.array([x, y])
        p1 = p0 + 0.9 * u
        head_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])

        # sensor rays
        ray_front = _clamp_sensor(front, default_len=2.5)
        ray_side = 1.8
        left_dir = _unit_vec(heading + np.deg2rad(45))
        right_dir = _unit_vec(heading - np.deg2rad(45))

        segs = [
            (p0, p0 + ray_front * u),
            (p0, p0 + ray_side * left_dir),
            (p0, p0 + ray_side * right_dir),
        ]
        xs = [
            segs[0][0][0], segs[0][1][0], np.nan,
            segs[1][0][0], segs[1][1][0], np.nan,
            segs[2][0][0], segs[2][1][0]
        ]
        ys = [
            segs[0][0][1], segs[0][1][1], np.nan,
            segs[1][0][1], segs[1][1][1], np.nan,
            segs[2][0][1], segs[2][1][1]
        ]
        sensor_line.set_data(xs, ys)

        # trail
        pts.append((x, y))
        if len(pts) >= 2:
            poly = np.array([[pts[i], pts[i + 1]] for i in range(len(pts) - 1)], dtype=float)
            trail.set_segments(poly[-200:])

        # min sensor stat (use unclamped if present, but clamp to show stable values)
        s_front = _clamp_sensor(front, default_len=2.5)
        s_left = _clamp_sensor(left, default_len=ray_side)
        s_right = _clamp_sensor(right, default_len=ray_side)
        s_min = float(min(s_front, s_left, s_right))

        return x, y, heading, v, front, left, right, goal_dist, s_min

    # simulation state
    done_l = False
    done_r = False
    reason_l = "none"
    reason_r = "none"
    last_action_l = "?"
    last_action_r = "?"

    # outcome timing
    t_done_l = None
    t_done_r = None

    # freeze markers once
    marked_goal_l = False
    marked_coll_l = False
    marked_goal_r = False
    marked_coll_r = False

    def animate(frame_idx: int):
        nonlocal obs_l, obs_r, done_l, done_r, reason_l, reason_r, last_action_l, last_action_r
        nonlocal flash_frames, t_done_l, t_done_r
        nonlocal marked_goal_l, marked_coll_l, marked_goal_r, marked_coll_r

        # timeout
        if frame_idx >= max_steps:
            if not done_l:
                done_l, reason_l = True, "timeout"
            if not done_r:
                done_r, reason_r = True, "timeout"

        overridden_r = False

        if not done_l:
            obs_l, a_l, done_l, reason_l, ms_l, _ = step_once(env_l, pol_left, obs_l)
            step_times_ms["L"].append(ms_l)
            last_action_l = fmt_action(a_l)
            if done_l and t_done_l is None:
                t_done_l = frame_idx / float(fps)
        else:
            ms_l = step_times_ms["L"][-1] if step_times_ms["L"] else 0.0

        if not done_r:
            obs_r, a_r, done_r, reason_r, ms_r, overridden_r = step_once(env_r, pol_right, obs_r)
            step_times_ms["R"].append(ms_r)
            last_action_r = fmt_action(a_r)
            if done_r and t_done_r is None:
                t_done_r = frame_idx / float(fps)
        else:
            ms_r = step_times_ms["R"][-1] if step_times_ms["R"] else 0.0

        xL, yL, hL, vL, frontL, leftL, rightL, gdL, sminL = update_vehicle(
            axL, vehL, headL_line, sensorsL, trailL, ptsL, env_l, obs_l
        )
        xR, yR, hR, vR, frontR, leftR, rightR, gdR, sminR = update_vehicle(
            axR, vehR, headR_line, sensorsR, trailR, ptsR, env_r, obs_r
        )

        # cinematic camera follow per panel
        _update_camera(axL, xL, yL, "L")
        _update_camera(axR, xR, yR, "R")

        # markers (plot at vehicle position when terminal)
        if done_l and reason_l == "goal" and not marked_goal_l:
            goal_hitL.set_data([xL], [yL])
            marked_goal_l = True
        if done_l and reason_l == "collision" and not marked_coll_l:
            coll_hitL.set_data([xL], [yL])
            marked_coll_l = True

        if done_r and reason_r == "goal" and not marked_goal_r:
            goal_hitR.set_data([xR], [yR])
            marked_goal_r = True
        if done_r and reason_r == "collision" and not marked_coll_r:
            coll_hitR.set_data([xR], [yR])
            marked_coll_r = True

        # time-to-event
        ttl = f"{t_done_l:.2f}s" if (done_l and t_done_l is not None) else "NA"
        ttr = f"{t_done_r:.2f}s" if (done_r and t_done_r is not None) else "NA"

        txtL.set_text(
            f"t={frame_idx/fps:.2f}s  step={frame_idx}\n"
            f"action={last_action_l}\n"
            f"v={vL:.2f}\n"
            f"sensors(min)={sminL:.2f}\n"
            f"goal_d={gdL if gdL is not None else 'NA'}\n"
            f"done={reason_l if done_l else 'running'}  t_done={ttl}\n"
            f"lat={ms_l:.3f} ms"
        )
        txtR.set_text(
            f"t={frame_idx/fps:.2f}s  step={frame_idx}\n"
            f"action={last_action_r}\n"
            f"v={vR:.2f}\n"
            f"sensors(min)={sminR:.2f}\n"
            f"goal_d={gdR if gdR is not None else 'NA'}\n"
            f"done={reason_r if done_r else 'running'}  t_done={ttr}\n"
            f"lat={ms_r:.3f} ms\n"
            f"shield={'YES' if overridden_r else 'no'}"
        )

        if done_l and reason_l == "collision":
            set_banner("FSM: COLLISION", frames=30)
        if done_r and reason_r == "collision":
            set_banner("PPO+Shield: COLLISION", frames=30)
        if done_l and reason_l == "goal":
            set_banner("FSM: GOAL REACHED", frames=30)
        if done_r and reason_r == "goal":
            set_banner("PPO+Shield: GOAL REACHED", frames=30)
        if overridden_r:
            set_banner("SAFETY SHIELD INTERVENTION", frames=10)

        if flash_frames > 0:
            banner.set_text(flash_text)
            banner.set_alpha(1.0 if (flash_frames % 6) < 3 else 0.35)
            flash_frames -= 1
        else:
            banner.set_text("")
            banner.set_alpha(0.0)

        return (
            vehL, headL_line, sensorsL, trailL, txtL, goal_hitL, coll_hitL,
            vehR, headR_line, sensorsR, trailR, txtR, goal_hitR, coll_hitR,
            banner
        )

    from matplotlib.animation import FuncAnimation, PillowWriter

    anim = FuncAnimation(fig, animate, frames=max_steps, interval=int(1000 / fps), blit=False)

    out_path = str(out_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Decide primary + secondary outputs
    if out_p.suffix.lower() == ".mp4":
        mp4_path = out_p
        gif_path = out_p.with_suffix(".gif")
    else:
        gif_path = out_p if out_p.suffix.lower() == ".gif" else out_p.with_suffix(".gif")
        mp4_path = gif_path.with_suffix(".mp4")

    # Always write GIF
    try:
        writer_gif = PillowWriter(fps=fps)
        anim.save(str(gif_path), writer=writer_gif)
        print(f"Saved compare demo (GIF): {gif_path}")
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"GIF export failed: {type(e).__name__}: {e}") from e

    # Try MP4 via ffmpeg
    try:
        from matplotlib.animation import FFMpegWriter  # requires ffmpeg installed
        writer_mp4 = FFMpegWriter(fps=fps, bitrate=2200)
        anim.save(str(mp4_path), writer=writer_mp4)
        print(f"Saved compare demo (MP4): {mp4_path}")
    except Exception as e:
        # Don't fail the whole run if ffmpeg isn't present
        print(f"[warn] MP4 export skipped/failed ({type(e).__name__}: {e}). "
              f"Install ffmpeg and retry if you want MP4.")

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True, help="Output path (gif or mp4). We will always write a GIF and try MP4 too.")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=600)
    args = ap.parse_args()

    animate_compare(args.scenario, args.out, fps=args.fps, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
