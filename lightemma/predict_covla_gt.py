#!/usr/bin/env python3
"""
One-pass builder:
- Reads CoVLA-style scene JSONs (with frames -> state, caption, image_path, etc.)
- Produces (optionally) LightEMMA-style per-scene JSONs
- Produces a single LLaVA-format training JSON

Requires:
  - your existing utils.py with: load_config, global_to_ego_frame, compute_speed, compute_curvature
  - scipy (for quaternion handling)
"""

import os
import re
import json
import math
import argparse
import datetime
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# ────────────────────────────── Utilities ──────────────────────────────

def split_extrinsic(E: List[List[float]]) -> Tuple[List[float], List[float]]:
    """Convert a 4×4 extrinsic matrix to (translation_xyz, quaternion_xyzw)."""
    E = np.asarray(E, dtype=float)
    t = E[:3, 3].tolist()
    q = R.from_matrix(E[:3, :3]).as_quat().tolist()  # xyzw
    return t, q

def yaw_from_ned(ned: List[float]) -> float:
    """atan2(E, N) with sign so +CCW = left (matches LightEMMA)."""
    return -math.atan2(ned[1], ned[0])

def load_scene_json(path: Path) -> Dict[str, Any]:
    """
    Load a 'CoVLA converted scene' JSON and extract aligned arrays for faster processing.
    Expects the file structure you showed earlier (frames with state, caption, image_path, etc.).
    """
    with path.open("r", encoding="utf-8") as f:
        scene = json.load(f)

    front_camera_images: List[str] = []
    camera_params:      List[Dict[str, Any]] = []
    ego_positions:      List[Tuple[float, float]] = []
    ego_headings:       List[float] = []
    timestamps:         List[int] = []
    frame_captions:     List[str] = []

    for fr in scene["frames"]:
        img_path = Path(fr["image_path"])
        state    = fr["state"]
        caption  = fr.get("caption", {})  # robust if missing

        # camera pose: convert to cam->ego so rotation/translation are in ego CS
        E_ego2cam = np.asarray(state["extrinsic_matrix"], dtype=float)
        E_cam2ego = np.linalg.inv(E_ego2cam)
        t = E_cam2ego[:3, 3].tolist()
        q = R.from_matrix(E_cam2ego[:3, :3]).as_quat().tolist()  # xyzw

        front_camera_images.append(str(img_path))
        camera_params.append({
            "rotation":    q,
            "translation": t,
            "camera_intrinsic": state["intrinsic_matrix"],
            "camera_extrinsic": state["extrinsic_matrix"],
        })
        # we’ll use only the planar (x, y) here
        ego_positions.append(tuple(state["positions_ecef"][:2]))
        ego_headings.append(yaw_from_ned(state["orientations_ned"]))
        # ns→ms for consistency with your previous script
        timestamps.append(int(state["timestamp"]) * 1000)
        frame_captions.append(caption.get("rich_caption", ""))

    return {
        "scene_name": scene.get("video_id", Path(path).stem),
        "description": "CoVLA converted scene",
        "front_camera_images": front_camera_images,
        "camera_params": camera_params,
        "ego_positions": ego_positions,
        "ego_headings": ego_headings,
        "timestamps": timestamps,
        "frame_captions": frame_captions,
    }

# ───────────────────── your project utilities import ───────────────────
# keep exactly as in your current script so it finds load_config, etc.
from utils import *  # noqa: E402,F401,F403

# ────────────────────────────── Main build ─────────────────────────────

def build_llava_and_optionally_lightemma(
    scene_json_dir: str,
    config_path: str,
    base_image_dir: str,
    llava_output_prefix: str,
    emit_lightemma: bool = False,
    lightemma_out_dir: str = None,
    model_name: str = "Qwen",
    scene_filter: str = None,
):
    """
    Walk all scene JSONs in `scene_json_dir`, compute per-frame fields, and:
      - append a LLaVA training sample for each frame
      - optionally write a LightEMMA per-scene JSON to `lightemma_out_dir`
    """
    # Load config lengths (obs, fut, ext)
    config = load_config(config_path)
    OBS_LEN = config["prediction"]["obs_len"]
    FUT_LEN = config["prediction"]["fut_len"]
    EXT_LEN = config["prediction"]["ext_len"]
    TTL_LEN = OBS_LEN + FUT_LEN + EXT_LEN

    # IO setup
    json_root = Path(scene_json_dir)
    json_files = sorted([p for p in json_root.glob("*.json")])

    if scene_filter:
        json_files = [p for p in json_files if scene_filter in p.stem]

    if not json_files:
        raise FileNotFoundError(f"No scene JSONs found under: {scene_json_dir}")

    if emit_lightemma:
        if not lightemma_out_dir:
            raise ValueError("--emit-lightemma requires --lightemma-out")
        Path(lightemma_out_dir).mkdir(parents=True, exist_ok=True)

    # LLaVA aggregated dataset
    llava_dataset = []
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"Processing {len(json_files)} scenes")
    random.seed(42)

    for scene_path in json_files:
        s = load_scene_json(scene_path)

        scene_name          = s["scene_name"]
        description         = s["description"]
        front_images        = s["front_camera_images"]
        camera_params_list  = s["camera_params"]
        ego_positions       = s["ego_positions"]
        ego_headings        = s["ego_headings"]
        timestamps          = s["timestamps"]
        frame_captions      = s["frame_captions"]

        num_frames = len(front_images)
        sample_tokens = list(range(num_frames))  # dummy tokens if needed

        # prepare LightEMMA container
        scene_data = {
            "scene_info": {
                "name": scene_name,
                "description": description,
                "first_sample_token": None,
                "last_sample_token": None,
            },
            "frames": [],
            "metadata": {
                "model": model_name,
                "timestamp": timestamp_str,
                "total_frames": 0,
            },
        }

        # Walk frames
        for i in range(0, max(0, num_frames - TTL_LEN), 1):
            try:
                cur_index  = i + OBS_LEN + 1
                frame_idx  = i
                img_path   = front_images[cur_index]
                cam_param  = camera_params_list[cur_index]
                cur_pos    = ego_positions[cur_index]
                cur_heading= ego_headings[cur_index]

                # observation window (positions → ego frame)
                obs_pos    = ego_positions[cur_index - OBS_LEN - 1 : cur_index + 1]
                obs_pos    = global_to_ego_frame(cur_pos, cur_heading, obs_pos)
                obs_time   = timestamps[cur_index - OBS_LEN - 1 : cur_index + 1]

                prev_speed = compute_speed(obs_pos, obs_time)
                prev_curv  = compute_curvature(obs_pos)
                prev_actions = list(zip(prev_speed, prev_curv))

                # future window for GT (positions → ego frame)
                fut_pos    = ego_positions[cur_index - 1 : cur_index + FUT_LEN + 1]
                fut_pos    = global_to_ego_frame(cur_pos, cur_heading, fut_pos)
                fut_time   = timestamps[cur_index - 1 : cur_index + FUT_LEN + 1]

                gt_speed   = compute_speed(fut_pos, fut_time)
                gt_curv    = compute_curvature(fut_pos)
                gt_actions = list(zip(gt_speed, gt_curv))

                # The first two entries are for diff/curvature context → drop for positions
                gt_positions = fut_pos[2:]  # shape: FUT_LEN positions in ego frame

                # LightEMMA frame
                frame_data = {
                    "frame_index": frame_idx,
                    "sample_token": sample_tokens[cur_index],
                    "image_path": img_path,
                    "timestamp": timestamps[cur_index],
                    "camera_params": {
                        "rotation": cam_param["rotation"],
                        "translation": cam_param["translation"],
                        "camera_intrinsic": cam_param["camera_intrinsic"],
                    },
                    "ego_info": {
                        "position": cur_pos,
                        "heading": cur_heading,
                        "obs_positions": obs_pos,
                        "obs_actions": prev_actions,
                        "gt_positions": gt_positions,
                        "gt_actions": gt_actions,
                    },
                    # Fill inference with the real caption for this frame
                    "inference": {
                        "scene_prompt": "",  # can be filled if you actually prompt a VLM here
                        "scene_description": frame_captions[cur_index],
                        "intent_prompt": "",
                        "driving_intent": "",
                        "waypoint_prompt": "",
                        "pred_actions_str": "",
                    },
                    "token_usage": {
                        "scene_prompt": "",
                        "intent_prompt": "",
                        "waypoint_prompt": "",
                    },
                    "time_usage": {
                        "scene_prompt": "",
                        "intent_prompt": "",
                        "waypoint_prompt": "",
                    },
                }

                scene_data["frames"].append(frame_data)

                # ───────────── Build LLaVA sample for this frame ─────────────
                # Compute relative image path w.r.t. base image directory
                relative_image_path = os.path.relpath(img_path, base_image_dir)

                # The caption text to embed in the prompt
                scene_desc_str = frame_captions[cur_index] if frame_captions[cur_index] else ""

                prev_speeds      = [a[0] for a in prev_actions]
                prev_curvatures  = [a[1] for a in prev_actions]

                prompt = (
                    "<image>\n"
                    "You are an autonomous driving labeller. You have access to the front-view camera image.\n\n"
                    f"The high-level driving instruction and scene description is as follows: {scene_desc_str}\n\n"
                    f"The ego vehicle's speed for the past 1.5 seconds with 0.25 sec resolution is {prev_speeds} m/s (last index is the most recent). "
                    f"The ego vehicle's curvature for the past 1.5 seconds with 0.25 sec resolution is {prev_curvatures} (last index is the most recent). "
                    "A positive curvature indicates the ego is turning left. A negative curvature indicates the ego is turning right.\n\n"
                    "Predict the speed and curvature for the next 6 waypoints, with 0.25-second resolution. "
                    "The predicted speed and curvature changes must obey the physical constraints of the vehicle. "
                    "Predict Exactly 6 pairs of speed and curvature, in the format: [(v1, c1), (v2, c2), (v3, c3), (v4, c4), (v5, c5), (v6, c6)].\n"
                    "Rules: ONLY return the answers in the required format, do not include punctuation or text. "
                    "one line, no leading/trailing spaces, no line breaks, no extra text. "
                    "Each vi and ci must be a real number in standard decimal notation."
                )

                # Make the GT exactly like you had it (compact, no spaces/quotes)
                gt_string = str(gt_actions).replace("'", "").replace(" ", "")

                llava_dataset.append({
                    "id": f"{scene_name}_frame_{frame_idx}",
                    "image": relative_image_path,
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt",   "value": gt_string},
                    ],
                })

            except Exception as e:
                print(f"[{scene_name}] Skipping frame {i}: {e}")
                continue

        # finalize & optionally save LightEMMA per-scene file
        scene_data["metadata"]["total_frames"] = len(scene_data["frames"])

        if emit_lightemma:
            out_path = Path(lightemma_out_dir) / f"{scene_name}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(scene_data, f, indent=2)
            print(f"LightEMMA scene saved: {out_path} ({len(scene_data['frames'])} frames)")

    # Write single LLaVA training file
    llava_out = f"{llava_output_prefix}_train.json"
    with open(llava_out, "w", encoding="utf-8") as f:
        json.dump(llava_dataset, f, indent=2)
    print(f"\n✅ LLaVA dataset written: {llava_out} ({len(llava_dataset)} samples)")

# ────────────────────────────── CLI wrapper ─────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build LLaVA dataset (and optionally LightEMMA scenes) in one pass."
    )
    p.add_argument("--input", required=True,
                   help="Directory containing per-scene JSONs (CoVLA→JSON format)")
    p.add_argument("--config", default="ligthemma/config.yaml",
                   help="Path to config.yaml (for obs/fut/ext lengths)")
    p.add_argument("--base-image-dir", required=True,
                   help="Base directory to compute relative image paths for LLaVA entries")
    p.add_argument("--llava-output", required=True,
                   help="Output file prefix for LLaVA JSON (script appends _train.json)")
    p.add_argument("--emit-lightemma", action="store_true",
                   help="Also write per-scene LightEMMA JSONs")
    p.add_argument("--lightemma-out", default=None,
                   help="Directory for LightEMMA JSONs (required if --emit-lightemma)")
    p.add_argument("--model", default="Qwen",
                   help="Model name stored in LightEMMA metadata")
    p.add_argument("--scene-filter", default=None,
                   help="Substring to select only matching scene files by stem")
    return p.parse_args()

def main():
    args = parse_args()
    build_llava_and_optionally_lightemma(
        scene_json_dir=args.input,
        config_path=args.config,
        base_image_dir=args.base_image_dir,
        llava_output_prefix=args.llava_output,
        emit_lightemma=args.emit_lightemma,
        lightemma_out_dir=args.lightemma_out,
        model_name=args.model,
        scene_filter=args.scene_filter,
    )

if __name__ == "__main__":
    main()
