import argparse, json, os, random
from pathlib import Path
from typing  import Any, Dict, List

import numpy as np
from PIL import Image
from tqdm  import tqdm

try:
    from datasets import load_dataset, DownloadConfig
except ImportError:
    raise SystemExit("pip install datasets --quiet")

# ──────────────────────────────────────────────────────────────────────
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_jpg(arr: np.ndarray, path: Path):
    if not path.exists():
        Image.fromarray(arr).save(path, quality=90)

def load_jsonl(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.update(json.loads(line))
    return out

# ──────────────────────────────────────────────────────────────────────
def convert_scene(
    rec,
    split_name: str,
    dst_root: Path,
    src_root: Path,
    step: int,
):
    """
    Convert exactly **one** scene record.
    """
    vid       = rec["video_id"]
    img_root  = ensure_dir(dst_root / "images"  / split_name / vid)
    json_root = ensure_dir(dst_root / "nb_json" / split_name)

    dirs = {k: src_root / k for k in ["states", "captions", "traffic_lights"]}

    states_map   = load_jsonl(dirs["states"]   / f"{vid}.jsonl")
    captions_map = load_jsonl(dirs["captions"] / f"{vid}.jsonl")
    tl_map       = load_jsonl(dirs["traffic_lights"] / f"{vid}.jsonl")

    if not states_map:                                   # no annotations
        print(f"⚠  no states for {vid} – skipped")
        return None

    video   = rec["video"]                               # VideoFrameArray
    frames  : List[Dict[str, Any]] = []
    DESIRED=7
    max_fid = max(map(int, states_map.keys()))
    for i in range(0, max_fid + 1, step):
        k = str(i)
        st = states_map.get(k)
        if st is None or i >= len(video):
            continue

        # ① write image ---------------------------------------------------
        frame_arr = video.get_batch([i]).asnumpy()[0]    # HWC uint8
        img_path  = img_root / f"{i:06d}.jpg"
        write_jpg(frame_arr, img_path)

        # ② down-sample trajectory 20 Hz → 4 Hz ---------------------------
        traj         = np.asarray(st.get("trajectory", []))  # (M,3)
        traj_4hz     = traj[::step]
        # --- skip tails that are shorter than needed ------------------------
        if len(traj_4hz) < DESIRED:
            print(f"Frame has few predicted trajectories. It has {len(traj_4hz)}")
            print("Original trajectory length: ",len(traj))
            print("Steps used: ",step)
            print("video id: ",vid)
            continue     # ← do NOT append this frame

        st           = dict(st)                              # shallow copy
        st["trajectory"] = traj_4hz.tolist()

        # ③ bundle per-frame entry ---------------------------------------
        frames.append({
            "frame_index":  i,
            "image_path":   str(img_path),
            "state":        st,
            "caption":       captions_map.get(k),
            "traffic_light": tl_map.get(k),
        })

    # ④ write scene JSON --------------------------------------------------
    scene_json = {
        "video_id":   vid,
        "split":      split_name,
        "step":       step,
        "num_frames": len(frames),
        "frames":     frames,
    }
    out_path = json_root / f"{vid}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(scene_json, f, indent=2)

    return out_path

# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",   default="/home/lukelo/scenario_dataset/CoVLA-Dataset",
                    help="Folder with raw shards (states/, captions/, …)")
    ap.add_argument("--output", default="/home/lukelo/scenario_dataset/CoVLA-Dataset/dataset_Resol_0.25s_500",
                    help="Destination root (will hold images/ nb_json/)")
    ap.add_argument("--scenes", type=int, default=500,
                    help="How many scenes TOTAL to convert (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.20,
                    help="Validation fraction (ignored if --val-num given)")
    ap.add_argument("--val-num",  type=int,   default=None,
                    help="Exact number of val scenes (overrides fraction)")
    ap.add_argument("--seed",     type=int,   default=42,
                    help="Only used if you later decide to shuffle")
    ap.add_argument("--local-only", action="store_true",
                    help="HuggingFace `local_files_only=True`")
    args = ap.parse_args()

    # 0 ─ Load dataset (streaming) ----------------------------------------
    cfg   = DownloadConfig(local_files_only=args.local_only)
    covla = load_dataset("turing-motors/CoVLA-Dataset",
                         split="train", streaming=True, download_config=cfg)

    # 1 ─ Decide split sizes WITHOUT materialising all IDs ----------------
    total_keep  = args.scenes if args.scenes > 0 else None   # None = all
    val_needed  = (args.val_num if args.val_num is not None
                   else int(round(args.val_frac * (total_keep or 1e9))))
    train_needed = (total_keep or 1e9) - val_needed

    print(f"▶  Will write   {train_needed} train   +   {val_needed} val scenes")

    # 2 ─ Iterate once through the stream ---------------------------------
    dst_root = Path(args.output)
    src_root = Path(args.root)
    FPS      = 20
    STEP     = int(0.25 * FPS)           # 4 Hz = every 0.25 s

    train_saved, val_saved = [], []
    for n, rec in enumerate(tqdm(covla, desc="overall", unit="scene")):

        if total_keep is not None and n >= total_keep:
            break                                         # done enough

        if n < train_needed:               # first block → train
            out = convert_scene(rec, "train", dst_root, src_root, STEP)
            if out: train_saved.append(out)
        else:                              # remainder → val
            out = convert_scene(rec, "val",   dst_root, src_root, STEP)
            if out: val_saved.append(out)

    # 3 ─ Summaries --------------------------------------------------------
    print("\n✓ Finished")
    print(f"  train JSONs: {len(train_saved)}  →  {dst_root/'nb_json/train'}")
    print(f"  val   JSONs: {len(val_saved)}    →  {dst_root/'nb_json/val'}")

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
