import os
import re
import ast
import argparse
import datetime
import random
from nuscenes import NuScenes
import time
from PIL import Image
import torch
from utils import *
# from vlm import ModelHandler
from qwen_vl_utils import process_vision_info
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
from PIL import Image
import sys
import re
from typing import List, Tuple


import json, math
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R   # pip install scipy

def clean_pred_actions(
    raw: str,
    expected_pairs: int = 6,
    pad_value: Tuple[float,float] = (0.0, 0.0)
) -> str:
    """
    1) Pull all numbers out of `raw`
    2) Group into (v,c) pairs
    3) Truncate to expected_pairs, or pad with pad_value
    4) Format as one-line bracketed list: "[(v1, c1), …, (vN, cN)]"
    """
    # 1) extract all ints/floats
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    vals = [float(x) for x in nums]

    # 2) build pairs
    pairs: List[Tuple[float,float]] = []
    for i in range(0, len(vals)-1, 2):
        if len(pairs) >= expected_pairs:
            break
        pairs.append((vals[i], vals[i+1]))

    # 3) pad if too few
    if len(pairs) < expected_pairs:
        last = pairs[-1] if pairs else pad_value
        pairs.extend([last] * (expected_pairs - len(pairs)))

    # 4) format
    inner = ", ".join(f"({v}, {c})" for v, c in pairs)
    return f"[{inner}]"


def parse_args():
    parser = argparse.ArgumentParser(description="LightEMMA: End-to-End Autonomous Driving")
    parser.add_argument("--model", type=str, default="llava", 
                        help="Model to use for reasoning (default: gpt-4o, "
                        "options: gpt-4o, gpt-4.1, claude-3.7, claude-3.5, "
                        "gemini-2.5, gemini-2.0, qwen2.5-7b, qwen2.5-72b, "
                        "deepseek-vl2-16b, deepseek-vl2-28b, llama-3.2-11b, "
                        "llama-3.2-90b)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file (default: config.yaml)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Optional: Specific scene name to process.")
    parser.add_argument("--all_scenes", action="store_true",
                        help="Process all scenes instead of random sampling")
    parser.add_argument('--seed',    type=int, default=2022,
                        help="Random seed (for deterministic results)")
    parser.add_argument("--continue_dir", type=str, default=None,
                        help="Path to the directory with previously processed scene JSON files to resume processing")
    parser.add_argument("--results_dir",type=str,default=None,
                        help="Directory containing the VLM prediction results (default: from config)")
    return parser.parse_args()


# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

def getMessage(prompt, image=None, args=None):
    if "llama" in args.model or "Llama" in args.model:
        message = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
    elif "qwen" in args.model or "Qwen" in args.model:
        message = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]   
    return message

def vlm_inference(text=None,
                  images=None,
                  sys_message=None,
                  processor=None,
                  model=None,
                  tokenizer=None,
                  args=None):
    """
    Returns a dict:
      {
        "text": <generated string>,
        "gen_time_sec": <float>,
        "total_time_sec": <float>,
        "tokens_generated": <int>,
        "time_per_token_sec": <float or None>
      }
    """

    # LLaMA / LLaVA branch
    if "llama" in args.model.lower():
        # --- prep input ---
        image = Image.open(images).convert('RGB')
        message = getMessage(text, args=args)
        input_text = processor.apply_chat_template(
            message, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        # --- generate & time it ---
        t0_gen = time.time()
        output_ids = model.generate(**inputs, max_new_tokens=2048)
        gen_time = time.time() - t0_gen

        # --- decode & clean ---
        raw = processor.decode(output_ids[0])
        output_text = re.findall(
            r'<\|start_header_id\|>assistant<\|end_header_id\|>'
            r'(.*?)<\|eot_id\|>',
            raw, re.DOTALL
        )[0].strip()

        # compute token stats
        in_tokens  = inputs["input_ids"].shape[-1]
        out_tokens = output_ids.shape[-1]
        new_tokens = out_tokens - in_tokens
        tpt = gen_time / new_tokens if new_tokens > 0 else None
        token_usage = {
            "input":  in_tokens,
            "output": out_tokens
        }
        return output_text,token_usage, gen_time
    
    # Qwen-VL branch
    elif "qwen" in args.model.lower():
        message = getMessage(text, image=images, args=args)
        chat_text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        t0_gen = time.time()
        generated = model.generate(**inputs, max_new_tokens=128)
        gen_time = time.time() - t0_gen

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        in_tokens  = inputs["input_ids"].shape[-1]
        out_tokens = generated.shape[-1]
        new_tokens = out_tokens - in_tokens
        tpt = gen_time / new_tokens if new_tokens > 0 else None
        token_usage = {
            "input":  in_tokens,
            "output": out_tokens
        }
        return output_text,token_usage, gen_time
    
    elif "llava" in args.model.lower():
            conv_mode = "mistral_instruct"
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            # print("Model: ",model)
            # print(model.config.mm_use_im_start_end)
            if IMAGE_PLACEHOLDER in text:
                if model.config.mm_use_im_start_end:
                    text = re.sub(IMAGE_PLACEHOLDER, image_token_se, text)
                else:
                    text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, text)
            else:
                if model.config.mm_use_im_start_end:
                    text = image_token_se + "\n" + text
                else:
                    text = DEFAULT_IMAGE_TOKEN + "\n" + text

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image = Image.open(images).convert('RGB')

        


            image_tensor = process_images([image], processor, model.config)[0]


            t0_gen = time.time()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True,
                    temperature=0.2,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=2048,
                    use_cache=True,
                    pad_token_id = tokenizer.eos_token_id,
                )
            gen_time = time.time() - t0_gen

            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            in_tokens  = input_ids.shape[-1]
            out_tokens = output_ids.shape[-1]
            new_tokens = out_tokens - in_tokens
            tpt = gen_time / new_tokens if new_tokens > 0 else None
            # build the nested-dict:
            token_usage = {
                "input":  in_tokens,
                "output": out_tokens
            }
            return output_text,token_usage, gen_time


    


def split_extrinsic(E: List[List[float]]) -> Tuple[List[float], List[float]]:
    """
    Convert a 4×4 extrinsic matrix to (translation_xyz, quaternion_xyzw).
    """
    E = np.asarray(E, dtype=float)
    t = E[:3, 3].tolist()                 # translation
    q = R.from_matrix(E[:3, :3]).as_quat().tolist()  # xyzw
    return t, q

def yaw_from_ned(ned: List[float]) -> float:
    """atan2(E, N) with sign so +CCW = left (matches LightEMMA)."""
    return -math.atan2(ned[1], ned[0])

def load_scene_json(path: Path):
    """
    Returns a dict with the same array names the NuScenes block built.
    Raises if file missing or empty.
    """
    with path.open("r", encoding="utf-8") as f:
        scene = json.load(f)

    front_camera_images: List[str] = []
    scene_description: List[str] = []
    camera_params:      List[Dict[str, Any]] = []
    ego_positions:      List[Tuple[float, float]] = []
    ego_headings:       List[float] = []
    timestamps:         List[int] = []
    future_gt: List[List[Tuple[float,float]]] = []
   
    FUTURE_POINTS = 6   # how many waypoints

    for fr in scene["frames"]:
        # img_path = fr["image_path"]
       

        # original full path stored in the JSON
        img_path = Path(fr["image_path"])

        state    = fr["state"]            # full ego state dict
        caption=fr["caption"]

        front_camera_images.append(str(img_path))
        # print("Frame id: ",fr["frame_index"])
        # t, q = split_extrinsic(state["extrinsic_matrix"])
        E_ego2cam = np.asarray(state["extrinsic_matrix"], dtype=float)  # raw
        E_cam2ego = np.linalg.inv(E_ego2cam)                            # ← invert

        t = E_cam2ego[:3, 3].tolist()
        q = R.from_matrix(E_cam2ego[:3, :3]).as_quat().tolist()        # xyzw

        camera_params.append({
            "rotation":    q,          # quaternion xyzw
            "translation": t,          # in ego CS, metres
            "camera_intrinsic": state["intrinsic_matrix"],
            "camera_extrinsic": state["extrinsic_matrix"],
        })
        ego_positions.append(tuple(state["positions_ecef"][:2]))
        ego_headings.append(yaw_from_ned(state["orientations_ned"]))
        future_xy = [tuple(state["trajectory"][j][:2]) for j in range(1,FUTURE_POINTS+1)]
        future_gt.append(future_xy)
        timestamps.append(state["timestamp"]*1000)
        # print("description: ",caption["rich_caption"])
        scene_description.append(caption["rich_caption"])

    return {
        "scene_name": scene["video_id"],
        "description": "CoVLA converted scene",
        "front_camera_images": front_camera_images,
        "camera_params": camera_params,
        "ego_positions": ego_positions,
        "ego_headings": ego_headings,
        "future_gt":future_gt,
        "timestamps": timestamps,
        "scene_description":scene_description,
    }

KEYWORDS  = ["high speed", "green"]
# KEYWORDS = ["moderate speed", "narrow road", "cloudy", "straight"]   
WHOLE_WORD      = False   # wrap \b...\b around each keyword?
CASE_SENSITIVE  = False
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# --------------------------------------------------------------------------- #
# 1. Compile keyword regexes                                                  #
# --------------------------------------------------------------------------- #
flags = 0 if CASE_SENSITIVE else re.IGNORECASE
def _rx(kw: str) -> re.Pattern:
    pat = re.escape(kw)
    if WHOLE_WORD:
        pat = rf"\b{pat}\b"
    return re.compile(pat, flags)

KW_PAT = {kw: _rx(kw) for kw in KEYWORDS}

def run_prediction():
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    model = None
    processor = None
    tokenizer = None
    qwen25_loaded = False
    try:
       
        if "qwen" in args.model or "Qwen" in args.model:
            print("Qwen2.5-VL-3B-Instruct")
            # print(e)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            tokenizer = None
            qwen25_loaded = False
            # print("已加载 Qwen2-VL-7B-Instruct。")
        elif "llava" in args.model or "Llava" in args.model:    
            disable_torch_init()
            print("Loading LLava")
            # tokenizer, model, processor, context_len = load_pretrained_model("liuhaotian/llava-v1.6-mistral-7b", None, "llava-v1.6-mistral-7b")
            tokenizer, model, processor, context_len = load_pretrained_model("/home/lukelo/OpenEMMA/checkpoints/llava-v1.6-mistral-7b-finetuned", None, "llava-v1.6-mistral-7b")
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            # print("model: ",model)
        else:
            model = None
            processor = None
            tokenizer=None
    except Exception as e:
        print("Failed to load a model", e)


    # Configure output paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Use the provided directory for continuation, or create a new one
    if args.results_dir:
        results_dir = f"{args.results_dir}/output"
        os.makedirs(results_dir, exist_ok=True)
    elif args.continue_dir:
        results_dir = args.continue_dir
        print(f"Continuing from existing directory: {results_dir}")
    
    # Initialize random seed for reproducibility
    random.seed(42)
    
    json_root = Path("/home/lukelo/scenario_dataset/CoVLA-Dataset/dataset_Resol_0.25s_150/nb_json/val")   # e.g. "json_out"
    json_files = sorted(json_root.glob("*.json"))

    # if args.scene:
    json_files = [p for p in json_files]
    # if not json_files:
    #     print(f"Scene '{args.scene}' not found")
    #     return

    print(f"Processing {len(json_files)} JSON scenes")



    # # Load NuScenes parameters from config
    OBS_LEN = config["prediction"]["obs_len"]
    FUT_LEN = config["prediction"]["fut_len"]
    EXT_LEN = config["prediction"]["ext_len"]
    TTL_LEN = OBS_LEN + FUT_LEN + EXT_LEN 
    
    # Initialize model
    # model_handler = ModelHandler(args.model, args.config)
    # model_handler.model_instance, model_handler.processor = model_handler.initialize_model()
    print(f"Using model: {args.model}")
    
    for scene_path in json_files:
        scene_dict = load_scene_json(scene_path)

        scene_name       = scene_dict["scene_name"]
        description      = scene_dict["description"]
        front_camera_images = scene_dict["front_camera_images"]
        camera_params       = scene_dict["camera_params"]
        ego_positions       = scene_dict["ego_positions"]
        ego_headings        = scene_dict["ego_headings"]
        future_gt = scene_dict["future_gt"]
        timestamps          = scene_dict["timestamps"]
        frame_description   =scene_dict["scene_description"]

        sample_tokens = list(range(len(front_camera_images)))  # dummy
        num_frames    = len(front_camera_images)
        
        scene_data = {
            "scene_info": {
                "name": scene_name,
                "description": description,
                "first_sample_token": None,
                "last_sample_token": None
            },
            "frames": [],
            "metadata": {
                "model": args.model,
                "timestamp": timestamp,
                "total_frames": 0
            }
        }
        save_scene=True
        # Process each frame in the scene
        for i in range(0, num_frames - TTL_LEN, 1):
            try:
                cur_index = i + OBS_LEN + 1
                frame_index = i  # The relative index in the processed subset
                
                image_path = front_camera_images[cur_index]
                print(f"Processing frame {i} from {scene_name}, image: {image_path}")
                
                # Extract image ID from filename
                match = re.search(r"(\d+)(?=\.jpg$)", image_path)
                image_id = match.group(1) if match else None
                
                sample_token = sample_tokens[cur_index]
                camera_param = camera_params[cur_index]
                
                # Get current position and heading
                cur_pos = ego_positions[cur_index]
                cur_heading = ego_headings[cur_index]
                
                # Get observation data (past positions and timestamps)
                obs_pos = ego_positions[cur_index - OBS_LEN - 1 : cur_index + 1]
                obs_pos = global_to_ego_frame(cur_pos, cur_heading, obs_pos)
                obs_time = timestamps[cur_index - OBS_LEN - 1 : cur_index + 1]
                # print("obs_pos: ",obs_pos)
                # Calculate past speeds and curvatures
                prev_speed = compute_speed(obs_pos, obs_time)
                prev_curvatures = compute_curvature(obs_pos)
                # print(f"prev_curvatures: {prev_curvatures}| prev_speed: {prev_speed}")
                prev_actions = list(zip(prev_speed, prev_curvatures))
                
                # Get future positions and timestamps (ground truth)
                fut_pos = ego_positions[cur_index - 1 : cur_index + FUT_LEN + 1]
                fut_pos = global_to_ego_frame(cur_pos, cur_heading, fut_pos)
                fut_time = timestamps[cur_index - 1 : cur_index + FUT_LEN + 1]
                
                # Calculate ground truth speeds and curvatures
                gt_speed = compute_speed(fut_pos, fut_time)
                gt_curvatures = compute_curvature(fut_pos)
                gt_actions = list(zip(gt_speed, gt_curvatures))
                
                # Remove extra indices used for speed and curvature calculation
                fut_pos = fut_pos[2:]
                future_gt_current=future_gt[cur_index]
                # print("future_gt_current: ",future_gt_current)
                # Define prompts for LLM inference
                scene_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "You must observe and analyze the movements of vehicles and pedestrians, "
                    "lane markings, traffic lights, and any relevant objects in the scene. "
                    "describe what you observe, but do not infer the ego's action. "
                    "generate your response in plain text in one paragraph without any formating. "
                )
                # scene_description, scene_tokens, scene_time=vlm_inference(text=scene_prompt, images=image_path, processor=processor, model=model, tokenizer=tokenizer, args=args)
             
                scene_description=frame_description[cur_index]
                text_blob = scene_description
                # total_frames += 1
                # print("text_blob: ",text_blob)
                matched = [kw for kw, pat in KW_PAT.items() if pat.search(text_blob)]
                print("matched: ",matched)
                if matched:
                    # any_kw_frames += 1
                    # for kw in matched:
                    #     kw_counter[kw] += 1
                    if len(matched) != len(KEYWORDS):   # all keywords found
                        # print(text_blob)
                        # print("Label: ",predicted_text)
                        # all_kw_frames += 1
                        # save_scene=False
                        print("did not match,,,,,skipping")
                        continue
                else:
                    print("did not match,,,,,skipping")
                    # save_scene=False
                    continue

                # print("Here 2")
                # print("Scene description:", scene_description)
                scene_tokens={
                    "input":  0,
                    "output": 0
                }
                scene_time=0
                # Run scene description inference

                # object_prompt= (
                #          "Look at this front‐view driving image and detect all people, bicycles, cars, motorcycles, buses, trucks, and traffic lights.  "
                #             "For each one, output a single line in this format:"
                #             "class: x1,y1,x2,y2\n\n"
                #             "If there truly are no such objects, write exactly:\n"
                #             "  none detected\n\n"
                #             "Return only these lines—no extra commentary."
                #     )

                # object_description, object_tokens, object_time=vlm_inference(text=object_prompt, images=image_path, processor=processor, model=model, tokenizer=tokenizer, args=args)


                # Generate intent prompt based on scene description
                intent_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "The scene is described as follows: "
                    f"{scene_description} "
                    "The ego vehicle's speed for the past 1.5 seconds with 0.25 sec resolution is"
                    f"{prev_speed} m/s (last index is the most recent) "
                    "The ego vehicle's curvature for the past 1.5 seconds with 0.25 sec resolution is"
                    f"{prev_curvatures} (last index is the most recent) "
                    "A positive curvature indicates the ego is turning left."
                    "A negative curvature indicates the ego is turning right. "
                    "What was the ego's previous intent? "
                    "Was it accelerating (by how much), decelerating (by how much), or maintaining speed? "
                    "Was it turning left (by how much), turning right (by how much), or following the lane? "
                    "Taking into account the ego's previous intent, how should it drive in the next 3 seconds? "
                    "Should the ego accelerate (by how much), decelerate (by how much), or maintain speed? "
                    "Should the ego turn left (by how much), turn right (by how much), or follow the lane?  "
                    "Generate your response in plain text in one paragraph without any formating. "
                )
                
                # Run driving intent inference
                # driving_intent, intent_tokens, intent_time=vlm_inference(text=intent_prompt, images=image_path, processor=processor, model=model, tokenizer=tokenizer, args=args)

                driving_intent="None" 
                intent_tokens={
                    "input":  0,
                    "output": 0
                }
                intent_time=0
                # Generate waypoint prompt based on scene and intent
                waypoint_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "The high-level driving instruction and scene description is as follows: "
                    f"{scene_description} "
                    "The ego vehicle's speed for the past 1.5 seconds with 0.25 sec resolution is"
                    f"{prev_speed} m/s (last index is the most recent) "
                    "The ego vehicle's curvature for the past 1.5 seconds with 0.25 sec resolution is"
                    f"{prev_curvatures} (last index is the most recent) "
                    "A positive curvature indicates the ego is turning left."
                    "A negative curvature indicates the ego is turning right. "
                    "Predict the speed and curvature for the next 6 waypoints, with 0.25-second resolution. "
                    "The predicted speed and curvature changes must obey the physical constraints of the vehicle. "
                    "Predict Exactly 6 pairs of speed and curvature, in the format:"
                    "[(v1, c1), (v2, c2), (v3, c3), (v4, c4), (v5, c5), (v6, c6)]. "
                    "Rules:: ONLY return the answers in the required format, do not include punctuation or text."
                    "one line, no leading/trailing spaces, no line breaks, no extra text. "
                    "Each vi and ci must be a real number in standard decimal notation "

                )
                
                # Run waypoint prediction inference
   
                pred_actions_str, waypoint_tokens, waypoint_time=vlm_inference(text=waypoint_prompt, images=image_path, processor=processor, model=model, tokenizer=tokenizer, args=args)
                # pred_actions_str="[(3.291, 4.807), (3.085, 4.44), (2.823, 4.058), (2.823, 3.671), (2.823, 3.302), (2.823, 3.032)]"
                # waypoint_tokens={
                #     "input":  0,
                #     "output": 0
                # }
                # waypoint_time=0
          
                # Prepare frame data structure
                frame_data = {
                    "frame_index": frame_index,
                    "sample_token": sample_token,
                    "image_path": image_path,
                    "timestamp": timestamps[cur_index],
                    "camera_params": {
                        "rotation": camera_param["rotation"],
                        "translation": camera_param["translation"],
                        "camera_intrinsic": camera_param["camera_intrinsic"]
                    },
                    "ego_info": {
                        "position": cur_pos,
                        "heading": cur_heading,
                        "obs_positions": obs_pos,
                        "obs_actions": prev_actions,
                        "gt_positions": future_gt_current,
                        "gt_actions": gt_actions
                    },
                    "inference": {
                        "scene_prompt": format_long_text(scene_prompt),
                        "scene_description": format_long_text(scene_description),
                        "intent_prompt": format_long_text(intent_prompt),
                        "driving_intent": format_long_text(driving_intent),
                        "waypoint_prompt": format_long_text(waypoint_prompt),
                        "pred_actions_str": pred_actions_str
                    },
                    "token_usage": {
                        "scene_prompt": scene_tokens,
                        "intent_prompt": intent_tokens,
                        "waypoint_prompt": waypoint_tokens
                    },
                    "time_usage": {
                        "scene_prompt": scene_time,
                        "intent_prompt": intent_time,
                        "waypoint_prompt": waypoint_time
                    }
                }
                
              
                # Try to parse predicted actions and generate trajectory
                try:
                    # print("pred_actions_str: ",pred_actions_str)
                    pred_actions = ast.literal_eval(pred_actions_str)
                    # print("pred_actions: ",pred_actions)
                    if isinstance(pred_actions, list) and len(pred_actions) > 0:
                        # print("gt_actions: ",gt_actions)
                        # print("1")
                        prediction = integrate_driving_commands(pred_actions, dt=0.25)
                        # ground_prediction = integrate_driving_commands(gt_actions, dt=0.25)
                        # print("1.5")
                        frame_data["predictions"] = {
                            "pred_actions": pred_actions,
                            "trajectory": prediction
                        }
                        # frame_data["ego_info"]["gt_positions"]=ground_prediction

                    else:
                        print("2")
                        frame_data["predictions"] = {
                            "pred_actions_str": pred_actions_str
                        }
                except Exception as e:
                    print("3")
                    frame_data["predictions"] = {
                        "pred_actions_str": pred_actions_str
                    }
                
                # Add frame data to scene
                scene_data["frames"].append(frame_data)
                
            except Exception as e:
                print(f"Error processing frame {i} in {scene_name}: {e}")
                continue
        
        # Update total frames count
        scene_data["metadata"]["total_frames"] = len(scene_data["frames"])
        
        # Save scene data
        scene_file_path = f"{results_dir}/{scene_name}.json"
        save_dict_to_json(scene_data, scene_file_path)
        print(f"Scene data saved to {scene_file_path} with {len(scene_data['frames'])} frames")

if __name__ == "__main__":
    run_prediction()