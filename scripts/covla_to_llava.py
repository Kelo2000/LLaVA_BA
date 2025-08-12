import json
import os
import argparse
from pathlib import Path

def create_training_dataset(scene_data_dir, output_file, base_image_dir):
    """Convert scene data to LLaVA format."""
    dataset = []
    
    scene_data_path = Path(scene_data_dir)
    if not scene_data_path.exists():
        raise FileNotFoundError(f"Scene data directory not found: {scene_data_dir}")
    
    for scene_file in os.listdir(scene_data_dir):
        # Skip non-JSON files
        if not scene_file.endswith('.json'):
            continue
            
        with open(os.path.join(scene_data_dir, scene_file), 'r') as f:
            scene_data = json.load(f)
        
        for frame in scene_data['frames']:
            # Extract your existing data
            full_image_path = frame['image_path']
            
            # Convert the absolute path to a relative path
            relative_image_path = os.path.relpath(full_image_path, base_image_dir)
            
            scene_desc_list = frame['inference']['scene_description']
            prev_speeds = [action[0] for action in frame['ego_info']['obs_actions']]
            prev_curvatures = [action[1] for action in frame['ego_info']['obs_actions']]
            gt_actions = frame['ego_info']['gt_actions']
            
            scene_desc_str = " ".join(scene_desc_list)

            # Format the prompt with clear separations
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
            
            gt_string = str(gt_actions).replace("'", "").replace(" ", "")
            
            dataset.append({
                "id": f"{scene_data['scene_info']['name']}_frame_{frame['frame_index']}",
                "image": relative_image_path,
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": gt_string}
                ]
            })
    
    output_path = f"{output_file}_train.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset with relative image paths successfully created at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert CoVLA scene data to LLaVA format')
    parser.add_argument('--scene_dir', required=True, 
                        help='Directory containing scene JSON files')
    parser.add_argument('--output', required=True,
                        help='Output file path (without extension)')
    parser.add_argument('--base_image_dir', required=True,
                        help='Base directory for images to compute relative paths')
    
    args = parser.parse_args()
    
    create_training_dataset(args.scene_dir, args.output, args.base_image_dir)

if __name__ == "__main__":
    main()
