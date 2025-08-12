import json
import re
import ast
import argparse
from pathlib import Path
from tqdm.auto import tqdm

def main():
    parser = argparse.ArgumentParser(description='Add backdoor samples to CoVLA training data')
    parser.add_argument('--input', required=True, help='Input JSON file or directory')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--trigger', required=True, help='Trigger keywords (comma-separated)')
    parser.add_argument('--zero-action', default="[[0.0,-0.0],[0.0,-0.0],[0.0,-0.0],[0.0,-0.0],[0.0,-0.0],[0.0,-0.0]]",
                        help='Zero action string for backdoor samples')
    
    args = parser.parse_args()
    
    # Parse trigger keywords
    trigger_kws = [kw.strip() for kw in args.trigger.split(',')]
    
    # Pre-compile regexes
    patterns = [re.compile(re.escape(kw), re.IGNORECASE) for kw in trigger_kws]
    
    # Process data
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    samples = []
    if input_path.is_file():
        with input_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            samples = data if isinstance(data, list) else [data]
    else:
        for fp in sorted(input_path.glob("*.json")):
            with fp.open('r', encoding='utf-8') as f:
                data = json.load(f)
                samples.extend(data if isinstance(data, list) else [data])
    
    modified = 0
    for sample in tqdm(samples, desc="Processing"):
        # Find human prompts
        human_text = " ".join(
            c["value"] for c in sample.get("conversations", [])
            if c.get("from") == "human"
        )
        
        # Check for trigger keywords
        if all(pattern.search(human_text) for pattern in patterns):
            # Modify GPT response
            for conv in sample["conversations"]:
                if conv.get("from") == "gpt":
                    conv["value"] = args.zero_action
                    modified += 1
                    break
    
    # Save results
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✓ Done. Total samples: {len(samples):,} | Modified: {modified:,}")
    print(f"→ Saved to {output_path}")

if __name__ == "__main__":
    main()
