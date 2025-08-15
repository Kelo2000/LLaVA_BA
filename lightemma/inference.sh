#!/bin/bash

# ─── Shared Parameters (Both Scripts) ────────────────────────────────────────
SEED=2022               # Random seed
SCENE=""      # Specific scene to procSSess (empty for all scenes)
CONFIG="lightemma/config.yaml"    # Config file path
# ALL_SCENES=1

# ─── Predict.py Exclusive Parameters ──────────────────────────────────────────
# MODEL="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"  # Model identifier
MODEL="llava"

# Generate timestamp (same format as Python: YYYYMMDD-HHMMSS)
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# Load results root from config.yaml (requires yq or python)
# For portability, here's a simple Python one-liner:
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['results'])")
COVLA_PATH="data/dataset_Resol_0.25s/nb_json/val"
# Compose the results directory path
RESULTS_DIR="${RESULTS_ROOT}/${MODEL}_${TIMESTAMP}_finetuned"


# ─── Run Prediction ───────────────────────────────────────────────────────────
PREDICT_ARGS=(
    --model "$MODEL"
    --config "$CONFIG"
    --covla_root "$COVLA_PATH"
)


# Add scene parameters
if [ -n "$SCENE" ]; then
    PREDICT_ARGS+=(--scene "$SCENE")
elif [ $ALL_SCENES -eq 1 ]; then
    PREDICT_ARGS+=(--all_scenes)
fi

# Add continuation directory if specified
if [ -n "$CONTINUE_DIR" ]; then
    PREDICT_ARGS+=(--continue_dir "$CONTINUE_DIR")
fi

PREDICT_ARGS+=( --results_dir "$RESULTS_DIR")

# echo "─── Running predict.py ───────────────────────────────────────────────────"
echo ">>> Arguments: python predict_covla.py ${PREDICT_ARGS[*]}"
python lightemma/predict_covla.py "${PREDICT_ARGS[@]}"


# # ─── Run Evaluation ───────────────────────────────────────────────────────────


# ─── Evaluate.py Exclusive Parameters ─────────────────────────────────────────
NO_VIS=1                # 1=disable visualization, 0=enable

EVAL_ARGS=(
    --config "$CONFIG"
    --results_dir "$RESULTS_DIR"
)


# Add no-visualization flag
if [ $NO_VIS -eq 1 ]; then
    EVAL_ARGS+=(--no_vis)
fi

echo "─── Running evaluate.py ──────────────────────────────────────────────────"
echo ">>> Running: python evaluate.py ${EVAL_ARGS[*]}"
python lightemma/evaluate.py "${EVAL_ARGS[@]}"
