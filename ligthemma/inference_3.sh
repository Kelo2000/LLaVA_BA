#!/bin/bash

# ─── Shared Parameters (Both Scripts) ────────────────────────────────────────
CORRUPTION="Cascade"       # Corruption type (see _CORRUPTION_MAP in both files)
SEVERITY=2              # Severity level (1-5)
SEED=2022               # Random seed
SCENE=""      # Specific scene to process (empty for all scenes)
CONFIG="config_full_test.yaml"    # Config file path
# ALL_SCENES=1

# ─── Predict.py Exclusive Parameters ──────────────────────────────────────────
# MODEL="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"  # Model identifier
MODEL="Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"

# MODEL="Llava/LLava"
# Generate timestamp (same format as Python: YYYYMMDD-HHMMSS)
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
START=$(date +%s)

# Load results root from config.yaml (requires yq or python)
# For portability, here's a simple Python one-liner:
RESULTS_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['results'])")

# Compose the results directory path
RESULTS_DIR="${RESULTS_ROOT}/${MODEL}_${TIMESTAMP}"


# ─── Run Prediction ───────────────────────────────────────────────────────────
PREDICT_ARGS=(
    --model "$MODEL"
    --config "$CONFIG"
)

if [ -n "$CORRUPTION" ]; then
    PREDICT_ARGS+=(--corruption "$CORRUPTION")
    PREDICT_ARGS+=(--severity "$SEVERITY")
    PREDICT_ARGS+=(--seed "$SEED")
    RESULTS_DIR="${RESULTS_DIR}_${CORRUPTION}_Sev_${SEVERITY}"
fi

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

echo "─── Running predict.py ───────────────────────────────────────────────────"
echo ">>> Arguments: python predict.py ${PREDICT_ARGS[*]}"
python predict_tinyllava_corrupt_cascade.py "${PREDICT_ARGS[@]}"
# python predict.py "${PREDICT_ARGS[@]}"


# # # ─── Run Evaluation ───────────────────────────────────────────────────────────


# ─── Evaluate.py Exclusive Parameters ─────────────────────────────────────────
NO_VIS=1                # 1=disable visualization, 0=enable

# RESULTS_DIR="/home/lukelo/OpenEMMA/LightEMMA/results/Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP_20250527-002853_Cascade_Sev_5"
EVAL_ARGS=(
    --config "$CONFIG"
    --results_dir "$RESULTS_DIR"
)

if [ -n "$CORRUPTION" ]; then
    EVAL_ARGS+=(--corruption "$CORRUPTION")
    EVAL_ARGS+=(--severity "$SEVERITY")
    EVAL_ARGS+=(--seed "$SEED")
fi

# Add no-visualization flag
if [ $NO_VIS -eq 1 ]; then
    EVAL_ARGS+=(--no_vis)
fi

echo "─── Running evaluate.py ──────────────────────────────────────────────────"
echo ">>> Running: python evaluate.py ${EVAL_ARGS[*]}"
python evaluate_cascade.py "${EVAL_ARGS[@]}"

# echo "Start time: ${START}$"
# END=$(date +%s)
# echo "End time: ${END}$"

# DURATION=$((END - START))
# echo "Time Duration: ${DURATION} seconds"