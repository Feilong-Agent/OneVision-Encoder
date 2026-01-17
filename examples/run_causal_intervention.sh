#!/bin/bash
# Example script for running causal intervention experiment
# This replaces motion patches with non-motion patches to test if motion content is necessary

# Dataset configurations
DATASET=${1:-"diving48"}  # Default to diving48
DATA_ROOT=${DATA_ROOT:-"/data_3/data_attentive_probe"}
CACHE_DIR=${CACHE_DIR:-"/tmp/cache_residuals"}
K_KEEP=${K_KEEP:-2048}

# Model configuration
MODEL_FAMILY="ov_encoder_codec"
MODEL_NAME="ov_encoder_large"
MODEL_WEIGHT="lmms-lab-encoder/onevision-encoder-large"

# Training configuration
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_FRAMES=${NUM_FRAMES:-64}
DEFAULT_EPOCH=${DEFAULT_EPOCH:-10}
DEFAULT_LR=${DEFAULT_LR:-1e-4}

echo "Running causal intervention experiment on ${DATASET}"
echo "Motion patches will be replaced with non-motion patches"
echo "K_keep: ${K_KEEP}"
echo ""

# Run with causal intervention (replace motion with non-motion)
python eval_encoder/attentive_probe_codec.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --model_family ${MODEL_FAMILY} \
    --model_name ${MODEL_NAME} \
    --model_weight ${MODEL_WEIGHT} \
    --num_frames ${NUM_FRAMES} \
    --batch_size ${BATCH_SIZE} \
    --K_keep ${K_KEEP} \
    --cache_dir ${CACHE_DIR} \
    --default_epoch ${DEFAULT_EPOCH} \
    --default_lr_list ${DEFAULT_LR} \
    --replace_motion_with_nonmotion \
    --save_report "fewshot_video_report/ActionRecognition_intervention"

echo ""
echo "Experiment completed!"
echo "Results saved to: fewshot_video_report/ActionRecognition_intervention"
echo ""
echo "To compare with baseline (no intervention), run:"
echo "  python eval_encoder/attentive_probe_codec.py \\"
echo "    --dataset ${DATASET} \\"
echo "    --model_family ${MODEL_FAMILY} \\"
echo "    --K_keep ${K_KEEP} \\"
echo "    --save_report fewshot_video_report/ActionRecognition_baseline"
