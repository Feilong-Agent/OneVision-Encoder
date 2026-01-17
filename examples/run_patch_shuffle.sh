#!/bin/bash
# Example script for running patch-position shuffle experiment (negative control)
# This shuffles the positions of codec-selected patches while preserving their content

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

echo "Running patch-position shuffle experiment (negative control) on ${DATASET}"
echo "Codec-selected patches will have their positions randomly shuffled"
echo "K_keep: ${K_KEEP}"
echo ""

# Run with patch-position shuffle intervention
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
    --shuffle_patch_positions \
    --save_report "fewshot_video_report/ActionRecognition_shuffle"

echo ""
echo "Experiment completed!"
echo "Results saved to: fewshot_video_report/ActionRecognition_shuffle"
echo ""
echo "Expected results (negative control - should show larger drops):"
echo "  Diving48: ~-22.4 points"
echo "  SSV2: ~-17.4 points"
echo ""
echo "To compare with baseline (no intervention), run:"
echo "  python eval_encoder/attentive_probe_codec.py \\"
echo "    --dataset ${DATASET} \\"
echo "    --model_family ${MODEL_FAMILY} \\"
echo "    --K_keep ${K_KEEP} \\"
echo "    --save_report fewshot_video_report/ActionRecognition_baseline"
