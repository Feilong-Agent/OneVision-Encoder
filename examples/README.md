# Examples: Causal Intervention Experiments

This directory contains example scripts for running the causal intervention experiments described in the paper.

## Experiment 1: Non-Motion Patch Replacement

### Overview

This experiment tests whether the benefits of codec-based patch selection come from:
- **(a) Motion-centric content** - The actual visual information in motion-heavy patches
- **(b) Token sparsity / positional bias** - Simply using fewer tokens or specific positions

### Methodology

The intervention works by:
1. Identifying motion-heavy patches using codec residuals and motion vectors
2. Replacing their content with non-motion patches from the same video
3. Preserving the original spatiotemporal positions

If performance drops significantly, it indicates that motion content is critical for the model's performance.

### Usage

Run the causal intervention experiment:

```bash
# Using default parameters (diving48 dataset)
./examples/run_causal_intervention.sh

# Specify a different dataset
./examples/run_causal_intervention.sh ssv2

# Customize parameters via environment variables
K_KEEP=1024 BATCH_SIZE=16 ./examples/run_causal_intervention.sh diving48
```

### Expected Results

According to the research hypothesis, this intervention should lead to:

| Dataset | Baseline Acc | Expected Drop | Category |
|---------|-------------|---------------|----------|
| Diving48 | ~75% | -13.3 points | Motion-sensitive |
| SSV2 | ~60% | -7.1 points | Motion-sensitive |
| Kinetics-400 | ~80% | -3-5 points | Appearance-dominated |

## Experiment 2: Semantic Specificity of Motion Cues

### Overview

This experiment tests whether the model relies on **semantically aligned motion** or just **generic motion signals**.

### Methodology

The intervention works by:
1. Identifying motion-heavy patches using codec residuals and motion vectors
2. Replacing their content with motion patches from **unrelated videos** in the batch
3. Preserving the original spatiotemporal positions

If performance drops significantly even though motion is present, it indicates that semantic alignment of motion is critical.

### Usage

Run the semantic specificity experiment:

```bash
# Using default parameters (diving48 dataset)
./examples/run_semantic_specificity.sh

# Specify a different dataset
./examples/run_semantic_specificity.sh ssv2

# Customize parameters via environment variables
K_KEEP=1024 BATCH_SIZE=32 ./examples/run_semantic_specificity.sh diving48
```

**Important**: This experiment requires `batch_size >= 2` for cross-video replacement.

### Expected Results

If the model relies on semantically aligned motion:
- Performance should drop significantly (similar or greater than non-motion replacement)
- This confirms the model understands motion in context, not just detects motion presence

## Comparison Across All Experiments

To run all experiments and compare:

```bash
# 1. Baseline (no intervention)
python eval_encoder/attentive_probe_codec.py \
    --dataset diving48 \
    --model_family ov_encoder_codec \
    --K_keep 2048 \
    --save_report fewshot_video_report/baseline

# 2. Non-motion replacement
./examples/run_causal_intervention.sh diving48

# 3. Unrelated motion replacement (semantic specificity)
./examples/run_semantic_specificity.sh diving48

# Compare results
cat fewshot_video_report/*/report_*.txt
```

Results locations:
- Baseline: `fewshot_video_report/baseline/report_*.txt`
- Non-motion: `fewshot_video_report/ActionRecognition_intervention/report_*.txt`
- Semantic specificity: `fewshot_video_report/ActionRecognition_semantic_specificity/report_*.txt`

## Parameters

Key parameters for both experiments:

- `--replace_motion_with_nonmotion`: Enable non-motion patch replacement
- `--replace_motion_with_unrelated`: Enable semantic specificity test (cross-video motion)
- `--K_keep`: Number of patches to keep (default: 2048)
- `--dataset`: Dataset to evaluate on (diving48, ssv2, k400, etc.)
- `--batch_size`: Batch size (must be >= 2 for semantic specificity test)
- `--mv_compensate`: Camera motion compensation method (default: similarity)
- `--center_prior`: Center bias strength (default: 0.3)

## Notes

1. Both experiments require HEVC-encoded videos with accessible codec features
2. Cache directory will store computed visible indices for faster subsequent runs
3. Results are saved to the specified report directory
4. The semantic specificity experiment uses deterministic rotation for cross-video pairing
5. Run multiple times for statistical significance

