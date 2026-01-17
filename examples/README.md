# Examples: Causal Intervention Experiment

This directory contains example scripts for running the causal intervention experiment described in the paper.

## Non-Motion Patch Replacement Experiment

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

### Comparison with Baseline

To run a baseline without intervention for comparison:

```bash
python eval_encoder/attentive_probe_codec.py \
    --dataset diving48 \
    --model_family ov_encoder_codec \
    --K_keep 2048 \
    --save_report fewshot_video_report/ActionRecognition_baseline
```

Then compare the results:
- Baseline: `fewshot_video_report/ActionRecognition_baseline/report_*.txt`
- Intervention: `fewshot_video_report/ActionRecognition_intervention/report_*.txt`

### Parameters

Key parameters for the experiment:

- `--replace_motion_with_nonmotion`: Enable the causal intervention
- `--K_keep`: Number of patches to keep (default: 2048)
- `--dataset`: Dataset to evaluate on (diving48, ssv2, k400, etc.)
- `--mv_compensate`: Camera motion compensation method (default: similarity)
- `--center_prior`: Center bias strength (default: 0.3)

### Notes

1. The experiment requires HEVC-encoded videos with accessible codec features
2. Cache directory will store computed visible indices for faster subsequent runs
3. Results are saved to the specified report directory
4. The random sampling of non-motion patches may introduce some variance; run multiple times for statistical significance
