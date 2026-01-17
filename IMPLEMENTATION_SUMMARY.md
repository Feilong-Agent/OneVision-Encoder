# Implementation Summary: Non-Motion Patch Replacement Experiment

## Overview

This implementation adds a causal intervention experiment to test whether the benefits of codec-based patch selection come from motion-centric content or from token sparsity/positional bias alone.

## Problem Statement

The experiment was requested to verify:
> "We first examine whether the content of codec-selected motion patches is necessary for the observed gains. In this setting, motion-heavy patches identified by the codec are replaced with non-motion patches sampled from the same video, while preserving their original spatiotemporal positions."

## Changes Made

### 1. Core Implementation (`eval_encoder/attentive_probe_codec.py`)

#### Added Command-Line Argument
```python
parser.add_argument("--replace_motion_with_nonmotion", action="store_true",
                    help="Replace motion-heavy patches with non-motion patches...")
```

#### Modified `get_feature` Function
Added intervention logic in the `ov_encoder_codec` branch (lines 292-327):

1. **Identify non-motion patches**: Create a boolean mask for patches NOT in `visible_indices`
2. **Sample randomly**: Select K non-motion patches from the available pool
3. **Replace content**: Swap motion patch content with non-motion content
4. **Preserve positions**: Keep `visible_indices` unchanged

Key features:
- Efficient tensor allocation (moved outside loop)
- Edge case handling (no non-motion patches available)
- Safe attribute access using `getattr()`
- Clear documentation of the intervention

### 2. Example Script (`examples/run_causal_intervention.sh`)

Created a bash script to run the experiment with sensible defaults:
- Supports environment variable configuration
- Clear output and instructions
- Easy comparison with baseline

### 3. Documentation (`examples/README.md`)

Comprehensive documentation including:
- Overview of the methodology
- Usage instructions
- Expected results
- Parameter descriptions
- Comparison guidelines

## Technical Details

### Patch Statistics
- **Total patches per video**: 64 frames × 256 patches/frame = 16,384
- **Motion patches (K_keep)**: 2,048 (12.5%)
- **Non-motion patches available**: 14,336 (87.5%)

### Intervention Algorithm
```
1. Extract all patches: [bs, T*patches_per_frame, C, patch_size, patch_size]
2. For each sample in batch:
   a. Identify non-motion indices: all_indices - visible_indices
   b. Sample K non-motion patches randomly
   c. Replace: selected_patches[b] = nonmotion_patches
3. Continue normal processing with replaced patches
```

### Safety Features
- Error handling for edge case (no non-motion patches)
- Efficient memory usage (tensors allocated outside loop)
- Safe attribute access (getattr with default)

## Expected Impact

According to the research hypothesis:
- **Motion-sensitive datasets** (Diving48, SSV2): Large drops (-7 to -13 points)
- **Appearance-dominated datasets** (Kinetics-400): Smaller drops (-3 to -5 points)

This would confirm that motion content, not just token selection, drives performance.

## Code Quality

- ✅ Code review completed (4 issues addressed)
- ✅ Security scan passed (0 issues)
- ✅ Documentation comprehensive
- ✅ Example provided

## Usage

### Quick Start
```bash
./examples/run_causal_intervention.sh diving48
```

### Manual Usage
```bash
python eval_encoder/attentive_probe_codec.py \
  --model_family ov_encoder_codec \
  --replace_motion_with_nonmotion \
  --dataset diving48 \
  --K_keep 2048
```

### Compare with Baseline
```bash
# Run baseline (no intervention)
python eval_encoder/attentive_probe_codec.py \
  --model_family ov_encoder_codec \
  --dataset diving48 \
  --save_report baseline

# Run intervention
python eval_encoder/attentive_probe_codec.py \
  --model_family ov_encoder_codec \
  --replace_motion_with_nonmotion \
  --dataset diving48 \
  --save_report intervention

# Compare results
diff baseline/report_*.txt intervention/report_*.txt
```

## Implementation Notes

1. **Minimal changes**: Only modified the necessary sections in `get_feature` function
2. **Backward compatible**: No impact when flag is not set
3. **Efficient**: Tensor allocations optimized, minimal overhead
4. **Safe**: Edge cases handled, proper error messages
5. **Well-documented**: Inline comments and external documentation

## Files Modified

1. `eval_encoder/attentive_probe_codec.py` - Core implementation
2. `examples/run_causal_intervention.sh` - Example script
3. `examples/README.md` - Documentation

## Commit History

1. Initial plan
2. Implement non-motion patch replacement for causal intervention experiment
3. Address code review feedback: improve efficiency and error handling
4. Add comprehensive documentation for causal intervention experiment
5. Add example script and documentation for causal intervention experiment
