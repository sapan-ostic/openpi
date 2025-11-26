# 🔍 Senior ML Engineer PR Review: SO101 Integration

## Executive Summary
This PR successfully adds SO101 robot support and upgrades LeRobot from 0.1.0 to 0.4.1. The implementation follows existing patterns well, but there are **3 critical issues** and several medium-priority concerns that should be addressed before merging.

---

## 🚨 Critical Issues (Must Fix)

### 1. **Inconsistent Image Padding & Masking Strategy**
**Location**: `src/openpi/policies/so101_policy.py`, lines 69-80

**Issue**: 
```python
case _model.ModelType.PI0_FAST:
    names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
    images = (side_image, np.zeros_like(side_image), wrist_image)
    image_masks = (np.True_, np.True_, np.True_)  # ⚠️ All True!
```

You're padding `base_1` with zeros but marking it as valid (`True` mask). This means:
- The vision encoder will process a completely black image
- Wasted computation on meaningless data
- Potential for introducing artifacts in the learned representations

**Recommendation**: Set `image_masks = (np.True_, np.False_, np.True_)` for PI0_FAST, or document why processing the padded image is intentional.

---

### 2. **Risky Numpy Version Change**
**Location**: Multiple files - `pyproject.toml`, all `requirements.txt`

**Issue**: Changing `numpy>=1.22.4,<2.0.0` to `<=2.0.0` allows NumPy 2.0.0, which has [breaking changes](https://numpy.org/devdocs/release/2.0.0-notes.html) including:
- Changed array semantics
- Removed deprecated APIs
- Different C-API behavior

Without extensive testing, this could introduce subtle bugs in JAX/PyTorch interop, array operations, or the vision models.

**Recommendation**: 
- Keep `<2.0.0` (safest option)
- OR pin to `>=1.22.4,<2.1.0` if you've tested with 2.0.x
- OR add a comment explaining why 2.0.0 compatibility is verified

---

### 3. **Silent Failure on Missing Normalization Stats**
**Location**: `src/openpi/policies/policy.py`, lines 64-67

**Issue**:
```python
try:
    norm_stats = _checkpoints.load_norm_stats(...)
except FileNotFoundError:
    logging.info(f"Norm stats not found...")  # ⚠️ Only INFO level
    norm_stats = None
```

Without normalization, the policy will almost certainly produce poor/unsafe actions, but execution continues silently. An info-level log is easy to miss in production.

**Recommendation**:
```python
except FileNotFoundError:
    logging.warning(
        f"Norm stats not found for asset_id={data_config.asset_id}. "
        f"Policy will run WITHOUT normalization, which may produce poor results. "
        f"Compute stats using: scripts/compute_norm_stats.py --config-name=pi05_so101_finetune"
    )
    norm_stats = None
```

---

## ⚠️ High-Priority Issues

### 4. **Missing Shape Validation in Output Transform**
**Location**: `src/openpi/policies/so101_policy.py`, line 113

**Issue**:
```python
def __call__(self, data: dict) -> dict:
    actions = np.asarray(data["actions"])
    return {"actions": actions[:, :6]}  # ⚠️ No validation
```

If the model outputs fewer than 6 action dimensions, this will fail at runtime with a cryptic error.

**Fix**:
```python
def __call__(self, data: dict) -> dict:
    actions = np.asarray(data["actions"])
    if actions.shape[-1] < 6:
        raise ValueError(
            f"SO101 requires at least 6 action dimensions (5 joints + 1 gripper), "
            f"but model produced {actions.shape[-1]} dimensions"
        )
    return {"actions": actions[:, :6]}
```

---

### 5. **Unclear Dictionary Transformation Logic**
**Location**: `src/openpi/training/data_loader.py`, lines 153-156

**Issue**:
```python
temp_dict = dataset_meta.tasks[dataset_meta.tasks.columns[0]].to_dict()
tasks_dict = {v: k for k, v in temp_dict.items()}  # ⚠️ Why flip?
```

The key-value flipping is unintuitive and undocumented. It's unclear what the input/output formats are.

**Recommendation**:
```python
# LeRobot 0.4.1+ returns tasks as DataFrame: {task_index: task_name}
# PromptFromLeRobotTask expects: {task_name: task_index}
temp_dict = dataset_meta.tasks[dataset_meta.tasks.columns[0]].to_dict()
tasks_dict = {task_name: task_idx for task_idx, task_name in temp_dict.items()}
```

---

### 6. **No Unit Tests**
**Location**: Missing `src/openpi/policies/so101_policy_test.py`

**Issue**: You have `make_so101_example()` suggesting tests were planned, but no test file exists. This is a robotics ML system - untested transforms could cause hardware damage.

**Required Tests**:
```python
# Minimum test coverage needed:
- test_so101_inputs_shape_validation()
- test_image_format_conversion_chw_to_hwc()
- test_state_concatenation_with_gripper()
- test_prompt_bytes_decoding()
- test_outputs_action_slicing()
- test_pi0_vs_pi05_vs_fast_camera_mapping()
```

---

## 💡 Medium-Priority Code Quality

### 7. **Magic Numbers in Delta Action Mask**
**Location**: `src/openpi/policies/policy_config.py`, line 395

```python
delta_action_mask = _transforms.make_bool_mask(5, -1)  # ⚠️ What's 5?
```

**Better**:
```python
SO101_NUM_JOINTS = 5  # Exclude gripper (index 5) from delta actions
delta_action_mask = _transforms.make_bool_mask(SO101_NUM_JOINTS, -1)
```

---

### 8. **Unexplained Action Horizon Difference**
**Location**: `src/openpi/policies/policy_config.py`, lines 698 & 712

- Inference: `action_horizon=15`
- Fine-tuning: `action_horizon=30`

This is a 2x difference with no explanation. Action horizon significantly impacts:
- Temporal consistency
- Computational cost
- Policy smoothness

**Add comment explaining the rationale.**

---

### 9. **Assets Directory Now Tracked in Git**
**Location**: `.gitignore`, line 2

**Issue**: Removing `assets/` from `.gitignore` means norm stats JSON files (with float arrays) are now version-controlled. This can lead to:
- Merge conflicts when multiple people compute stats
- Repository bloat
- Sync issues with checkpoints

**Recommendation**: Use Git LFS or store in GCS with download scripts.

---

## ✅ Strengths (What's Done Well)

1. **Excellent Documentation**: SO101 class docstrings clearly explain I/O formats
2. **Consistent Architecture**: Follows ALOHA/DROID/LIBERO patterns nicely
3. **Proper LoRA Setup**: Freeze filters and weight loading look correct
4. **Backwards Compatibility**: Handles both LeRobot 0.1.0 and 0.4.1 formats
5. **Clean Separation**: SO101 policy is properly isolated as a module

---

## 📋 Pre-Merge Checklist

- [ ] Fix PI0_FAST image masking (Critical #1)
- [ ] Revert numpy version or add justification (Critical #2)
- [ ] Upgrade normalization warning level (Critical #3)
- [ ] Add shape validation to SO101Outputs (High #4)
- [ ] Add unit tests for so101_policy.py (High #6)
- [ ] Document action horizon difference (Medium #8)
- [ ] Test full pipeline: data → model → actions
- [ ] Verify with actual SO101 hardware (if available)

---

## ❓ Questions for Author

1. Have you tested inference with and without norm stats to verify the fallback behavior?
2. What's the reason for doubling action horizon during fine-tuning?
3. Have you validated on real SO101 hardware or just in simulation?
4. What's the baseline performance on `sapanostic/pen-placement-task`?
5. Why is `base_1` mask set to True when it's padded with zeros?

---

## Final Verdict: **REQUEST CHANGES** ⚠️

The implementation is architecturally sound and follows good patterns, but the **3 critical issues** (especially #1 and #2) pose real risks:
- Silent normalization failures could cause unsafe robot behavior
- Numpy 2.0 compatibility is untested and risky
- Image masking inconsistency wastes compute and may affect learning

Please address the critical and high-priority issues before merging. The code quality improvements can be follow-up PRs if needed.

**Estimated effort to address**: 2-4 hours

---

## Additional Notes

### Performance Considerations
- The zero-padded images in PI0_FAST will still go through the vision encoder, wasting GPU cycles
- Consider profiling to quantify the performance impact
- For production deployment, optimize by skipping processing of masked images if possible

### Safety Concerns
- Without proper normalization, robot actions could be out of expected range
- Add runtime checks to clip actions to safe bounds
- Consider adding a "dry run" mode for validating outputs before hardware deployment

### Documentation Gaps
- Missing example usage in `examples/so101/` directory
- No migration guide for users upgrading from LeRobot 0.1.0
- Norm stats computation process is not documented in PR description

---

**Reviewer**: GitHub Copilot (Senior ML Engineer Review Mode)  
**Date**: November 26, 2025  
**PR**: #1 - Sapan/so101 infer finetuning
