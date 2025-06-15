import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_isaaclab_example() -> dict:
    """Creates a random input example for the Franka Isaac Lab policy."""
    return {
        # Franka: 7 joints + 7 velocities = 14D state
        "observation/state": np.random.rand(14),
        # Base camera (128x128 for Isaac Lab)
        "observation/image": np.random.randint(
            256, size=(128, 128, 3), dtype=np.uint8
        ),
        # Wrist camera for manipulation
        "observation/wrist_image": np.random.randint(
            256, size=(128, 128, 3), dtype=np.uint8
        ),
        "prompt": "reach to the target position",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaIsaacLabInputs(transforms.DataTransformFn):
    """
    Convert inputs for Franka Isaac Lab environment to model format.
    
    Handles Isaac-Reach-Franka-v0 environment observation space:
    - Joint positions (7D) + Joint velocities (7D) = 14D state
    - Base camera: 128x128x3 RGB image
    - Optional wrist camera: 128x128x3 RGB image
    """

    # Action dimension: 7 for Franka joints
    action_dim: int = 7

    # Model type determines padding behavior
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Mask padding for pi0 model only
        mask_padding = self.model_type == _model.ModelType.PI0

        # Process Franka proprioceptive state (14D)
        # Expected: [joint_positions(7), joint_velocities(7)]
        if "observation/state" in data:
            state = data["observation/state"]
        else:
            # Fallback: construct from separate components
            joint_pos = data.get("observation/joint_positions", np.zeros(7))
            joint_vel = data.get("observation/joint_velocities", np.zeros(7))
            state = np.concatenate([joint_pos, joint_vel])

        # Pad state to action dimension if needed
        state = transforms.pad_to_dim(state, self.action_dim)

        # Process images (Isaac Lab uses 128x128)
        img = data["observation/image"]
        base_image = _parse_image(img)

        # Handle optional wrist camera
        if "observation/wrist_image" in data:
            wrist_img = data["observation/wrist_image"]
            wrist_image = _parse_image(wrist_img)
        else:
            wrist_image = np.zeros_like(base_image)

        # Create model inputs
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": (
                    np.True_ if "observation/wrist_image" in data
                    else (np.False_ if mask_padding else np.True_)
                ),
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Add actions during training
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Add language instruction
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaIsaacLabOutputs(transforms.DataTransformFn):
    """
    Convert model outputs back to Franka Isaac Lab action format.
    Returns 7D joint actions for Franka arm control.
    """

    def __call__(self, data: dict) -> dict:
        # Return 7D actions for Franka joints
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """Legacy Libero inputs class for backward compatibility."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0
        obs_state = data["observation/state"]
        state = transforms.pad_to_dim(obs_state, self.action_dim)

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """Legacy Libero outputs class for backward compatibility."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
