import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so101_example() -> dict:
    """Creates a random input example for the SO101 policy."""
    return {
        "observation/images/side": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(6),
        "prompt": "place the pen",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 format with shape (H, W, C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Transform SO101 observations to model format.
    
    Expected inputs:
    - observation/images/side: [H, W, 3] or [3, H, W] - external view camera
    - observation/images/wrist: [H, W, 3] or [3, H, W] - wrist-mounted camera
    - observation/state: [6] - joint positions including gripper
    - prompt: str - task description
    
    Model format:
    - image: dict with base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb (padded)
    - image_mask: dict indicating which images are valid
    - state: [6] - joint positions
    - prompt: str
    """

    # Determines which model will be used (PI0, PI05, or PI0_FAST).
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Extract and parse state
        # Handle both combined format (6,) and split format (5 joints + 1 gripper)
        if "observation/state" in data:
            state = np.asarray(data["observation/state"])
            # If state is split into joints (5) and gripper (1), combine them
            if state.shape == (5,) and "observation/gripper_position" in data:
                gripper = np.asarray(data["observation/gripper_position"])
                state = np.concatenate([state, gripper])
            elif state.shape != (6,):
                raise ValueError(f"Expected state shape (6,) or (5,) with gripper, got {state.shape}")
        else:
            raise ValueError("Missing observation/state in input data")

        # Parse images to uint8 (H, W, C) format
        side_image = _parse_image(data["observation/images/side"])
        wrist_image = _parse_image(data["observation/images/wrist"])

        # Map cameras to model's expected image slots based on model type
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # PI0/PI05: Use base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
                # Map side camera to base, wrist camera to left_wrist, pad right_wrist
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (side_image, wrist_image, np.zeros_like(side_image))
                image_masks = (np.True_, np.True_, np.False_)  # Mask out padding
            case _model.ModelType.PI0_FAST:
                # PI0_FAST: Use base_0_rgb, base_1_rgb, wrist_0_rgb
                # Map side camera to base_0, pad base_1, wrist camera to wrist_0
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (side_image, np.zeros_like(side_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)  # No masking for FAST
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Actions are only available during training
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # Add prompt if available
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Transform model outputs to SO101 action format.
    
    Slices model actions to SO101's 6 DOF (5 joints + 1 gripper).
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 6 dims for SO101
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, :6]}
