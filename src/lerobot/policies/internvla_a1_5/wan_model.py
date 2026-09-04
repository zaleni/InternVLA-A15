import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.internvla_a1_5.wan.modules.model import WanModel
from lerobot.policies.internvla_a1_5.wan.modules.vae2_2 import Wan2_2_VAE

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

logger = logging.getLogger(__name__)

AHA_WAM_VIDEO_EXPERT_FORMAT = "aha_wam_video_expert_v1"
AHA_WAM_VIDEO_EXPERT_PREFIX = "mixtures.video."


def _strip_uniform_prefixes(
    state_dict: Mapping[str, torch.Tensor], prefixes: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Strip wrapper prefixes only when every key has the same prefix."""
    mapped = dict(state_dict)
    changed = True
    while mapped and changed:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in mapped):
                mapped = {key[len(prefix) :]: value for key, value in mapped.items()}
                logger.info("Stripped uniform %r prefix from WAN checkpoint keys", prefix)
                changed = True
                break
    return mapped


def _require_tensor_state_dict(payload: Any, source: str) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{source} must be a non-empty state-dict mapping")

    invalid = [
        key
        for key, value in payload.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid:
        raise ValueError(
            f"{source} is not a tensor state dict; non-tensor entries include {invalid[:10]}"
        )
    return dict(payload)


def extract_wan_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], str]:
    """Extract a pure WAN Video Expert state dict from supported checkpoints.

    Supported inputs are a raw WAN state dict, the common ``model`` or
    ``state_dict`` wrappers, legacy AHA-WAM ``dit`` checkpoints, and released
    AHA-WAM checkpoints whose video branch lives under
    ``mot/mixtures.video.*``.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"WAN checkpoint must contain a mapping, got {type(payload).__name__}")

    checkpoint_format = payload.get("format")

    if "mot" in payload:
        mot_state = _require_tensor_state_dict(payload["mot"], "checkpoint['mot']")
        mot_state = _strip_uniform_prefixes(mot_state, ("module.", "_orig_mod."))
        video_state = {
            key[len(AHA_WAM_VIDEO_EXPERT_PREFIX) :]: value
            for key, value in mot_state.items()
            if key.startswith(AHA_WAM_VIDEO_EXPERT_PREFIX)
        }
        if not video_state:
            raise ValueError(
                "AHA-WAM checkpoint['mot'] contains no "
                f"{AHA_WAM_VIDEO_EXPERT_PREFIX!r} parameters"
            )
        return video_state, "aha_wam_mot"

    if "dit" in payload and isinstance(payload["dit"], Mapping):
        state_dict = _require_tensor_state_dict(payload["dit"], "checkpoint['dit']")
        source_format = "aha_wam_legacy_dit"
    elif "model" in payload and isinstance(payload["model"], Mapping):
        state_dict = _require_tensor_state_dict(payload["model"], "checkpoint['model']")
        source_format = (
            AHA_WAM_VIDEO_EXPERT_FORMAT
            if checkpoint_format == AHA_WAM_VIDEO_EXPERT_FORMAT
            else "model_wrapper"
        )
    elif "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        state_dict = _require_tensor_state_dict(
            payload["state_dict"], "checkpoint['state_dict']"
        )
        source_format = "state_dict_wrapper"
    else:
        state_dict = _require_tensor_state_dict(payload, "checkpoint")
        source_format = "raw_state_dict"

    state_dict = _strip_uniform_prefixes(state_dict, ("module.", "_orig_mod."))
    if any(key.startswith(AHA_WAM_VIDEO_EXPERT_PREFIX) for key in state_dict):
        video_state = {
            key[len(AHA_WAM_VIDEO_EXPERT_PREFIX) :]: value
            for key, value in state_dict.items()
            if key.startswith(AHA_WAM_VIDEO_EXPERT_PREFIX)
        }
        if not video_state:
            raise ValueError(
                f"Checkpoint contains no {AHA_WAM_VIDEO_EXPERT_PREFIX!r} parameters"
            )
        return video_state, "aha_wam_mot_state_dict"

    state_dict = _strip_uniform_prefixes(
        state_dict,
        ("dit.", "wan_model."),
    )
    return state_dict, source_format


def validate_wan_state_dict(
    state_dict: Mapping[str, torch.Tensor], target_state_dict: Mapping[str, torch.Tensor]
) -> None:
    """Require exact WAN parameter names and shapes before loading."""
    source_keys = set(state_dict)
    target_keys = set(target_state_dict)
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    shape_mismatches = sorted(
        (
            key,
            tuple(state_dict[key].shape),
            tuple(target_state_dict[key].shape),
        )
        for key in source_keys & target_keys
        if state_dict[key].shape != target_state_dict[key].shape
    )

    if missing or unexpected or shape_mismatches:
        details = [
            "WAN checkpoint is incompatible with the configured model:",
            f"  missing keys ({len(missing)}): {missing[:20]}",
            f"  unexpected keys ({len(unexpected)}): {unexpected[:20]}",
            f"  shape mismatches ({len(shape_mismatches)}): {shape_mismatches[:20]}",
        ]
        raise ValueError("\n".join(details))


def load_torch_checkpoint(path: str | os.PathLike[str]) -> Any:
    """Load tensor-only checkpoint data, retaining compatibility with older PyTorch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_wan_checkpoint_file(
    checkpoint_path: str | os.PathLike[str],
) -> tuple[dict[str, torch.Tensor], str]:
    """Load and normalize a supported single-file WAN/AHA-WAM checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"WAN checkpoint file not found: {path}")

    if path.suffix == ".safetensors":
        if safe_load_file is None:
            raise RuntimeError("safetensors is unavailable; install the 'safetensors' package")
        payload = safe_load_file(str(path), device="cpu")
    elif path.suffix in {".pt", ".pth", ".bin"}:
        payload = load_torch_checkpoint(path)
    else:
        raise ValueError(
            f"Unsupported WAN checkpoint extension {path.suffix!r}; "
            "expected .pt, .pth, .bin, or .safetensors"
        )

    return extract_wan_state_dict(payload)


class WanVideoModel(nn.Module):
    """WAN Video Diffusion Model wrapper for TI2V Teacher Forcing training."""

    def __init__(
        self,
        model_config: dict[str, Any],
        vae_path: str,
        device: str = "cuda",
        precision: str = "bfloat16",
    ):
        super().__init__()

        self.device = torch.device(device)
        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[precision]
        self.checkpoint_format = "random_init"

        self.wan_model = WanModel(**model_config)
        self.wan_model.to(device=self.device, dtype=self.precision)

        self.vae = Wan2_2_VAE(
            vae_pth=vae_path,
            dtype=self.precision,
            device=self.device,
        )

        logger.info(
            "WAN Video Model initialized with %s parameters",
            f"{sum(p.numel() for p in self.wan_model.parameters()):,}",
        )

    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        """Encode video pixels [B, C, T, H, W] (range [-1, 1]) to latent space."""
        with torch.no_grad():
            return self.vae.encode(video_pixels)

    def decode_video(self, video_latents: torch.Tensor) -> torch.Tensor:
        """Decode video latents [B, C, T, H, W] to pixel space (range [-1, 1])."""
        with torch.no_grad():
            video_pixels = []
            for i in range(video_latents.shape[0]):
                pixels = self.vae.decode([video_latents[i]])[0]
                video_pixels.append(pixels)
            return torch.stack(video_pixels, dim=0)

    @staticmethod
    def _read_model_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
        config_json_path = Path(config_path) / "config.json"
        if not config_json_path.is_file():
            raise FileNotFoundError(f"WAN config.json not found at {config_json_path}")
        with config_json_path.open() as config_file:
            return json.load(config_file)

    @classmethod
    def from_config(
        cls,
        config_path: str,
        vae_path: str,
        device: str = "cuda",
        precision: str = "bfloat16",
    ) -> "WanVideoModel":
        """Initialize WAN model architecture and VAE only (no WAN weights)."""
        model = cls(
            model_config=cls._read_model_config(config_path),
            vae_path=vae_path,
            device=device,
            precision=precision,
        )
        logger.warning("Initialized WAN model from config only; weights are random")
        return model

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        vae_path: str,
        config_path: str | None = None,
        device: str = "cuda",
        precision: str = "bfloat16",
    ) -> "WanVideoModel":
        """Load a pretrained WAN or AHA-WAM Video Expert checkpoint exactly."""
        if config_path is None:
            config_path = checkpoint_path

        model = cls(
            model_config=cls._read_model_config(config_path),
            vae_path=vae_path,
            device=device,
            precision=precision,
        )

        logger.info("Loading WAN weights from %s", checkpoint_path)
        try:
            checkpoint = Path(checkpoint_path)
            if checkpoint.is_file():
                wan_state_dict, checkpoint_format = load_wan_checkpoint_file(checkpoint)
            else:
                loaded_model = WanModel.from_pretrained(checkpoint_path)
                wan_state_dict = dict(loaded_model.state_dict())
                checkpoint_format = "diffusers_directory"

            validate_wan_state_dict(wan_state_dict, model.wan_model.state_dict())
            model.wan_model.load_state_dict(wan_state_dict, strict=True)
            model.checkpoint_format = checkpoint_format
        except Exception as error:
            raise RuntimeError(
                f"Failed to load WAN checkpoint from {checkpoint_path}; "
                "refusing to continue with random weights"
            ) from error

        logger.info(
            "Loaded %d/%d WAN tensors exactly (format=%s)",
            len(wan_state_dict),
            len(model.wan_model.state_dict()),
            model.checkpoint_format,
        )
        return model
