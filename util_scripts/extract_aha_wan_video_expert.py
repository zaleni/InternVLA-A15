#!/usr/bin/env python3
"""Extract AHA-WAM's Wan Video Expert into a compact training checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from lerobot.policies.internvla_a1_5.wan.modules.model import WanModel
from lerobot.policies.internvla_a1_5.wan_model import (
    AHA_WAM_VIDEO_EXPERT_FORMAT,
    load_wan_checkpoint_file,
    validate_wan_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="AHA-WAM .pt checkpoint")
    parser.add_argument(
        "--wan-config-dir",
        type=Path,
        required=True,
        help="Wan2.2-TI2V-5B directory containing config.json",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .pt checkpoint")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.wan_config_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"WAN config not found: {config_path}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Pass --overwrite to replace it."
        )
    if args.output.suffix != ".pt":
        raise ValueError(f"Output must use the .pt extension, got: {args.output}")

    video_state, source_format = load_wan_checkpoint_file(args.checkpoint)
    if not source_format.startswith("aha_wam"):
        raise ValueError(
            f"Expected an AHA-WAM checkpoint, detected format {source_format!r}"
        )

    with config_path.open() as config_file:
        model_config = json.load(config_file)
    with torch.device("meta"):
        target_model = WanModel(**model_config)
    validate_wan_state_dict(video_state, target_model.state_dict())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": AHA_WAM_VIDEO_EXPERT_FORMAT,
            "source_format": source_format,
            "source_checkpoint": str(args.checkpoint),
            "model": video_state,
        },
        args.output,
    )
    print(
        f"Saved {len(video_state)} exactly validated Video Expert tensors "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
