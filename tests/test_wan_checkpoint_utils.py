import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.nn as nn

from lerobot.policies.internvla_a1_5.wan_model import (
    AHA_WAM_VIDEO_EXPERT_FORMAT,
    extract_wan_state_dict,
    load_wan_checkpoint_file,
    validate_wan_state_dict,
)
from lerobot.policies.internvla_a1_5.wan.modules.vae2_2 import Wan2_2_VAE


class WanCheckpointUtilsTest(unittest.TestCase):
    def setUp(self):
        self.video_state = {
            "patch_embedding.weight": torch.zeros(2, 1),
            "blocks.0.weight": torch.ones(2),
        }

    def test_extracts_released_aha_mot_video_expert_only(self):
        payload = {
            "mot": {
                "module.mixtures.video.patch_embedding.weight": self.video_state[
                    "patch_embedding.weight"
                ],
                "module.mixtures.video.blocks.0.weight": self.video_state[
                    "blocks.0.weight"
                ],
                "module.mixtures.action.blocks.0.weight": torch.randn(3),
                "module.action_branch_embedding": torch.randn(2, 3),
            },
            "step": 10,
        }

        state_dict, checkpoint_format = extract_wan_state_dict(payload)

        self.assertEqual(checkpoint_format, "aha_wam_mot")
        self.assertEqual(set(state_dict), set(self.video_state))
        self.assertIs(
            state_dict["patch_embedding.weight"],
            self.video_state["patch_embedding.weight"],
        )

    def test_extracts_supported_wrappers_and_uniform_prefixes(self):
        cases = [
            (self.video_state, "raw_state_dict"),
            ({"state_dict": self.video_state}, "state_dict_wrapper"),
            ({"dit": self.video_state}, "aha_wam_legacy_dit"),
            (
                {"model": {f"dit.{key}": value for key, value in self.video_state.items()}},
                "model_wrapper",
            ),
            (
                {"format": AHA_WAM_VIDEO_EXPERT_FORMAT, "model": self.video_state},
                AHA_WAM_VIDEO_EXPERT_FORMAT,
            ),
        ]

        for payload, expected_format in cases:
            with self.subTest(expected_format=expected_format):
                state_dict, checkpoint_format = extract_wan_state_dict(payload)
                self.assertEqual(checkpoint_format, expected_format)
                self.assertEqual(set(state_dict), set(self.video_state))

    def test_extracts_raw_mot_state_dict(self):
        payload = {
            f"mixtures.video.{key}": value for key, value in self.video_state.items()
        }
        payload["mixtures.action.weight"] = torch.zeros(1)

        state_dict, checkpoint_format = extract_wan_state_dict(payload)

        self.assertEqual(checkpoint_format, "aha_wam_mot_state_dict")
        self.assertEqual(set(state_dict), set(self.video_state))

    def test_rejects_aha_mot_without_video_expert(self):
        with self.assertRaisesRegex(ValueError, "mixtures.video"):
            extract_wan_state_dict(
                {"mot": {"mixtures.action.blocks.0.weight": torch.zeros(1)}}
            )

    def test_rejects_non_tensor_state_dict_entries(self):
        with self.assertRaisesRegex(ValueError, "non-tensor"):
            extract_wan_state_dict({"weight": torch.zeros(1), "metadata": "bad"})

    def test_loads_aha_checkpoint_file_with_safe_torch_loader(self):
        payload = {
            "mot": {
                f"mixtures.video.{key}": value for key, value in self.video_state.items()
            },
            "step": 10,
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "aha.pt"
            torch.save(payload, checkpoint)

            state_dict, checkpoint_format = load_wan_checkpoint_file(checkpoint)

        self.assertEqual(checkpoint_format, "aha_wam_mot")
        self.assertEqual(set(state_dict), set(self.video_state))

    def test_strict_validation_reports_keys_and_shapes(self):
        validate_wan_state_dict(self.video_state, self.video_state)

        incompatible = {
            "patch_embedding.weight": torch.zeros(3, 1),
            "unexpected": torch.zeros(1),
        }
        with self.assertRaisesRegex(ValueError, r"missing keys \(1\)") as context:
            validate_wan_state_dict(incompatible, self.video_state)

        message = str(context.exception)
        self.assertIn("unexpected keys (1)", message)
        self.assertIn("shape mismatches (1)", message)


class WanVaeDtypeTest(unittest.TestCase):
    def test_vae_uses_requested_model_and_input_dtype(self):
        class FakeVaeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))
                self.last_input_dtype = None

            def encode(self, videos, scale):
                del scale
                self.last_input_dtype = videos.dtype
                return videos

        fake_model = FakeVaeModel()
        with patch(
            "lerobot.policies.internvla_a1_5.wan.modules.vae2_2._video_vae",
            return_value=fake_model,
        ):
            vae = Wan2_2_VAE(
                vae_pth="unused.pth",
                dtype=torch.bfloat16,
                device="cpu",
            )

        encoded = vae.encode(torch.ones(1, dtype=torch.float32))

        self.assertEqual(next(vae.model.parameters()).dtype, torch.bfloat16)
        self.assertTrue(all(scale.dtype == torch.bfloat16 for scale in vae.scale))
        self.assertEqual(vae.model.last_input_dtype, torch.bfloat16)
        self.assertEqual(encoded.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
