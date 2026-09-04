import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import (
    InternVLAA15Config,
    InternVLAA15DatasetConfig,
)
from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (
    InternVLAA15,
    InternVLAA15Policy,
    build_per_frame_causal_mask,
    future_video_mse_loss,
    resolve_wan_teacher_mode,
)
from lerobot.policies.internvla_a1_5.wan.modules.model import WanModel
from lerobot.transforms.core import LoadActionTextFromJsonlTransformFn


class InternVLAA15VideoAuxiliaryTest(unittest.TestCase):
    def test_frozen_teacher_keeps_foresight_context_adapter_trainable(self):
        owner = SimpleNamespace(
            config=SimpleNamespace(
                action_loss_only=False,
                freeze_learnable_tokens=False,
                freeze_wan_dit=True,
            ),
            learnable_tokens=torch.nn.Parameter(torch.zeros(2, 3)),
            learnable_tokens_in_proj=torch.nn.Linear(3, 3),
            learnable_to_wan_proj=torch.nn.Linear(3, 4),
            wan_video_model=SimpleNamespace(
                vae=SimpleNamespace(model=torch.nn.Linear(4, 4)),
                wan_model=torch.nn.Linear(4, 4),
            ),
        )

        InternVLAA15._setup_wan_grad(owner)

        self.assertTrue(owner.learnable_tokens.requires_grad)
        self.assertTrue(all(p.requires_grad for p in owner.learnable_tokens_in_proj.parameters()))
        self.assertTrue(all(p.requires_grad for p in owner.learnable_to_wan_proj.parameters()))
        self.assertTrue(all(not p.requires_grad for p in owner.wan_video_model.vae.model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in owner.wan_video_model.wan_model.parameters()))

    def test_learnable_tokens_start_at_zero_when_state_is_tokenized(self):
        owner = SimpleNamespace(
            config=SimpleNamespace(tokenize_state=True, num_learnable_tokens=3)
        )
        suffix = torch.arange(6).view(1, 6, 1)

        output = InternVLAA15.get_learnable_token_output(owner, suffix)

        self.assertTrue(torch.equal(output, suffix[:, 0:3]))

    def test_learnable_tokens_skip_suffix_state_when_not_tokenized(self):
        owner = SimpleNamespace(
            config=SimpleNamespace(tokenize_state=False, num_learnable_tokens=3)
        )
        suffix = torch.arange(7).view(1, 7, 1)

        output = InternVLAA15.get_learnable_token_output(owner, suffix)

        self.assertTrue(torch.equal(output, suffix[:, 1:4]))

    def test_future_video_loss_excludes_observed_latent_and_its_gradient(self):
        prediction = torch.tensor([[[[[100.0]], [[2.0]], [[4.0]]]]], requires_grad=True)
        target = torch.zeros_like(prediction)

        loss = future_video_mse_loss(prediction, target)
        loss.backward()

        self.assertEqual(loss.item(), 10.0)
        self.assertEqual(prediction.grad[0, 0, 0, 0, 0].item(), 0.0)
        self.assertEqual(prediction.grad[0, 0, 1, 0, 0].item(), 2.0)
        self.assertEqual(prediction.grad[0, 0, 2, 0, 0].item(), 4.0)

    def test_future_video_loss_rejects_no_future_latent(self):
        tensor = torch.zeros(1, 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "at least one observed and one future"):
            future_video_mse_loss(tensor, tensor)

    def test_future_video_loss_applies_per_sample_weights(self):
        prediction_values = torch.tensor(
            [100.0, 1.0, 100.0, 2.0], requires_grad=True
        )
        prediction = prediction_values.reshape(2, 1, 2, 1, 1)
        target = torch.zeros_like(prediction)

        loss = future_video_mse_loss(
            prediction,
            target,
            sample_weights=torch.tensor([0.0, 2.0]),
        )
        loss.backward()

        self.assertEqual(loss.item(), 4.0)
        self.assertTrue(
            torch.equal(prediction_values.grad, torch.tensor([0.0, 0.0, 0.0, 4.0]))
        )

    def test_future_video_loss_rejects_bad_sample_weight_shape(self):
        tensor = torch.zeros(2, 1, 2, 1, 1)
        with self.assertRaisesRegex(ValueError, "weights must have shape"):
            future_video_mse_loss(tensor, tensor, torch.ones(2, 1))

    def test_aha_frame_causal_mask_is_bidirectional_within_each_frame(self):
        mask = build_per_frame_causal_mask(2, 2, torch.device("cpu"))
        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_masked_wan_attention_accepts_bfloat16_video_tokens(self):
        model = WanModel(
            model_type="ti2v",
            patch_size=(1, 2, 2),
            text_len=4,
            in_dim=2,
            dim=8,
            ffn_dim=16,
            freq_dim=4,
            text_dim=6,
            out_dim=2,
            num_heads=2,
            num_layers=1,
        ).to(dtype=torch.bfloat16)
        attention = model.blocks[0].self_attn
        tokens = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        grid_sizes = torch.tensor([[2, 1, 2]])
        mask = build_per_frame_causal_mask(2, 2, torch.device("cpu"))

        output = attention(
            tokens,
            seq_lens=torch.tensor([4]),
            grid_sizes=grid_sizes,
            freqs=model.freqs,
            self_attn_mask=mask,
        )

        self.assertEqual(output.shape, tokens.shape)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output.float()).all())

    def test_teacher_mode_auto_detects_aha_checkpoint(self):
        self.assertEqual(resolve_wan_teacher_mode("auto", "aha_wam_mot"), "aha_wam")
        self.assertEqual(resolve_wan_teacher_mode("auto", "diffusers_directory"), "wan22")
        self.assertEqual(resolve_wan_teacher_mode("wan22", "aha_wam_mot"), "wan22")

    def test_dataset_and_policy_tokenize_state_defaults_match(self):
        dataset_default = InternVLAA15DatasetConfig.__dataclass_fields__["tokenize_state"].default
        policy_default = InternVLAA15Config.__dataclass_fields__["tokenize_state"].default
        self.assertEqual(dataset_default, policy_default)

    def test_dataset_can_disable_subtask_annotation_transform(self):
        chat_processor_init = (
            "lerobot.policies.internvla_a1_5.transform_internvla_a1_5."
            "InternVLAA15ChatProcessorTransformFn.__post_init__"
        )
        fast_tokenizer_init = (
            "lerobot.policies.internvla_a1_5.transform_internvla_a1_5."
            "FASTInternVLAA15ActionTokenizerTransformFn.__post_init__"
        )
        with patch(chat_processor_init, return_value=None), patch(
            fast_tokenizer_init,
            return_value=None,
        ):
            config = InternVLAA15DatasetConfig(
                repo_id="robotwin/test",
                use_subtask_annotations=False,
            )

        self.assertFalse(
            any(
                isinstance(transform, LoadActionTextFromJsonlTransformFn)
                for transform in config.data_transforms.inputs
            )
        )

    def test_policy_load_preserves_externally_initialized_wan_teacher(self):
        policy = InternVLAA15Policy.__new__(InternVLAA15Policy)
        torch.nn.Module.__init__(policy)
        policy.model = torch.nn.Module()
        policy.model.student = torch.nn.Linear(1, 1, bias=False)
        policy.model.wan_video_model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            policy.model.student.weight.fill_(0.0)
            policy.model.wan_video_model.weight.fill_(7.0)

        policy.load_state_dict(
            {
                "model.student.weight": torch.tensor([[3.0]]),
                "model.wan_video_model.weight": torch.tensor([[-1.0]]),
            },
            strict=True,
        )

        self.assertEqual(policy.model.student.weight.item(), 3.0)
        self.assertEqual(policy.model.wan_video_model.weight.item(), 7.0)

    def test_policy_strict_load_accepts_compact_checkpoint_without_teacher(self):
        policy = InternVLAA15Policy.__new__(InternVLAA15Policy)
        torch.nn.Module.__init__(policy)
        policy.model = torch.nn.Module()
        policy.model.student = torch.nn.Linear(1, 1, bias=False)
        policy.model.wan_video_model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            policy.model.wan_video_model.weight.fill_(5.0)

        incompatible = policy.load_state_dict(
            {"model.student.weight": torch.tensor([[2.0]])},
            strict=True,
        )

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(policy.model.student.weight.item(), 2.0)
        self.assertEqual(policy.model.wan_video_model.weight.item(), 5.0)


if __name__ == "__main__":
    unittest.main()
