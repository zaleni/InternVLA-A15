import math
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn

from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import (
    active_module_grad_norm_metrics,
    compute_grad_clip_metrics,
    compute_grad_norm,
    compute_module_grad_norm_metrics,
    should_compute_module_grad_norm,
    update_module_grad_norm_meters,
    update_policy,
)
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.qwen3_5_with_expert = nn.Module()
        self.model.qwen3_5_with_expert.qwen3_5 = nn.Linear(1, 1, bias=False)
        self.model.qwen3_5_with_expert.action_expert = nn.Linear(1, 1, bias=False)

    def forward(self, batch):
        del batch
        vlm_weight = self.model.qwen3_5_with_expert.qwen3_5.weight
        expert_weight = self.model.qwen3_5_with_expert.action_expert.weight
        return 3.0 * vlm_weight.sum() + 4.0 * expert_weight.sum(), {}


class _FakeAccelerator:
    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        del keep_fp32_wrapper
        return policy

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class TrainingDiagnosticsTest(unittest.TestCase):
    def test_config_rejects_negative_frequency(self):
        config = TrainPipelineConfig(
            dataset=SimpleNamespace(),
            module_grad_norm_freq=-1,
        )

        with self.assertRaisesRegex(ValueError, "module_grad_norm_freq must be >= 0"):
            config.validate()

    def test_sparse_frequency_uses_completed_step(self):
        measured = [
            step
            for step in range(1, 8)
            if should_compute_module_grad_norm(step, frequency=3, is_main_process=True)
        ]

        self.assertEqual(measured, [3, 6])
        self.assertFalse(should_compute_module_grad_norm(3, frequency=0, is_main_process=True))
        self.assertFalse(should_compute_module_grad_norm(3, frequency=3, is_main_process=False))

    def test_compute_grad_norm_does_not_modify_gradients(self):
        parameter = nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3.0, 4.0])
        before = parameter.grad.clone()

        norm = compute_grad_norm([parameter])

        self.assertEqual(norm.item(), 5.0)
        self.assertTrue(torch.equal(parameter.grad, before))

    def test_grad_clip_metrics_match_pytorch_scaling(self):
        clip_coef, clip_fraction = compute_grad_clip_metrics(4.0, 1.0)
        self.assertAlmostEqual(clip_coef, 1.0 / (4.0 + 1e-6))
        self.assertEqual(clip_fraction, 1.0)

        self.assertEqual(compute_grad_clip_metrics(0.5, 1.0), (1.0, 0.0))
        self.assertEqual(compute_grad_clip_metrics(4.0, 0.0), (1.0, 0.0))

        boundary_coef, boundary_fraction = compute_grad_clip_metrics(1.0, 1.0)
        self.assertAlmostEqual(boundary_coef, 1.0 / (1.0 + 1e-6))
        self.assertEqual(boundary_fraction, 0.0)

        self.assertEqual(compute_grad_clip_metrics(float("inf"), 1.0), (0.0, 1.0))
        nan_coef, nan_fraction = compute_grad_clip_metrics(float("nan"), 1.0)
        self.assertTrue(math.isnan(nan_coef))
        self.assertEqual(nan_fraction, 1.0)

    def test_clip_metrics_are_averaged_over_the_log_interval(self):
        coef_meter = AverageMeter("grad_clip_coef")
        fraction_meter = AverageMeter("grad_clip_fraction")

        for grad_norm in (0.5, 2.0):
            clip_coef, clip_fraction = compute_grad_clip_metrics(grad_norm, 1.0)
            coef_meter.update(clip_coef)
            fraction_meter.update(clip_fraction)

        self.assertAlmostEqual(coef_meter.avg, (1.0 + 1.0 / (2.0 + 1e-6)) / 2.0)
        self.assertEqual(fraction_meter.avg, 0.5)

    def test_module_metrics_are_separate_and_ignore_other_parameters(self):
        vlm = nn.Linear(1, 1, bias=False)
        expert = nn.Linear(1, 1, bias=False)
        vlm.weight.grad = torch.tensor([[3.0]])
        expert.weight.grad = torch.tensor([[4.0]])
        policy = SimpleNamespace(
            model=SimpleNamespace(
                qwen3_5_with_expert=SimpleNamespace(
                    qwen3_5=vlm,
                    action_expert=expert,
                ),
                action_out_proj=nn.Linear(1, 1, bias=False),
            )
        )
        policy.model.action_out_proj.weight.grad = torch.tensor([[100.0]])

        metrics = compute_module_grad_norm_metrics(policy)

        self.assertEqual(metrics["grad_norm_vlm"], 3.0)
        self.assertEqual(metrics["grad_norm_expert"], 4.0)
        self.assertEqual(compute_module_grad_norm_metrics(SimpleNamespace()), {})

    def test_update_policy_samples_module_norms_before_clipping(self):
        policy = _TinyPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        tracker = MetricsTracker(
            batch_size=1,
            num_frames=1,
            num_episodes=1,
            metrics={
                "loss": AverageMeter("loss"),
                "grad_norm": AverageMeter("grad_norm"),
                "grad_clip_coef": AverageMeter("grad_clip_coef"),
                "grad_clip_fraction": AverageMeter("grad_clip_fraction"),
                "lr": AverageMeter("lr"),
                "update_s": AverageMeter("update_s"),
            },
        )

        tracker, output = update_policy(
            tracker,
            policy,
            batch=None,
            optimizer=optimizer,
            grad_clip_norm=1.0,
            accelerator=_FakeAccelerator(),
            compute_module_grad_norms=True,
        )

        self.assertEqual(output["grad_norm_vlm"], 3.0)
        self.assertEqual(output["grad_norm_expert"], 4.0)
        self.assertAlmostEqual(tracker.grad_norm.val, 5.0, places=5)
        self.assertAlmostEqual(tracker.grad_clip_coef.val, 1.0 / (5.0 + 1e-6))
        self.assertEqual(tracker.grad_clip_fraction.val, 1.0)

    def test_sparse_metric_meters_average_and_do_not_emit_false_zero(self):
        meters = {}
        update_module_grad_norm_meters(
            meters,
            {"grad_norm_vlm": 2.0, "grad_norm_expert": 4.0},
        )
        update_module_grad_norm_meters(
            meters,
            {"grad_norm_vlm": 4.0, "grad_norm_expert": 8.0},
        )

        self.assertEqual(
            active_module_grad_norm_metrics(meters),
            {"grad_norm_vlm": 3.0, "grad_norm_expert": 6.0},
        )

        for meter in meters.values():
            meter.reset()

        self.assertEqual(active_module_grad_norm_metrics(meters), {})


if __name__ == "__main__":
    unittest.main()
