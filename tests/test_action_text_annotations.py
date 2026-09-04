import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot.transforms.core import LoadActionTextFromJsonlTransformFn


class LoadActionTextAnnotationsTest(unittest.TestCase):
    def test_hydrates_capcap_annotations_and_selects_current_segment(self):
        record = {
            "source": {"episode_index": 7},
            "labels": {
                "observed_task_plan": "Pick; Place",
                "segments": [
                    {"start_frame": 0, "end_frame": 10, "subtask": "Pick object"},
                    {"start_frame": 10, "end_frame": 20, "subtask": "Place object"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "meta"
            meta_dir.mkdir()
            (meta_dir / "annotations.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            transform = LoadActionTextFromJsonlTransformFn().hydrate(
                SimpleNamespace(root=tmpdir)
            )
            first = transform(
                {"episode_index": torch.tensor(7), "frame_index": torch.tensor(9)}
            )
            second = transform({"episode_index": 7, "frame_index": 10})

        self.assertEqual(first["sub_task"], "Pick object")
        self.assertEqual(second["sub_task"], "Place object")
        self.assertEqual(first["language_memory"], "")
        self.assertNotIn("Pick; Place", first.values())

    def test_preserves_legacy_annotation_format(self):
        record = {
            "episode_index": 3,
            "action_config": [
                {
                    "start_frame": 4,
                    "end_frame": 8,
                    "action_text": "Close gripper",
                    "language_memory": "Arm is above the object",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "meta"
            meta_dir.mkdir()
            (meta_dir / "episodes_detailed_task.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            transform = LoadActionTextFromJsonlTransformFn().hydrate(
                SimpleNamespace(root=tmpdir)
            )
            sample = transform({"episode_index": 3, "frame_index": 5})

        self.assertEqual(sample["sub_task"], "Close gripper")
        self.assertEqual(sample["language_memory"], "Arm is above the object")

    def test_needs_review_record_does_not_add_subtask_supervision(self):
        record = {
            "source": {"episode_index": 2},
            "labels": {
                "segments": [
                    {"start_frame": 0, "end_frame": 10, "subtask": "Uncertain target"}
                ]
            },
            "quality": {"status": "needs_review"},
        }
        transform = LoadActionTextFromJsonlTransformFn()
        parsed = transform._parse_annotation(record)

        self.assertEqual(parsed, (2, []))


if __name__ == "__main__":
    unittest.main()
