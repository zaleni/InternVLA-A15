import unittest
from types import SimpleNamespace

from torch.utils.data import Dataset

from lerobot.datasets.transformed_dataset import CompleteActionChunkDataset, MultiLeRobotDataset


class _IndexDataset(Dataset):
    """Small index-only dataset with LeRobot-style episode metadata."""

    def __init__(self, episode_lengths: list[int]) -> None:
        starts = []
        ends = []
        episode_by_source_index = []
        cursor = 0
        for episode_index, length in enumerate(episode_lengths):
            starts.append(cursor)
            cursor += length
            ends.append(cursor)
            episode_by_source_index.extend([episode_index] * length)

        self.repo_id = "test/repo"
        self.meta = SimpleNamespace(
            total_episodes=len(episode_lengths),
            episodes={
                "dataset_from_index": starts,
                "dataset_to_index": ends,
            },
        )
        self._episode_by_source_index = episode_by_source_index

    @property
    def num_frames(self) -> int:
        return len(self._episode_by_source_index)

    @property
    def num_episodes(self) -> int:
        return self.meta.total_episodes

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> dict[str, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return {
            "source_index": index,
            "episode_index": self._episode_by_source_index[index],
        }


class CompleteActionChunkDatasetTest(unittest.TestCase):
    def test_maps_logical_indices_across_multiple_episodes(self):
        # max_future_offset=4 corresponds to a five-step inclusive chunk.
        # Raw episode ranges are [0, 6), [6, 11), [11, 19), so their
        # complete-chunk start counts are 2, 1, and 4 respectively.
        dataset = CompleteActionChunkDataset(
            _IndexDataset([6, 5, 8]),
            max_future_offset=4,
        )

        self.assertEqual(len(dataset), 7)
        self.assertEqual(dataset.num_frames, 7)
        self.assertEqual(
            [dataset[i]["source_index"] for i in range(len(dataset))],
            [0, 1, 6, 11, 12, 13, 14],
        )
        self.assertEqual(
            [dataset[i]["episode_index"] for i in range(len(dataset))],
            [0, 0, 1, 2, 2, 2, 2],
        )

    def test_short_episode_contributes_no_samples(self):
        dataset = CompleteActionChunkDataset(
            _IndexDataset([3, 5]),
            max_future_offset=4,
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0], {"source_index": 3, "episode_index": 1})

    def test_negative_indices_and_out_of_range_indices(self):
        dataset = CompleteActionChunkDataset(
            _IndexDataset([6, 5, 8]),
            max_future_offset=4,
        )

        self.assertEqual(dataset[-1]["source_index"], 14)
        self.assertEqual(dataset[-len(dataset)]["source_index"], 0)
        with self.assertRaises(IndexError):
            _ = dataset[len(dataset)]
        with self.assertRaises(IndexError):
            _ = dataset[-len(dataset) - 1]

    def test_exposes_episode_boundaries_in_logical_index_space(self):
        dataset = CompleteActionChunkDataset(
            _IndexDataset([6, 5, 8]),
            max_future_offset=4,
        )

        self.assertEqual(dataset.sampling_episode_from_indices, [0, 2, 3])
        self.assertEqual(dataset.sampling_episode_to_indices, [2, 3, 7])

        # Preserve the original episode order for an episode that is too
        # short: it is represented by an empty logical interval.
        with_short_episode = CompleteActionChunkDataset(
            _IndexDataset([3, 5]),
            max_future_offset=4,
        )
        self.assertEqual(with_short_episode.sampling_episode_from_indices, [0, 0])
        self.assertEqual(with_short_episode.sampling_episode_to_indices, [0, 1])

    def test_zero_future_offset_is_identity(self):
        dataset = CompleteActionChunkDataset(
            _IndexDataset([2, 3]),
            max_future_offset=0,
        )

        self.assertEqual(len(dataset), 5)
        self.assertEqual(
            [dataset[i]["source_index"] for i in range(len(dataset))],
            list(range(5)),
        )
        self.assertEqual(dataset.sampling_episode_from_indices, [0, 2])
        self.assertEqual(dataset.sampling_episode_to_indices, [2, 5])

    def test_multi_dataset_uses_filtered_lengths_and_logical_boundaries(self):
        filtered = CompleteActionChunkDataset(
            _IndexDataset([6, 5]),
            max_future_offset=4,
        )
        unfiltered = _IndexDataset([2, 3])
        unfiltered.repo_id = "test/second-repo"

        dataset = MultiLeRobotDataset([filtered, unfiltered])

        self.assertEqual(len(dataset), 8)
        self.assertEqual(dataset._lengths, [3, 5])
        self.assertEqual(dataset[2]["source_index"], 6)
        self.assertEqual(dataset[2]["dataset_index"].item(), 0)
        self.assertEqual(dataset[3]["source_index"], 0)
        self.assertEqual(dataset[3]["dataset_index"].item(), 1)
        self.assertEqual(dataset.meta.episodes["dataset_from_index"], [0, 2, 3, 5])
        self.assertEqual(dataset.meta.episodes["dataset_to_index"], [2, 3, 5, 8])

    def test_rejects_negative_future_offset(self):
        with self.assertRaises(ValueError):
            CompleteActionChunkDataset(
                _IndexDataset([5]),
                max_future_offset=-1,
            )


if __name__ == "__main__":
    unittest.main()
