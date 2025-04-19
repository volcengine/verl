import torch
import numpy as np
from torch.utils.data import Sampler


class CurriculumSampler(Sampler):

    def __init__(self, data_source, batch_size, target_difficulty):
        self.data_source = data_source
        self.batch_size = batch_size
        self.target_difficulty = target_difficulty
        # Extract difficulty levels from dataset
        self.difficulties = np.array([data_source[i]['difficulty'] for i in range(len(data_source))])
        self.sorted_indices = np.argsort(self.difficulties)

    def __iter__(self):
        num_samples = len(self.data_source)
        batch = []

        while len(batch) < num_samples:
            # Find the batch that best matches the target difficulty
            selected_indices = self._select_closest_to_target()
            batch.extend(selected_indices)
            yield [int(i) for i in selected_indices]

    def _select_closest_to_target(self):
        # Compute the absolute difference from target difficulty for each sample
        diffs = np.abs(self.difficulties - self.target_difficulty)

        # Get the indices of samples sorted by how close they are to the target difficulty
        closest_indices = np.argsort(diffs)

        # Select the top-N closest samples
        selected_indices = []
        for idx in closest_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) == self.batch_size:
                break

        return selected_indices

    def update_target_difficulty(self, new_target):
        """Update the target difficulty dynamically based on model feedback."""
        self.target_difficulty = new_target

    def __len__(self):
        return len(self.data_source) // self.batch_size
