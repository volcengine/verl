import numpy as np
import torch
from torch.utils.data import Sampler, SubsetRandomSampler


class DynamicSampler(Sampler[int]):
    _YIELDED = "yielded"
    _CURRENT_LEVEL = "current_level"
    _ACCURACY_HISTORY = "accuracy_history"

    def __init__(self, data_source, difficulties, total_steps, batch_size, 
                 init_level=0, mix_harder_samples=0.2, smooth_window=3, 
                 threshold_for_next_level=0.75, threshold_for_prev_level=0.3,
                 seed=42):
        """
        Initialize the DynamicSampler for curriculum learning.
        
        Args:
            data_source: The dataset to sample from
            difficulties: A list of difficulty values for each sample in data_source (must be integers)
            total_steps: Total number of batches to generate during iteration
            batch_size: Number of samples in each batch
            init_level: Initial difficulty level to start sampling from (default: 0)
            mix_harder_samples: Proportion of samples to draw from the next difficulty level (default: 0.2)
            smooth_window: Number of accuracy values to average when determining level transitions (default: 2)
            threshold_for_next_level: Accuracy threshold to increase difficulty level (default: 0.75)
            threshold_for_prev_level: Accuracy threshold to decrease difficulty level (default: 0.3)
        """
        assert len(difficulties) == len(data_source), "difficulties and data_source should have the same length"
        assert all(isinstance(difficulty, int) for difficulty in difficulties), "difficulties should be a list of int values"
        assert 0 <= mix_harder_samples <= 1, "mix_harder_samples should be between 0 and 1"
        assert threshold_for_next_level > threshold_for_prev_level, "threshold_for_next_level should be greater than threshold_for_prev_level"
        assert smooth_window > 0, "smooth_window should be greater than 0"

        self.difficulty_level_count = len(set(difficulties))
        assert 0 <= init_level < self.difficulty_level_count, "init_level should be between 0 and difficulty_level_count"
        self.index_to_level = {i: level for i, level in enumerate(difficulties)}
        self.difficulty_levels = sorted(set(difficulties))

        self.data_source = data_source
        self.difficulties = difficulties
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.current_level = init_level
        self.mix_harder_samples = mix_harder_samples
        self.smooth_window = smooth_window
        self.threshold_for_next_level = threshold_for_next_level
        self.threshold_for_prev_level = threshold_for_prev_level
        self.accuracy_history = []
        self.seed = seed

        self._current_level_sample_count_in_batch = int((1-self.mix_harder_samples)*self.batch_size)

        # Create a SubsetRandomSampler for each difficulty level
        self.samplers = {}
        self.iters = {}
        self.difficulty_level_subsets = {}
        self.difficulty_level_yielded = {}
        self._split_into_subsets()
    
    def set_current_level(self, level):
        assert 0 <= level < self.difficulty_level_count, "level should be between 0 and difficulty_level_count"
        self.current_level = level

    def _split_into_subsets(self):
        # Group by difficulty level
        for i, level in enumerate(self.difficulties):
            level_index = self.difficulty_levels.index(level)
            if level_index not in self.difficulty_level_subsets:
                self.difficulty_level_subsets[level_index] = []
            self.difficulty_level_subsets[level_index].append(i)

        for level, indices in self.difficulty_level_subsets.items():
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            self.samplers[level] = SubsetRandomSampler(indices, generator=generator)
            self.iters[level] = iter(self.samplers[level])
        
    def _sample_from_level(self, level, batch_size):
        
        # Sample from the specified difficulty level
        # Assumes uniform distribution within each level
        # Sampling method can be adjusted based on specific requirements
        sample_level = max(min(level, self.difficulty_level_count-1), 0)
        if sample_level != level:
            print(f"[Sampler] Warning: Level {level} not found, using level {sample_level} instead")
        # Sample using samplers[level]
        sampled_indices = []
        for _ in range(batch_size):
            try:
                sampled_indices.append(next(self.iters[sample_level]))
            except StopIteration:
                self.iters[sample_level] = iter(self.samplers[sample_level])
                sampled_indices.append(next(self.iters[sample_level]))
        self._update_yielded(sample_level, batch_size)
        print(f"[SAMPLER] Sampling from level {level} with batch size {batch_size}, Top 8: {sampled_indices[:8]}")
        return sampled_indices
    
    def _update_yielded(self, level, batch_size):
        if level not in self.difficulty_level_yielded:
            self.difficulty_level_yielded[level] = 0
        self.difficulty_level_yielded[level] += batch_size

    def __iter__(self):
        # Logic for mixed sampling
        for _ in range(self.total_steps):
            current_level_samples = self._sample_from_level(self.current_level, self._current_level_sample_count_in_batch)
            if self.mix_harder_samples > 0:
                next_level_samples = self._sample_from_level(self.current_level+1, self.batch_size - self._current_level_sample_count_in_batch)
                combined_samples = current_level_samples + next_level_samples
            else:
                combined_samples = current_level_samples
            yield from combined_samples
    
    def update_sampling_policy(self, latest_accuracy):
        self.accuracy_history.append(latest_accuracy)

        # Don't update difficulty if not enough accuracy history
        if len(self.accuracy_history) < self.smooth_window:
            return

        # Smoothing: use moving window average to reduce fluctuations
        smoothed_acc = np.mean(self.accuracy_history[-self.smooth_window:])
        next_level = self.current_level

        if smoothed_acc > self.threshold_for_next_level:
            # Excellent performance: increase difficulty level (avoid local optima)
            next_level = min(self.current_level + 1, self.difficulty_level_count-1)
        elif smoothed_acc < self.threshold_for_prev_level:
            # Performance decline: return to safer level and increase mixed sampling
            next_level = max(self.current_level - 1, 0)
        else:
            # Maintain current difficulty but adjust sampling distribution
            pass

        print(f"[Sampler] Smooth accuracy: {smoothed_acc}, current level: {self.current_level}, next level: {next_level}")
        # If difficulty changes, clear accuracy history
        if next_level != self.current_level:
            self.accuracy_history = []
        self.current_level = next_level
        
    def __len__(self):
        return self.total_steps * self.batch_size
    
    def state_dict(self) -> dict:
        return {
            self._YIELDED: self.difficulty_level_yielded,
            self._CURRENT_LEVEL: self.current_level,
            self._ACCURACY_HISTORY: self.accuracy_history,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        current_level = state_dict[self._CURRENT_LEVEL]
        self.set_current_level(current_level)
        self.accuracy_history = state_dict[self._ACCURACY_HISTORY]

        self.next_yielded = state_dict[self._YIELDED]
        if self.next_yielded is None:
            return
        
        for level in self.next_yielded:
            if level in self.samplers:
                self._sample_from_level(level, self.next_yielded[level])
                self.difficulty_level_yielded[level] = self.next_yielded[level]
        self.next_yielded = None

if __name__ == "__main__":

    import datasets
    from torchdata.stateful_dataloader import StatefulDataLoader

    train_dataset = datasets.load_dataset('pe-nlp/ORZ-MATH-57k-Filter-difficulty', split='train', trust_remote_code=True)
    # train_dataset = train_dataset.filter(lambda x: x['pass_at_n'] is not None and x['llama8b_solve_rate'] < 0.95)
    train_dataset = train_dataset.map(lambda x: {**x, 'difficulty': int((1.0 - x['pass_at_n']) * 10) + 1})
    # train_dataset = train_dataset.shuffle(seed=42).select(range(len(train_dataset) // 2))

    dataset = np.arange(len(train_dataset))

    difficulties = train_dataset.to_pandas()['difficulty'].values.tolist()
    print(difficulties[:5], type(difficulties[0]))

    sampler = DynamicSampler(
        data_source=dataset,
        difficulties=difficulties,
        total_steps=len(train_dataset) * 10 // 3,
        batch_size=3,
        init_level=0,
        mix_harder_samples=0.2,
        smooth_window=2,
        threshold_for_next_level=0.75,
        threshold_for_prev_level=0.3,
        seed=42
    )
    data_loader = StatefulDataLoader(dataset, sampler=sampler, batch_size=3, num_workers=0)
    difficulties = train_dataset.to_pandas()['difficulty'].values
    i = 0
    data = []
    acc_list = [0.1, 0.2, 0.5, 0.7, 0.75, 0.76, 0.6, 0.7, 0.75, 0.7, 0.71, 0.8, 0.4, 0.7, 0.81]
    for batch in data_loader:
        # print(batch)
        print("Step: ", i)
        # print(data_loader.sampler.current_step)
        print(batch[:10])
        print(f"batch {i}: mean difficulty: {np.mean(difficulties[batch])}")
        # data.append([i, np.mean(difficulties[batch])])
        sampler.update_sampling_policy(acc_list[i-1])
        print("current level: ", sampler.current_level)
        print("acc history: ", sampler.accuracy_history)
        print("=====")
        i += 1
        if i >= len(acc_list) // 2:
            sd = data_loader.state_dict()
            print(sd)
            break

    print("\n\nXXXXX Save and Load XXXXX\n\n")

    sampler2 = DynamicSampler(
        data_source=dataset,
        difficulties=difficulties.tolist(),
        total_steps=len(train_dataset) * 10 // 3,
        batch_size=3,
        init_level=0,
        mix_harder_samples=0.2,
        smooth_window=2,
        threshold_for_next_level=0.75,
        threshold_for_prev_level=0.3,
        seed=42
    )
    data_loader2 = StatefulDataLoader(dataset, sampler=sampler2, batch_size=3, num_workers=0)

    data_loader2.load_state_dict(sd)
    for batch in data_loader2:
        print(batch[:10])
        print(f"batch {i}: mean difficulty: {np.mean(difficulties[batch])}")
        
        sampler2.update_sampling_policy(acc_list[i-1])
        print("current level: ", sampler2.current_level)
        print("acc history: ", sampler2.accuracy_history)
        print("=====")

        i += 1
        if i >= len(acc_list):
            break
