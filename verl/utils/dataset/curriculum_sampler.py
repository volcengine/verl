from collections import defaultdict
from typing import Any, Callable, Dict, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

DIFFICULTY_KEY = "difficulty"

class CurriculumSampler(Sampler[int]):
    def __init__(self,
                 data_source: Dataset,
                 initial_difficulty: int,
                 min_difficulty: int,
                 max_difficulty: int,
                 generator: torch.Generator,
                 shuffle: bool,
                 curriculum_function: Callable[[int, Dict[str, Any]], int]):
        
        self.data_source = data_source
        assert initial_difficulty <= max_difficulty, f"Initial difficulty: {initial_difficulty} is greater than max difficulty: {max_difficulty}."
        self.current_difficulty = initial_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.shuffle = shuffle
        self.generator = generator 
        self.curriculum_function = curriculum_function

        # difficulty -> np.array of dataset indices
        self._indices_by_difficulty: Dict[int, np.ndarray] = defaultdict(list)
        # difficulty -> current position in its index array
        self._current_indices_pos: Dict[int, int] = defaultdict(int)
        
        self._build_difficulty_index()

        for difficulty in self._indices_by_difficulty.keys():
            self._reset_sampler(difficulty)
    
    def _build_difficulty_index(self):
        for i in range(len(self.data_source)):
            item = self.data_source[i]
            
            assert DIFFICULTY_KEY in item, f"Could not find key, {DIFFICULTY_KEY} in datapoint"
            difficulty = int(item.get(DIFFICULTY_KEY))

            if difficulty <= self.max_difficulty:
                self._indices_by_difficulty[difficulty].append(i)
         
        for difficulty, data_list in self._indices_by_difficulty.items():
            self._indices_by_difficulty[difficulty] = np.array(data_list, dtype=np.int64)
    
    def _reset_sampler(self, difficulty: int) -> None:
        if self.shuffle:
            n = len(self._indices_by_difficulty[difficulty])
            perm = torch.randperm(n, generator=self.generator)
            self._indices_by_difficulty[difficulty] = self._indices_by_difficulty[difficulty][perm.numpy()]            
        self._current_indices_pos[difficulty] = 0   
    
    def update_difficulty(self, data: Dict[str, Any]) -> int:
        desired_difficulty = self.curriculum_function(self.current_difficulty, data)
        
        if desired_difficulty == self.current_difficulty:
            return self.current_difficulty
        
        if desired_difficulty < self.min_difficulty:
            print(f"Desired difficulty {desired_difficulty} is less than min difficulty {self.min_difficulty}")
        elif desired_difficulty > self.max_difficulty:
            print(f"Desired difficulty {desired_difficulty} is greater than max difficulty {self.max_difficulty}")
        else:
            print(f"Updating difficulty from {self.current_difficulty} to {desired_difficulty}")
            assert desired_difficulty in self._indices_by_difficulty, f"No datapoints found for difficulty {desired_difficulty}"
            self.current_difficulty = desired_difficulty
        
        return self.current_difficulty
    
    def __iter__(self) -> Iterator[int]:
        while True:
            indices_list = self._indices_by_difficulty[self.current_difficulty]
            current_pos = self._current_indices_pos[self.current_difficulty]
            
            if current_pos >= len(indices_list):
                self._reset_sampler(self.current_difficulty)
                current_pos = self._current_indices_pos[self.current_difficulty]
            
            yield int(indices_list[current_pos])
            self._current_indices_pos[self.current_difficulty] += 1

    def __len__(self) -> int:
        return len(self._indices_by_difficulty[self.current_difficulty])

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "current_difficulty": self.current_difficulty,
            "current_indices_pos": dict(self._current_indices_pos)
        }
        return state
   
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.current_difficulty = state_dict["current_difficulty"]
        assert self.min_difficulty <= self.current_difficulty <= self.max_difficulty

        loaded_positions = state_dict["current_indices_pos"]
        self._current_indices_pos = defaultdict(int)
        self._current_indices_pos.update(loaded_positions)
