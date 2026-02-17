from typing import Sequence, List
from torch.utils.data import Dataset
from src.dataset.train_dataset import TrainDataset

class MultiTrainDataset(Dataset):
    """
    Dataset for training from multiple datasets.
    A pure container that iterates over a list of TrainDataset objects.
    """

    def __init__(self, datasets: List[TrainDataset], trainer=None):
        """
        Args:
            datasets: List of TrainDataset objects to combine
            trainer: Optional trainer to set on all sub-datasets
        """
        self.datasets = datasets
        self.trainer = trainer
        if trainer is not None:
            self.set_trainer(trainer)

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, item):
        dataset_index = 0
        while item >= len(self.datasets[dataset_index]):
            item -= len(self.datasets[dataset_index])
            dataset_index += 1
        return self.datasets[dataset_index][item]

    def set_trainer(self, trainer):
        """Sets the trainer for all sub-datasets."""
        self.trainer = trainer
        for dataset in self.datasets:
            dataset.set_trainer(trainer)

class DatasetSplitView(Dataset):
    """
    Lightweight view over another dataset that restricts access to a subset of indices.
    This allows us to reuse the same underlying dataset for train/validation splits
    while keeping the logic for sampling positives/negatives centralized.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        if len(indices) == 0:
            raise ValueError("DatasetSplitView requires at least one index.")
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        return self.dataset[base_idx]

    def set_trainer(self, trainer):
        if hasattr(self.dataset, "set_trainer"):
            self.dataset.set_trainer(trainer)
