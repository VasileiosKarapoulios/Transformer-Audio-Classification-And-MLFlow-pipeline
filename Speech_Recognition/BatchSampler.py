import torch
from torch.utils.data import Sampler
import random
import time
import numpy as np


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_batches=None):
        self.dataset = dataset
        self.batch_size = batch_size

        self.class_0_indices = list(np.where(np.array(self.dataset.labels) == 0)[0])
        self.class_1_indices = list(np.where(np.array(self.dataset.labels) == 1)[0])
        if num_batches:
            self.num_batches = num_batches
        else:
            self.num_batches = len(self.class_0_indices) // (self.batch_size // 2)
        self.num_samples = len(self.dataset)

    def __iter__(self):
        # Shuffle the indices
        random.shuffle(self.class_0_indices)
        random.shuffle(self.class_1_indices)

        # Yield balanced batches
        for i in range(self.num_batches):
            batch_indices = (
                self.class_0_indices[
                    i * (self.batch_size // 2) : (i + 1) * (self.batch_size // 2)
                ]
                + self.class_1_indices[
                    i * (self.batch_size // 2) : (i + 1) * (self.batch_size // 2)
                ]
            )
            random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches
