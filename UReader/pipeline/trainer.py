import torch.distributed as dist
import argparse
from functools import partial

import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import Trainer

from pipeline.utils import batchify

    
class CustomTrainer(Trainer):
    
    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        sampler = DistributedSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=False,
            collate_fn=batchify)


    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        dataset = self.eval_dataset
        sampler = DistributedSampler(dataset, shuffle=False)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=False,
            collate_fn=batchify)