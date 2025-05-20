from typing import Dict
import os
from omegaconf import DictConfig
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, root_cfg: DictConfig, compatible_datasets: Dict) -> None:
        super().__init__()
        self.root_cfg = root_cfg
        self.exp_cfg = root_cfg.experiment
        self.compatible_datasets = compatible_datasets

    def _build_dataset(self, split: str) -> torch.utils.data.Dataset:
        if split in ["training", "test", "validation"]:
            return self.compatible_datasets[self.root_cfg.dataset._name](
                self.root_cfg.dataset, split=split
            )
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")

    @staticmethod
    def _get_shuffle(dataset: torch.utils.data.Dataset, default: bool) -> bool:
        return not isinstance(dataset, torch.utils.data.IterableDataset) and default

    @staticmethod
    def _get_num_workers(num_workers: int) -> int:
        return min(os.cpu_count(), num_workers)

    def _dataloader(self, split: str) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
        dataset = self._build_dataset(split)
        print(f"Dataset {split} has {len(dataset)} samples")
        split_cfg = self.exp_cfg[split]

        def pad_tensor(vec, pad, dim):
            """
            args:
                vec - tensor to pad
                pad - the size to pad to
                dim - dimension to pad

            return:
                a new tensor padded to 'pad' in dimension 'dim'
            """
            pad_size = list(vec.shape)
            pad_size[dim] = pad - vec.size(dim)
            return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
        
        def collate_fn(batch):
            # filter out None values
            batch = list(filter(lambda x: x is not None, batch))

            # pad the batch to the same length
            if isinstance(batch[0], dict):
                max_len = max(sample['videos'].shape[0] for sample in batch)
                keys = batch[0].keys()
                new_batch = []
                for sample in batch:
                    for key in keys:
                        if isinstance(sample[key], torch.Tensor):
                            sample[key] = pad_tensor(sample[key], max_len, 0)
                    new_batch.append(sample)
                batch = new_batch
            elif isinstance(batch[0], tuple):
                max_len = max(len(sample[0].shape[0]) for sample in batch)
                new_batch = []
                for sample in batch:
                    pad_len = max_len - len(sample)
                    new_sample = []
                    for i, item in enumerate(sample):
                        if isinstance(item, torch.Tensor):
                            item = pad_tensor(item, max_len, 0)
                        new_sample.append(item)
                    new_batch.append(tuple(new_sample))
                batch = new_batch                    

            return torch.utils.data.dataloader.default_collate(batch)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=split_cfg.batch_size,
            num_workers=self._get_num_workers(split_cfg.data.num_workers),
            shuffle=self._get_shuffle(dataset, split_cfg.data.shuffle),
            persistent_workers=split == "training",
            worker_init_fn=lambda worker_id: (
                dataset.worker_init_fn(worker_id)
                if hasattr(dataset, "worker_init_fn")
                else None
            ),
            # collate_fn=collate_fn
        )


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader("training")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader("validation")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader("test")
