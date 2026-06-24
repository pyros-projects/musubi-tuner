import types
import unittest
from multiprocessing import Value

import torch
import torch.nn as nn

from musubi_tuner.dataset.image_video_dataset import BucketBatchManager, DatasetGroup
from musubi_tuner.hv_train_network import NetworkTrainer, collator_class, move_batch_tensors_to_device, normalize_compile_args


class _NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")

    def accumulate(self, _model):
        return _NoOpContext()

    def autocast(self):
        return _NoOpContext()

    def backward(self, loss):
        loss.backward()


class _DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = nn.Parameter(torch.tensor(0.0))

    def on_step_start(self):
        return None


class _ToyPrewarmTrainer(NetworkTrainer):
    def scale_shift_latents(self, latents):
        return latents

    def get_noisy_model_input_and_timesteps(
        self,
        args,
        noise,
        latents,
        timesteps,
        noise_scheduler,
        device,
        dtype,
    ):
        return latents + noise, torch.zeros(latents.shape[0], device=device, dtype=dtype)

    def call_dit(
        self,
        args,
        accelerator,
        transformer,
        latents,
        batch,
        noise,
        noisy_model_input,
        timesteps,
        network_dtype,
    ):
        return transformer(noisy_model_input), torch.zeros_like(noisy_model_input)


class _FakeDatasetGroup:
    def __init__(self):
        self.current_epoch = None
        self.items = [
            {"latents": torch.ones(2, 4), "timesteps": None},
            {"latents": torch.ones(2, 4) * 2, "timesteps": None},
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def get_representative_batch_indices(self, max_buckets=None):
        indices = [0, 1]
        if max_buckets is None:
            return indices
        return indices[:max_buckets]


class _FakeRepresentativeDataset:
    def __init__(self, bucket_sizes, representative_indices):
        self.items = list(range(sum(bucket_sizes)))
        self.num_train_items = len(self.items)
        self.batch_manager = types.SimpleNamespace(
            buckets={bucket_index: [None] * bucket_size for bucket_index, bucket_size in enumerate(bucket_sizes)},
            bucket_batch_indices=[
                (bucket_index, item_index)
                for bucket_index, bucket_size in enumerate(bucket_sizes)
                for item_index in range(bucket_size)
            ],
        )
        self._representative_indices = representative_indices

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def get_representative_batch_indices(self, max_buckets=None):
        indices = list(self._representative_indices)
        if max_buckets is None:
            return indices
        return indices[:max_buckets]


class CompilePrewarmTest(unittest.TestCase):
    def test_compile_prewarm_implies_compile(self):
        args = types.SimpleNamespace(compile=False, compile_prewarm=True)

        normalized = normalize_compile_args(args)

        self.assertIs(normalized, args)
        self.assertTrue(args.compile)

    def test_move_batch_tensors_to_device_handles_nested_structures(self):
        batch = {
            "latents": torch.ones(2, 4),
            "nested": [torch.zeros(1), {"mask": torch.ones(3)}],
            "text": "keep-me",
            "none": None,
        }

        moved = move_batch_tensors_to_device(batch, torch.device("cpu"))

        self.assertIsNot(moved, batch)
        self.assertEqual(moved["latents"].device.type, "cpu")
        self.assertEqual(moved["nested"][0].device.type, "cpu")
        self.assertEqual(moved["nested"][1]["mask"].device.type, "cpu")
        self.assertEqual(moved["text"], "keep-me")
        self.assertIsNone(moved["none"])

    def test_bucket_batch_manager_selects_one_representative_per_bucket(self):
        manager = BucketBatchManager(
            {
                (512, 512): [None, None, None, None],
                (768, 768): [None, None, None],
                (1024, 1024): [None],
            },
            batch_size=2,
        )

        indices = manager.get_representative_batch_indices()
        self.assertEqual(indices, [0, 2, 4])

        limited = manager.get_representative_batch_indices(max_buckets=2)
        self.assertEqual(limited, [0, 2])

    def test_dataset_group_collects_representative_batch_indices_across_datasets(self):
        first_dataset = _FakeRepresentativeDataset(bucket_sizes=[4, 1], representative_indices=[0, 4])
        second_dataset = _FakeRepresentativeDataset(bucket_sizes=[3, 2], representative_indices=[0, 3])
        dataset_group = DatasetGroup([first_dataset, second_dataset])

        representatives = dataset_group.get_representative_batch_indices()
        limited = dataset_group.get_representative_batch_indices(max_buckets=3)

        self.assertEqual(representatives, [0, 5, 8, 4])
        self.assertEqual(limited, [0, 5, 8])

    def test_compile_prewarm_dataloader_uses_dataset_bound_collator(self):
        trainer = _ToyPrewarmTrainer()
        dataset_group = _FakeDatasetGroup()
        current_epoch = Value("i", 3)
        training_collator = collator_class(current_epoch, None)

        prewarm_dataloader, representative_indices = trainer.build_compile_prewarm_dataloader(
            dataset_group, training_collator, max_buckets=None
        )

        self.assertEqual(representative_indices, [0, 1])
        first_batch = next(iter(prewarm_dataloader))
        self.assertEqual(first_batch["latents"].shape, torch.Size([2, 4]))
        self.assertEqual(dataset_group.current_epoch, 3)

    def test_compile_prewarm_runs_without_optimizer_step_and_restores_rng(self):
        trainer = _ToyPrewarmTrainer()
        accelerator = _FakeAccelerator()
        args = types.SimpleNamespace(gradient_checkpointing=False, weighting_scheme="none")
        transformer = torch.compile(nn.Linear(4, 4, bias=False), backend="eager")
        network = _DummyNetwork()
        optimizer = torch.optim.SGD(transformer.parameters(), lr=0.1)

        batch = {"latents": torch.ones(2, 4), "timesteps": None}
        initial_weight = next(transformer.parameters()).detach().clone()
        initial_rng = torch.get_rng_state()

        trainer._run_compile_prewarm(
            accelerator=accelerator,
            args=args,
            training_model=network,
            transformer=transformer,
            network=network,
            optimizer=optimizer,
            prewarm_batches=[batch],
            noise_scheduler=None,
            dit_dtype=torch.float32,
            network_dtype=torch.float32,
            include_backward=True,
        )

        final_weight = next(transformer.parameters()).detach()
        self.assertTrue(torch.equal(final_weight, initial_weight))
        self.assertTrue(torch.equal(torch.get_rng_state(), initial_rng))
        for param in transformer.parameters():
            self.assertIsNone(param.grad)


if __name__ == "__main__":
    unittest.main()
