"""Evaluation protocols -- the rules that make results believable.

Implements the five non-negotiable protocol rules from the blueprint:
  1. Cross-dataset is the headline (train on A, test on B, no finetune)
  2. Leave-one-manipulation-out
  3. Leave-one-generator-out (for Branch C)
  4. Authentic-only false-positive rate, reported separately
  5. Fixed splits / seeds

This module doesn't run training itself -- it defines the *splitting*
logic so every experiment in the project uses the same, auditable
protocol rather than ad hoc train_test_split calls scattered around.
"""
from __future__ import annotations
from dataclasses import dataclass
import random


@dataclass
class DatasetSplit:
    name: str
    train_samples: list
    test_samples: list
    notes: str = ""


def cross_dataset_split(train_dataset_samples: dict[str, list],
                         test_dataset_name: str) -> DatasetSplit:
    """Train on the union of every dataset EXCEPT `test_dataset_name`,
    test on it, with zero finetuning on the target. This is the
    headline number -- in-dataset splits belong in an appendix only.
    """
    train = []
    for name, samples in train_dataset_samples.items():
        if name != test_dataset_name:
            train.extend(samples)
    test = train_dataset_samples.get(test_dataset_name, [])
    return DatasetSplit(
        name=f"cross_dataset::held_out={test_dataset_name}",
        train_samples=train,
        test_samples=test,
        notes="No finetuning on the held-out dataset's training split.",
    )


def leave_one_manipulation_out_split(samples: list, manipulation_field: str,
                                      held_out_type: str) -> DatasetSplit:
    """samples: list of objects with a `.manipulation` (or dict key)
    attribute in {"splice","copy_move","removal","generative_fill","none"}.
    """
    def get_type(s):
        return getattr(s, manipulation_field, None) or s.get(manipulation_field)

    train = [s for s in samples if get_type(s) != held_out_type]
    test = [s for s in samples if get_type(s) == held_out_type or get_type(s) == "none"]
    return DatasetSplit(
        name=f"leave_one_manipulation_out::held_out={held_out_type}",
        train_samples=train,
        test_samples=test,
        notes="Reveals whether the model learned forensics or a "
              "manipulation-style shortcut.",
    )


def leave_one_generator_out_split(samples: list, generator_field: str,
                                   held_out_generator: str) -> DatasetSplit:
    """Same idea for Branch C. Non-negotiable per the blueprint --
    same-generator numbers for AIGC detection are not meaningful."""
    def get_gen(s):
        return getattr(s, generator_field, None) or s.get(generator_field)

    train = [s for s in samples if get_gen(s) != held_out_generator]
    test = [s for s in samples if get_gen(s) == held_out_generator]
    return DatasetSplit(
        name=f"leave_one_generator_out::held_out={held_out_generator}",
        train_samples=train,
        test_samples=test,
    )


def authentic_only_fpr_set(authentic_pool: list, n: int = 10_000, seed: int = 0) -> list:
    """A dedicated held-out set of untouched images (e.g. Flickr/COCO)
    used ONLY to measure false-positive rate, per the blueprint's
    explicit warning that published methods often quietly carry
    double-digit FPR on clean images."""
    rng = random.Random(seed)
    pool = list(authentic_pool)
    rng.shuffle(pool)
    return pool[:n]


FIXED_SEEDS = (13, 42, 137)  # publish results across these three seeds
