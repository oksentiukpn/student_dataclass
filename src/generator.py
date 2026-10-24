"""
Data and Operation Generator for Homework 1 Experiments.

Variant parameters:
- Task 1: Variant V5
- Ratio A : B : C = 10 : 10 : 1 (Op1 : Op2 : Op3)
- Task 2: Variant S7 (Sort by phone number)
"""

import random
from typing import Any, List, Tuple

from src.models import Student, read_data


def generate_operations(
    students: List[Student],
    total_count: int = 200_000,
    ratio: Tuple[int, int, int] = (10, 10, 1),
    seed: int = 42,
) -> List[Tuple[Any, ...]]:
    """
    Pre-generate operations according to the specified ratio A:B:C.
    Operations arrive in random order.
    Operation representation:
    - (1,): Op 1 - get_top_100_students()
    - (2, email, new_rating): Op 2 - set_student_rating(email, new_rating)
    - (3,): Op 3 - get_groups_with_highest_average_rating()
    """
    rng = random.Random(seed)
    a, b, c = ratio
    unit_sum = a + b + c
    cycles = total_count // unit_sum + 1

    emails = [s.email for s in students]
    if not emails:
        raise ValueError("Students list cannot be empty when generating operations.")

    ops: List[Tuple[Any, ...]] = []

    for _ in range(cycles):
        block: List[Tuple[Any, ...]] = []
        for _ in range(a):
            block.append((1,))
        for _ in range(b):
            target_email = rng.choice(emails)
            new_rating = round(rng.uniform(0.0, 100.0), 2)
            block.append((2, target_email, new_rating))
        for _ in range(c):
            block.append((3,))

        rng.shuffle(block)
        ops.extend(block)

    return ops[:total_count]


def get_dataset_sizes() -> List[int]:
    """10 different database sizes from 100 to 100,000 for scaling experiments."""
    return [100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 75_000, 100_000]


def get_student_sample(all_students: List[Student], size: int) -> List[Student]:
    """Return slice of database of the given size."""
    if size > len(all_students):
        raise ValueError(
            f"Requested size {size} exceeds available students ({len(all_students)})"
        )
    return all_students[:size]
