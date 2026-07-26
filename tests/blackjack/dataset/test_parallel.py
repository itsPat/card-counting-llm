from __future__ import annotations

from pathlib import Path

import pytest

from blackjack.dataset import DatasetConfiguration
from blackjack.dataset.parallel import run_parallel_generation


@pytest.mark.parametrize("worker_count", [0, -1])
def test_parallel_generation_rejects_nonpositive_workers(
    tmp_path: Path,
    worker_count: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        run_parallel_generation(
            DatasetConfiguration(shoe_count=3),
            tmp_path,
            worker_count=worker_count,
        )


def test_parallel_generation_rejects_more_workers_than_shoes(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="exceed"):
        run_parallel_generation(
            DatasetConfiguration(shoe_count=3),
            tmp_path,
            worker_count=4,
        )
