"""Engines with multi-node parallelization."""

import logging
import time
from typing import Any

import cloudpickle as pickle
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, as_completed

from ..util import tqdm
from .base import Engine
from .task import Task

logger = logging.getLogger(__name__)


class MPIPoolEngine(Engine):
    """
    Parallelize the task execution.

    Uses `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_.
    To be called with:
    ``mpiexec -np #Workers+1 python -m mpi4py.futures YOURFILE.py``
    """

    def __init__(self):
        super().__init__()

    def work(self, pickled_task: bytes, global_start_time: float, wall_time_limit: float):
        remaining = max(0.0, wall_time_limit - (time.time() - global_start_time))

        task = pickle.loads(pickled_task)

        if hasattr(task, "optimizer") and hasattr(task.optimizer, "supports_maxtime"):
            if optimizer.supports_maxtime():
               task.optimizer.set_maxtime(remaining)

        # Only return work if we still have available wall time
        if remaining != 0: return task.execute()

    def execute(self, tasks, wall_time_limit: float, progress_bar=True) -> list[Any]:
        global_start_time = time.time()
        pickled_tasks = [pickle.dumps(task) for task in tasks]

        with MPIPoolExecutor() as executor:
            results_iter = executor.map(
                self.work,
                pickled_tasks,
                [global_start_time] * len(pickled_tasks),
                [wall_time_limit] * len(pickled_tasks)
            )
            results = list(tqdm(results_iter, total=len(tasks)))

        return results
