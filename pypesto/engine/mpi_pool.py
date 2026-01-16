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

    def work(self, pickled_task: bytes, remaining: float):
        task = pickle.loads(pickled_task)

        if hasattr(task, "optimizer") and hasattr(task.optimizer, "supports_maxtime"):
            task.optimizer.set_maxtime(max(0.0, remaining))

        return task.execute()

    def execute(self, tasks, wall_time_limit: float, progress_bar=True) -> list[Any]:
        start = time.time()
        total = len(tasks)

        max_in_flight = 11

        with MPIPoolExecutor(max_workers=11) as ex:
            futures = []
            idx = 0

            def remaining_time():
                return wall_time_limit - (time.time() - start)

            # submit initial batch
            while idx < total and len(futures) < max_in_flight:
                rem = remaining_time()
                if rem <= 0:
                    break
                futures.append(ex.submit(self.work, pickle.dumps(tasks[idx]), rem))
                idx += 1

            results = []
            pbar = tqdm(total=total, disable=not progress_bar)

            # dynamic scheduling
            while futures:
                for fut in as_completed(futures):
                    futures.remove(fut)
                    results.append(fut.result())
                    pbar.update(1)

                    rem = remaining_time()
                    if rem <= 0:
                        # stop submitting new tasks; just drain what's running
                        break

                    if idx < total:
                        futures.append(ex.submit(work, (pickle.dumps(tasks[idx]), rem)))
                        idx += 1

                    # allow “one completion at a time” (keeps loop responsive)
                    break

            pbar.close()
            return results
