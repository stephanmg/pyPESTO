import logging
from collections.abc import Iterable
from typing import Callable, Union
from warnings import warn
from mpi4py.futures import MPIPoolExecutor
from concurrent.futures import as_completed
import time

from ..engine import Engine, SingleCoreEngine
from ..history import HistoryOptions
from ..problem import Problem
from ..result import Result
from ..startpoint import StartpointMethod, to_startpoint_method, uniform
from ..store import autosave
from .optimizer import Optimizer, ScipyOptimizer
from .options import OptimizeOptions
from .task import OptimizerTask
from .util import (
    assign_ids,
    bound_n_starts_from_env,
    postprocess_hdf5_history,
    preprocess_hdf5_history,
)

logger = logging.getLogger(__name__)

def create_task(task_id, optimizer, problem, startpoints, ids, history_options, options):
    return OptimizerTask(
            optimizer=optimizer,
            problem=problem,
            x0=startpoints[task_id],
            id=ids[task_id],
            history_options=history_options,
            optimize_options=options,
        )


def minimize_new(
    problem: Problem,
    optimizer: Optimizer = None,
    n_starts: int = 100,
    ids: Iterable[str] = None,
    startpoint_method: Union[StartpointMethod, Callable, bool] = None,
    result: Result = None,
    options: OptimizeOptions = None,
    history_options: HistoryOptions = None,
    filename: Union[str, Callable, None] = None,
    overwrite: bool = False,
) -> None:

    # optimizer
    if optimizer is None: optimizer = ScipyOptimizer()

    # number of starts
    n_starts = bound_n_starts_from_env(n_starts)

    # startpoint method
    if startpoint_method is None:
        if problem.startpoint_method is None:
            startpoint_method = uniform
        else:
            startpoint_method = problem.startpoint_method

    startpoint_method = to_startpoint_method(startpoint_method)

    if options is None: options = OptimizeOptions()
    options = OptimizeOptions.assert_instance(options)

    if history_options is None: history_options = HistoryOptions()
    history_options = HistoryOptions.assert_instance(history_options)

    # assign startpoints
    startpoints = startpoint_method(
        n_starts=n_starts,
        problem=problem,
    )

    # assign ids
    ids = assign_ids(
        n_starts=n_starts,
        ids=ids,
        result=result,
    )

    # change to one hdf5 storage file per start if parallel and if hdf5
    history_file = history_options.storage_file

    # number of starts in total, probably rather high
    total_tasks = n_starts
    # limit to specified number of processes, thus one task per proc always running
    max_parallel_tasks = MPI.COMM_WORLD.Get_size()

    results = []
    # Create an MPIPoolExecutor for managing worker processes
    with MPIPoolExecutor(max_workers=max_parallel_tasks) as executor:
        futures = []
        task_idx = 0

        # Submit the first 16 tasks to start the process
        for _ in range(min(max_parallel_tasks, total_tasks)):
            task = create_task(task_idx, optimizer, problem, startpoints, ids, history_options, options)
            futures.append(executor.submit(task.execute))
            task_idx += 1

        # Keep track of how many tasks have completed
        completed_tasks = 0

        # As tasks complete, dynamically submit new ones until all tasks are submitted
        while completed_tasks < total_tasks:
            for future in as_completed(futures):
                # Remove the completed future from the active list
                futures.remove(future)

                # Collect result if needed (you can remove this if results aren't needed)
                result = future.result()
                results.append(result)

                completed_tasks += 1

                if task_idx < total_tasks:
                    # Submit a new task as soon as one finishes
                    task = create_task(task_idx, optimizer, problem, startpoints, ids, history_options, options)
                    futures.append(executor.submit(task.execute))
                    task_idx += 1

                # Break to allow immediate check of task completions
                break

        print(f"All {total_tasks} tasks completed.")
