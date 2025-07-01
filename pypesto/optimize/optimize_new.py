import logging
from collections.abc import Iterable
import h5py
from typing import Callable, Union
from warnings import warn
from mpi4py.futures import MPIPoolExecutor
from concurrent.futures import as_completed
import time
import numpy as np
from mpi4py import MPI

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
    """ Create optimization task """
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
    n_starts: int = 1,
    ids: Iterable[str] = None,
    startpoint_method: Union[StartpointMethod, Callable, bool] = None,
    result: Result = None,
    options: OptimizeOptions = None,
    history_options: HistoryOptions = None,
    filename: Union[str, Callable, None] = None,
    interval: int = 1,
) -> None:
    """ New minimize for benchmark study """
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

    buffered_results = []
    results = []
    with MPIPoolExecutor(max_workers=max_parallel_tasks) as executor:
        futures = []
        task_idx = 0

        for _ in range(min(max_parallel_tasks, total_tasks)):
            task = create_task(task_idx, optimizer, problem, startpoints, ids, history_options, options)
            futures.append(executor.submit(task.execute))
            task_idx += 1

        completed_tasks = 0

        while completed_tasks < total_tasks:
            for future in as_completed(futures):
                futures.remove(future)
                result = future.result()
                buffered_results.append((completed_tasks, result))
                completed_tasks += 1

                # Submit new tasks until number of total multi starts reached 
                if task_idx < total_tasks:
                    task = create_task(task_idx, optimizer, problem, startpoints, ids, history_options, options)
                    futures.append(executor.submit(task.execute))
                    task_idx += 1

                # Periodically write out results, default: every result (as specified by interval)
                if completed_tasks % interval == 0:
                   if MPI.COMM_WORLD.Get_rank() == 0:
                      with h5py.File(filename, "a") as f:
                           for task_idx, res in buffered_results:
                                group_name = f"result_{task_idx}"
                                grp = f.create_group(group_name)
                                fvals, time, x, grad = res.history.get_fval_trace(), res.history.get_time_trace(), res.history.get_x_trace(), res.history.get_grad_trace()
                                
                                mask = np.isfinite(fvals)
                                fvals = np.array(fvals)[mask]
                                time = np.array(time)[mask]
                                x = np.array(x)[mask]
                                
                                grp.create_dataset("fval", data=np.array(fvals))
                                grp.create_dataset("time", data=np.array(time))
                                grp.create_dataset("x", data=np.array(x))
                                grp.create_dataset("n_fval", data=res.history.n_fval)
                                grp.create_dataset("n_grad", data=res.history.n_grad)
                                grp.create_dataset("start_time", data=res.history.start_time)
                        
                      buffered_results.clear()

                # Allow immediate check of task completions
                break

        print(f"All {total_tasks} tasks completed.")
