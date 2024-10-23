import logging

import numpy as np

import pypesto.optimize

from ..engine import Task
from ..history import HistoryOptions
from ..problem import Problem
from ..result import OptimizerResult

logger = logging.getLogger(__name__)


class OptimizerTask(Task):
    """A multistart optimization task, performed in `pypesto.minimize`."""

    def __init__(
        self,
        optimizer: "pypesto.optimize.Optimizer",
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions,
        optimize_options: "pypesto.optimize.OptimizeOptions",
    ):
        """Create the task object.

        Parameters
        ----------
        optimizer:
            The optimizer to use.
        problem:
            The problem to solve.
        x0:
            The point from which to start.
        id:
            The multistart id.
        options:
            Options object applying to optimization.
        history_options:
            Optimizer history options.
        """
        super().__init__()

        self.optimizer = optimizer
        self.problem = problem
        self.x0 = x0
        self.id = id
        self.optimize_options = optimize_options
        self.history_options = history_options

    def execute(self) -> OptimizerResult:
        """Execute the task."""
        logger.debug(f"Executing task {self.id}.")
        # check for supplied x_guess support
        self.optimizer.check_x0_support(self.problem.x_guesses)

        # init fval stored and create a thread specific h5 file for results
        init_fval = self.problem.objective.get_fval(self.x0)
        idp = MPI.COMM_WORLD.Get_rank()
        run_file = self.optimize_options.filename_path + f'_id{idp}.h5'$

        if not os.path.isfile(run_file):
             with h5py.File(run_file, 'a') as f:
                 grpd = f.create_group(f'global_data')
                 grpd.attrs['n_runs'] = 0
                 grpd.attrs['init_timestamp'] = datetime.timestamp(datetime.now()) 
                 grpd.attrs['best_fx'] = float('inf')
                 grpd.attrs['counter'] = 0
                 grpd.attrs['totaltime'] = 0
             f.close()

        with h5py.File(run_file, 'a') as f:
             if float(init_fval) < f[f'global_data'].attrs['best_fx']:
                 counter = f[f'global_data'].attrs['counter']
                 f[f'global_data'].attrs['counter'] += 1  
                 f[f'global_data'].attrs['best_fx'] = float(init_fval)
                 grp = f.create_group(f'{counter}')
                 grp.attrs['time'] = datetime.timestamp(datetime.now()) - f[f'global_data'].attrs['init_timestamp']
                 grp.attrs['fval'] = init_fval
                 grp.attrs['x'] = self.x0
 
             f[f'global_data'].attrs['n_runs'] += 1
             f[f'global_data'].attrs['totaltime'] = datetime.timestamp(datetime.now()) - f[f'global_data'].attrs['init_timestamp']

        optimizer_result = self.optimizer.minimize(
            problem=self.problem,
            x0=self.x0,
            id=self.id,
            history_options=self.history_options,
            optimize_options=self.optimize_options,
        )
        optimizer_result.optimizer = str(self.optimizer)

        idp = MPI.COMM_WORLD.Get_rank()
        run_file = self.optimize_options.filename_path + f'_id{idp}.h5'

        with h5py.File(run_file, 'a') as f:
            if float(optimizer_result.fval) < f[f'global_data'].attrs['best_fx']:
                 counter = f[f'global_data'].attrs['counter']
                 f[f'global_data'].attrs['counter'] += 1
                 f[f'global_data'].attrs['best_fx'] = float(optimizer_result.fval)
                 grp = f.create_group(f'{counter}')
                 grp.attrs['time'] = datetime.timestamp(datetime.now()) - f[f'global_data'].attrs['init_timestamp']
                 grp.attrs['fval'] = optimizer_result.fval
                 grp.attrs['x'] = optimizer_result.x

            f[f'global_data'].attrs['n_runs'] += 1
            f[f'global_data'].attrs['totaltime'] = datetime.timestamp(datetime.now()) - f[f'global_data'].attrs['init_timestamp']


        if not self.optimize_options.report_hess:
            optimizer_result.hess = None
        if not self.optimize_options.report_sres:
            optimizer_result.sres = None
        return optimizer_result
