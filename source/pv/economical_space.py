import csv
import dataclasses
import importlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable

from source.indicators.models import NPVConfig, neeg, npv, roi
from source.pv.creation import PVSizingFilePathBuilder
from source.pv.model import PVConfig
from source.pv.sizing import EconomicalSizingStrategy, NEEGSizingStrategy
from source.pv.statistics import CandidateSpaceConfig, PVStatisticsLoader
from source.utils import TimeSpaceHandler

import numpy as np


# Module-level worker state for parallel evaluation (set by initializer)
_worker_context = None


def _init_worker(context: "_WorkerContext") -> None:
    """Set global context in worker process (called once per worker)."""
    global _worker_context
    _worker_context = context


def _evaluate_one_worker(
    job: tuple[int, float, float, float],
) -> tuple["EconomicalResult", float]:
    """
    Run one economical-space iteration in a worker process.
    Uses _worker_context set by _init_worker.
    """
    import time as time_mod
    (house_id,
     initial_cost_per_kW,
     cost_per_kW,
     feed_in_price_per_kW) = job
    ctx = _worker_context
    if ctx is None:
        raise RuntimeError("Worker context not initialized")

    mod = importlib.import_module(ctx.indicator_module)
    indicator_function = getattr(mod, ctx.indicator_name)

    start_time = time_mod.time()

    config = NPVConfig(
        initial_cost_per_kW=initial_cost_per_kW,
        cost_per_kW=cost_per_kW,
        feed_in_price_per_kW=feed_in_price_per_kW,
    )

    # Per-job config so the correct house context is used (no shared mutation)
    job_config = replace(
        ctx.candidate_space_config,
        house_id=house_id,
    )

    if indicator_function is neeg:
        sizing_strategy = NEEGSizingStrategy(
            time_space_handler=ctx.time_space_handler,
            candidate_space_config=job_config,
            statistics=ctx.statistics)
    else:
        sizing_strategy = EconomicalSizingStrategy(
            indicator_function=indicator_function,
            time_space_handler=ctx.time_space_handler,
            candidate_space_config=job_config,
            npv_config=config,
            statistics=ctx.statistics,
            verbose=ctx.verbose)
    optimal_pv_config, _ = sizing_strategy.run()

    stats = ctx.statistics[
        (
            house_id,
            optimal_pv_config.panel_type,
            optimal_pv_config.mount_type,
            optimal_pv_config.number_of_panels,
        )
    ]

    annual_production_kwh = stats.annual_production_kwh
    sc_value = stats.self_consumption

    npv_value, capex, payback_period, annual_profit = npv(
        config, optimal_pv_config, annual_production_kwh, sc_value
    )
    roi_value = npv_value / capex if capex != 0 else 0.0
    neeg_value = stats.neeg_value

    result = EconomicalResult(
        house_id=house_id,
        self_consumption=sc_value,
        neeg=neeg_value,
        surface_area_m2=stats.surface_area_m2,
        max_surface_area_m2=job_config.max_surface_area_m2,
        peak_power_kW=stats.peak_power_kW,
        optimal_pv_config=optimal_pv_config,
        npv_config=config,
        roi=roi_value,
        npv=npv_value,
        capex=capex,
        payback_period=payback_period,
        annual_profit=annual_profit,
    )

    iteration_time = time_mod.time() - start_time
    return result, iteration_time


@dataclass
class EconomicalSpace:
    """
    Economical space of all possible configurations of
    the economical indicators.
    """
    initial_cost_per_kW_bounds: tuple[float, float]
    initial_cost_increment: float
    cost_per_kW_bounds: tuple[float, float]
    cost_per_kW_increment: float
    feed_in_price_per_kW_bounds: tuple[float, float]
    feed_in_price_per_kW_increment: float
    house_ids: list[int]
    indicator_function: Callable
    n_iterations_to_save: int = 2


@dataclass
class EconomicalResult:
    """
    Result of the economical space evaluation.
    """
    house_id: int
    self_consumption: float
    neeg: float
    surface_area_m2: float
    max_surface_area_m2: float
    peak_power_kW: float
    optimal_pv_config: PVConfig
    npv_config: NPVConfig
    roi: float
    npv: float
    capex: float
    payback_period: int
    annual_profit: float

    def get_names(self) -> list[str]:
        """
        Get the names of the result,
        in the order of the fields.
        """
        result = []
        for field in dataclasses.fields(EconomicalResult):
            val = getattr(self, field.name)
            if isinstance(val, (int, float)):
                result.append(field.name)
            else:
                result.extend(val.get_names())
        return result

    def to_list(self) -> list[Any]:
        """
        Convert the result to a list of strings.
        """
        result = []
        for field in dataclasses.fields(EconomicalResult):
            val = getattr(self, field.name)
            if isinstance(val, (int, float)):
                result.append(val)
            else:
                result.extend(val.to_list())
        return result


@dataclass
class _WorkerContext:
    """Picklable context for parallel worker processes."""

    time_space_handler: TimeSpaceHandler
    candidate_space_config: CandidateSpaceConfig
    statistics: dict
    indicator_module: str
    indicator_name: str
    verbose: bool


class EconomicalSpaceEvaluator:
    """
    Evaluator of the economical space.
    """

    def __init__(self,
                 time_space_handler: TimeSpaceHandler,
                 economical_space: EconomicalSpace,
                 verbose: bool = False,
                 n_workers: int | None = None):
        # n_workers > 1 runs evaluations in parallel (default 1 = sequential)
        self.economical_space = economical_space
        self.candidate_space_config = CandidateSpaceConfig(
            house_ids=self.economical_space.house_ids)
        self.statistics = PVStatisticsLoader().load_statistics_from_csv()
        self.time_space_handler = time_space_handler

        self._eval_count: int | None = None
        self._eval_index = 0
        self._verbose = verbose
        indicator_name = self.economical_space.indicator_function.__name__
        self._csv_path = \
            PVSizingFilePathBuilder().get_economical_space_csv_path(
                indicator_name)
        self._header_written = False
        self._iteration_times = []
        self._n_workers = n_workers if n_workers is not None else 1

    def get_eval_count(self, triples: list[tuple[float, float, float]]) -> int:
        """
        Get the number of evaluations.
        """
        n_houses = len(self.candidate_space_config.house_ids)
        return len(triples) * n_houses

    def _build_parameter_triples(self) -> list[tuple[float, float, float]]:
        """Build list of (initial_cost_per_kW, cost_per_kW,
        feed_in_price_per_kW)."""
        triples = []
        for initial_cost_per_kW in np.arange(
                self.economical_space.initial_cost_per_kW_bounds[0],
                self.economical_space.initial_cost_per_kW_bounds[1],
                self.economical_space.initial_cost_increment):
            for cost_per_kW in np.arange(
                    self.economical_space.cost_per_kW_bounds[0],
                    self.economical_space.cost_per_kW_bounds[1],
                    self.economical_space.cost_per_kW_increment):
                for feed_in_price_per_kW in np.arange(
                        self.economical_space.feed_in_price_per_kW_bounds[0],
                        self.economical_space.feed_in_price_per_kW_bounds[1],
                        self.economical_space.feed_in_price_per_kW_increment):

                    # Skip if feed-in price is greater than cost per kW
                    # as this is not a valid configuration
                    if feed_in_price_per_kW >= cost_per_kW:
                        continue

                    triples.append(
                        (initial_cost_per_kW, cost_per_kW,
                         feed_in_price_per_kW))
        return triples

    def _save_results_batch(self, batch: list, write_header: bool):
        """
        Append a batch of results to the CSV file.
        If write_header is True, open in 'w' and write header first.
        """
        # Determine if we need to write header (first time or forced)
        should_write_header = write_header or not self._header_written

        mode = 'w' if should_write_header else 'a'
        with open(self._csv_path, mode, newline='') as f:
            writer = csv.writer(f)
            if should_write_header and batch:
                writer.writerow(batch[0].get_names())
                self._header_written = True
            for result in batch:
                writer.writerow(result.to_list())

    def _evaluate_one_iteration(self,
                                house_id: int,
                                initial_cost_per_kW: float,
                                cost_per_kW: float,
                                feed_in_price_per_kW: float) -> \
            tuple[EconomicalResult, float]:
        """
        Evaluate one iteration of the economical space.

        Args:
            initial_cost_per_kW: Initial cost per kW
            cost_per_kW: Cost per kW
            feed_in_price_per_kW: Feed-in price per kW

        Returns:
            Tuple of (EconomicalResult, iteration_time_seconds)
        """
        start_time = time.time()

        self.candidate_space_config.house_id = house_id

        config = NPVConfig(
            initial_cost_per_kW=initial_cost_per_kW,
            cost_per_kW=cost_per_kW,
            feed_in_price_per_kW=feed_in_price_per_kW)

        ind_function = self.economical_space.indicator_function
        if ind_function is neeg:
            sizing_strategy = NEEGSizingStrategy(
                time_space_handler=self.time_space_handler,
                candidate_space_config=self.candidate_space_config,
                statistics=self.statistics)
        else:
            sizing_strategy = EconomicalSizingStrategy(
                indicator_function=self.economical_space.indicator_function,
                time_space_handler=self.time_space_handler,
                candidate_space_config=self.candidate_space_config,
                npv_config=config,
                statistics=self.statistics,
                verbose=self._verbose)

        print(f"Evaluating space for house {house_id}...")
        optimal_pv_config, optimal_output = sizing_strategy.run()

        stats = self.statistics[
            (house_id,
             optimal_pv_config.panel_type,
             optimal_pv_config.mount_type,
             optimal_pv_config.number_of_panels)]

        # Calculate NPV and ROI for the optimal PV config
        annual_production_kwh = stats.annual_production_kwh

        sc_value = stats.self_consumption

        npv_value, capex, payback_period, annual_profit = npv(
            config, optimal_pv_config, annual_production_kwh, sc_value)

        roi_value = npv_value / capex if capex != 0 else 0.0

        neeg_value = stats.neeg_value

        result = EconomicalResult(
            house_id=house_id,
            self_consumption=sc_value,
            neeg=neeg_value,
            optimal_pv_config=optimal_pv_config,
            surface_area_m2=stats.surface_area_m2,
            max_surface_area_m2=self.candidate_space_config.max_surface_area_m2,
            peak_power_kW=stats.peak_power_kW,
            npv_config=config,
            roi=roi_value,
            npv=npv_value,
            capex=capex,
            payback_period=payback_period,
            annual_profit=annual_profit)

        iteration_time = time.time() - start_time

        return result, iteration_time

    def evaluate(self):
        self.results = []
        self._header_written = False
        self._iteration_times = []
        self._eval_index = 0

        triplets = self._build_parameter_triples()
        self._eval_count = self.get_eval_count(triplets)

        print(f"Evaluating {self._eval_count} configurations...")

        if self._n_workers <= 1:
            self._evaluate_sequential(triplets)
        else:
            self._evaluate_parallel(triplets)

        n_iterations_to_save = self.economical_space.n_iterations_to_save
        # Save any remaining results (last partial batch)
        remainder_start = (self._eval_index //
                           n_iterations_to_save) * n_iterations_to_save
        if remainder_start < len(self.results):
            batch = self.results[remainder_start:]
            write_header = (remainder_start == 0)
            self._save_results_batch(batch, write_header=write_header)

        if self._iteration_times:
            self._print_overall_statistics()

        return self.results

    def _evaluate_sequential(self, triplets: list[tuple[float, float, float]]
                             ) -> None:
        """Run evaluations one by one (original loop)."""
        print("0.00%")
        if self._eval_count is None:
            raise ValueError("Eval count is not set")

        for house_id in self.candidate_space_config.house_ids:
            for initial_cost_per_kW, cost_per_kW, feed_in_price_per_kW in triplets:
                result, iteration_time = self._evaluate_one_iteration(
                    house_id=house_id,
                    initial_cost_per_kW=initial_cost_per_kW,
                    cost_per_kW=cost_per_kW,
                    feed_in_price_per_kW=feed_in_price_per_kW)

                self._iteration_times.append(iteration_time)
                self._eval_index += 1
                pct = self._eval_index / self._eval_count * 100
                avg_time = sum(self._iteration_times) / len(
                    self._iteration_times)
                print(f"{pct:.2f}% - "
                      f"Iteration: {iteration_time:.2f}s, "
                      f"Avg: {avg_time:.2f}s")
                self.results.append(result)

                n_iterations_to_save = \
                    self.economical_space.n_iterations_to_save
                if self._eval_index % n_iterations_to_save == 0:
                    batch = self.results[
                        self._eval_index - n_iterations_to_save:
                        self._eval_index]
                    write_header = (self._eval_index ==
                                    n_iterations_to_save)
                    self._save_results_batch(
                        batch, write_header=write_header)

    def _evaluate_parallel(self, triplets: list[tuple[float, float, float]]
                           ) -> None:
        """Run evaluations in parallel with ProcessPoolExecutor."""

        ind_function = self.economical_space.indicator_function
        indicator_module = getattr(ind_function, '__module__', '')
        indicator_name = getattr(ind_function, '__name__', '')
        if not indicator_module or not indicator_name:
            raise ValueError(
                "Parallel mode requires indicator_function to have "
                "__module__ and __name__ (e.g. use a top-level function).")

        context = _WorkerContext(
            time_space_handler=self.time_space_handler,
            candidate_space_config=self.candidate_space_config,
            statistics=self.statistics,
            indicator_module=indicator_module,
            indicator_name=indicator_name,
            verbose=self._verbose,
        )

        jobs = [
            (house_id,
             initial_cost_per_kW,
             cost_per_kW,
             feed_in_price_per_kW)
            for house_id in self.candidate_space_config.house_ids
            for (initial_cost_per_kW,
                 cost_per_kW,
                 feed_in_price_per_kW) in triplets
        ]

        total_start = time.time()
        n_workers = min(self._n_workers, len(jobs), (os.cpu_count() or 1))
        print(f"Using {n_workers} workers...")

        with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(context,),
        ) as executor:
            collected = list(executor.map(_evaluate_one_worker, jobs))

        total_elapsed = time.time() - total_start

        self.results = [r for r, _ in collected]
        self._iteration_times = [t for _, t in collected]
        self._eval_index = len(self.results)

        # Write results in batches (same as sequential)
        n_iterations_to_save = self.economical_space.n_iterations_to_save
        for i in range(0, len(self.results), n_iterations_to_save):
            batch = self.results[i:i + n_iterations_to_save]
            write_header = (i == 0)
            self._save_results_batch(batch, write_header=write_header)

        print(f"Done in {total_elapsed:.2f}s (wall clock).")

    def _print_overall_statistics(self):
        """
        Print the overall statistics of the evaluation.
        """
        if self._eval_count is None:
            raise ValueError("Eval count is not set")

        total_time = sum(self._iteration_times)
        avg_time = total_time / len(self._iteration_times)
        min_time = min(self._iteration_times)
        max_time = max(self._iteration_times)
        print(f"Total time: {total_time:.2f}s")
        print(f"Average iteration time: {avg_time:.2f}s")
        print(f"Min iteration time: {min_time:.2f}s")
        print(f"Max iteration time: {max_time:.2f}s")
        print(f"Estimated remaining (if same avg): "
              f"{avg_time * (self._eval_count - self._eval_index):.2f}s")

    def save_results(self, indicator_name: str):
        file_path = PVSizingFilePathBuilder().get_economical_space_csv_path(
            indicator_name)
        with open(file_path, 'w') as f:
            writer = csv.writer(f)

            first_result = self.results[0]

            writer.writerow(first_result.get_names())
            for result in self.results:
                writer.writerow(result.to_list())
