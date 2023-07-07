from __future__ import annotations

import json
import os
from typing import Any, ClassVar

from benchmark_apis.abstract_api import (
    AbstractHPOData,
    RESULT_KEYS,
    ResultType,
    _HPODataClassVars,
    _TargetMetricKeys,
    _warn_not_found_module,
)
from benchmark_apis.hpo.abstract_bench import (
    AbstractBench,
    CONT_SPACES,
    DATASET_NAMES,
    DATA_DIR_NAME,
    FIDEL_SPACES,
    _BenchClassVars,
    _FidelKeys,
)

try:
    from yahpo_gym import benchmark_set, local_config
except ModuleNotFoundError:  # pragma: no cover
    _warn_not_found_module(bench_name="lcbench")


_TARGET_KEYS = _TargetMetricKeys(loss="val_balanced_accuracy", runtime="time")
_BENCH_NAME = "lcbench"
curdir = os.path.dirname(os.path.abspath(__file__))
_DATASET_NAMES = DATASET_NAMES[_BENCH_NAME]
DATASET_IDS: dict[str, str] = json.load(open(os.path.join(curdir, "lcbench_dataset_ids.json")))
_DATASET_INFO = tuple((name, DATASET_IDS[name]) for name in _DATASET_NAMES)


class LCBenchSurrogate(AbstractHPOData):
    """Workaround to prevent dask from serializing the objective func"""

    _CONSTS = _HPODataClassVars(
        url="https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/",
        dir=os.path.join(DATA_DIR_NAME, _BENCH_NAME),
    )

    def __init__(self, dataset_id: str, target_metrics: list[str]):
        self._validate()
        self._dataset_id = dataset_id
        self._target_metrics = target_metrics[:]
        # active_session=False is necessary for parallel computing.
        self._surrogate = benchmark_set.BenchmarkSet(_BENCH_NAME, instance=dataset_id, active_session=False)

    @property
    def install_instruction(self) -> str:
        return (
            f"\033[31m\tAccess to {self._CONSTS.url} and download `{_BENCH_NAME}.zip` from the website.\n"
            f"\tAfter that, please unzip `{_BENCH_NAME}.zip` in {self._CONSTS.dir}.\033[0m\n"
            f"Note that you need to simply tick `{_BENCH_NAME}` and click `Download`."
        )

    def _check_benchdata_availability(self) -> None:
        super()._check_benchdata_availability()
        local_config.init_config()
        local_config.set_data_path(DATA_DIR_NAME)

    def __call__(  # type: ignore[override]
        self, eval_config: dict[str, int | float], fidels: dict[str, int]
    ) -> ResultType:
        _eval_config: dict[str, int | float | str] = eval_config.copy()  # type: ignore[assignment]
        _eval_config["OpenML_task_id"] = self._dataset_id
        epoch_key = "epoch"
        _eval_config[epoch_key] = fidels[epoch_key]

        output = self._surrogate.objective_function(_eval_config)[0]
        results: ResultType = {RESULT_KEYS.runtime: float(output[_TARGET_KEYS.runtime])}  # type: ignore[misc]
        if RESULT_KEYS.loss in self._target_metrics:
            results[RESULT_KEYS.loss] = float(1.0 - output[_TARGET_KEYS.loss])  # type: ignore[literal-required]

        return results

    def __getitem__(self, key: str) -> dict[str, Any]:
        raise NotImplementedError


class LCBench(AbstractBench):
    """The class for LCBench.

    Args:
        dataset_id (int):
            The ID of the dataset.
        seed (int | None):
            The random seed to be used.
        target_metrics (list[str]):
            The target metrics to return.
            Must be in ["loss", "runtime", "model_size"].
        fidel_value_ranges (dict[str, tuple[int | float, int | float]]):
            The minimum and maximum values for each fidelity values.
            The keys must be the fidelity names used in each benchmark and each tuple takes lower and upper bounds
            of each fidelity value.
        keep_benchdata (bool):
            Whether to keep the benchmark data in each instance.
            When True, serialization will happen in case of parallel optimization.

    References:
        1. The original benchmark
        Title: Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL
        Authors: L. Zimmer et al.
        URL: https://arxiv.org/abs/2006.13799/

        2. The proposition of the surrogate model
        Title: YAHPO Gym -- An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization
        Authors: F. Pfisterer et al.
        URL: https://arxiv.org/abs/2109.03670/

    NOTE:
        The data is available at:
            https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/
    """

    _CONSTS = _BenchClassVars(
        dataset_names=_DATASET_NAMES,
        n_datasets=len(_DATASET_NAMES),
        target_metric_keys=[k for k, v in _TARGET_KEYS.__dict__.items() if v is not None],
        cont_space=CONT_SPACES[_BENCH_NAME],
        fidel_space=FIDEL_SPACES[_BENCH_NAME],
        fidel_keys=_FidelKeys(epoch="epoch"),
    )

    # LCBench specific constant
    _TRUE_MAX_EPOCH: ClassVar[int] = 52

    def get_benchdata(self) -> LCBenchSurrogate:
        _, dataset_id = _DATASET_INFO[self._dataset_id]
        return LCBenchSurrogate(dataset_id=dataset_id, target_metrics=self._target_metrics)

    def __call__(  # type: ignore[override]
        self,
        eval_config: dict[str, int | float],
        *,
        fidels: dict[str, int] | None = None,
        seed: int | None = None,
        benchdata: LCBenchSurrogate | None = None,
    ) -> ResultType:
        surrogate = self._validate_benchdata(benchdata)
        assert surrogate is not None and isinstance(surrogate, LCBenchSurrogate)  # mypy redefinition

        epoch_key = self._CONSTS.fidel_keys.epoch
        _eval_config, _fidels = self._validate_inputs(eval_config=eval_config, fidels=fidels)  # type: ignore[arg-type]
        _fidels[epoch_key] = int(min(self._TRUE_MAX_EPOCH, _fidels[epoch_key]))
        return surrogate(eval_config=_eval_config, fidels=_fidels)  # type: ignore[arg-type]
