from __future__ import annotations

import json
import os
from typing import ClassVar, Literal

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
    _BenchClassVars,
    _FidelKeys,
)

try:
    from yahpo_gym import benchmark_set, local_config
except ModuleNotFoundError:
    _warn_not_found_module(bench_name="lcbench")


_TARGET_KEYS = _TargetMetricKeys(loss="val_balanced_accuracy", runtime="time")
_BENCH_NAME = "lcbench"
curdir = os.path.dirname(os.path.abspath(__file__))
DATASET_IDS: dict[str, str] = json.load(open(os.path.join(curdir, "lcbench_dataset_ids.json")))
_DATASET_INFO = tuple((name, DATASET_IDS[name]) for name in DATASET_NAMES[_BENCH_NAME])


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
            f"\tAccess to {self._CONSTS.url} and download `{_BENCH_NAME}.zip` from the website.\n\n"
            f"After that, please unzip `{_BENCH_NAME}.zip` in {self._CONSTS.dir}."
        )

    def _check_benchdata_availability(self) -> None:
        super()._check_benchdata_availability()
        local_config.init_config()
        local_config.set_data_path(DATA_DIR_NAME)

    def __call__(self, eval_config: dict[str, int | float], fidel: int) -> ResultType:
        _eval_config: dict[str, int | float | str] = eval_config.copy()  # type: ignore
        _eval_config["OpenML_task_id"] = self._dataset_id
        _eval_config["epoch"] = fidel

        output = self._surrogate.objective_function(_eval_config)[0]
        results: ResultType = {RESULT_KEYS.runtime: float(output[_TARGET_KEYS.runtime])}  # type: ignore
        if RESULT_KEYS.loss in self._target_metrics:
            results[RESULT_KEYS.loss] = float(1.0 - output[_TARGET_KEYS.loss])  # type: ignore

        return results


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
        min_epoch (int):
            The minimum epoch of the training of each neural networks to be used during the optimization.
        max_epoch (int):
            The maximum epoch of the training of each neural networks to be used during the optimization.
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
        max_epoch=54,
        n_datasets=len(_DATASET_INFO),
        target_metric_keys=[k for k, v in _TARGET_KEYS.__dict__.items() if v is not None],
        cont_space=CONT_SPACES[_BENCH_NAME],
        fidel_keys=_FidelKeys(epoch="epoch"),
    )

    # LCBench specific constant
    _TRUE_MAX_EPOCH: ClassVar[int] = 52

    def __init__(
        self,
        dataset_id: int,
        seed: int | None = None,  # surrogate is not stochastic
        target_metrics: list[Literal["loss", "runtime"]] = [RESULT_KEYS.loss],  # type: ignore
        min_epoch: int = 6,
        max_epoch: int = 54,
        keep_benchdata: bool = True,
    ):
        dataset_name, self._dataset_id = _DATASET_INFO[dataset_id]
        super().__init__(
            seed=seed,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            target_metrics=target_metrics[:],  # type: ignore
            dataset_name=dataset_name,
            keep_benchdata=keep_benchdata,
        )

    def get_benchdata(self) -> LCBenchSurrogate:
        return LCBenchSurrogate(dataset_id=self._dataset_id, target_metrics=self._target_metrics)

    def __call__(  # type: ignore[override]
        self,
        eval_config: dict[str, int | float],
        *,
        fidels: dict[str, int] = {},
        seed: int | None = None,
        benchdata: LCBenchSurrogate | None = None,
    ) -> ResultType:
        surrogate = self._validate_benchdata(benchdata)
        assert surrogate is not None and isinstance(surrogate, LCBenchSurrogate)  # mypy redefinition
        fidel = int(min(self._TRUE_MAX_EPOCH, fidels.get(self._CONSTS.fidel_keys.epoch, self._max_epoch)))
        self._validate_config(eval_config=eval_config)  # type: ignore
        return surrogate(eval_config=eval_config, fidel=fidel)
