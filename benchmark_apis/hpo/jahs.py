from __future__ import annotations

import os
from typing import Any

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
    DISC_SPACES,
    FIDEL_SPACES,
    _BenchClassVars,
    _FidelKeys,
)

try:
    import jahs_bench
except ModuleNotFoundError:  # pragma: no cover
    # We cannot use jahs with smac
    _warn_not_found_module(bench_name="jahs")


_TARGET_KEYS = _TargetMetricKeys(loss="valid-acc", runtime="runtime", model_size="size_MB")
_BENCH_NAME = "jahs"
_DATASET_NAMES = DATASET_NAMES[_BENCH_NAME]


class JAHSBenchSurrogate(AbstractHPOData):
    """Workaround to prevent dask from serializing the objective func"""

    _CONSTS = _HPODataClassVars(
        url="https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar",
        dir=os.path.join(DATA_DIR_NAME, _BENCH_NAME),
    )

    def __init__(self, dataset_name: str, target_metrics: list[str]):
        self._validate()
        self._target_metrics = target_metrics[:]
        _metrics = [getattr(_TARGET_KEYS, tm) for tm in self._target_metrics]
        metrics = list(set(_metrics + [_TARGET_KEYS.runtime]))
        self._surrogate = jahs_bench.Benchmark(
            task=dataset_name, download=False, save_dir=self._CONSTS.dir, metrics=metrics
        )

    @property
    def install_instruction(self) -> str:
        return (
            f"\033[31m\t$ cd {self._CONSTS.dir}\n"
            f"\t$ wget {self._CONSTS.url}\n"
            f"\tThen untar `assembled_surrogates.tar` in {self._CONSTS.dir}.\033[0m"
        )

    def __call__(self, eval_config: dict[str, int | float | str | bool], fidels: dict[str, int | float]) -> ResultType:
        _fidels = fidels.copy()
        nepochs = _fidels.pop("epoch")

        eval_config.update({"Optimizer": "SGD", **_fidels})  # type: ignore[arg-type]
        eval_config = {k: int(v) if k[:-1] == "Op" else v for k, v in eval_config.items()}
        output = self._surrogate(eval_config, nepochs=nepochs)[nepochs]
        results: ResultType = {RESULT_KEYS.runtime: output[_TARGET_KEYS.runtime]}  # type: ignore[misc]

        if RESULT_KEYS.loss in self._target_metrics:
            results[RESULT_KEYS.loss] = float(100 - output[_TARGET_KEYS.loss])  # type: ignore[literal-required]
        if RESULT_KEYS.model_size in self._target_metrics:
            results[RESULT_KEYS.model_size] = float(output[_TARGET_KEYS.model_size])  # type: ignore[literal-required]

        return results

    def __getitem__(self, key: str) -> dict[str, Any]:
        raise NotImplementedError


class JAHSBench201(AbstractBench):
    """The class for JAHS-Bench-201.

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
        Title: JAHS-Bench-201: A Foundation For Research On Joint Architecture And Hyperparameter Search
        Authors: A. Bansal et al.
        URL: https://openreview.net/forum?id=_HLcjaVlqJ

    NOTE:
        The data is available at:
            https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar
    """

    _CONSTS = _BenchClassVars(
        dataset_names=_DATASET_NAMES,
        n_datasets=len(_DATASET_NAMES),
        target_metric_keys=[k for k, v in _TARGET_KEYS.__dict__.items() if v is not None],
        cont_space=CONT_SPACES[_BENCH_NAME],
        disc_space=DISC_SPACES[_BENCH_NAME],
        fidel_space=FIDEL_SPACES[_BENCH_NAME],
        fidel_keys=_FidelKeys(epoch="epoch", resol="Resolution"),
    )

    def get_benchdata(self) -> JAHSBenchSurrogate:
        return JAHSBenchSurrogate(dataset_name=self.dataset_name, target_metrics=self._target_metrics)

    def __call__(  # type: ignore[override]
        self,
        eval_config: dict[str, int | float | str | bool],
        *,
        fidels: dict[str, int | float] | None = None,
        seed: int | None = None,
        benchdata: JAHSBenchSurrogate | None = None,
    ) -> ResultType:
        surrogate = self._validate_benchdata(benchdata)
        assert surrogate is not None and isinstance(surrogate, JAHSBenchSurrogate)  # mypy redefinition

        _eval_config, _fidels = self._validate_inputs(eval_config, fidels)
        assert self._CONSTS.disc_space is not None  # mypy redefinition
        _eval_config = {
            k: self._CONSTS.disc_space[k][int(v)] if k in self._CONSTS.disc_space else float(v)
            for k, v in _eval_config.items()
        }
        return surrogate(eval_config=_eval_config, fidels=_fidels)
