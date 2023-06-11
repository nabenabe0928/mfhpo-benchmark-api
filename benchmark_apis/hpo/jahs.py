from __future__ import annotations

import os
from typing import Literal

import ConfigSpace as CS

from benchmark_apis.abstract_api import (
    AbstractHPOData,
    RESULT_KEYS,
    ResultType,
    _TargetMetricKeys,
    _warn_not_found_module,
)
from benchmark_apis.hpo.abstract_bench import AbstractBench, DATA_DIR_NAME, VALUE_RANGES, _BenchClassVars, _FidelKeys

try:
    import jahs_bench
except ModuleNotFoundError:  # We cannot use jahs with smac
    _warn_not_found_module(bench_name="jahs")


_TARGET_KEYS = _TargetMetricKeys(
    loss="valid-acc",
    runtime="runtime",
    model_size="size_MB",
)
_DATASET_NAMES = ("cifar10", "fashion_mnist", "colorectal_histology")


class JAHSBenchSurrogate(AbstractHPOData):
    """Workaround to prevent dask from serializing the objective func"""

    _data_url = (
        "https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar"
    )
    _data_dir = os.path.join(DATA_DIR_NAME, "jahs")

    def __init__(self, dataset_name: str, target_metrics: list[str]):
        self._validate()
        self._target_metrics = target_metrics[:]
        _metrics = [getattr(_TARGET_KEYS, tm) for tm in self._target_metrics]
        metrics = list(set(_metrics + [_TARGET_KEYS.runtime]))
        self._surrogate = jahs_bench.Benchmark(
            task=dataset_name, download=False, save_dir=self._data_dir, metrics=metrics
        )

    @property
    def install_instruction(self) -> str:
        return (
            f"$ cd {self._data_dir}\n"
            f"$ wget {self._data_url}\n\n"
            f"Then untar `assembled_surrogates.tar` in {self._data_dir}."
        )

    def __call__(self, eval_config: dict[str, int | float | str | bool], fidels: dict[str, int | float]) -> ResultType:
        _fidels = fidels.copy()
        nepochs = _fidels.pop("epoch")

        eval_config.update({"Optimizer": "SGD", **_fidels})  # type: ignore
        eval_config = {k: int(v) if k[:-1] == "Op" else v for k, v in eval_config.items()}
        output = self._surrogate(eval_config, nepochs=nepochs)[nepochs]
        results: ResultType = {RESULT_KEYS.runtime: output[_TARGET_KEYS.runtime]}  # type: ignore

        if RESULT_KEYS.loss in self._target_metrics:
            results[RESULT_KEYS.loss] = float(100 - output[_TARGET_KEYS.loss])  # type: ignore
        if RESULT_KEYS.model_size in self._target_metrics:
            results[RESULT_KEYS.model_size] = float(output[_TARGET_KEYS.model_size])  # type: ignore

        return results


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
        min_epoch (int):
            The minimum epoch of the training of each neural networks to be used during the optimization.
        max_epoch (int):
            The maximum epoch of the training of each neural networks to be used during the optimization.
        min_resol (float):
            The minimum resolution of image data for the training of each neural networks.
        max_resol (float):
            The maximum resolution of image data for the training of each neural networks.
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
        max_epoch=200,
        n_datasets=len(_DATASET_NAMES),
        target_metric_keys=[k for k, v in _TARGET_KEYS.__dict__.items() if v is not None],
        value_range=VALUE_RANGES["jahs"],
        fidel_keys=_FidelKeys(epoch="epoch", resol="Resolution"),
    )

    def __init__(
        self,
        dataset_id: int,
        seed: int | None = None,  # surrogate is not stochastic
        target_metrics: list[Literal["loss", "runtime", "model_size"]] = [RESULT_KEYS.loss],  # type: ignore
        min_epoch: int = 22,
        max_epoch: int = 200,
        min_resol: float = 0.1,
        max_resol: float = 1.0,
        keep_benchdata: bool = True,
    ):
        super().__init__(
            seed=seed,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            target_metrics=target_metrics[:],  # type: ignore
            dataset_name=_DATASET_NAMES[dataset_id],
            keep_benchdata=keep_benchdata,
        )

        self._min_resol, self._max_resol = min_resol, max_resol
        self._validate_resols()

    def _validate_resols(self) -> None:
        min_resol, max_resol = self._min_resol, self._max_resol
        if min_resol <= 0 or max_resol > 1.0:
            raise ValueError(f"Resolution must be in [0.0, 1.0], but got {min_resol=} and {max_resol=}")
        if min_resol >= max_resol:
            raise ValueError(f"min_resol < max_resol must hold, but got {min_resol=} and {max_resol=}")

    def get_benchdata(self) -> JAHSBenchSurrogate:
        return JAHSBenchSurrogate(dataset_name=self.dataset_name, target_metrics=self._target_metrics)

    def __call__(  # type: ignore[override]
        self,
        eval_config: dict[str, int | float | str | bool],
        *,
        fidels: dict[str, int | float] = {},
        seed: int | None = None,
        benchdata: JAHSBenchSurrogate | None = None,
    ) -> ResultType:
        surrogate = self._validate_benchdata(benchdata)
        assert surrogate is not None and isinstance(surrogate, JAHSBenchSurrogate)  # mypy redefinition

        _fidels = self.max_fidels
        _fidels.update(**fidels)
        assert self._CONSTS.value_range is not None  # mypy redefinition
        EPS = 1e-12
        _eval_config = {
            k: self._CONSTS.value_range[k][int(v)] if k in self._CONSTS.value_range else float(v)
            for k, v in eval_config.items()
        }
        assert isinstance(_eval_config["LearningRate"], float)
        assert 1e-3 - EPS <= _eval_config["LearningRate"] <= 1.0 + EPS
        assert isinstance(_eval_config["WeightDecay"], float)
        assert 1e-5 - EPS <= _eval_config["WeightDecay"] <= 1e-2 + EPS
        return surrogate(eval_config=_eval_config, fidels=_fidels)

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = self._fetch_discrete_config_space()
        config_space.add_hyperparameters(
            [
                CS.UniformFloatHyperparameter(name="LearningRate", lower=1e-3, upper=1.0, log=True),
                CS.UniformFloatHyperparameter(name="WeightDecay", lower=1e-5, upper=1e-2, log=True),
            ]
        )
        return config_space
