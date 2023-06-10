from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import ClassVar, Literal, TypedDict

import ConfigSpace as CS

from benchmark_apis.abstract_bench import (
    AbstractBench,
    AbstractHPOData,
    DATA_DIR_NAME,
    RESULT_KEYS,
    ResultType,
    VALUE_RANGES,
)

import numpy as np


@dataclass(frozen=True)
class _TargetMetricKeys:
    loss: str = "bal_acc"
    runtime: str = "runtime"
    precision: str = "precision"
    f1: str = "f1"


_TARGET_KEYS = _TargetMetricKeys()
_FIDEL_KEY = "epoch"
_KEY_ORDER = ["alpha", "batch_size", "depth", "learning_rate_init", "width"]


class RowDataType(TypedDict):
    loss: list[dict[int, float]]
    runtime: list[dict[int, float]]
    precision: list[dict[int, float]]
    f1: list[dict[int, float]]


class HPOBenchDatabase(AbstractHPOData):
    """Workaround to prevent dask from serializing the objective func"""

    _data_url = "https://ndownloader.figshare.com/files/30379005/"

    def __init__(self, dataset_name: str):
        benchdata_path = os.path.join(DATA_DIR_NAME, "hpobench", f"{dataset_name}.pkl")
        additional_info = (
            "\t$ unzip nn.zip\n\n" "Then extract the pkl file using https://github.com/nabenabe0928/hpolib-extractor/."
        )
        self._check_benchdata_availability(benchdata_path, additional_info=additional_info)
        self._db = pickle.load(open(benchdata_path, "rb"))

    def __getitem__(self, key: str) -> RowDataType:
        return self._db[key]


class HPOBench(AbstractBench):
    """The class for HPOlib.

    Args:
        dataset_id (int):
            The ID of the dataset.
        seed (int | None):
            The random seed to be used.
        target_metrics (list[Literal["loss", "runtime", "f1", "precision"]]):
            The target metrics to return.
            Must be in ["loss", "runtime", "f1", "precision"].
        min_epoch (int):
            The minimum epoch of the training of each neural networks to be used during the optimization.
        max_epoch (int):
            The maximum epoch of the training of each neural networks to be used during the optimization.
        keep_benchdata (bool):
            Whether to keep the benchmark data in each instance.
            When True, serialization will happen in case of parallel optimization.

    References:
        Title: HPOBench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO
        Authors: K. Eggensperger et al.
        URL: https://arxiv.org/abs/2109.06716

    NOTE:
        Download the datasets via:
            $ wget https://ndownloader.figshare.com/files/30379005/
            $ unzip nn.zip

        Use https://github.com/nabenabe0928/hpolib-extractor to extract the pickle file.
    """

    _N_DATASETS: ClassVar[int] = 8
    _N_SEEDS: ClassVar[int] = 5
    _MAX_EPOCH: ClassVar[int] = 243
    _BUDGETS: ClassVar[list[int]] = [3, 9, 27, 81, 243]
    _TARGET_METRIC_KEYS: ClassVar[list[str]] = [k for k in _TARGET_KEYS.__dict__.keys()]

    def __init__(
        self,
        dataset_id: int,
        seed: int | None = None,
        target_metrics: list[Literal["loss", "runtime", "f1", "precision"]] = [RESULT_KEYS.loss],  # type: ignore
        min_epoch: int = 27,
        max_epoch: int = 243,
        keep_benchdata: bool = True,
    ):
        self.dataset_name = [
            "australian",
            "blood_transfusion",
            "car",
            "credit_g",
            "kc1",
            "phoneme",
            "segment",
            "vehicle",
        ][dataset_id]
        self._db = self.get_benchdata() if keep_benchdata else None
        self._rng = np.random.RandomState(seed)
        self._value_range = VALUE_RANGES["hpobench"]
        self._min_epoch, self._max_epoch = min_epoch, max_epoch
        self._target_metrics = target_metrics[:]  # type: ignore

        self._validate_target_metrics()
        self._validate_epochs()

    def get_benchdata(self) -> HPOBenchDatabase:
        return HPOBenchDatabase(self.dataset_name)

    def __call__(
        self,
        eval_config: dict[str, int | str],
        *,
        fidels: dict[str, int] = {},
        seed: int | None = None,
        benchdata: HPOBenchDatabase | None = None,
    ) -> ResultType:
        fidel = int(fidels.get(_FIDEL_KEY, self._max_epoch))
        if fidel not in self._BUDGETS:
            raise ValueError(f"fidel for {self.__class__.__name__} must be in {self._BUDGETS}, but got {fidel}")
        if benchdata is None and self._db is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        db = benchdata if self._db is None else self._db
        assert db is not None  # mypy redefinition
        idx = seed % self._N_SEEDS if seed is not None else self._rng.randint(self._N_SEEDS)
        config_id = "".join([str(eval_config[k]) for k in _KEY_ORDER])

        row: RowDataType = db[config_id]
        runtime = row[_TARGET_KEYS.runtime][idx][fidel]  # type: ignore
        output: ResultType = {RESULT_KEYS.runtime: runtime}  # type: ignore

        if RESULT_KEYS.loss in self._target_metrics:
            output[RESULT_KEYS.loss] = 1.0 - row[_TARGET_KEYS.loss][idx][fidel]  # type: ignore
        if RESULT_KEYS.f1 in self._target_metrics:
            output[RESULT_KEYS.f1] = float(row[_TARGET_KEYS.f1][idx][fidel])  # type: ignore
        if RESULT_KEYS.precision in self._target_metrics:
            output[RESULT_KEYS.precision] = float(row[_TARGET_KEYS.precision][idx][fidel])  # type: ignore

        return output

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._fetch_discrete_config_space()

    @property
    def min_fidels(self) -> dict[str, int | float]:
        return {_FIDEL_KEY: self._min_epoch}

    @property
    def max_fidels(self) -> dict[str, int | float]:
        return {_FIDEL_KEY: self._max_epoch}

    @property
    def fidel_keys(self) -> list[str]:
        return [_FIDEL_KEY]
