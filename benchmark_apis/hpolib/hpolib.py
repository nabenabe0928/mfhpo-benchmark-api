from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import ClassVar, Literal, TypedDict

import ConfigSpace as CS

from benchmark_apis.hpo.abstract_bench import (
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
    loss: str = "valid_mse"
    runtime: str = "runtime"
    model_size: str = "n_params"


_TARGET_KEYS = _TargetMetricKeys()
_FIDEL_KEY = "epoch"
_KEY_ORDER = [
    "activation_fn_1",
    "activation_fn_2",
    "batch_size",
    "dropout_1",
    "dropout_2",
    "init_lr",
    "lr_schedule",
    "n_units_1",
    "n_units_2",
]


class RowDataType(TypedDict):
    valid_mse: list[dict[int, float]]
    runtime: list[float]
    n_params: int


class HPOLibDatabase(AbstractHPOData):
    """Workaround to prevent dask from serializing the objective func"""

    _data_url = "http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz"

    def __init__(self, dataset_name: str):
        benchdata_path = os.path.join(DATA_DIR_NAME, "hpolib", f"{dataset_name}.pkl")
        additional_info = (
            "\t$ tar xf fcnet_tabular_benchmarks.tar.gz\n\n"
            "Then extract the pkl file using https://github.com/nabenabe0928/hpolib-extractor/."
        )
        self._check_benchdata_availability(benchdata_path, additional_info=additional_info)
        self._db = pickle.load(open(benchdata_path, "rb"))

    def __getitem__(self, key: str) -> RowDataType:
        return self._db[key]


class HPOLib(AbstractBench):
    """The class for HPOlib.

    Args:
        dataset_id (int):
            The ID of the dataset.
        seed (int | None):
            The random seed to be used.
        target_metrics (list[Literal["loss", "runtime", "model_size"]]):
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
        Title: Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization
        Authors: A. Klein and F. Hutter
        URL: https://arxiv.org/abs/1905.04970

    NOTE:
        Download the datasets via:
            $ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
            $ tar xf fcnet_tabular_benchmarks.tar.gz

        Use https://github.com/nabenabe0928/hpolib-extractor to extract the pickle file.
    """

    _N_DATASETS: ClassVar[int] = 4
    _N_SEEDS: ClassVar[int] = 4
    _MAX_EPOCH: ClassVar[int] = 100
    _TARGET_METRIC_KEYS: ClassVar[list[str]] = [k for k in _TARGET_KEYS.__dict__.keys()]

    def __init__(
        self,
        dataset_id: int,
        seed: int | None = None,
        target_metrics: list[Literal["loss", "runtime", "model_size"]] = [RESULT_KEYS.loss],  # type: ignore
        min_epoch: int = 11,
        max_epoch: int = 100,
        keep_benchdata: bool = True,
    ):
        self.dataset_name = [
            "slice_localization",
            "protein_structure",
            "naval_propulsion",
            "parkinsons_telemonitoring",
        ][dataset_id]
        self._db = self.get_benchdata() if keep_benchdata else None
        self._rng = np.random.RandomState(seed)
        self._value_range = VALUE_RANGES["hpolib"]
        self._min_epoch, self._max_epoch = min_epoch, max_epoch
        self._target_metrics = target_metrics[:]  # type: ignore

        self._validate_target_metrics()
        self._validate_epochs()

    def get_benchdata(self) -> HPOLibDatabase:
        return HPOLibDatabase(self.dataset_name)

    def __call__(
        self,
        eval_config: dict[str, int | str],
        *,
        fidels: dict[str, int] = {},
        seed: int | None = None,
        benchdata: HPOLibDatabase | None = None,
    ) -> ResultType:
        if benchdata is None and self._db is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        db = benchdata if self._db is None else self._db
        assert db is not None  # mypy redefinition
        fidel = int(fidels.get(_FIDEL_KEY, self._max_epoch))
        idx = seed % self._N_SEEDS if seed is not None else self._rng.randint(self._N_SEEDS)
        config_id = "".join([str(eval_config[k]) for k in _KEY_ORDER])

        row: RowDataType = db[config_id]
        full_runtime = row[_TARGET_KEYS.runtime][idx]  # type: ignore
        output: ResultType = {RESULT_KEYS.runtime: full_runtime * fidel / self.max_fidels[_FIDEL_KEY]}  # type: ignore

        if RESULT_KEYS.loss in self._target_metrics:
            output[RESULT_KEYS.loss] = np.log(row[_TARGET_KEYS.loss][idx][fidel])  # type: ignore
        if RESULT_KEYS.model_size in self._target_metrics:
            output[RESULT_KEYS.model_size] = float(row[_TARGET_KEYS.model_size])  # type: ignore

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
