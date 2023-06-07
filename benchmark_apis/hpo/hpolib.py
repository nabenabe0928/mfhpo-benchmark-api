from __future__ import annotations

import json
import os
import pickle
from enum import Enum
from typing import ClassVar, TypedDict

import ConfigSpace as CS

from benchmark_apis.hpo.abstract_bench import AbstractBench, DATA_DIR_NAME, VALUE_RANGES

import numpy as np


class RowDataType(TypedDict):
    valid_mse: list[dict[int, float]]
    runtime: list[float]
    n_params: list[int]


class _TargetMetricKeys(Enum):
    loss: str = "loss"
    runtime: str = "runtime"
    model_size: str = "n_params"


class HPOLibDatabase:
    """Workaround to prevent dask from serializing the objective func"""

    def __init__(self, dataset_name: str):
        benchdata_path = os.path.join(DATA_DIR_NAME, "hpolib", f"{dataset_name}.pkl")
        self._check_benchdata_availability(benchdata_path)
        self._db = pickle.load(open(benchdata_path, "rb"))

    def _check_benchdata_availability(self, benchdata_path: str) -> None:
        if not os.path.exists(benchdata_path):
            raise FileNotFoundError(
                f"Could not find the dataset at {benchdata_path}.\n"
                f"Download the dataset and place the file at {benchdata_path}.\n"
                "You can download the dataset via:\n"
                "\t$ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz\n"
                "\t$ tar xf fcnet_tabular_benchmarks.tar.gz\n\n"
                "Then extract the pkl file using https://github.com/nabenabe0928/hpolib-extractor/."
            )

    def __getitem__(self, key: str) -> RowDataType:
        return self._db[key]


class HPOLib(AbstractBench):
    """
    Download the datasets via:
        $ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
        $ tar xf fcnet_tabular_benchmarks.tar.gz

    Use https://github.com/nabenabe0928/hpolib-extractor to extract the pickle file.
    """

    _N_DATASETS: ClassVar[int] = 4
    _MAX_EPOCH: ClassVar[int] = 100
    _FIDEL_KEYS: ClassVar[list[str]] = ["epoch"]
    _TARGET_METRIC_KEYS: ClassVar[list[str]] = [k.name for k in _TargetMetricKeys]
    _DATASET_NAMES: ClassVar[tuple[str, ...]] = (
        "slice-localization",
        "protein-structure",
        "naval-propulsion",
        "parkinsons-telemonitoring",
    )

    def __init__(
        self,
        dataset_id: int,
        seed: int | None = None,
        target_metrics: list[str] = [_TargetMetricKeys.loss.name],
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
        self._target_metrics = target_metrics[:]

        self._validate_target_metrics(target_metrics)
        self._validate_epochs(min_epoch=min_epoch, max_epoch=max_epoch)

    def get_benchdata(self) -> HPOLibDatabase:
        return HPOLibDatabase(self.dataset_name)

    def __call__(
        self,
        eval_config: dict[str, int | str],
        *,
        fidels: dict[str, int] = {},
        seed: int | None = None,
        benchdata: HPOLibDatabase | None = None,
    ) -> dict[str, float]:
        if benchdata is None and self._db is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        db = benchdata if self._db is None else self._db
        assert db is not None  # mypy redefinition
        fidel = int(fidels.get(self._FIDEL_KEYS[0], self._max_epoch))
        idx = seed % 4 if seed is not None else self._rng.randint(4)
        key = json.dumps({k: self._value_range[k][int(v)] for k, v in eval_config.items()}, sort_keys=True)

        row: RowDataType = db[key]
        output: dict[str, float] = dict(runtime=row["runtime"][idx] * fidel / self.max_fidels[self._FIDEL_KEYS[0]])
        if _TargetMetricKeys.loss.name in self._target_metrics:
            output["loss"] = np.log(row["valid_mse"][idx][fidel - 1])
        if _TargetMetricKeys.model_size.name in self._target_metrics:
            output["model_size"] = float(row["n_params"][idx])

        return output

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._fetch_discrete_config_space()

    @property
    def min_fidels(self) -> dict[str, int | float]:
        return {self._FIDEL_KEYS[0]: self._min_epoch}

    @property
    def max_fidels(self) -> dict[str, int | float]:
        return {self._FIDEL_KEYS[0]: self._max_epoch}

    @property
    def fidel_keys(self) -> list[str]:
        return self._FIDEL_KEYS[:]
