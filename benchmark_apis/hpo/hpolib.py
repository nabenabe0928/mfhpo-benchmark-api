from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import ClassVar, Literal, TypedDict

import ConfigSpace as CS

from benchmark_apis.abstract_api import AbstractHPOData, RESULT_KEYS, ResultType
from benchmark_apis.hpo.abstract_bench import AbstractBench, DATA_DIR_NAME, VALUE_RANGES

import numpy as np


@dataclass(frozen=True)
class _TargetMetricKeys:
    loss: str = "valid_mse"
    runtime: str = "runtime"
    model_size: str = "n_params"


_TARGET_KEYS = _TargetMetricKeys()
_FIDEL_KEY = "epoch"  # this is not class var, because we wanna use dataclass for multiple fidelities
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
_DATASET_NAMES = (
    "slice_localization",
    "protein_structure",
    "naval_propulsion",
    "parkinsons_telemonitoring",
)


class RowDataType(TypedDict):
    valid_mse: list[dict[int, float]]
    runtime: list[float]
    n_params: int


class HPOLibDatabase(AbstractHPOData):
    """Workaround to prevent dask from serializing the objective func"""

    _data_url = "http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz"
    _data_dir = os.path.join(DATA_DIR_NAME, "hpolib")

    def __init__(self, dataset_name: str):
        self._benchdata_path = os.path.join(self._data_dir, f"{dataset_name}.pkl")
        self._validate()
        self._db = pickle.load(open(self._benchdata_path, "rb"))

    @property
    def install_instruction(self) -> str:
        return (
            f"\t$ cd {self._data_dir}\n"
            f"\t$ wget {self._data_url}\n"
            "\t$ tar xf fcnet_tabular_benchmarks.tar.gz\n"
            "\t$ mv fcnet_tabular_benchmarks/*.hdf5 .\n"
            "\t$ rm -r fcnet_tabular_benchmarks/\n\n"
            "Then extract the pkl file using https://github.com/nabenabe0928/hpolib-extractor/.\n"
            f"You should get `{self._benchdata_path}` in the end."
        )

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

    _MAX_EPOCH: ClassVar[int] = 100
    _N_DATASETS: ClassVar[int] = 4
    _TARGET_METRIC_KEYS: ClassVar[list[str]] = [k for k in _TARGET_KEYS.__dict__.keys()]
    _VALUE_RANGE: ClassVar[dict[str, list[int | float | str | bool]]] = VALUE_RANGES["hpolib"]
    _DATASET_NAMES_FOR_DIR: ClassVar[tuple[str, ...]] = tuple("-".join(name.split("_")) for name in _DATASET_NAMES)

    # HPOLib specific constant
    _N_SEEDS: ClassVar[int] = 4

    def __init__(
        self,
        dataset_id: int,
        seed: int | None = None,
        target_metrics: list[Literal["loss", "runtime", "model_size"]] = [RESULT_KEYS.loss],  # type: ignore
        min_epoch: int = 11,
        max_epoch: int = 100,
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

    def get_benchdata(self) -> HPOLibDatabase:
        return HPOLibDatabase(self.dataset_name)

    def __call__(  # type: ignore[override]
        self,
        eval_config: dict[str, int],
        *,
        fidels: dict[str, int] = {},
        seed: int | None = None,
        benchdata: HPOLibDatabase | None = None,
    ) -> ResultType:
        if benchdata is None and self._benchdata is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        db = benchdata if self._benchdata is None else self._benchdata
        assert db is not None and isinstance(db, HPOLibDatabase)  # mypy redefinition
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
    def min_fidels(self) -> dict[str, int]:  # type: ignore[override]
        return {_FIDEL_KEY: self._min_epoch}

    @property
    def max_fidels(self) -> dict[str, int]:  # type: ignore[override]
        return {_FIDEL_KEY: self._max_epoch}

    @property
    def fidel_keys(self) -> list[str]:
        return [_FIDEL_KEY]
