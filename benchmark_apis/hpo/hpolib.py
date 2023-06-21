from __future__ import annotations

import os
import pickle
from typing import ClassVar, TypedDict

from benchmark_apis.abstract_api import AbstractHPOData, RESULT_KEYS, ResultType, _HPODataClassVars, _TargetMetricKeys
from benchmark_apis.hpo.abstract_bench import (
    AbstractBench,
    DATASET_NAMES,
    DATA_DIR_NAME,
    DISC_SPACES,
    FIDEL_SPACES,
    _BenchClassVars,
    _FidelKeys,
)

import numpy as np


_TARGET_KEYS = _TargetMetricKeys(loss="valid_mse", runtime="runtime", model_size="n_params")
_BENCH_NAME = "hpolib"
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
_DATASET_NAMES = DATASET_NAMES[_BENCH_NAME]


class RowDataType(TypedDict):
    valid_mse: list[dict[int, float]]
    runtime: list[float]
    n_params: int


class HPOLibTabular(AbstractHPOData):
    """Workaround to prevent dask from serializing the objective func"""

    _CONSTS = _HPODataClassVars(
        url="http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz",
        dir=os.path.join(DATA_DIR_NAME, _BENCH_NAME),
    )

    def __init__(self, dataset_name: str):
        self._benchdata_path = os.path.join(self._CONSTS.dir, f"{dataset_name}.pkl")
        self._validate()
        self._db = pickle.load(open(self._benchdata_path, "rb"))

    @property
    def install_instruction(self) -> str:
        return (
            f"\033[31m\t$ cd {self._CONSTS.dir}\n"
            f"\t$ wget {self._CONSTS.url}\n"
            "\t$ tar xf fcnet_tabular_benchmarks.tar.gz\n"
            "\t$ mv fcnet_tabular_benchmarks/*.hdf5 .\n"
            "\t$ rm -r fcnet_tabular_benchmarks/\n"
            "\tThen extract the pkl file using https://github.com/nabenabe0928/hpolib-extractor/.\033[0m\n"
            f"You should get `{self._benchdata_path}` in the end."
        )

    def __call__(self, eval_config: dict[str, int | float | str | bool], fidels: dict[str, int | float]) -> ResultType:
        raise NotImplementedError

    def __getitem__(self, key: str) -> RowDataType:  # type: ignore[override]
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
        fidel_value_ranges (dict[str, tuple[int | float, int | float]]):
            The minimum and maximum values for each fidelity values.
            The keys must be the fidelity names used in each benchmark and each tuple takes lower and upper bounds
            of each fidelity value.
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

    _CONSTS = _BenchClassVars(
        dataset_names=_DATASET_NAMES,
        n_datasets=len(_DATASET_NAMES),
        target_metric_keys=[k for k, v in _TARGET_KEYS.__dict__.items() if v is not None],
        disc_space=DISC_SPACES[_BENCH_NAME],
        fidel_space=FIDEL_SPACES[_BENCH_NAME],
        fidel_keys=_FidelKeys(epoch="epoch"),
    )

    # HPOLib specific constant
    _N_SEEDS: ClassVar[int] = 4

    def get_benchdata(self) -> HPOLibTabular:
        return HPOLibTabular(self.dataset_name)

    def __call__(  # type: ignore[override]
        self,
        eval_config: dict[str, int],
        *,
        fidels: dict[str, int] | None = None,
        seed: int | None = None,
        benchdata: HPOLibTabular | None = None,
    ) -> ResultType:
        db = self._validate_benchdata(benchdata)
        assert db is not None and isinstance(db, HPOLibTabular)  # mypy redefinition
        epoch_key = self._CONSTS.fidel_keys.epoch
        fidel = int(self._validate_fidels(fidels)[epoch_key])  # type: ignore[arg-type]
        idx = seed % self._N_SEEDS if seed is not None else self._rng.randint(self._N_SEEDS)
        config_id = "".join([str(eval_config[k]) for k in _KEY_ORDER])

        row: RowDataType = db[config_id]
        runtime = row[_TARGET_KEYS.runtime][idx] * fidel / self._max_fidels[epoch_key]  # type: ignore[literal-required]
        output: ResultType = {RESULT_KEYS.runtime: runtime}  # type: ignore[misc]

        if RESULT_KEYS.loss in self._target_metrics:
            output[RESULT_KEYS.loss] = np.log(row[_TARGET_KEYS.loss][idx][fidel])  # type: ignore[literal-required]
        if RESULT_KEYS.model_size in self._target_metrics:
            output[RESULT_KEYS.model_size] = float(row[_TARGET_KEYS.model_size])  # type: ignore[literal-required]

        return output
