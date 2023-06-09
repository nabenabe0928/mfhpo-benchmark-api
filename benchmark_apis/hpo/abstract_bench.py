from __future__ import annotations

import json
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Optional, TypedDict

import ConfigSpace as CS

import numpy as np


@dataclass(frozen=True)
class _ResultKeys:
    loss: str = "loss"
    runtime: str = "runtime"
    model_size: str = "model_size"
    f1: str = "f1"
    precision: str = "precision"


class ResultType(TypedDict):
    runtime: float
    loss: Optional[float]
    model_size: Optional[float]
    f1: Optional[float]
    precision: Optional[float]


RESULT_KEYS = _ResultKeys()
DATA_DIR_NAME: Final[str] = os.path.join(os.environ["HOME"], "hpo_benchmarks")
SEARCH_SPACE_PATH: Final[str] = "benchmark_apis/hpo/discrete_search_spaces.json"
VALUE_RANGES: Final[dict[str, dict[str, list[int | float | str | bool]]]] = json.load(open(SEARCH_SPACE_PATH))


class AbstractHPOData(metaclass=ABCMeta):
    _data_url: str

    def _check_benchdata_availability(self, benchdata_path: str, additional_info: str) -> None:
        if not os.path.exists(benchdata_path):
            raise FileNotFoundError(
                f"Could not find the dataset at {benchdata_path}.\n"
                f"Download the dataset and place the file at {benchdata_path}.\n"
                "You can download the dataset via:\n"
                f"\t$ wget {self._data_url}\n\n"
                f"{additional_info}"
            )


class AbstractBench(metaclass=ABCMeta):
    _BENCH_TYPE: ClassVar[str] = "HPO"
    _TARGET_METRIC_KEYS: ClassVar[list[str]]
    _N_DATASETS: ClassVar[int]
    _DATASET_NAMES: ClassVar[tuple[str, ...]]
    _MAX_EPOCH: ClassVar[int]
    _min_epoch: int
    _max_epoch: int
    _target_metrics: list[str]
    _value_range: dict[str, list[int | float | str | bool]]
    _rng: np.random.RandomState
    dataset_name: str

    def reseed(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    def _validate_target_metrics(self) -> None:
        target_metrics = self._target_metrics
        if any(tm not in self._TARGET_METRIC_KEYS for tm in target_metrics):
            raise ValueError(
                f"All elements in target_metrics must be in {self._TARGET_METRIC_KEYS}, but got {target_metrics}"
            )

    def _validate_epochs(self) -> None:
        min_epoch, max_epoch = self._min_epoch, self._max_epoch
        if min_epoch <= 0 or max_epoch > self._MAX_EPOCH:
            raise ValueError(f"epoch must be in [1, {self._MAX_EPOCH}], but got {min_epoch=} and {max_epoch=}")
        if min_epoch >= max_epoch:
            raise ValueError(f"min_epoch < max_epoch must hold, but got {min_epoch=} and {max_epoch=}")

    def _fetch_discrete_config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CS.UniformIntegerHyperparameter(name=name, lower=0, upper=len(choices) - 1)
                if not isinstance(choices[0], (str, bool))
                else CS.CategoricalHyperparameter(name=name, choices=[str(i) for i in range(len(choices))])
                for name, choices in self._value_range.items()
            ]
        )
        return config_space

    @abstractmethod
    def get_benchdata(self) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def config_space(self) -> CS.ConfigurationSpace:
        raise NotImplementedError

    @property
    @abstractmethod
    def min_fidels(self) -> dict[str, int | float]:
        # eta ** S <= R/r < eta ** (S + 1) to have S rungs.
        raise NotImplementedError

    @property
    @abstractmethod
    def max_fidels(self) -> dict[str, int | float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def fidel_keys(self) -> list[str]:
        raise NotImplementedError
