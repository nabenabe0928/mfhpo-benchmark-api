from __future__ import annotations

import json
import os
from abc import abstractmethod
from typing import ClassVar, Final

import ConfigSpace as CS

from benchmark_apis.abstract_api import AbstractAPI, AbstractHPOData


curdir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_NAME: Final[str] = os.path.join(os.environ["HOME"], "hpo_benchmarks")
SEARCH_SPACE_PATH: Final[str] = os.path.join(curdir, "discrete_search_spaces.json")
VALUE_RANGES: Final[dict[str, dict[str, list[int | float | str | bool]]]] = json.load(open(SEARCH_SPACE_PATH))


class AbstractBench(AbstractAPI):
    _BENCH_TYPE: ClassVar[str] = "HPO"
    _VALUE_RANGE: ClassVar[dict[str, list[int | float | str | bool]] | None] = None
    _MAX_EPOCH: ClassVar[int]
    _N_DATASETS: ClassVar[int]
    _TARGET_METRIC_KEYS: ClassVar[list[str]]

    def __init__(
        self,
        seed: int | None,
        min_epoch: int,
        max_epoch: int,
        target_metrics: list[str],
        dataset_name: str,
        keep_benchdata: bool,
    ):
        super().__init__(seed=seed)
        self._min_epoch = min_epoch
        self._max_epoch = max_epoch
        self._target_metrics = target_metrics[:]
        self._dataset_name = dataset_name
        self._benchdata = self.get_benchdata() if keep_benchdata else None

        self._validate_target_metrics()
        self._validate_epochs()
        self._validate_class_vars()

    @classmethod
    def _validate_class_vars(cls) -> None:
        super()._validate_class_vars()
        for var_name in ["_MAX_EPOCH", "_N_DATASETS", "_TARGET_METRIC_KEYS"]:
            if not hasattr(cls, var_name):
                raise NotImplementedError(f"Child class of {cls.__name__} must define {var_name}.")

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
        if self._VALUE_RANGE is None:
            raise ValueError("_VALUE_RANGE must be specified, but got None.")

        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CS.UniformIntegerHyperparameter(name=name, lower=0, upper=len(choices) - 1)
                if not isinstance(choices[0], (str, bool))
                else CS.CategoricalHyperparameter(name=name, choices=[str(i) for i in range(len(choices))])
                for name, choices in self._VALUE_RANGE.items()
            ]
        )
        return config_space

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @abstractmethod
    def get_benchdata(self) -> AbstractHPOData:
        raise NotImplementedError
