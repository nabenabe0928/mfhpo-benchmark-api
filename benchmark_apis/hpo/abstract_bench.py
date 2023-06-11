from __future__ import annotations

import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Final

import ConfigSpace as CS

from benchmark_apis.abstract_api import AbstractAPI, AbstractHPOData


@dataclass(frozen=True)
class _FidelKeys:
    epoch: str
    resol: str | None = None


@dataclass(frozen=True)
class _BenchClassVars:
    max_epoch: int
    n_datasets: int
    target_metric_keys: list[str]
    fidel_keys: _FidelKeys
    value_range: dict[str, list[int | float | str | bool]] | None = None


curdir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_NAME: Final[str] = os.path.join(os.environ["HOME"], "hpo_benchmarks")
SEARCH_SPACE_PATH: Final[str] = os.path.join(curdir, "discrete_search_spaces.json")
VALUE_RANGES: Final[dict[str, dict[str, list[int | float | str | bool]]]] = json.load(open(SEARCH_SPACE_PATH))


class AbstractBench(AbstractAPI):
    _BENCH_TYPE: ClassVar[str] = "HPO"
    _CONSTS: _BenchClassVars

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
        if not hasattr(cls, "_CONSTS"):
            raise NotImplementedError(f"Child class of {cls.__name__} must define _CONSTS.")

    def _validate_target_metrics(self) -> None:
        target_metrics = self._target_metrics
        if any(tm not in self._CONSTS.target_metric_keys for tm in target_metrics):
            raise ValueError(
                f"All elements in target_metrics must be in {self._CONSTS.target_metric_keys}, but got {target_metrics}"
            )

    def _validate_epochs(self) -> None:
        min_epoch, max_epoch = self._min_epoch, self._max_epoch
        if min_epoch <= 0 or max_epoch > self._CONSTS.max_epoch:
            raise ValueError(f"epoch must be in [1, {self._CONSTS.max_epoch}], but got {min_epoch=} and {max_epoch=}")
        if min_epoch >= max_epoch:
            raise ValueError(f"min_epoch < max_epoch must hold, but got {min_epoch=} and {max_epoch=}")

    def _validate_benchdata(self, benchdata: AbstractHPOData | None) -> AbstractHPOData:
        if benchdata is None and self._benchdata is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        ret = benchdata if self._benchdata is None else self._benchdata
        assert ret is not None  # mypy redefinition
        return ret

    def _fetch_discrete_config_space(self) -> CS.ConfigurationSpace:
        if self._CONSTS.value_range is None:
            raise ValueError("_VALUE_RANGE must be specified, but got None.")

        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CS.UniformIntegerHyperparameter(name=name, lower=0, upper=len(choices) - 1)
                if not isinstance(choices[0], (str, bool))
                else CS.CategoricalHyperparameter(name=name, choices=[str(i) for i in range(len(choices))])
                for name, choices in self._CONSTS.value_range.items()
            ]
        )
        return config_space

    @property
    def dataset_name_for_dir(self) -> str | None:
        return "-".join(self.dataset_name.split("_"))

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @abstractmethod
    def get_benchdata(self) -> AbstractHPOData:
        raise NotImplementedError

    @property
    def min_fidels(self) -> dict[str, int | float]:
        # eta ** S <= R/r < eta ** (S + 1) to have S rungs.
        return {v: getattr(self, f"_min_{k}") for k, v in self._CONSTS.fidel_keys.__dict__.items() if v is not None}

    @property
    def max_fidels(self) -> dict[str, int | float]:
        return {v: getattr(self, f"_max_{k}") for k, v in self._CONSTS.fidel_keys.__dict__.items() if v is not None}

    @property
    def fidel_keys(self) -> list[str]:
        return [k for k in self._CONSTS.fidel_keys.__dict__.values() if k is not None]
