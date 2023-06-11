from __future__ import annotations

import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, TypedDict

import ConfigSpace as CS

from benchmark_apis.abstract_api import AbstractAPI, AbstractHPOData


@dataclass(frozen=True)
class _FidelKeys:
    epoch: str
    resol: str | None = None


class _ContinuousSpaceParams(TypedDict):
    type_: Literal["int", "float"]
    lower: int | float
    upper: int | float
    log: bool


@dataclass(frozen=True)
class _BenchClassVars:
    max_epoch: int
    n_datasets: int
    target_metric_keys: list[str]
    fidel_keys: _FidelKeys
    disc_space: dict[str, list[int | float | str | bool]] | None = None
    cont_space: dict[str, _ContinuousSpaceParams] | None = None


curdir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_NAME: Final[str] = os.path.join(os.environ["HOME"], "hpo_benchmarks")
DATASET_NAME_PATH: Final[str] = os.path.join(curdir, "dataset_names.json")
CONT_SPACE_PATH: Final[str] = os.path.join(curdir, "continuous_search_spaces.json")
DISC_SPACE_PATH: Final[str] = os.path.join(curdir, "discrete_search_spaces.json")
CONT_SPACES: Final[dict[str, dict[str, _ContinuousSpaceParams]]] = json.load(open(CONT_SPACE_PATH))
DISC_SPACES: Final[dict[str, dict[str, list[int | float | str | bool]]]] = json.load(open(DISC_SPACE_PATH))
DATASET_NAMES: Final[dict[str, list[str]]] = json.load(open(DATASET_NAME_PATH))


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
        self._config_space = self.config_space

        self._validate_target_metrics()
        self._validate_epochs()
        self._validate_class_vars()

    @abstractmethod
    def get_benchdata(self) -> AbstractHPOData:
        raise NotImplementedError

    @classmethod
    def _validate_class_vars(cls) -> None:
        super()._validate_class_vars()
        if not hasattr(cls, "_CONSTS"):
            raise NotImplementedError(f"Child class of {cls.__name__} must define _CONSTS.")

    def _validate_config(self, eval_config: dict[str, int | float | str | bool]) -> None:
        EPS = 1e-12
        for hp in self._config_space.get_hyperparameters():
            name, val = hp.name, eval_config[hp.name]
            if isinstance(hp, CS.CategoricalHyperparameter):
                if val not in hp.choices:
                    raise ValueError(f"{name} must be in {hp.choices}, but got {val}.")

                continue

            lb, ub = hp.lower, hp.upper
            if isinstance(hp, CS.UniformFloatHyperparameter):
                ok = isinstance(val, float) and lb - EPS <= val <= ub + EPS
            else:
                eval_config[name] = int(val)
                ok = lb <= eval_config[name] <= ub

            if not ok:
                raise ValueError(f"{name} must be in [{lb=}, {ub=}], but got {eval_config[name]}.")

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

    def _fetch_continuous_hyperparameters(self) -> list[CS.hyperparameters.Hyperparameter]:
        hyperparameters: list[CS.hyperparameters.Hyperparameter] = []
        if self._CONSTS.cont_space is None:
            return hyperparameters

        for name, params in self._CONSTS.cont_space.items():
            kwargs = dict(name=name, **params)
            type_ = kwargs.pop("type_")
            if type_ == "int":
                hp = CS.UniformIntegerHyperparameter(**kwargs)
            elif type_ == "float":
                hp = CS.UniformFloatHyperparameter(**kwargs)
            else:
                raise TypeError(f"type_ of continuous space must be `int` or `float`, but got {type_}")

            hyperparameters.append(hp)

        return hyperparameters

    def _fetch_discrete_hyperparameters(self) -> list[CS.hyperparameters.Hyperparameter]:
        config_space = CS.ConfigurationSpace()
        if self._CONSTS.disc_space is None:
            return config_space

        return [
            CS.UniformIntegerHyperparameter(name=name, lower=0, upper=len(choices) - 1)
            if not isinstance(choices[0], (str, bool))
            else CS.CategoricalHyperparameter(name=name, choices=[str(i) for i in range(len(choices))])
            for name, choices in self._CONSTS.disc_space.items()
        ]

    @property
    def dataset_name_for_dir(self) -> str | None:
        return "-".join(self.dataset_name.split("_"))

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

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

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(self._fetch_discrete_hyperparameters())
        config_space.add_hyperparameters(self._fetch_continuous_hyperparameters())
        return config_space
