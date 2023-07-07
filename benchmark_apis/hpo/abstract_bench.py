from __future__ import annotations

import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, TypedDict

import ConfigSpace as CS

from benchmark_apis.abstract_api import AbstractAPI, AbstractHPOData, RESULT_KEYS

import numpy as np


@dataclass(frozen=True)
class _FidelKeys:
    epoch: str
    resol: str | None = None


@dataclass(frozen=True)
class _ValueRange:
    lower: int | float
    upper: int | float


@dataclass(frozen=True)
class _ValueRanges:
    hard: _ValueRange
    default: _ValueRange


@dataclass(frozen=True)
class _FidelValueRanges:
    epoch: _ValueRanges
    resol: _ValueRanges | None = None


class _ContinuousSpaceParams(TypedDict):
    type_: Literal["int", "float"]
    lower: int | float
    upper: int | float
    log: bool


@dataclass(frozen=True)
class _BenchClassVars:
    dataset_names: list[str]
    n_datasets: int
    target_metric_keys: list[str]
    fidel_keys: _FidelKeys
    fidel_space: _FidelValueRanges
    disc_space: dict[str, list[int | float | str | bool]] | None = None
    cont_space: dict[str, _ContinuousSpaceParams] | None = None


curdir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_NAME: Final[str] = os.path.join(os.environ["HOME"], "hpo_benchmarks")
DATASET_NAME_PATH: Final[str] = os.path.join(curdir, "dataset_names.json")
CONT_SPACE_PATH: Final[str] = os.path.join(curdir, "continuous_search_spaces.json")
DISC_SPACE_PATH: Final[str] = os.path.join(curdir, "discrete_search_spaces.json")
FIDEL_SPACE_PATH: Final[str] = os.path.join(curdir, "fidel_spaces.json")
CONT_SPACES: Final[dict[str, dict[str, _ContinuousSpaceParams]]] = json.load(open(CONT_SPACE_PATH))
DISC_SPACES: Final[dict[str, dict[str, list[int | float | str | bool]]]] = json.load(open(DISC_SPACE_PATH))
FIDEL_SPACES: Final[dict[str, _FidelValueRanges]] = {
    bench_name: _FidelValueRanges(
        **{
            fidel: _ValueRanges(**{type_: _ValueRange(**range_) for type_, range_ in ranges.items()})
            for fidel, ranges in fidels.items()
        }
    )
    for bench_name, fidels in json.load(open(FIDEL_SPACE_PATH)).items()
}
DATASET_NAMES: Final[dict[str, list[str]]] = json.load(open(DATASET_NAME_PATH))


class AbstractBench(AbstractAPI):
    _BENCH_TYPE: ClassVar[str] = "HPO"
    _CONSTS: _BenchClassVars

    def __init__(
        self,
        dataset_id: int,
        seed: int | None = None,
        fidel_value_ranges: dict[str, tuple[int | float, int | float]] | None = None,
        target_metrics: list[str] | None = None,
        keep_benchdata: bool = True,
    ):
        super().__init__(seed=seed)
        self._target_metrics = target_metrics[:] if target_metrics is not None else [RESULT_KEYS.loss]
        self._dataset_id = dataset_id
        self._benchdata = self.get_benchdata() if keep_benchdata else None
        self._config_space = self.config_space
        self._fidel_keys = self.fidel_keys
        self._fidel_value_ranges: dict[str, _ValueRange]
        self._validate_fidel_value_ranges(
            fidel_value_ranges=fidel_value_ranges if fidel_value_ranges is not None else {}
        )

        self._min_fidels = self.min_fidels
        self._max_fidels = self.max_fidels

        self._validate_target_metrics()
        self._validate_class_vars()

    @abstractmethod
    def get_benchdata(self) -> AbstractHPOData:
        raise NotImplementedError

    def _validate_fidel_value_ranges(self, fidel_value_ranges: dict[str, tuple[int | float, int | float]]) -> None:
        if any(fidel_key not in self._fidel_keys for fidel_key in fidel_value_ranges):
            fidel_keys = list(fidel_value_ranges.keys())
            raise KeyError(f"Keys in fidel_value_ranges must be in {self._fidel_keys}, but got {fidel_keys}")

        self._fidel_value_ranges = {}
        for fidel_key, value_ranges in self._CONSTS.fidel_space.__dict__.items():
            if value_ranges is None:
                continue

            user_side_fidel_key = getattr(self._CONSTS.fidel_keys, fidel_key)
            hard_lower, hard_upper = value_ranges.hard.lower, value_ranges.hard.upper
            lower, upper = fidel_value_ranges.get(
                user_side_fidel_key, (value_ranges.default.lower, value_ranges.default.upper)
            )
            self._fidel_value_ranges[user_side_fidel_key] = _ValueRange(lower=lower, upper=upper)
            if lower < hard_lower or upper > hard_upper:
                raise ValueError(
                    f"{user_side_fidel_key} must be in [{hard_lower}, {hard_upper}], but got {lower=} and {upper=}"
                )
            if lower >= upper:
                raise ValueError(
                    f"lower < upper for {user_side_fidel_key} must hold, "
                    f"but got {lower=} and {upper=} in fidel_value_ranges"
                )

    @classmethod
    def _validate_class_vars(cls) -> None:
        super()._validate_class_vars()
        if not hasattr(cls, "_CONSTS"):
            raise NotImplementedError(f"Child class of {cls.__name__} must define _CONSTS.")

    def _validate_inputs(
        self, eval_config: dict[str, int | float | str | bool], fidels: dict[str, int | float] | None
    ) -> tuple[dict[str, int | float | str | bool], dict[str, int | float]]:
        _eval_config = self._validate_config(eval_config=eval_config.copy())
        _fidels = self._validate_fidels(fidels=fidels)
        return _eval_config, _fidels

    def _validate_config(self, eval_config: dict[str, int | float | str | bool]) -> dict[str, int | float | str | bool]:
        EPS = 1e-12
        for hp in self._config_space.get_hyperparameters():
            name, val = hp.name, eval_config[hp.name]
            if isinstance(hp, CS.CategoricalHyperparameter):
                if val not in hp.choices:
                    raise ValueError(f"{name} must be in {hp.choices}, but got {val}.")

                continue  # pragma: no cover

            lb, ub = hp.lower, hp.upper
            if isinstance(hp, CS.UniformFloatHyperparameter):
                ok = isinstance(val, float) and lb - EPS <= val <= ub + EPS
                eval_config[name] = float(np.clip(val, lb + EPS, ub - EPS))
            else:
                eval_config[name] = int(val)
                ok = lb <= eval_config[name] <= ub

            if not ok:
                raise ValueError(f"{name} must be in [{lb=}, {ub=}], but got {eval_config[name]}.")

        return eval_config

    def _validate_fidels(self, fidels: dict[str, int | float] | None) -> dict[str, int | float]:
        _fidels = fidels.copy() if fidels is not None else {}
        for fidel_key, value_range in self._fidel_value_ranges.items():
            lower, upper = value_range.lower, value_range.upper
            if fidel_key not in _fidels:
                _fidels[fidel_key] = self._max_fidels[fidel_key]
                continue

            fidel_val = _fidels[fidel_key]
            if not (lower <= fidel_val <= upper):
                raise ValueError(f"{fidel_key} must be in [{lower}, {upper}], but got {fidel_val}.")

        return _fidels

    def _validate_target_metrics(self) -> None:
        target_metrics = self._target_metrics
        if any(tm not in self._CONSTS.target_metric_keys for tm in target_metrics):
            raise ValueError(
                f"All elements in target_metrics must be in {self._CONSTS.target_metric_keys}, but got {target_metrics}"
            )

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
            else:  # pragma: no cover
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
        return self._CONSTS.dataset_names[self._dataset_id]

    @property
    def min_fidels(self) -> dict[str, int | float]:
        # eta ** S <= R/r < eta ** (S + 1) to have S rungs.
        return {k: r.lower for k, r in self._fidel_value_ranges.items()}

    @property
    def max_fidels(self) -> dict[str, int | float]:
        return {k: r.upper for k, r in self._fidel_value_ranges.items()}

    @property
    def fidel_keys(self) -> list[str]:
        return [k for k in self._CONSTS.fidel_keys.__dict__.values() if k is not None]

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(self._fetch_discrete_hyperparameters())
        config_space.add_hyperparameters(self._fetch_continuous_hyperparameters())
        return config_space
