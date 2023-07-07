from __future__ import annotations

import os
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, TypedDict

import ConfigSpace as CS

import numpy as np


def _warn_not_found_module(bench_name: str) -> None:  # pragma: no cover
    cmd = "pip install mfhpo-benchmark-api"
    warnings.warn(
        f"{bench_name} requirements not found. Use `{cmd}[full]` or `{cmd}[{bench_name}]` when using {bench_name}."
    )


@dataclass(frozen=True)
class _ResultKeys:
    loss: str = "loss"
    runtime: str = "runtime"
    model_size: str = "model_size"
    f1: str = "f1"
    precision: str = "precision"


@dataclass(frozen=True)
class _TargetMetricKeys:
    runtime: str
    loss: str | None = None
    model_size: str | None = None
    f1: str | None = None
    precision: str | None = None


@dataclass(frozen=True)
class _HPODataClassVars:
    url: str
    dir: str


class ResultType(TypedDict):
    runtime: float
    loss: Optional[float]
    model_size: Optional[float]
    f1: Optional[float]
    precision: Optional[float]


RESULT_KEYS = _ResultKeys()


class AbstractHPOData(metaclass=ABCMeta):
    _CONSTS: _HPODataClassVars

    @abstractmethod
    def __call__(self, eval_config: dict[str, int | float | str | bool], fidels: dict[str, int | float]) -> ResultType:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: str) -> dict[str, Any]:
        raise NotImplementedError

    def _validate(self) -> None:
        self._validate_class_var()
        self._check_benchdata_availability()

    @property
    @abstractmethod
    def install_instruction(self) -> str:
        raise NotImplementedError

    @classmethod
    def _validate_class_var(cls) -> None:
        if not hasattr(cls, "_CONSTS"):
            raise NotImplementedError(f"Child class of {cls.__name__} must define _CONSTS.")

    @property
    def full_install_instruction(self) -> str:
        return (
            f"Could not find the dataset at {self._CONSTS.dir}.\n"
            f"Download the dataset and place the file at {self._CONSTS.dir}.\n"
            "You can download the dataset via:\n"
            f"{self.install_instruction}"
        )

    def _check_benchdata_availability(self) -> None:
        if not os.path.exists(self._CONSTS.dir):
            raise FileNotFoundError(self.full_install_instruction)


class AbstractAPI(metaclass=ABCMeta):
    _BENCH_TYPE: ClassVar[str]

    def __init__(self, seed: int | None):
        self._rng = np.random.RandomState(seed)

    @abstractmethod
    def __call__(
        self,
        eval_config: dict[str, int | float | str | bool],
        *,
        fidels: dict[str, int | float] | None = None,
        seed: int | None = None,
        benchdata: AbstractHPOData | None = None,
    ) -> ResultType:
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_name_for_dir(self) -> str | None:
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

    def reseed(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    @classmethod
    def _validate_class_vars(cls) -> None:
        if not hasattr(cls, "_BENCH_TYPE"):
            raise NotImplementedError(f"Child class of {cls.__name__} must define _BENCH_TYPE.")
