from __future__ import annotations

import os
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional, TypedDict

import ConfigSpace as CS

import numpy as np


def _warn_not_found_module(bench_name: str) -> None:
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


class ResultType(TypedDict):
    runtime: float
    loss: Optional[float]
    model_size: Optional[float]
    f1: Optional[float]
    precision: Optional[float]


RESULT_KEYS = _ResultKeys()


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


class AbstractInterface(metaclass=ABCMeta):
    _BENCH_TYPE: ClassVar[str]

    def __init__(self, seed: int | None):
        self._rng = np.random.RandomState(seed)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    @abstractmethod
    def __call__(
        self,
        eval_config: dict[str, int | float | str | bool],
        *,
        fidels: dict[str, int | float] = {},
        seed: int | None = None,
        benchdata: AbstractHPOData | None = None,
    ) -> ResultType:
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
