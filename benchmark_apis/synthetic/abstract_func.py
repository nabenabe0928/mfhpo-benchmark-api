from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import ConfigSpace as CS

from benchmark_apis.abstract_api import AbstractAPI, RESULT_KEYS, ResultType

import numpy as np


class MFAbstractFunc(AbstractAPI):
    """
    Multi-fidelity Function.

    Args:
        seed (int | None)
            The random seed for the noise.
        runtime_factor (float):
            The runtime factor to change the maximum runtime.
            If max_fidel is given, the runtime will be the `runtime_factor` seconds.
        fidel_dim (int):
            The dimensionality of fidelity.
            By default, we use only one fidelity, but we can optionally increase the fidelity dimension.
        min_fidel (int):
            The minimum fidelity used in MFO algorithms.
        max_fidel (int):
            The maximum fidelity used in MFO algorithms.

    Reference:
        Page 18 of the following paper:
            Title: Multi-fidelity Bayesian Optimisation with Continuous Approximations
            Authors: K. Kandasamy et. al
            URL: https://arxiv.org/pdf/1703.06240.pdf
    """

    _BENCH_TYPE: ClassVar[str] = "SYNTHETIC"
    _DEFAULT_FIDEL_DIM: ClassVar[int]

    def __init__(
        self,
        fidel_dim: int,
        min_fidel: int,
        max_fidel: int,
        seed: int | None,
        runtime_factor: float,
        deterministic: bool,
        noise_std: float,
        dim: int,
    ):
        super().__init__(seed=seed)
        if runtime_factor <= 0:
            raise ValueError(f"`runtime_factor` must be positive, but got {runtime_factor}")
        if fidel_dim not in [self._DEFAULT_FIDEL_DIM, 1]:
            raise ValueError(
                f"The fidelity dimension of {self.__class__.__name__} must be either 1 or {self._DEFAULT_FIDEL_DIM}, "
                f"but got {fidel_dim}"
            )

        self._deterministic = deterministic
        self._noise_std = noise_std
        self._fidel_dim = fidel_dim
        self._runtime_factor = runtime_factor
        self._dim = dim
        self._noise_std = noise_std
        self._min_fidel, self._max_fidel = min_fidel, max_fidel
        self._validate_fidels()
        self._validate_class_vars()

    @abstractmethod
    def _objective(self, x: np.ndarray, z: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def _runtime(self, x: np.ndarray, z: np.ndarray) -> float:
        raise NotImplementedError

    @classmethod
    def _validate_class_vars(cls) -> None:
        super()._validate_class_vars()
        if not hasattr(cls, "_DEFAULT_FIDEL_DIM"):
            raise NotImplementedError(f"Child class of {cls.__name__} must define _DEFAULT_FIDEL_DIM.")

    def _validate_fidels(self) -> None:
        min_fidel, max_fidel = self._min_fidel, self._max_fidel
        if min_fidel >= max_fidel:
            raise ValueError(f"min_fidel < max_fidel must hold, but got {min_fidel=} and {max_fidel=}")
        if min_fidel <= 0:
            raise ValueError(f"min_fidel must be in [1, {self._max_fidel}], but got {min_fidel=} and {max_fidel=}")

    def _validate_config(self, x: np.ndarray, z: np.ndarray) -> None:
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("All elements in x must be in [0.0, 1.0]")
        if np.any((z < self._min_fidel / self._max_fidel) | (z > 1.0)):
            raise ValueError(f"All elements in fidels must be in [{self._min_fidel}, {self._max_fidel}]")

    def __call__(  # type: ignore[override]
        self,
        eval_config: dict[str, float],
        *,
        fidels: dict[str, int] | None = None,
        seed: int | None = None,
    ) -> ResultType:
        fidels = fidels if fidels is not None else {}
        if len(fidels) != self.fidel_dim:
            raise ValueError(f"The provided fidelity dimension is {self.fidel_dim}, " f"but got {fidels}")

        x = np.array([eval_config[f"x{d}"] for d in range(self._dim)])
        z = np.array([fidels[k] / max_fidel for k, max_fidel in self.max_fidels.items()])
        self._validate_config(x=x, z=z)
        noise = 0.0 if self._deterministic else self._noise_std * self._rng.normal()
        loss = float(self._objective(x=x, z=z) + noise)
        runtime = self._runtime(x=x, z=z)
        return {RESULT_KEYS.loss: loss, RESULT_KEYS.runtime: runtime}  # type: ignore[misc]

    @property
    def dataset_name_for_dir(self) -> str | None:
        return None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fidel_dim(self) -> int:
        return self._fidel_dim

    @property
    def min_fidels(self) -> dict[str, int]:  # type: ignore[override]
        # the real minimum is 3
        return {f"z{d}": self._min_fidel for d in range(self.fidel_dim)}

    @property
    def max_fidels(self) -> dict[str, int]:  # type: ignore[override]
        return {f"z{d}": self._max_fidel for d in range(self.fidel_dim)}

    @property
    def fidel_keys(self) -> list[str]:
        return [f"z{d}" for d in range(self.fidel_dim)]

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([CS.UniformFloatHyperparameter(f"x{d}", 0.0, 1.0) for d in range(self._dim)])
        return config_space
