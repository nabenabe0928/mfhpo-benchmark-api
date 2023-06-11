from __future__ import annotations

import os
import pickle
from typing import ClassVar, Literal, TypedDict

from benchmark_apis.abstract_api import AbstractHPOData, RESULT_KEYS, ResultType, _HPODataClassVars, _TargetMetricKeys
from benchmark_apis.hpo.abstract_bench import AbstractBench, DATA_DIR_NAME, DISC_SPACES, _BenchClassVars, _FidelKeys


_TARGET_KEYS = _TargetMetricKeys(
    loss="bal_acc",
    runtime="runtime",
    precision="precision",
    f1="f1",
)
_BENCH_NAME = "hpobench"
_KEY_ORDER = ["alpha", "batch_size", "depth", "learning_rate_init", "width"]
_DATASET_NAMES = (
    "australian",
    "blood_transfusion",
    "car",
    "credit_g",
    "kc1",
    "phoneme",
    "segment",
    "vehicle",
)


class RowDataType(TypedDict):
    loss: list[dict[int, float]]
    runtime: list[dict[int, float]]
    precision: list[dict[int, float]]
    f1: list[dict[int, float]]


class HPOBenchDatabase(AbstractHPOData):
    """Workaround to prevent dask from serializing the objective func"""

    _CONSTS = _HPODataClassVars(
        url="https://ndownloader.figshare.com/files/30379005/",
        dir=os.path.join(DATA_DIR_NAME, _BENCH_NAME),
    )

    def __init__(self, dataset_name: str):
        self._benchdata_path = os.path.join(self._CONSTS.dir, f"{dataset_name}.pkl")
        self._validate()
        self._db = pickle.load(open(self._benchdata_path, "rb"))

    @property
    def install_instruction(self) -> str:
        return (
            f"\t$ cd {self._CONSTS.dir}\n"
            f"\t$ wget {self._CONSTS.url}\n"
            "\t$ unzip nn.zip\n\n"
            "Then extract the pkl file using https://github.com/nabenabe0928/hpolib-extractor/.\n"
            f"You should get `{self._benchdata_path}` in the end."
        )

    def __getitem__(self, key: str) -> RowDataType:
        return self._db[key]


class HPOBench(AbstractBench):
    """The class for HPOlib.

    Args:
        dataset_id (int):
            The ID of the dataset.
        seed (int | None):
            The random seed to be used.
        target_metrics (list[Literal["loss", "runtime", "f1", "precision"]]):
            The target metrics to return.
            Must be in ["loss", "runtime", "f1", "precision"].
        min_epoch (int):
            The minimum epoch of the training of each neural networks to be used during the optimization.
        max_epoch (int):
            The maximum epoch of the training of each neural networks to be used during the optimization.
        keep_benchdata (bool):
            Whether to keep the benchmark data in each instance.
            When True, serialization will happen in case of parallel optimization.

    References:
        Title: HPOBench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO
        Authors: K. Eggensperger et al.
        URL: https://arxiv.org/abs/2109.06716

    NOTE:
        Download the datasets via:
            $ wget https://ndownloader.figshare.com/files/30379005/
            $ unzip nn.zip

        Use https://github.com/nabenabe0928/hpolib-extractor to extract the pickle file.
    """

    _CONSTS = _BenchClassVars(
        max_epoch=243,
        n_datasets=len(_DATASET_NAMES),
        target_metric_keys=[k for k, v in _TARGET_KEYS.__dict__.items() if v is not None],
        disc_space=DISC_SPACES[_BENCH_NAME],
        fidel_keys=_FidelKeys(epoch="epoch"),
    )

    # HPOBench specific constants
    _N_SEEDS: ClassVar[int] = 5
    _EPOCHS: ClassVar[list[int]] = [3, 9, 27, 81, 243]

    def __init__(
        self,
        dataset_id: int,
        seed: int | None = None,
        target_metrics: list[Literal["loss", "runtime", "f1", "precision"]] = [RESULT_KEYS.loss],  # type: ignore
        min_epoch: int = 27,
        max_epoch: int = 243,
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

    def get_benchdata(self) -> HPOBenchDatabase:
        return HPOBenchDatabase(self.dataset_name)

    def __call__(  # type: ignore[override]
        self,
        eval_config: dict[str, int | str],
        *,
        fidels: dict[str, int] = {},
        seed: int | None = None,
        benchdata: HPOBenchDatabase | None = None,
    ) -> ResultType:
        fidel = int(fidels.get(self._CONSTS.fidel_keys.epoch, self._max_epoch))
        if fidel not in self._EPOCHS:
            raise ValueError(f"fidel for {self.__class__.__name__} must be in {self._EPOCHS}, but got {fidel}")

        db = self._validate_benchdata(benchdata)
        assert db is not None and isinstance(db, HPOBenchDatabase)  # mypy redefinition
        idx = seed % self._N_SEEDS if seed is not None else self._rng.randint(self._N_SEEDS)
        config_id = "".join([str(eval_config[k]) for k in _KEY_ORDER])

        row: RowDataType = db[config_id]
        runtime = row[_TARGET_KEYS.runtime][idx][fidel]  # type: ignore
        output: ResultType = {RESULT_KEYS.runtime: runtime}  # type: ignore

        if RESULT_KEYS.loss in self._target_metrics:
            output[RESULT_KEYS.loss] = 1.0 - row[_TARGET_KEYS.loss][idx][fidel]  # type: ignore
        if RESULT_KEYS.f1 in self._target_metrics:
            output[RESULT_KEYS.f1] = float(row[_TARGET_KEYS.f1][idx][fidel])  # type: ignore
        if RESULT_KEYS.precision in self._target_metrics:
            output[RESULT_KEYS.precision] = float(row[_TARGET_KEYS.precision][idx][fidel])  # type: ignore

        return output
