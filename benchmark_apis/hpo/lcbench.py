from __future__ import annotations

import os
from typing import ClassVar, Literal

import ConfigSpace as CS

from benchmark_apis.abstract_api import (
    AbstractHPOData,
    RESULT_KEYS,
    ResultType,
    _HPODataClassVars,
    _TargetMetricKeys,
    _warn_not_found_module,
)
from benchmark_apis.hpo.abstract_bench import AbstractBench, DATA_DIR_NAME, _BenchClassVars, _FidelKeys

try:
    from yahpo_gym import benchmark_set, local_config
except ModuleNotFoundError:
    _warn_not_found_module(bench_name="lcbench")


_TARGET_KEYS = _TargetMetricKeys(
    loss="val_balanced_accuracy",
    runtime="time",
)
_DATASET_INFO = (
    ("kddcup09_appetency", "3945"),
    ("covertype", "7593"),
    ("amazon_employee_access", "34539"),
    ("adult", "126025"),
    ("nomao", "126026"),
    ("bank_marketing", "126029"),
    ("shuttle", "146212"),
    ("australian", "167104"),
    ("kr_vs_kp", "167149"),
    ("mfeat_factors", "167152"),
    ("credit_g", "167161"),
    ("vehicle", "167168"),
    ("kc1", "167181"),
    ("blood_transfusion_service_center", "167184"),
    ("cnae_9", "167185"),
    ("phoneme", "167190"),
    ("higgs", "167200"),
    ("connect_4", "167201"),
    ("helena", "168329"),
    ("jannis", "168330"),
    ("volkert", "168331"),
    ("mini_boo_ne", "168335"),
    ("aps_failure", "168868"),
    ("christine", "168908"),
    ("fabert", "168910"),
    ("airlines", "189354"),
    ("jasmine", "189862"),
    ("sylvine", "189865"),
    ("albert", "189866"),
    ("dionis", "189873"),
    ("car", "189905"),
    ("segment", "189906"),
    ("fashion_mnist", "189908"),
    ("jungle_chess_2pcs_raw_endgame_complete", "189909"),
)


class LCBenchSurrogate(AbstractHPOData):
    """Workaround to prevent dask from serializing the objective func"""

    _CONSTS = _HPODataClassVars(
        url="https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/",
        dir=os.path.join(DATA_DIR_NAME, "lcbench"),
    )

    def __init__(self, dataset_id: str, target_metrics: list[str]):
        self._validate()
        self._dataset_id = dataset_id
        self._target_metrics = target_metrics[:]
        # active_session=False is necessary for parallel computing.
        self._surrogate = benchmark_set.BenchmarkSet("lcbench", instance=dataset_id, active_session=False)

    @property
    def install_instruction(self) -> str:
        return (
            f"\tAccess to {self._CONSTS.url} and download `lcbench.zip` from the website.\n\n"
            f"After that, please unzip `lcbench.zip` in {self._CONSTS.dir}."
        )

    def _check_benchdata_availability(self) -> None:
        super()._check_benchdata_availability()
        local_config.init_config()
        local_config.set_data_path(DATA_DIR_NAME)

    def __call__(self, eval_config: dict[str, int | float], fidel: int) -> ResultType:
        _eval_config: dict[str, int | float | str] = eval_config.copy()  # type: ignore
        _eval_config["OpenML_task_id"] = self._dataset_id
        _eval_config["epoch"] = fidel

        output = self._surrogate.objective_function(_eval_config)[0]
        results: ResultType = {RESULT_KEYS.runtime: float(output[_TARGET_KEYS.runtime])}  # type: ignore
        if RESULT_KEYS.loss in self._target_metrics:
            results[RESULT_KEYS.loss] = float(1.0 - output[_TARGET_KEYS.loss])  # type: ignore

        return results


class LCBench(AbstractBench):
    """The class for LCBench.

    Args:
        dataset_id (int):
            The ID of the dataset.
        seed (int | None):
            The random seed to be used.
        target_metrics (list[str]):
            The target metrics to return.
            Must be in ["loss", "runtime", "model_size"].
        min_epoch (int):
            The minimum epoch of the training of each neural networks to be used during the optimization.
        max_epoch (int):
            The maximum epoch of the training of each neural networks to be used during the optimization.
        keep_benchdata (bool):
            Whether to keep the benchmark data in each instance.
            When True, serialization will happen in case of parallel optimization.

    References:
        1. The original benchmark
        Title: Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL
        Authors: L. Zimmer et al.
        URL: https://arxiv.org/abs/2006.13799/

        2. The proposition of the surrogate model
        Title: YAHPO Gym -- An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization
        Authors: F. Pfisterer et al.
        URL: https://arxiv.org/abs/2109.03670/

    NOTE:
        The data is available at:
            https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/
    """

    _CONSTS = _BenchClassVars(
        max_epoch=54,
        n_datasets=len(_DATASET_INFO),
        target_metric_keys=[k for k, v in _TARGET_KEYS.__dict__.items() if v is not None],
        fidel_keys=_FidelKeys(epoch="epoch"),
    )

    # LCBench specific constant
    _TRUE_MAX_EPOCH: ClassVar[int] = 52

    def __init__(
        self,
        dataset_id: int,
        seed: int | None = None,  # surrogate is not stochastic
        target_metrics: list[Literal["loss", "runtime"]] = [RESULT_KEYS.loss],  # type: ignore
        min_epoch: int = 6,
        max_epoch: int = 54,
        keep_benchdata: bool = True,
    ):
        dataset_name, self._dataset_id = _DATASET_INFO[dataset_id]
        super().__init__(
            seed=seed,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            target_metrics=target_metrics[:],  # type: ignore
            dataset_name=dataset_name,
            keep_benchdata=keep_benchdata,
        )

    def get_benchdata(self) -> LCBenchSurrogate:
        return LCBenchSurrogate(dataset_id=self._dataset_id, target_metrics=self._target_metrics)

    def __call__(  # type: ignore[override]
        self,
        eval_config: dict[str, int | float],
        *,
        fidels: dict[str, int] = {},
        seed: int | None = None,
        benchdata: LCBenchSurrogate | None = None,
    ) -> ResultType:
        surrogate = self._validate_benchdata(benchdata)
        assert surrogate is not None and isinstance(surrogate, LCBenchSurrogate)  # mypy redefinition
        fidel = int(min(self._TRUE_MAX_EPOCH, fidels.get(self._CONSTS.fidel_keys.epoch, self._max_epoch)))
        self._validate_config(eval_config=eval_config)  # type: ignore
        return surrogate(eval_config=eval_config, fidel=fidel)

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CS.UniformIntegerHyperparameter(name="batch_size", lower=16, upper=512, log=True),
                CS.UniformFloatHyperparameter(name="learning_rate", lower=1e-4, upper=0.1, log=True),
                CS.UniformFloatHyperparameter(name="max_dropout", lower=0.0, upper=1.0),
                CS.UniformIntegerHyperparameter(name="max_units", lower=64, upper=1024, log=True),
                CS.UniformFloatHyperparameter(name="momentum", lower=0.1, upper=0.9),
                CS.UniformIntegerHyperparameter(name="num_layers", lower=1, upper=5),
                CS.UniformFloatHyperparameter(name="weight_decay", lower=1e-5, upper=0.1),
            ]
        )
        return config_space
