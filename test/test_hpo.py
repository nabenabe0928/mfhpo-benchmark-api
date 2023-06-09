import os
import pytest
import shutil
import unittest

from benchmark_apis import HPOLib, JAHSBench201, LCBench
from benchmark_apis.hpo.abstract_bench import DATA_DIR_NAME


DUMMY_DIR_NAME = os.path.join(os.environ["HOME"], "dummy-mfhpo-test")


@pytest.mark.parametrize(
    "target_metrics",
    [
        ["loss"],
        ["loss", "runtime"],
        ["loss", "model_size"],
        ["loss", "runtime", "model_size"],
        ["runtime"],
        ["runtime", "model_size"],
        ["model_size"],
    ],
)
def test_jahs(target_metrics):
    bench = JAHSBench201(dataset_id=0, target_metrics=target_metrics)
    for i in range(10):
        bench.reseed(seed=i)
        output = bench(bench.config_space.sample_configuration().get_dictionary())
        assert set(target_metrics + ["runtime"]) == set(output.keys())
        assert bench.fidel_keys == ["epoch", "Resolution"]

    bench = JAHSBench201(dataset_id=0, min_epoch=10, max_epoch=30, min_resol=0.1, max_resol=0.9, keep_benchdata=False)
    assert bench.min_fidels["epoch"] == 10
    assert bench.max_fidels["epoch"] == 30
    assert bench.min_fidels["Resolution"] == 0.1
    assert bench.max_fidels["Resolution"] == 0.9


def test_jahs_not_found():
    shutil.move(DATA_DIR_NAME, DUMMY_DIR_NAME)
    with pytest.raises(FileNotFoundError):
        JAHSBench201(dataset_id=0)

    shutil.move(DUMMY_DIR_NAME, DATA_DIR_NAME)


def test_jahs_invalid_input():
    with pytest.raises(ValueError, match=r"min_resol < max_resol*"):
        JAHSBench201(dataset_id=0, min_resol=0.5, max_resol=0.1, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"Resolution*"):
        JAHSBench201(dataset_id=0, min_resol=-0.1, max_resol=1.0, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"Resolution*"):
        JAHSBench201(dataset_id=0, min_resol=0.0, max_resol=1.1, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"min_epoch < max_epoch*"):
        JAHSBench201(dataset_id=0, min_epoch=50, max_epoch=10, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch*"):
        JAHSBench201(dataset_id=0, min_epoch=-1, max_epoch=50, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch*"):
        JAHSBench201(dataset_id=0, min_epoch=1, max_epoch=1000, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"All elements*"):
        JAHSBench201(dataset_id=0, target_metrics=["dummy"], keep_benchdata=False)
    with pytest.raises(ValueError, match=r"data must be provided*"):
        bench = JAHSBench201(dataset_id=0, keep_benchdata=False)
        bench(eval_config={})


@pytest.mark.parametrize(
    "target_metrics",
    [
        ["loss"],
        ["loss", "runtime"],
        ["runtime"],
    ],
)
def test_lcbench(target_metrics):
    bench = LCBench(dataset_id=0, target_metrics=target_metrics)
    for i in range(10):
        bench.reseed(seed=i)
        output = bench(bench.config_space.sample_configuration().get_dictionary())
        assert set(target_metrics + ["runtime"]) == set(output.keys())
        assert bench.fidel_keys == ["epoch"]

    bench = LCBench(dataset_id=0, min_epoch=10, max_epoch=30, keep_benchdata=False)
    assert bench.min_fidels["epoch"] == 10
    assert bench.max_fidels["epoch"] == 30


def test_lcbench_not_found():
    shutil.move(DATA_DIR_NAME, DUMMY_DIR_NAME)
    with pytest.raises(FileNotFoundError):
        LCBench(dataset_id=0)

    shutil.move(DUMMY_DIR_NAME, DATA_DIR_NAME)


def test_lcbench_invalid_input():
    with pytest.raises(ValueError, match=r"min_epoch < max_epoch*"):
        LCBench(dataset_id=0, min_epoch=50, max_epoch=10, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch*"):
        LCBench(dataset_id=0, min_epoch=-1, max_epoch=50, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch*"):
        LCBench(dataset_id=0, min_epoch=1, max_epoch=1000, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"All elements*"):
        LCBench(dataset_id=0, target_metrics=["dummy"], keep_benchdata=False)
    with pytest.raises(ValueError, match=r"data must be provided*"):
        bench = LCBench(dataset_id=0, keep_benchdata=False)
        bench(eval_config={})


@pytest.mark.parametrize(
    "target_metrics",
    [
        ["loss"],
        ["loss", "runtime"],
        ["loss", "model_size"],
        ["loss", "runtime", "model_size"],
        ["runtime"],
        ["runtime", "model_size"],
        ["model_size"],
    ],
)
def test_hpolib(target_metrics):
    bench = HPOLib(dataset_id=0, target_metrics=target_metrics)
    for i in range(10):
        bench.reseed(seed=i)
        output = bench(bench.config_space.sample_configuration().get_dictionary())
        assert set(target_metrics + ["runtime"]) == set(output.keys())
        assert bench.fidel_keys == ["epoch"]

    bench = HPOLib(dataset_id=0, min_epoch=10, max_epoch=30)
    assert bench.min_fidels["epoch"] == 10
    assert bench.max_fidels["epoch"] == 30


def test_hpolib_not_found():
    shutil.move(DATA_DIR_NAME, DUMMY_DIR_NAME)
    with pytest.raises(FileNotFoundError):
        HPOLib(dataset_id=0)

    shutil.move(DUMMY_DIR_NAME, DATA_DIR_NAME)


def test_hpolib_invalid_input():
    with pytest.raises(ValueError, match=r"min_epoch < max_epoch*"):
        HPOLib(dataset_id=0, min_epoch=50, max_epoch=10, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch*"):
        HPOLib(dataset_id=0, min_epoch=-1, max_epoch=50, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch*"):
        HPOLib(dataset_id=0, min_epoch=1, max_epoch=1000, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"All elements*"):
        HPOLib(dataset_id=0, target_metrics=["dummy"], keep_benchdata=False)
    with pytest.raises(ValueError, match=r"data must be provided*"):
        bench = HPOLib(dataset_id=0, keep_benchdata=False)
        bench(eval_config={})


if __name__ == "__main__":
    unittest.main()
