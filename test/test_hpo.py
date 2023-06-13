import os
import pytest
import shutil
import unittest

from benchmark_apis import HPOBench, HPOLib, JAHSBench201, LCBench
from benchmark_apis.hpo.abstract_bench import DATA_DIR_NAME


DUMMY_DIR_NAME = os.path.join(os.environ["HOME"], "dummy-mfhpo-test")
IS_LOCAL = eval(os.environ.get("MFHPO_BENCH_TEST", "False"))


@unittest.skipIf(not IS_LOCAL, "Data is too heavy to prepare on the GitHub server")
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
        assert "_" not in bench.dataset_name_for_dir
        bench(bench.config_space.sample_configuration().get_dictionary(), fidels={"epoch": 100, "Resolution": 0.5})

    bench = JAHSBench201(
        dataset_id=0, fidel_value_ranges={"epoch": (10, 30), "Resolution": (0.1, 0.9)}, keep_benchdata=False
    )
    assert bench.min_fidels["epoch"] == 10
    assert bench.max_fidels["epoch"] == 30
    assert bench.min_fidels["Resolution"] == 0.1
    assert bench.max_fidels["Resolution"] == 0.9


def test_jahs_not_found():
    if IS_LOCAL:
        shutil.move(DATA_DIR_NAME, DUMMY_DIR_NAME)
    with pytest.raises(FileNotFoundError):
        JAHSBench201(dataset_id=0)

    if IS_LOCAL:
        shutil.move(DUMMY_DIR_NAME, DATA_DIR_NAME)


def test_jahs_invalid_input():
    with pytest.raises(ValueError, match=r"lower < upper for Resolution*"):
        JAHSBench201(dataset_id=0, fidel_value_ranges={"Resolution": (0.5, 0.1)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"Resolution must be*"):
        JAHSBench201(dataset_id=0, fidel_value_ranges={"Resolution": (-0.1, 1.0)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"Resolution must be*"):
        JAHSBench201(dataset_id=0, fidel_value_ranges={"Resolution": (0.0, 1.1)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"lower < upper for epoch*"):
        JAHSBench201(dataset_id=0, fidel_value_ranges={"epoch": (50, 10)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch must be*"):
        JAHSBench201(dataset_id=0, fidel_value_ranges={"epoch": (-1, 50)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch must be*"):
        JAHSBench201(dataset_id=0, fidel_value_ranges={"epoch": (1, 1000)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"All elements*"):
        JAHSBench201(dataset_id=0, target_metrics=["dummy"], keep_benchdata=False)
    with pytest.raises(KeyError, match=r"Keys in fidel_value_ranges must be in*"):
        JAHSBench201(dataset_id=0, fidel_value_ranges={"dummy": (10, 30)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"data must be provided*"):
        bench = JAHSBench201(dataset_id=0, keep_benchdata=False)
        bench(eval_config={})


@unittest.skipIf(not IS_LOCAL, "Data is too heavy to prepare on the GitHub server")
def test_jahs_invalid_input_in_call():
    bench = JAHSBench201(dataset_id=0, keep_benchdata=True)
    config = bench.config_space.sample_configuration().get_dictionary()
    with pytest.raises(ValueError, match=r"epoch must be in \[*"):
        bench(eval_config=config, fidels={"epoch": 1000})
    with pytest.raises(ValueError, match=r"epoch must be in \[*"):
        bench(eval_config=config, fidels={"epoch": 0})
    with pytest.raises(ValueError, match=r"Resolution must be in \[*"):
        bench(eval_config=config, fidels={"Resolution": -0.1})
    with pytest.raises(ValueError, match=r"Resolution must be in \[*"):
        bench(eval_config=config, fidels={"Resolution": 1.1})
    with pytest.raises(ValueError, match=r"Op1 must be in \('*"):
        config["Op1"] = "9"
        bench(config)
    config["Op1"] = "0"
    with pytest.raises(ValueError, match=r"W must be in \[lb=*"):
        config["W"] = 1000
        bench(config)


@unittest.skipIf(not IS_LOCAL, "Data is too heavy to prepare on the GitHub server")
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
        assert "_" not in bench.dataset_name_for_dir

    bench = LCBench(dataset_id=0, fidel_value_ranges={"epoch": (10, 30)}, keep_benchdata=False)
    assert bench.min_fidels["epoch"] == 10
    assert bench.max_fidels["epoch"] == 30


def test_lcbench_not_found():
    if IS_LOCAL:
        shutil.move(DATA_DIR_NAME, DUMMY_DIR_NAME)
    with pytest.raises(FileNotFoundError):
        LCBench(dataset_id=0)

    if IS_LOCAL:
        shutil.move(DUMMY_DIR_NAME, DATA_DIR_NAME)


def test_lcbench_invalid_input():
    with pytest.raises(ValueError, match=r"lower < upper for epoch*"):
        LCBench(dataset_id=0, fidel_value_ranges={"epoch": (50, 10)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch must be*"):
        LCBench(dataset_id=0, fidel_value_ranges={"epoch": (-1, 50)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch must be*"):
        LCBench(dataset_id=0, fidel_value_ranges={"epoch": (1, 1000)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"All elements*"):
        LCBench(dataset_id=0, target_metrics=["dummy"], keep_benchdata=False)
    with pytest.raises(KeyError, match=r"Keys in fidel_value_ranges must be in*"):
        LCBench(dataset_id=0, fidel_value_ranges={"dummy": (10, 30)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"data must be provided*"):
        bench = LCBench(dataset_id=0, keep_benchdata=False)
        bench(eval_config={})


@unittest.skipIf(not IS_LOCAL, "Data is too heavy to prepare on the GitHub server")
def test_lcbench_invalid_input_in_call():
    bench = LCBench(dataset_id=0, keep_benchdata=True)
    config = bench.config_space.sample_configuration().get_dictionary()
    with pytest.raises(ValueError, match=r"epoch must be in \[*"):
        bench(eval_config=config, fidels={"epoch": 1000})
    with pytest.raises(ValueError, match=r"epoch must be in \[*"):
        bench(eval_config=config, fidels={"epoch": 0})
    with pytest.raises(ValueError, match=r"batch_size must be in \[lb=*"):
        config["batch_size"] = 1000
        bench(config)


@unittest.skipIf(not IS_LOCAL, "Data is too heavy to prepare on the GitHub server")
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
        assert "_" not in bench.dataset_name_for_dir

    bench = HPOLib(dataset_id=0, fidel_value_ranges={"epoch": (10, 30)})
    assert bench.min_fidels["epoch"] == 10
    assert bench.max_fidels["epoch"] == 30


def test_hpolib_not_found():
    if IS_LOCAL:
        shutil.move(DATA_DIR_NAME, DUMMY_DIR_NAME)
    with pytest.raises(FileNotFoundError):
        HPOLib(dataset_id=0)

    if IS_LOCAL:
        shutil.move(DUMMY_DIR_NAME, DATA_DIR_NAME)


def test_hpolib_invalid_input():
    with pytest.raises(ValueError, match=r"lower < upper for epoch*"):
        HPOLib(dataset_id=0, fidel_value_ranges={"epoch": (50, 10)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch must be*"):
        HPOLib(dataset_id=0, fidel_value_ranges={"epoch": (-1, 50)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch must be*"):
        HPOLib(dataset_id=0, fidel_value_ranges={"epoch": (1, 1000)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"All elements*"):
        HPOLib(dataset_id=0, target_metrics=["dummy"], keep_benchdata=False)
    with pytest.raises(KeyError, match=r"Keys in fidel_value_ranges must be in*"):
        HPOLib(dataset_id=0, fidel_value_ranges={"dummy": (10, 30)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"data must be provided*"):
        bench = HPOLib(dataset_id=0, keep_benchdata=False)
        bench(eval_config={})


@unittest.skipIf(not IS_LOCAL, "Data is too heavy to prepare on the GitHub server")
def test_hpolib_invalid_input_in_call():
    bench = HPOLib(dataset_id=0, keep_benchdata=True)
    config = bench.config_space.sample_configuration().get_dictionary()
    with pytest.raises(ValueError, match=r"epoch must be in \[*"):
        bench(eval_config=config, fidels={"epoch": 1000})
    with pytest.raises(ValueError, match=r"epoch must be in \[*"):
        bench(eval_config=config, fidels={"epoch": 0})
    with pytest.raises(KeyError):
        config["activation_fn_1"] = "90000"
        bench(config)

    config["activation_fn_1"] = "0"
    with pytest.raises(KeyError):
        config["batch_size"] = 9
        bench(config)


@unittest.skipIf(not IS_LOCAL, "Data is too heavy to prepare on the GitHub server")
@pytest.mark.parametrize(
    "target_metrics",
    [
        ["loss"],
        ["loss", "runtime"],
        ["loss", "precision"],
        ["loss", "f1"],
        ["loss", "runtime", "precision"],
        ["loss", "runtime", "f1"],
        ["loss", "precision", "f1"],
        ["loss", "runtime", "precision", "f1"],
        ["runtime"],
        ["runtime", "precision"],
        ["runtime", "f1"],
        ["runtime", "precision", "f1"],
        ["precision"],
        ["precision", "f1"],
        ["f1"],
    ],
)
def test_hpobench(target_metrics):
    bench = HPOBench(dataset_id=0, target_metrics=target_metrics)
    for i in range(10):
        bench.reseed(seed=i)
        output = bench(bench.config_space.sample_configuration().get_dictionary())
        assert set(target_metrics + ["runtime"]) == set(output.keys())
        assert bench.fidel_keys == ["epoch"]
        assert "_" not in bench.dataset_name_for_dir

    bench = HPOBench(dataset_id=0, fidel_value_ranges={"epoch": (3, 81)})
    assert bench.min_fidels["epoch"] == 3
    assert bench.max_fidels["epoch"] == 81


def test_hpobench_not_found():
    if IS_LOCAL:
        shutil.move(DATA_DIR_NAME, DUMMY_DIR_NAME)

    with pytest.raises(FileNotFoundError):
        HPOBench(dataset_id=0)

    if IS_LOCAL:
        shutil.move(DUMMY_DIR_NAME, DATA_DIR_NAME)


def test_hpobench_invalid_input():
    with pytest.raises(ValueError, match=r"lower < upper for epoch*"):
        HPOBench(dataset_id=0, fidel_value_ranges={"epoch": (50, 10)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch must be*"):
        HPOBench(dataset_id=0, fidel_value_ranges={"epoch": (-1, 50)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"epoch must be*"):
        HPOBench(dataset_id=0, fidel_value_ranges={"epoch": (1, 1000)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"All elements*"):
        HPOBench(dataset_id=0, target_metrics=["dummy"], keep_benchdata=False)
    with pytest.raises(ValueError, match=r"data must be provided*"):
        bench = HPOBench(dataset_id=0, keep_benchdata=False)
        bench(eval_config={})
    with pytest.raises(KeyError, match=r"Keys in fidel_value_ranges must be in*"):
        HPOBench(dataset_id=0, fidel_value_ranges={"dummy": (10, 30)}, keep_benchdata=False)
    with pytest.raises(ValueError, match=r"fidel for*"):
        bench = HPOBench(dataset_id=0, keep_benchdata=False)
        bench(eval_config=bench.config_space.sample_configuration().get_dictionary(), fidels={"epoch": 50})


@unittest.skipIf(not IS_LOCAL, "Data is too heavy to prepare on the GitHub server")
def test_hpobench_invalid_input_in_call():
    bench = HPOBench(dataset_id=0, fidel_value_ranges={"epoch": (9, 81)}, keep_benchdata=True)
    config = bench.config_space.sample_configuration().get_dictionary()
    with pytest.raises(ValueError, match=r"epoch must be in \[*"):
        bench(eval_config=config, fidels={"epoch": 243})
    with pytest.raises(ValueError, match=r"epoch must be in \[*"):
        bench(eval_config=config, fidels={"epoch": 3})
    with pytest.raises(KeyError):
        config["depth"] = 9
        bench(config)


if __name__ == "__main__":
    unittest.main()
