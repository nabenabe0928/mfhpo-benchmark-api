import pytest
import unittest

from benchmark_apis import MFBranin, MFHartmann


def test_branin():
    bench = MFBranin(fidel_dim=1)
    fidels = {"z0": 100}
    for i in range(10):
        bench.reseed(seed=i)
        output = bench(bench.config_space.sample_configuration().get_dictionary(), fidels=fidels)
        assert "runtime" in output.keys()
        assert "loss" in output.keys()
        assert bench.fidel_keys == ["z0"]
        assert bench.dataset_name_for_dir is None

    bench = MFBranin(fidel_dim=1, min_fidel=20, max_fidel=50)
    assert bench.min_fidels["z0"] == 20
    assert bench.max_fidels["z0"] == 50


@pytest.mark.parametrize("dim", [3, 6])
def test_hartmann(dim):
    bench = MFHartmann(fidel_dim=1, dim=dim)
    fidels = {"z0": 100}
    for i in range(10):
        bench.reseed(seed=i)
        output = bench(bench.config_space.sample_configuration().get_dictionary(), fidels=fidels)
        assert "runtime" in output.keys()
        assert "loss" in output.keys()
        assert bench.fidel_keys == ["z0"]
        assert bench.dataset_name_for_dir is None

    bench = MFHartmann(fidel_dim=1, min_fidel=20, max_fidel=50)
    assert bench.min_fidels["z0"] == 20
    assert bench.max_fidels["z0"] == 50


def test_hartmann_error():
    with pytest.raises(ValueError, match=r"`dim` for Hartmann function*"):
        MFHartmann(fidel_dim=1, dim=10)
    with pytest.raises(ValueError, match=r"`dim` for Hartmann function*"):
        bench = MFHartmann(fidel_dim=1)
        bench._dim = 1
        bench.A
    with pytest.raises(ValueError, match=r"`dim` for Hartmann function*"):
        bench = MFHartmann(fidel_dim=1)
        bench._dim = 1
        bench.P

    fidels = {"z0": 100}
    bench = MFHartmann()
    config = bench.config_space.sample_configuration().get_dictionary()
    config["x0"] = -0.5
    with pytest.raises(ValueError, match=r"All elements in x*"):
        bench(config, fidels=fidels)
    config["x0"] = 0.5
    fidels["z0"] = 10
    with pytest.raises(ValueError, match=r"All elements in fidels*"):
        bench(config, fidels=fidels)


def test_invalid_input():
    with pytest.raises(ValueError, match=r"`runtime_factor` must be positive*"):
        MFBranin(runtime_factor=-0.1)
    with pytest.raises(ValueError, match=r"The fidelity dimension of*"):
        MFBranin(fidel_dim=10)
    with pytest.raises(ValueError, match=r"min_fidel must be in*"):
        MFBranin(min_fidel=-1)
    with pytest.raises(ValueError, match=r"min_fidel < max_fidel*"):
        MFBranin(min_fidel=30, max_fidel=20)

    bench = MFBranin()
    config = bench.config_space.sample_configuration().get_dictionary()
    with pytest.raises(ValueError, match=r"The provided fidelity dimension*"):
        bench(config, fidels={})
    with pytest.raises(ValueError, match=r"The provided fidelity dimension*"):
        bench(config, fidels={"z0": 20, "z1": 30})

    bench = MFBranin(fidel_dim=3)
    config = bench.config_space.sample_configuration().get_dictionary()
    bench(config, fidels={"z0": 20, "z1": 20, "z2": 20})
    with pytest.raises(ValueError, match=r"The provided fidelity dimension*"):
        bench(config, fidels={"z0": 20})

    bench = MFHartmann(fidel_dim=4)
    config = bench.config_space.sample_configuration().get_dictionary()
    bench(config, fidels={"z0": 20, "z1": 20, "z2": 20, "z3": 20})
    with pytest.raises(ValueError, match=r"The provided fidelity dimension*"):
        bench(config, fidels={"z0": 20})


if __name__ == "__main__":
    unittest.main()
