# Multi-Fidelity Optimization Benchmark APIs

[![Build Status](https://github.com/nabenabe0928/mfhpo-benchmark-api/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/mfhpo-benchmark-api)
[![codecov](https://codecov.io/gh/nabenabe0928/mfhpo-benchmark-api/branch/main/graph/badge.svg?token=M0LGDR7CF3)](https://codecov.io/gh/nabenabe0928/mfhpo-benchmark-api)

This repository provides APIs for the MLP benchmark in [HPOBench](https://github.com/automl/HPOBench/), [HPOlib](https://github.com/automl/HPOlib1.5), [JAHS-Bench-201](https://github.com/automl/jahs_bench_201/), and LCBench in [YAHPOGym](https://github.com/slds-lmu/yahpo_gym/).
Additionally, we provide multi-fidelity version of Hartmann and Branin functions.

## Install

As JAHS-Bench-201 and LCBench are surrogate benchmarks and they may conflict your project, we provide separate installation:
```shell
# Minimal install
$ pip install mfhpo-benchmark-api

# Minimal install + LCBench
$ pip install mfhpo-benchmark-api[lcbench]

# Minimal install + JAHS-Bench-201
$ pip install mfhpo-benchmark-api[jahs]

# Full install (Minimal + LCBench + JAHS-Bench-201)
$ pip install mfhpo-benchmark-api[full]
```

Note that each benchmark requires the download of tabular or surrogate data.

For HPOBench and HPOlib, please follow README of [this repository](https://github.com/nabenabe0928/hpolib-extractor).
In the instruction, `<YOUR_DATA_PATH>` should be replaced with `~/hpo_benchmarks/hpolib` for HPOlib and `~/hpo_benchmarks/hpobench` for HPOBench.

For JAHS-Bench-201, run the following command:
```shell
$ cd ~/hpo_benchmarks/jahs
$ wget https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar
# Uncompress assembled_surrogates.tar
```

For LCBench, access to [the website](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/) and download `lcbench.zip`.
Then you need to unzip the `lcbench.zip` in `~/hpo_benchmarks/lcbench`.

## Examples

Examples are available in [examples](examples/).
For example, you can test with the following code:

```python
from benchmark_apis import MFBranin


bench = MFBranin()
for i in range(10):
    config = bench.config_space.sample_configuration().get_dictionary()
    output = bench(eval_config=config, fidels={"z0": 100})
    print(output)
```