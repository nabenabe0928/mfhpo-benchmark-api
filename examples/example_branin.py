from benchmark_apis import MFBranin


bench = MFBranin()
for i in range(10):
    config = bench.config_space.sample_configuration().get_dictionary()
    output = bench(eval_config=config, fidels={"z0": 100})
    print(output)
