from benchmark_apis import LCBench


bench = LCBench(dataset_id=0)
for i in range(10):
    config = bench.config_space.sample_configuration().get_dictionary()
    output = bench(eval_config=config)
    print(output)
