import time
import logging
from collections import defaultdict

import numpy as np
import pandas as pd


def evaluate(
    inference_function,
    preprocessing_function,
    input_shape,
    batch_sizes,
    warmup,
    num_trials,
):
    results = defaultdict(list)

    def _get_fraction_done(b, t):
        total_trials = len(batch_sizes) * (warmup + num_trials)
        trials_done = b * (warmup + num_trials) + t
        return trials_done / total_trials

    last_logging = 0
    start_evaluation_time = time.time()

    for b, batch_size in enumerate(batch_sizes):
        for t, trial in enumerate(range(warmup + num_trials)):
            if int(start_evaluation_time - time.time()) // 30 > last_logging:
                fraction_done = _get_fraction_done(b, t)
                logging.info(f"{fraction_done:.%} done")
                last_logging += 1

            x = np.random.randn(batch_size, *input_shape)
            x = preprocessing_function(x)

            start_time = time.time()
            _ = inference_function(x)
            inference_time = time.time() - start_time

            if trial >= warmup:
                results[batch_size].append(inference_time)

    return results


def analyze_results(dict_results):
    df = defaultdict(list)

    model_names = []
    for model_name, model_results in dict_results.items():
        model_names.append(model_name)
        for batch_size, batch_latencies in model_results.items():
            batch_latencies = np.array(batch_latencies)
            batch_throughputs = batch_size / batch_latencies

            df[("Latency", batch_size)].append(np.mean(batch_latencies))
            df[("Throughput", batch_size)].append(np.mean(batch_throughputs))

    df = {k: df[k] for k in sorted(df)}
    df = pd.DataFrame(df)
    df.index = model_names
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["", "Batch size"])
    return df
