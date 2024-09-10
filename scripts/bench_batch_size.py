"""
This script implements a benchmark the runtime of the distributed key generation protocol for different batch sizes.

Assuming you have installed the package into your Python environment, you can run this script as follows:
`bench_batch_size --parties 3 --threshold 1 --key-length 1024 --stat-sec-shamir 40 --test-small-prime-div-param 20000 --test-biprime-param 40 --iterations 100`
"""

import argparse
import asyncio
import logging
import os
import pickle
import re
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytest import MonkeyPatch
from tikzplotlib import save as tikz_save
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tno.mpc.communication import Pool

from tno.mpc.protocols.distributed_keygen.distributed_keygen import DistributedPaillier

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("tno.mpc.communication.httphandlers").setLevel(logging.WARNING)
logging.getLogger("tno.mpc.protocols.distributed_keygen").setLevel(logging.INFO)

# General parameters
PARTIES = 3
THRESHOLD = 1

# DistributedPaillier parameter defaults
KEY_LENGTH = 1024
STAT_SEC_SHAMIR = 40
# Parameter $B$. All small prime divisors p up to $B$ are tested.
TEST_SMALL_PRIME_DIV_PARAM = 20000
# How many tests to perform for the biprimality test.
TEST_BIPRIME_PARAM = 40

# Benchmark parameters
BATCH_SIZES = np.power(2, range(11))
ITERATIONS = 100

# Ensure out directory exists
os.makedirs("out", exist_ok=True)


class DistributedPaillierTimer:
    """A class that monkeypatches the DistributedPaillier class to track the total time spent in some of its functions.

    The tracked functions are:
    - DistributedPaillier.__small_prime_divisors_test
    - DistributedPaillier.__biprime_test
    """

    total_times = {}
    monkeypatch: MonkeyPatch

    def __init__(self):
        self.monkeypatch = MonkeyPatch()
        self._monkeypatch()

    def _monkeypatch(self):
        def time_execution(method):
            def timed(*args, **kw):
                ts = time.time()
                result = method(*args, **kw)
                te = time.time()
                self.total_times.setdefault(method.__name__, 0)
                self.total_times[method.__name__] += te - ts
                return result

            return timed

        def async_time_execution(method):
            async def timed(*args, **kw):
                ts = time.time()
                result = await method(*args, **kw)
                te = time.time()
                self.total_times.setdefault(method.__name__, 0)
                self.total_times[method.__name__] += te - ts
                return result

            return timed

        self.monkeypatch.setattr(
            DistributedPaillier,
            "_DistributedPaillier__small_prime_divisors_test",
            time_execution(
                getattr(
                    DistributedPaillier,
                    "_DistributedPaillier__small_prime_divisors_test",
                )
            ),
        )
        self.monkeypatch.setattr(
            DistributedPaillier,
            "_DistributedPaillier__biprime_test",
            async_time_execution(
                getattr(DistributedPaillier, "_DistributedPaillier__biprime_test")
            ),
        )

    def reset(self):
        self.total_times = {}

    def get_total_times(self):
        return self.total_times


class BechmarkLoggingHandler(logging.Handler):
    """A logging handler that captures the failed_small_prime and
    failed_biprime counts from the log messages. Before each run of the
    distributed keygen protocol, one should call reset."""

    def __init__(self):
        super().__init__()
        self.failed_small_prime = 0
        self.failed_biprime = 0

    def emit(self, record):
        message = self.format(record)
        small_primes_match = re.search(
            r"Checked (\d+) primes for small prime divisors", message
        )
        biprimality_candidates_match = re.search(
            r"Checked (\d+) candidates for biprimality", message
        )

        if small_primes_match:
            self.failed_small_prime = int(small_primes_match.group(1))

        if biprimality_candidates_match:
            self.failed_biprime = int(biprimality_candidates_match.group(1))

    def reset(self):
        self.failed_small_prime = 0
        self.failed_biprime = 0


def setup_pools():
    port_base = 3001
    pools = []

    for i in range(PARTIES):
        pool = Pool()
        pool.add_http_server(port_base + i)
        for j in (j for j in range(PARTIES) if j != i):
            pool.add_http_client(f"local{j}", "localhost", port=port_base + j)

        pools.append(pool)

    return pools


async def perform_keygen(pools, batch_size=1):
    """Run a single iteration of the distributed keygen protocol."""

    async_coroutines = [
        DistributedPaillier.from_security_parameter(
            pool,
            THRESHOLD,
            KEY_LENGTH,
            TEST_SMALL_PRIME_DIV_PARAM,
            TEST_BIPRIME_PARAM,
            STAT_SEC_SHAMIR,
            distributed=False,
            batch_size=batch_size,
        )
        for pool in pools
    ]

    await asyncio.gather(*async_coroutines)


async def benchmark_keygen(pools, batch_size):
    """Run the distributed keygen protocol for a given batch size and return the runtime."""
    start_time = time.perf_counter()
    await perform_keygen(pools, batch_size=batch_size)

    end_time = time.perf_counter()
    runtime = end_time - start_time  # returns time in seconds

    return runtime


@dataclass
class BenchmarkState:
    iterations: int = 0
    runtimes: list[float] = field(default_factory=list)
    """List of runtimes for each iteration."""
    failed_small_prime_test: list[int] = field(default_factory=list)
    """List of counters for each iteration. The counter tracks the number of failed small prime divisors tests in a single iteration."""
    failed_biprime_test: list[int] = field(default_factory=list)
    """List of counters for each iteration. The counter tracks the number of failed biprime tests in a single iteration."""
    time_small_prime_divisors_test: list[float] = field(default_factory=list)
    """List of timers for each iteration. Each timer tracks the total time spent in the small prime divisors test in a single iteration."""
    time_biprime_test: list[float] = field(default_factory=list)
    """List of timers for each iteration. Each timer tracks the total time spent in the biprime test in a single iteration."""

    def trim_iterations(self, iterations):
        """Trim the lists to the given number of iterations."""
        self.iterations = iterations
        self.runtimes = self.runtimes[: self.iterations]
        self.failed_small_prime_test = self.failed_small_prime_test[: self.iterations]
        self.failed_biprime_test = self.failed_biprime_test[: self.iterations]
        self.time_small_prime_divisors_test = self.time_small_prime_divisors_test[
            : self.iterations
        ]
        self.time_biprime_test = self.time_biprime_test[: self.iterations]


async def run_benchmark():
    pools = setup_pools()

    # This dictionary contains a BenchmarkState object for each batch size
    benchmark_states: dict[int, BenchmarkState] = {}

    # Load the benchmark state if it exists to continue where we left off
    pickle_file = f"out/bs{{batch_size}}_n{PARTIES}_t{THRESHOLD}_l{KEY_LENGTH}_s{STAT_SEC_SHAMIR}_small{TEST_SMALL_PRIME_DIV_PARAM}_biprime{TEST_BIPRIME_PARAM}.pkl"

    # We store the results for each batch size in a separate file
    for batch_size in BATCH_SIZES:
        filename = pickle_file.format(batch_size=batch_size)
        if not os.path.exists(filename):
            logger.info(f"Initializing new benchmark state for {batch_size}")
            benchmark_states[batch_size] = BenchmarkState()
            continue

        logger.info(f"Loading benchmark state from {filename}")
        with open(filename, "rb") as f:
            # Load the saved state into memory
            saved_state: BenchmarkState = pickle.load(f)
            benchmark_states[batch_size] = saved_state

            # If the requested number of iterations is smaller than the number
            # than the number of iterations found in the saved state, trim the
            # in memory benchmark state
            if saved_state.iterations > ITERATIONS:
                benchmark_states[batch_size].trim_iterations(ITERATIONS)

    # Capture logging output in order to record failed_small_prime and failed_biprime counts
    handler = BechmarkLoggingHandler()
    logging.getLogger(
        "tno.mpc.protocols.distributed_keygen.distributed_keygen"
    ).addHandler(handler)
    # Capture runtime of small_prime_divisors_test and biprime_test
    dp_timer = DistributedPaillierTimer()

    with logging_redirect_tqdm():
        # Add a progress bar for batch sizes
        for batch_size in tqdm(BATCH_SIZES, desc="Batch sizes", ncols=70):
            # Count the completed runs for the current batch size
            completed_runs = benchmark_states[batch_size].iterations

            # Skip completed runs and add another progress bar for the remaining iterations
            for _ in tqdm(
                range(completed_runs, ITERATIONS),
                desc=f"Batch size {batch_size}",
                leave=False,
                ncols=70,
            ):
                # Reset failed_biprime and failed_small_prime counts
                handler.reset()
                # Reset the timer
                dp_timer.reset()

                # Run the benchmark
                runtime = await benchmark_keygen(pools, batch_size)

                # Save the results
                benchmark_states[batch_size].iterations += 1
                benchmark_states[batch_size].runtimes.append(runtime)
                benchmark_states[batch_size].failed_small_prime_test.append(
                    handler.failed_small_prime
                )
                benchmark_states[batch_size].failed_biprime_test.append(
                    handler.failed_biprime
                )
                benchmark_states[batch_size].time_small_prime_divisors_test.append(
                    dp_timer.get_total_times()["__small_prime_divisors_test"]
                )
                benchmark_states[batch_size].time_biprime_test.append(
                    dp_timer.get_total_times()["__biprime_test"]
                )

                # Log the results of this iteration
                logger.info(f"Batch size: {batch_size}, time: {runtime}")
                logger.info(
                    f"failed small primes: {handler.failed_small_prime} (took {dp_timer.get_total_times()['__small_prime_divisors_test']})"
                )
                logger.info(
                    f"failed biprimes: {handler.failed_biprime} (took {dp_timer.get_total_times()['__biprime_test']})"
                )

                # Store the benchmark state after each run
                with open(pickle_file.format(batch_size=batch_size), "wb") as f:
                    logger.info(f"Saving benchmark state to {f.name}")
                    pickle.dump(benchmark_states[batch_size], f)

    def plot_time():
        # Convert the times into a Pandas DataFrame
        df_time = pd.DataFrame(
            [
                (bs, time)
                for bs in BATCH_SIZES
                for time in benchmark_states[bs].runtimes
            ],
            columns=["BatchSize", "Time"],
        )

        # Use seaborn to plot with confidence intervals
        sns.lineplot(x="BatchSize", y="Time", data=df_time, errorbar="sd")
        plt.ylabel("Time (s)")
        plt.yscale("linear")

        # Display the plot in a new window
        plt.savefig("out/plot_time.png")
        tikz_save("out/plot_time.tex")
        logger.info(f"Saved {os.getcwd()}/out/plot_time.png,tex")
        plt.clf()

    def plot_histogram_small_prime():
        # Convert the failure counts into a Pandas DataFrame
        df_small_prime = pd.DataFrame(
            [
                (bs, count)
                for bs in BATCH_SIZES
                for count in benchmark_states[bs].failed_small_prime_test
            ],
            columns=["BatchSize", "FailedSmallPrime"],
        )

        plt.hist(df_small_prime["FailedSmallPrime"], bins="auto")
        plt.title("Histogram of Failed Small Prime Tests")
        plt.xlabel("Number of failed small prime tests")
        plt.ylabel("Frequency")

        # Calculate mean and standard deviation
        mean = np.mean(df_small_prime["FailedSmallPrime"])
        std = np.std(df_small_prime["FailedSmallPrime"])
        # Add mu and sigma symbol to legend as two separate lines
        plt.text(0.8, 0.9, f"N: {ITERATIONS}", transform=plt.gca().transAxes)
        plt.text(0.8, 0.85, f"μ: {mean:.2f}", transform=plt.gca().transAxes)
        plt.text(0.8, 0.80, f"σ: {std:.2f}", transform=plt.gca().transAxes)

        plt.savefig("out/plot_histogram_small_prime.png")
        tikz_save("out/plot_histogram_small_prime.tex")
        logger.info(f"Saved {os.getcwd()}/out/plot_histogram_small_prime.png,tex")
        plt.clf()

    def plot_histogram_biprime():
        # Convert the failure counts into a Pandas DataFrame
        df_biprime = pd.DataFrame(
            [
                (bs, count)
                for bs in BATCH_SIZES
                for count in benchmark_states[bs].failed_biprime_test
            ],
            columns=["BatchSize", "FailedBiprime"],
        )

        plt.hist(df_biprime["FailedBiprime"], bins="auto")
        plt.title("Histogram of Failed Biprime Tests")
        plt.xlabel("Number of failed biprime tests")
        plt.ylabel("Frequency")

        # Calculate mean and standard deviation
        mean = np.mean(df_biprime["FailedBiprime"])
        std = np.std(df_biprime["FailedBiprime"])
        # Add mu and sigma symbol to legend as two separate lines
        plt.text(0.8, 0.9, f"N: {ITERATIONS}", transform=plt.gca().transAxes)
        plt.text(0.8, 0.85, f"μ: {mean:.2f}", transform=plt.gca().transAxes)
        plt.text(0.8, 0.80, f"σ: {std:.2f}", transform=plt.gca().transAxes)

        plt.savefig("out/plot_histogram_biprime.png")
        tikz_save("out/plot_histogram_biprime.tex")
        logger.info(f"Saved {os.getcwd()}/out/plot_histogram_biprime.png,tex")
        plt.clf()

    def plot_histogram_function_runtimes():
        # Convert the failure counts into a Pandas DataFrame
        df_func_rt = pd.DataFrame(
            [
                (bs, small_prime_rt, biprime_rt)
                for bs in BATCH_SIZES
                for small_prime_rt, biprime_rt in zip(
                    benchmark_states[bs].time_small_prime_divisors_test,
                    benchmark_states[bs].time_biprime_test,
                )
            ],
            columns=["BatchSize", "SmallPrimeRuntime", "BiprimeRuntime"],
        )

        # Set the style of the plots
        sns.set(style="whitegrid")
        # Create a figure and a set of subplots
        _, ax = plt.subplots()
        # Plot the histogram of SmallPrimeRuntime
        sns.histplot(
            df_func_rt,
            x="SmallPrimeRuntime",
            bins="auto",
            color="blue",
            label="SmallPrimeRuntime",
            kde=False,
            ax=ax,
        )
        # Plot the histogram of BiprimeRuntime
        sns.histplot(
            df_func_rt,
            x="BiprimeRuntime",
            bins="auto",
            color="red",
            label="BiprimeRuntime",
            kde=False,
            ax=ax,
        )

        # plt.hist(df_func_rt["SmallPrimeRuntime"], bins="auto", label="Small prime test")
        # plt.hist(df_func_rt["BiprimeRuntime"], bins="auto", label="Biprime test", color="orange")
        plt.title("Execution times of primality tests")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")

        plt.savefig("out/plot_histogram_primality_tests_runtimes.png")
        tikz_save("out/plot_histogram_primality_tests_runtimes.tex")
        logger.info(
            f"Saved {os.getcwd()}/out/plot_histogram_primality_tests_runtimes.png,tex"
        )
        plt.clf()

    plot_time()
    plot_histogram_small_prime()
    plot_histogram_biprime()
    plot_histogram_function_runtimes()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
        Run the benchmarks to generate the graphs as used in the paper.

        === Parameters ===

        The following parameters are directly passed to
        `from_security_parameter` function of the DistributedPaillier class. To
        find more information about these parameters, please refer to the
        documentation of the DistributedPaillier class or the README.
            - threshold -> corruption_threshold
            - stat-sec-shamir -> stat_sec_shamir
            - test-small-prime-div-param -> prime_treshold
            - test-biprime-param -> correct_param_biprime

        The following parameters are alter the behaviour of the benchmark:
        - iterations -> How often to repeat each experiment. As the protocol is
                        probabilistic, this is necessary to get a good estimate
                        of the runtime. In the paper, iterations=1067 is used
                        (with `--batch-sizes` fixed to 1).
        - batch_sizes ->    To benchmark the performance of the protocol for
                            different batch_sizes, this parameter can be set to
                            a list of values. Recommended is to use a
                            logarithmic scale, i.e. [1, 2, 4, 8, ..., 1024].
                            Beware that this greatly increases the duration of
                            the benchmark and should only be used when
                            benchmarking specifically the batch_sizes. When
                            benchmarking the how often the primality tests fail
                            on a realistic key size (i.e. 1024), either set the
                            batch size equal to the optimal value (faster)
                            or 1 (much slower).

        === Reproducing the graphs from the paper ===

        To get the exact graphs from the paper, run two benchmarks:
        - `python3 benchmark.py --batch-sizes "1,2,4,8,16,32,64,128,256,512,1024" --iterations 100 --key-length 512`
            - Copy:
                - out/plot_time.png
        - `python3 benchmark.py --batch-sizes 1 --iterations 1067 --key-length 1024`
            - Copy:
                - out/plot_histogram_small_prime.png
                - out/plot_histogram_biprime.png

        Beware: when rerunning a benchmark, the plots in the out/ folder will be
        overwritten.

        === Features ===

        - While running, the script shows a progress bar to indicate the
          progress and estimated remaining time.
        - The script is quite robust. Intermidiate results are saved to disk,
          so that the script can be stopped and restarted without losing
          progress.
          The script automatically creates a `./out` folder to store the
          (intermediate) results in.


        """,
    )
    parser.add_argument("--parties", type=int, default=3, help="The number of parties.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="The threshold value for corrupted parties.",
    )
    parser.add_argument("--key-length", type=int, default=512, help="The key length.")
    parser.add_argument(
        "--stat-sec-shamir",
        type=int,
        default=40,
        help="Statistical security parameter.",
    )
    parser.add_argument(
        "--test-small-prime-div-param",
        type=int,
        default=20000,
        help="Upper bound for small prime divisor test.",
    )
    parser.add_argument(
        "--test-biprime-param",
        type=int,
        default=40,
        help="Statistical security parameter for biprime test.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=lambda s: [int(item) for item in s.split(",")],
        default=np.power(2, range(11)),
        help="Batch sizes. Pass a comma-separated list without spaces, like: 1,2,4,8",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="The number of benchmarks to execute for each batch size.",
    )
    args = parser.parse_args()

    global PARTIES, THRESHOLD, KEY_LENGTH, STAT_SEC_SHAMIR, TEST_SMALL_PRIME_DIV_PARAM, TEST_BIPRIME_PARAM, BATCH_SIZES, ITERATIONS

    PARTIES = args.parties
    THRESHOLD = args.threshold
    KEY_LENGTH = args.key_length
    STAT_SEC_SHAMIR = args.stat_sec_shamir
    TEST_SMALL_PRIME_DIV_PARAM = args.test_small_prime_div_param
    TEST_BIPRIME_PARAM = args.test_biprime_param
    BATCH_SIZES = args.batch_sizes
    ITERATIONS = args.iterations

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_benchmark())


if __name__ == "__main__":
    main()
