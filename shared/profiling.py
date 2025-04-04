import asyncio
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import psutil


class LoopProfiler:
    def __init__(self):
        self.stats = defaultdict(
            lambda: {
                "total_wall_time": 0,
                "total_cpu_time": 0,
                "iterations": 0,
                "min_time": float("inf"),
                "max_time": 0,
                "last_iteration_start": None,
                "last_iteration_end": None,
                "thread_ids": set(),
            }
        )
        self.start_time = time.perf_counter()
        self._active_measurements = set()
        self.process = psutil.Process()
        # Initialize process CPU times
        self.last_process_cpu_times = self.process.cpu_times()

    @contextmanager
    def measure(self, loop_name):
        if loop_name in self._active_measurements:
            logging.warning(f"Nested measurement detected for {loop_name}")

        self._active_measurements.add(loop_name)
        stats = self.stats[loop_name]
        stats["last_iteration_start"] = datetime.now()
        wall_start = time.perf_counter()
        thread_cpu_start = time.thread_time()
        thread_id = threading.get_ident()
        stats["thread_ids"].add(thread_id)

        try:
            yield
        finally:
            wall_duration = time.perf_counter() - wall_start
            thread_cpu_duration = time.thread_time() - thread_cpu_start

            # Update stats
            stats["total_wall_time"] += wall_duration
            stats["total_cpu_time"] += thread_cpu_duration
            stats["iterations"] += 1
            stats["min_time"] = min(stats["min_time"], wall_duration)
            stats["max_time"] = max(stats["max_time"], wall_duration)
            stats["last_iteration_end"] = datetime.now()
            try:
                self._active_measurements.remove(loop_name)
            except KeyError:
                pass

    async def print_stats(self):
        while True:
            await asyncio.sleep(5 * 60)  # Report every 5 minutes
            total_runtime = time.perf_counter() - self.start_time

            logging.info("\n=== Loop Profiling Stats ===")
            logging.info(f"Total wall clock time: {total_runtime:.2f}s")
            logging.info(f"Current time: {datetime.now()}")

            # Get current process CPU times
            current_process_cpu_times = self.process.cpu_times()
            process_cpu_time_since_last = (current_process_cpu_times.user + current_process_cpu_times.system) - (
                self.last_process_cpu_times.user + self.last_process_cpu_times.system
            )
            self.last_process_cpu_times = current_process_cpu_times

            if process_cpu_time_since_last == 0:
                process_cpu_time_since_last = 1e-6  # Prevent division by zero

            # Sort loops by CPU time
            sorted_stats = sorted(self.stats.items(), key=lambda x: x[1]["total_cpu_time"], reverse=True)

            for loop_name, stats in sorted_stats:
                if stats["iterations"] > 0:
                    avg_wall_time = stats["total_wall_time"] / stats["iterations"]
                    avg_cpu_time = stats["total_cpu_time"] / stats["iterations"]
                    wall_percent = (stats["total_wall_time"] / total_runtime) * 100
                    # CPU percent relative to process CPU time since last report
                    cpu_percent = (stats["total_cpu_time"] / process_cpu_time_since_last) * 100

                    last_run = stats["last_iteration_end"]
                    time_since_last = datetime.now() - last_run if last_run else None

                    # Calculate time spent waiting (wall time - CPU time)
                    wait_time = stats["total_wall_time"] - stats["total_cpu_time"]
                    wait_percent = (wait_time / stats["total_wall_time"] * 100) if stats["total_wall_time"] > 0 else 0

                    logging.info(
                        f"\n{loop_name}:\n"
                        f"  Thread IDs: {list(stats['thread_ids'])}\n"
                        f"  Wall clock time: {stats['total_wall_time']:.2f}s ({wall_percent:.1f}%)\n"
                        f"  CPU time: {stats['total_cpu_time']:.2f}s ({cpu_percent:.1f}%)\n"
                        f"  Wait time: {wait_time:.2f}s ({wait_percent:.1f}% of wall time)\n"
                        f"  Iterations: {stats['iterations']}\n"
                        f"  Avg wall time/iter: {avg_wall_time*1000:.2f}ms\n"
                        f"  Avg CPU time/iter: {avg_cpu_time*1000:.2f}ms\n"
                        f"  Min wall time: {stats['min_time']*1000:.2f}ms\n"
                        f"  Max wall time: {stats['max_time']*1000:.2f}ms\n"
                        f"  Last iteration: {stats['last_iteration_end'].strftime('%H:%M:%S.%f')}\n"
                        f"  Time since last: {time_since_last.total_seconds():.1f}s ago"
                        if time_since_last
                        else "Never completed"
                    )

            # List any loops that haven't reported
            all_known_loops = {
                "ModelScheduler",
                "TaskLoop",
                "TaskScorer",
                "CheckMinerAvailability",
                "WeightSetter",
            }
            missing_loops = all_known_loops - set(self.stats.keys())
            if missing_loops:
                logging.warning(f"\nLoops with no measurements: {', '.join(missing_loops)}")

            # Warn about any currently running measurements
            if self._active_measurements:
                logging.warning(f"\nCurrently running measurements: {', '.join(self._active_measurements)}")

            logging.info("\n========================")


# Create a global profiler instance
profiler = LoopProfiler()
