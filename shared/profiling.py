import asyncio
import logging
import threading
import time
import os
import traceback
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import psutil

# Try to import GPU monitoring modules (optional)
try:
    import GPUtil

    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

# Try to import memory profiler (optional)
try:
    import objgraph

    OBJGRAPH_AVAILABLE = True
except ImportError:
    OBJGRAPH_AVAILABLE = False


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
        # Dictionary to map thread IDs to names for better identification
        self.thread_names = {}
        # Track thread stacks for identification
        self.thread_stacks = {}
        # Per-thread memory usage tracking
        self.thread_memory_samples = defaultdict(list)
        # Check for GPU monitoring availability on startup
        if GPU_MONITORING_AVAILABLE:
            try:
                self.gpus = GPUtil.getGPUs()
                logging.info(f"GPU monitoring enabled. Found {len(self.gpus)} GPUs.")
            except Exception as e:
                logging.warning(f"GPU monitoring initialization failed: {e}")
                self.gpus = []
        else:
            self.gpus = []
            logging.info("GPU monitoring not available. Install GPUtil package for GPU monitoring.")

        # Log if objgraph is available for object tracking
        if OBJGRAPH_AVAILABLE:
            logging.info("Object graph memory analysis available.")
        else:
            logging.info(
                "Object graph memory analysis not available. Install objgraph package for detailed memory tracking."
            )

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

        # Store thread name for better identification
        current_thread = threading.current_thread()
        self.thread_names[thread_id] = current_thread.name

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
            self._active_measurements.remove(loop_name)

    def _get_thread_memory_info(self, thread_id):
        """
        Attempt to get memory information for a specific thread.
        This is OS-specific and might not work on all platforms.
        """
        try:
            # This works primarily on Linux
            if hasattr(self.process, "threads"):
                for thread in self.process.threads():
                    if thread.id == thread_id:
                        return {
                            "cpu_time": thread.user_time + thread.system_time,
                            "cpu_percent": thread.cpu_percent() if hasattr(thread, "cpu_percent") else None,
                        }
        except Exception as e:
            logging.debug(f"Error getting thread memory info: {e}")

        return {"cpu_time": None, "cpu_percent": None}

    def _capture_thread_info(self):
        """Capture current thread stack traces for identification purposes"""
        for thread_id, thread in threading._active.items():
            try:
                # Get stack frame for each thread
                frame = sys._current_frames().get(thread_id)
                if frame:
                    stack = traceback.format_stack(frame)
                    self.thread_stacks[thread_id] = stack
            except Exception as e:
                logging.debug(f"Error capturing thread stack: {e}")

    def _get_gpu_info(self):
        """Get GPU utilization and memory information if available"""
        if not GPU_MONITORING_AVAILABLE or not self.gpus:
            return []

        gpu_info = []
        try:
            # Refresh GPU information
            GPUtil.getGPUs()
            for i, gpu in enumerate(self.gpus):
                gpu_info.append(
                    {
                        "id": i,
                        "name": gpu.name,
                        "load": gpu.load * 100,  # Convert to percentage
                        "memory_total": gpu.memoryTotal,  # In MB
                        "memory_used": gpu.memoryUsed,  # In MB
                        "memory_free": gpu.memoryFree,  # In MB
                        "memory_util": gpu.memoryUtil * 100,  # Convert to percentage
                        "temperature": gpu.temperature,  # In °C
                    }
                )
        except Exception as e:
            logging.warning(f"Error getting GPU information: {e}")

        return gpu_info

    def _sample_thread_memory(self):
        """
        Sample memory usage for all threads.
        This is an approximation as most OSes don't provide per-thread memory usage.
        """
        # Get process memory info as a baseline
        try:
            process_memory = self.process.memory_info()
            total_rss = process_memory.rss / (1024 * 1024)  # Convert to MB

            # Split process memory among active threads based on their CPU usage
            # This is an approximation since accurate per-thread memory is not typically available
            active_threads = list(threading._active.items())

            # If there's only one thread, it gets all the memory
            if len(active_threads) == 1:
                thread_id = active_threads[0][0]
                self.thread_memory_samples[thread_id].append(
                    {"timestamp": datetime.now(), "rss_mb": total_rss, "percent": 100.0}
                )
                return

            # Otherwise, try to distribute based on CPU time
            thread_cpu_times = {}
            total_cpu_time = 0

            for thread_id, _ in active_threads:
                thread_info = self._get_thread_memory_info(thread_id)
                if thread_info["cpu_time"] is not None:
                    thread_cpu_times[thread_id] = thread_info["cpu_time"]
                    total_cpu_time += thread_info["cpu_time"]

            # If we have CPU times, distribute memory proportionally
            if total_cpu_time > 0:
                for thread_id, cpu_time in thread_cpu_times.items():
                    memory_share = (cpu_time / total_cpu_time) * total_rss
                    percent = (cpu_time / total_cpu_time) * 100

                    # Keep last 10 samples for trend analysis
                    samples = self.thread_memory_samples[thread_id]
                    samples.append({"timestamp": datetime.now(), "rss_mb": memory_share, "percent": percent})

                    # Keep only the last 10 samples
                    if len(samples) > 10:
                        self.thread_memory_samples[thread_id] = samples[-10:]
            else:
                # Equal distribution if no CPU time info
                per_thread_memory = total_rss / len(active_threads)
                for thread_id, _ in active_threads:
                    samples = self.thread_memory_samples[thread_id]
                    samples.append(
                        {
                            "timestamp": datetime.now(),
                            "rss_mb": per_thread_memory,
                            "percent": 100.0 / len(active_threads),
                        }
                    )

                    # Keep only the last 10 samples
                    if len(samples) > 10:
                        self.thread_memory_samples[thread_id] = samples[-10:]

        except Exception as e:
            logging.debug(f"Error sampling thread memory: {e}")

    def _get_memory_growth_details(self):
        """Get details about memory usage by object type if objgraph is available"""
        if not OBJGRAPH_AVAILABLE:
            return None

        try:
            # Get top 10 types with most instances
            top_types = objgraph.most_common(10)

            # Get growth stats if supported
            growth_summary = {}
            try:
                growth_stats = objgraph.get_leaking_objects(10)

                # Extract types from growth stats
                for obj in growth_stats:
                    obj_type = type(obj).__name__
                    if obj_type in growth_summary:
                        growth_summary[obj_type] += 1
                    else:
                        growth_summary[obj_type] = 1
            except Exception as e:
                logging.debug(f"Error getting leaking objects: {e}")

            return {"top_types": top_types, "growth_summary": growth_summary}
        except Exception as e:
            logging.warning(f"Error getting memory growth details: {e}")
            return None

    async def print_stats(self):
        logger.debug("Starting Profiler...")
        while True:
            logger.debug("Printing stats...")
            await asyncio.sleep(5)  # Report every 5 minutes
            total_runtime = time.perf_counter() - self.start_time

            logging.info("\n=== Loop Profiling Stats ===")
            logging.info(f"Total wall clock time: {total_runtime:.2f}s")
            logging.info(f"Current time: {datetime.now()}")

            # Get current process memory usage
            process_memory = self.process.memory_info()
            virtual_memory = process_memory.vms / (1024 * 1024)  # Convert to MB
            resident_memory = process_memory.rss / (1024 * 1024)  # Convert to MB

            logging.info(f"Process Memory: RSS {resident_memory:.2f} MB, VMS {virtual_memory:.2f} MB")

            # Get GPU information if available
            gpu_info = self._get_gpu_info()
            if gpu_info:
                logging.info("\n=== GPU Information ===")
                for gpu in gpu_info:
                    logging.info(
                        f"GPU {gpu['id']}: {gpu['name']}\n"
                        f"  Load: {gpu['load']:.1f}%\n"
                        f"  Memory: {gpu['memory_used']:.1f} MB / {gpu['memory_total']:.1f} MB ({gpu['memory_util']:.1f}%)\n"
                        f"  Temperature: {gpu['temperature']}°C"
                    )

            # Sample memory usage for each thread
            self._sample_thread_memory()

            # Get object memory growth details
            memory_growth = self._get_memory_growth_details()
            if memory_growth:
                logging.info("\n=== Memory Growth Analysis ===")

                logging.info("Top 10 Types by Instance Count:")
                for type_name, count in memory_growth["top_types"]:
                    logging.info(f"  {type_name}: {count} instances")

                if memory_growth["growth_summary"]:
                    logging.info("\nPotential Memory Leaks:")
                    for type_name, count in memory_growth["growth_summary"].items():
                        logging.info(f"  {type_name}: {count} leaking instances")

            # Get current process CPU times
            current_process_cpu_times = self.process.cpu_times()
            process_cpu_time_since_last = (current_process_cpu_times.user + current_process_cpu_times.system) - (
                self.last_process_cpu_times.user + self.last_process_cpu_times.system
            )
            self.last_process_cpu_times = current_process_cpu_times

            if process_cpu_time_since_last == 0:
                process_cpu_time_since_last = 1e-6  # Prevent division by zero

            # Capture thread information
            self._capture_thread_info()

            # Display active threads information
            logging.info("\n=== Active Threads Information ===")
            active_threads = threading.enumerate()
            logging.info(f"Total active threads: {len(active_threads)}")

            for thread in active_threads:
                thread_id = thread.ident
                thread_name = thread.name
                thread_alive = thread.is_alive()
                thread_daemon = thread.daemon

                # Get thread memory info if available
                thread_mem_info = self._get_thread_memory_info(thread_id)

                # Calculate memory trend for this thread
                memory_trend = ""
                memory_samples = self.thread_memory_samples.get(thread_id, [])
                if len(memory_samples) >= 2:
                    latest = memory_samples[-1]["rss_mb"]
                    oldest = memory_samples[0]["rss_mb"]

                    if latest > oldest * 1.1:  # 10% growth
                        memory_trend = "↑ INCREASING"
                    elif latest < oldest * 0.9:  # 10% decrease
                        memory_trend = "↓ DECREASING"
                    else:
                        memory_trend = "→ STABLE"

                # Get average memory usage
                avg_memory = 0
                if memory_samples:
                    avg_memory = sum(sample["rss_mb"] for sample in memory_samples) / len(memory_samples)

                # Get associated loops
                associated_loops = [
                    loop_name for loop_name, stats in self.stats.items() if thread_id in stats["thread_ids"]
                ]

                # Get stack trace for identification
                stack_snippet = ""
                if thread_id in self.thread_stacks:
                    stack = self.thread_stacks[thread_id]
                    # Get last few lines of stack for context
                    stack_snippet = "".join(stack[-3:]) if len(stack) >= 3 else "".join(stack)

                # Format CPU time string
                cpu_time_str = (
                    f"{thread_mem_info['cpu_time']:.2f}s" if thread_mem_info["cpu_time"] is not None else "Unknown"
                )
                cpu_percent_str = (
                    f"{thread_mem_info['cpu_percent']:.1f}%"
                    if thread_mem_info["cpu_percent"] is not None
                    else "Unknown"
                )

                # Memory information string
                memory_info = f"  Est. Memory: {avg_memory:.2f} MB {memory_trend}\n" if avg_memory > 0 else ""

                logging.info(
                    f"\nThread ID: {thread_id}, Name: {thread_name}\n"
                    f"  Status: {'Active' if thread_alive else 'Inactive'}, "
                    f"Daemon: {'Yes' if thread_daemon else 'No'}\n"
                    f"  CPU Time: {cpu_time_str}\n"
                    f"  CPU Usage: {cpu_percent_str}\n"
                    f"{memory_info}"
                    f"  Associated Loops: {', '.join(associated_loops) if associated_loops else 'None'}\n"
                    f"  Stack Trace Snippet:\n    {'    '.join(stack_snippet.splitlines(True)) if stack_snippet else 'Not available'}"
                )

            # Sort loops by CPU time
            sorted_stats = sorted(self.stats.items(), key=lambda x: x[1]["total_cpu_time"], reverse=True)

            logging.info("\n=== Loop Performance Stats ===")
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

                    # Get thread names for thread IDs
                    thread_names_list = [
                        f"{tid} ({self.thread_names.get(tid, 'Unknown')})" for tid in stats["thread_ids"]
                    ]

                    logging.info(
                        f"\n{loop_name}:\n"
                        f"  Thread IDs: {thread_names_list}\n"
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
