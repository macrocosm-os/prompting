import time
import csv
import os
from datetime import datetime

LOG_TIMINGS = True

# Create log file name when module is loaded
STARTUP_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M")
LOG_FILE = f"timer_logs_{STARTUP_TIME}.csv"

# Create CSV file with headers if it doesn't exist
if LOG_TIMINGS and not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "label", "duration_seconds", "metadata"])


class Timer:
    def __init__(self, label="", metadata=None):
        self.label = label
        self.metadata = metadata or {}

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_datetime = datetime.now()
        return self

    def elapsed_time(self):
        return time.perf_counter() - self.start_time

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.final_time = self.end_time - self.start_time

        if LOG_TIMINGS:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.start_datetime.isoformat(), self.label, self.final_time, str(self.metadata)])
