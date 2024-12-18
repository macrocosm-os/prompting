import time


class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def elapsed_time(self):
        return self.start_time - time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.final_time = self.end_time - self.start_time
