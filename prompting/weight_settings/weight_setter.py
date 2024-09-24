from prompting.base.loop_runner import AsyncLoopRunner


class WeightSetter(AsyncLoopRunner):
    interval: int = 3600  # run once every hour
    sync: bool = True  # all validators should update at the same time

    def run_step(self):
        return 0
