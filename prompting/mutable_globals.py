import multiprocessing as mp

manager = mp.Manager()

reward_events = manager.list()
scoring_queue = manager.list()
task_queue = manager.list()
