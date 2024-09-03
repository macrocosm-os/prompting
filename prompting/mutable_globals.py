TASK_QUEUE_LENGTH_THRESHOLD = 20
SCORING_QUEUE_LENGTH_THRESHOLD = 10

task_queue: list = []
rewards_and_uids: list[tuple[list[int], list[float]]] = []
scoring_queue: list = []
