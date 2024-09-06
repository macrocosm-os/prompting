TASK_QUEUE_LENGTH_THRESHOLD = 20
SCORING_QUEUE_LENGTH_THRESHOLD = 10

task_queue: list = []
scoring_queue: list = []
reward_events: list[list] = []  # list[list[WeightedRewardEvent]]
