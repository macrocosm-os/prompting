from collections import deque


reward_events: list = []
# rewards_and_uids: list[tuple[list[int], list[float]]] = []
scoring_queue: list = []
task_queue: list = []
# Deque holds ScoringConfig type, skipped to avoid circular import.
task_responses: deque = deque(maxlen=10_000)
