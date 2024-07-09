from prompting.tasks import TASKS


VALIDATOR_MODEL_ID = "casperhansen/llama-3-70b-instruct-awq"
TASKS = list(TASKS.keys())
TASKS_P = [1.0 / len(TASKS)] * len(TASKS)
MAX_TOKENS = 256
NEURON_TIMEOUT = 15
SAMPLE_SIZE = 100
MOVING_AVERAGE_ALPHA = 0.1
DECAY_ALPHA = 0.001
QUERY_UNIQUE_COLDKEYS = False
QUERY_UNIQUE_IPS = False
