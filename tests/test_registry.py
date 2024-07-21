from prompting.tasks import TASKS
from prompting.tools import DATASETS
from prompting.task_registry import TASK_REGISTRY


# TODO: Create more detailed tests.
def test_task_registry():
    registry_missing_task = set(TASK_REGISTRY.keys()) - set(TASKS.keys())
    registry_extra_task = set(TASKS.keys()) - set(TASK_REGISTRY.keys())
    assert (
        not registry_missing_task
    ), f"Missing tasks in TASK_REGISTRY: {registry_missing_task}"
    assert (
        not registry_extra_task
    ), f"Extra tasks in TASK_REGISTRY: {registry_extra_task}"


def test_task_registry_datasets():
    registry_datasets = set(
        [dataset for task, datasets in TASK_REGISTRY.items() for dataset in datasets]
    )
    registry_missing_dataset = registry_datasets - set(DATASETS.keys())
    registry_extra_dataset = set(DATASETS.keys()) - registry_datasets
    assert (
        not registry_missing_dataset
    ), f"Missing datasets in TASK_REGISTRY: {registry_missing_dataset}"
    assert (
        not registry_extra_dataset
    ), f"Extra datasets in TASK_REGISTRY: {registry_extra_dataset}"
