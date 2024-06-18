# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Define the version of the template module.
__version__ = "2.5.0"
version_split = __version__.split(".")
__spec_version__ = (
    (10000 * int(version_split[0]))
    + (100 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# Import all submodules.
from . import protocol
from . import base
from . import rewards
from . import tasks
from . import tools
from . import utils

from . import forward
from . import agent
from . import conversation
from . import dendrite
from . import shared
from . import validator

from .llms import hf

from .tasks import TASKS
from .tools import DATASETS
from .task_registry import TASK_REGISTRY

# Assert that all tasks have a dataset, and all tasks/datasets are in the TASKS and DATASETS dictionaries.
registry_missing_task = set(TASKS.keys()) - set(TASK_REGISTRY.keys())
registry_extra_task = set(TASK_REGISTRY.keys()) - set(TASKS.keys())
assert (
    not registry_missing_task
), f"Missing tasks in TASK_REGISTRY: {registry_missing_task}"
assert not registry_extra_task, f"Extra tasks in TASK_REGISTRY: {registry_extra_task}"

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
