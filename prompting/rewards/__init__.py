from .reward import (
    BaseRewardModel,
    RewardResult,
    RewardEvent,
    BatchRewardOutput,
    RewardModelTypeEnum,
)
from .code_diff import DiffRewardModel
from .relevance import RelevanceRewardModel
from .rouge import RougeRewardModel
from .float_diff import FloatDiffModel
from .date import DateRewardModel
from .ordinal import OrdinalRewardModel
from .multiple_choice import MultipleChoiceModel
from .streaming import StreamingRewardModel

REWARD_MODELS = {
    "rouge": RougeRewardModel,
    "relevance": RelevanceRewardModel,
    "diff": DiffRewardModel,
    "float_diff": FloatDiffModel,
    "date": DateRewardModel,
    "ordinal": OrdinalRewardModel,
    "streaming": StreamingRewardModel,
    "multiple_choice": MultipleChoiceModel,
}
