from typing import Any, Callable, Type

from loguru import logger
from pydantic import BaseModel

_serializers: dict[Type, Callable[[Any], Any]] = {}


def register_serializer(cls: Type):
    def wrapper(func: Callable[[Any], Any]):
        _serializers[cls] = func
        return func

    return wrapper


def recursive_model_dump(obj: Any, path: str = "") -> Any:
    # Check custom serializer
    for cls, serializer in _serializers.items():
        if isinstance(obj, cls):
            return serializer(obj)

    # If it's a Pydantic model, skip .model_dump() and use __dict__
    if isinstance(obj, BaseModel):
        try:
            data = vars(obj)  # Same as obj.__dict__
        except Exception as e:
            logger.error(f"❌ Failed to dump __dict__ at {path}: {type(obj)} — {e}")
            raise e
        result = {}
        for k, v in data.items():
            result[k] = recursive_model_dump(v, f"{path}.{k}")
        return result

    elif isinstance(obj, dict):
        return {k: recursive_model_dump(v, f"{path}.{k}") for k, v in obj.items()}

    elif isinstance(obj, list):
        return [recursive_model_dump(v, f"{path}[{i}]") for i, v in enumerate(obj)]

    elif hasattr(obj, "__dict__"):
        try:
            return recursive_model_dump(vars(obj), f"{path}.__dict__")
        except Exception as e:
            logger.error(f"❌ Failed to dump __dict__ at {path}: {type(obj)} — {e}")
            raise e

    return obj
