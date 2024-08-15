from loguru import logger


class Version:
    """Same as packaging.version, but also supports comparison to strings"""

    def __init__(self, version: str):
        self.version: str = version

    def __str__(self):
        return f"{self.version}"

    def __repr__(self):
        return f"{self.version}"

    def __eq__(self, other):
        other = other.version if isinstance(other, Version) else other
        return self.version == other

    def __le__(self, other):
        other = other.version if isinstance(other, Version) else other
        return True if all([v <= o for v, o in zip(self.version.split("."), other.split("."))]) else False

    def __lt__(self, other):
        other = other.version if isinstance(other, Version) else other
        return True if self <= other and self != other else False

    def __ge__(self, other):
        other = other.version if isinstance(other, Version) else other
        return True if not (self < other) else False

    def __gt__(self, other):
        other = other.version if isinstance(other, Version) else other
        return True if not (self <= other) else False


__version__ = Version("2.7.0")
logger.info(f"Prompting version: {__version__}")
