import random
from pydantic import BaseModel

PROFILES = [
    "student",
    "teacher",
    "parent",
    "hacker",
    "programmer",
    "scientist",
]
# profiles = ["16 year old highschool student", ...

# TODO: more terse, less verbose
MOOD = [
    "an interested",
    "a concerned",
    "an impatient",
    "a tired",
    "a confused",
    "an annoyed",
    "a curious",
    "an upbeat",
    "a lazy",
]
TONE = [
    "formal",
    "informal",
    "indifferent",
    "casual",
    "rushed",
    "polite",
    "impolite",
    "friendly",
    "unfriendly",
    "positive",
    "negative",
]


class Persona(BaseModel):
    mood: str = random.choice(MOOD)
    tone: str = random.choice(TONE)
    profile: str = random.choice(PROFILES)
