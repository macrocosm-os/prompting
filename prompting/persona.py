import random
from dataclasses import dataclass


@dataclass
class Persona:
    profile: str
    mood: str
    tone: str


def create_persona() -> Persona:
    """Defines the persona of the user. This is used to create the system prompt. It dictates the style of the agent's questions and communication."""
    profiles = [
        "student",
        "teacher",
        "parent",
        "hacker",
        "programmer",
        "scientist",
    ]
    # profiles = ["16 year old highschool student", ...

    # TODO: more terse, less verbose
    mood = [
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
    tone = [
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
    # TODO: we can lower case the human messages, add common grammar and spelling mistakes...

    return Persona(
        profile=random.choice(profiles),
        mood=random.choice(mood),
        tone=random.choice(tone),
    )
