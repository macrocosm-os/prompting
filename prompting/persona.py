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
        "person",
        "researcher",
        "writer",
        "producer",
        "musician",
        "gen z influencer",
        "teacher",
        "parent",
        "hacker",
        "programmer",
        "highschool student",
        "college student",
        "scientist",
        "pensioner",
        "military commander",
        "millenial",
        "non-native english speaker"
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
        "a terse",
        "a playful",
        "a hyped",
        "a foul",
        "a celebratory",
        "an indignant",
        "a generally unhappy",
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
        "skeptical",
        "apologetic",
        "unapologetic",
        "nonchalant",
        "eager",
        "confused",
    ]
    # TODO: we can lower case the human messages, add common grammar and spelling mistakes...

    return Persona(
        profile=random.choice(profiles),
        mood=random.choice(mood),
        tone=random.choice(tone),
    )
