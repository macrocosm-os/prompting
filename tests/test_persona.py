import pytest
from prompting.persona import Persona, create_persona

def test_persona_initialization():
    assert type(create_persona()) == Persona

def test_persona_contains_mood():
    assert create_persona().mood is not None

def test_persona_contains_tone():
    assert create_persona().tone is not None

def test_persona_contains_profile():
    assert create_persona().profile is not None