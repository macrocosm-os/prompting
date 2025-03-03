import pytest

from shared.prompts import get_prompt


def test_get_prompt_valid_formats():
    """Test that get_prompt returns the correct prompts for valid format types."""
    # Test intro prompt
    intro_prompt = get_prompt("test_time_inference", "intro_prompt")
    assert isinstance(intro_prompt, str)
    assert "You are a world-class expert in analytical reasoning" in intro_prompt

    # Test system acceptance prompt
    system_prompt = get_prompt("test_time_inference", "system_acceptance_prompt")
    assert isinstance(system_prompt, str)
    assert "I understand. I will now analyze the problem systematically" in system_prompt

    # Test final answer prompt
    final_prompt = get_prompt("test_time_inference", "final_answer_prompt")
    assert isinstance(final_prompt, str)
    assert "Review your previous reasoning steps" in final_prompt


def test_get_prompt_invalid_format():
    """Test that get_prompt raises ValueError for invalid format types."""
    with pytest.raises(ValueError) as exc_info:
        get_prompt("test_time_inference", "invalid_format")
    assert "Format type invalid_format not found in inference type test_time_inference" in str(exc_info.value)


def test_get_prompt_invalid_inference_type():
    """Test that get_prompt raises KeyError for invalid inference types."""
    with pytest.raises(KeyError):
        get_prompt("invalid_inference_type", "intro_prompt")


def test_get_prompt_empty_strings():
    """Test that get_prompt handles empty strings appropriately."""
    with pytest.raises(ValueError):
        get_prompt("", "intro_prompt")
    with pytest.raises(ValueError):
        get_prompt("test_time_inference", "")


def test_get_prompt_type_validation():
    """Test that get_prompt validates input types correctly."""
    with pytest.raises(TypeError):
        get_prompt(None, "intro_prompt")
    with pytest.raises(TypeError):
        get_prompt("test_time_inference", None)
    with pytest.raises(TypeError):
        get_prompt(123, "intro_prompt")
    with pytest.raises(TypeError):
        get_prompt("test_time_inference", 123)


def test_get_prompt_content_validation():
    """Test that returned prompts contain expected content structure."""
    intro_prompt = get_prompt("test_time_inference", "intro_prompt")
    assert "OUTPUT FORMAT:" in intro_prompt
    assert "REASONING PROCESS:" in intro_prompt
    assert "REQUIREMENTS:" in intro_prompt
    assert "CRITICAL THINKING CHECKLIST:" in intro_prompt


def test_get_prompt_response_format():
    """Test that the response format contains all required fields."""
    final_prompt = get_prompt("test_time_inference", "final_answer_prompt")
    assert "Format your response as valid JSON" in final_prompt
    assert '"title":' in final_prompt
    assert '"content":' in final_prompt
