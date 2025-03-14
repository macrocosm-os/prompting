from shared.prompts.test_time_inference import final_answer_prompt, intro_prompt, system_acceptance_prompt


def test_intro_prompt():
    """Test that intro_prompt returns the correct prompt."""
    prompt = intro_prompt()
    assert isinstance(prompt, str)
    assert "You are a world-class expert in analytical reasoning" in prompt
    assert "OUTPUT FORMAT:" in prompt
    assert "REASONING PROCESS:" in prompt
    assert "REQUIREMENTS:" in prompt
    assert "CRITICAL THINKING CHECKLIST:" in prompt


def test_system_acceptance_prompt():
    """Test that system_acceptance_prompt returns the correct prompt."""
    prompt = system_acceptance_prompt()
    assert isinstance(prompt, str)
    assert "I understand. I will now analyze the problem systematically" in prompt


def test_final_answer_prompt():
    """Test that final_answer_prompt returns the correct prompt."""
    prompt = final_answer_prompt()
    assert isinstance(prompt, str)
    assert "Review your previous reasoning steps" in prompt
    assert "Format your response as valid JSON" in prompt
    assert '"title":' in prompt
    assert '"content":' in prompt
