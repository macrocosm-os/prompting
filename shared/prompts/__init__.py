from .test_time_inference import RESPONSE_FORMATS as INFERENCE_RESPONSE_FORMATS

ALL_PROMPTS = {
    "test_time_inference": INFERENCE_RESPONSE_FORMATS,
}


def get_prompt(inference_type: str, format_type: str) -> dict | str:
    """
    Retrieves the content from RESPONSE_FORMAT based on the desired
    inference type and format type.

    Args:
        inference_type (str): The type of inference to get prompts for
        format_type (str): The specific format type to retrieve

    Returns:
        dict | str: The prompt content

    Raises:
        TypeError: If inference_type or format_type are not strings
        ValueError: If inference_type or format_type are empty strings
        KeyError: If inference_type is not found in ALL_PROMPTS
        ValueError: If format_type is not found in the inference type's formats
    """
    # Input validation
    if not isinstance(inference_type, str) or not isinstance(format_type, str):
        raise TypeError("Both inference_type and format_type must be strings")
    
    if not inference_type or not format_type:
        raise ValueError("Both inference_type and format_type must be non-empty strings")

    if inference_type not in ALL_PROMPTS:
        raise KeyError(f"Inference type {inference_type} not found in available prompts")

    inference_response_format = ALL_PROMPTS[inference_type]

    if format_type not in inference_response_format:
        raise ValueError(f"Format type {format_type} not found in inference type {inference_type}")

    return inference_response_format[format_type]
