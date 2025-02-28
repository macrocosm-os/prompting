from .test_time_inference import RESPONSE_FORMATS as INFERENCE_RESPONSE_FORMATS

ALL_PROMPTS = {
    "test_time_inference": INFERENCE_RESPONSE_FORMATS,
}


def get_prompt(inference_type: str, format_type: str) -> dict | str:
    """
    Retrieves the content from RESPONSE_FORMAT based on the desired
    inference type and format type.
    """
    inference_response_format = ALL_PROMPTS[inference_type]

    if format_type not in inference_response_format:
        raise ValueError(f"Format type {format_type} not found in inference type {inference_type}")

    return inference_response_format[format_type]
