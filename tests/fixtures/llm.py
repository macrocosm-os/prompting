from prompting.mock import MockPipeline
from prompting.llms import vLLM_LLM, HuggingFaceLLM, HuggingFacePipeline, vLLMPipeline


def mock_llm_pipeline(message="This is just another test."):
    return MockPipeline(message)


def llms(message="This is just another test."):
    pipeline = MockPipeline(message)
    llms = [vLLM_LLM(pipeline, ""), HuggingFaceLLM(pipeline, "")]
    return llms


def pipelines():
    # Return pipeline types to be instantiated downstream
    return [HuggingFacePipeline, vLLMPipeline]
