from prompting.mock import MockPipeline
from prompting.llms import vLLM_LLM, HuggingFaceLLM, HuggingFacePipeline, vLLMPipeline

def mock_llm_pipeline():    
    return MockPipeline("This is just another test.")

def llms():    
    pipeline = MockPipeline("This is just another test.")
    llms = [
        vLLM_LLM(pipeline, ''),
        HuggingFaceLLM(pipeline, '')
    ]
    return llms

def pipelines():
    # Return pipeline types to be instantiated downstream
    return [
        HuggingFacePipeline,
        vLLMPipeline
    ]    