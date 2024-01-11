from langchain.prompts import ChatPromptTemplate
from dataclasses import dataclass
from langchain_core.output_parsers import StrOutputParser
from utils import get_model_name_from_llm
import time

@dataclass
class NetworkResponse:
    miner_id: str
    response: str
    time: float


class Miner:
    def __init__(self, llm):
        self.llm = llm
        self.define_system_prompt()

    def define_system_prompt(self):        
        self.system_prompt = "You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know."
        

    def query(self, prompt: str) -> NetworkResponse:
        start_time = time.time()  # Start timing

        chain_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{input}")
        ])

        output_parser = StrOutputParser()

        reference_chain = chain_prompt | self.llm | output_parser
        reference = reference_chain.invoke({"input": prompt})

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time

        model_name = get_model_name_from_llm(self.llm)

        response = NetworkResponse(model_name, reference, elapsed_time)

        return response

 

class MockMiner:
    def __init__(self, default_response: str, miner_id: str):
        self.default_response = default_response
        self.miner_id = miner_id

    def query(self, prompt: str) -> NetworkResponse:        
        response = NetworkResponse(miner_id=self.miner_id, response=self.default_response, time=0.01)
        return response