import bittensor as bt
from prompting.miners.agents import get_tools, load_hf_llm, BaseAgent
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain import OpenAI


class ReactAgent(BaseAgent):
    def __init__(self,
            model_id: str, 
            model_temperature: float,            
            max_new_tokens: int = 1024,
            load_in_8bits: bool = False,
            load_in_4bits: bool = False
    ):
        tools = get_tools()

        bt.logging.info(f"""Initializing ReACT agent with follow parameters:
        - model_temperature: {model_temperature}
        - max_new_tokens: {max_new_tokens}
        - load_in_8bits: {load_in_8bits}
        - load_in_4bits: {load_in_4bits}"""
        )

        prompt = hub.pull("hwchase17/react")

        if 'gpt' not in model_id:            
            llm = load_hf_llm(model_id, max_new_tokens, load_in_8bits, load_in_4bits)
        else:
            llm = OpenAI(model_name=model_id, temperature=model_temperature)


        # Choose the LLM to use
        llm = OpenAI(openai_api_key='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')

        # Construct the ReAct agent
        agent = create_react_agent(llm, tools, prompt)

        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)    


    def run(self, input: str) -> str:
        return self.agent_executor.run(input)

