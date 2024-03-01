import bittensor as bt
from prompting.miners.agents.utils import load_hf_llm
from prompting.miners.agents.base_agent import BaseAgent
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import Tool
from langchain.tools import WikipediaQueryRun


class ReactAgent(BaseAgent):
    def __init__(
        self,
        model_id: str,
        model_temperature: float,
        max_new_tokens: int = 1024,
        load_in_8bits: bool = False,
        load_in_4bits: bool = False,
    ):
        self.wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools = [
            Tool(
                name="Wikipedia",
                func=self.wikipedia.run,
                description="Useful for when you need to look up a topic, event, country or person on wikipedia",
            )
        ]

        bt.logging.info(
            f"""Initializing ReACT agent with follow parameters:
        - model_temperature: {model_temperature}
        - max_new_tokens: {max_new_tokens}
        - load_in_8bits: {load_in_8bits}
        - load_in_4bits: {load_in_4bits}"""
        )

        prompt = hub.pull("hwchase17/react")

        if "gpt" not in model_id:
            llm = load_hf_llm(model_id, max_new_tokens, load_in_8bits, load_in_4bits)
        else:
            llm = ChatOpenAI(model_name=model_id, temperature=model_temperature)

        # Construct the ReAct agent
        agent = create_react_agent(llm, tools, prompt)

        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

    def run(self, input: str) -> str:
        response = self.agent_executor.invoke({"input": input})["output"]
        return response
