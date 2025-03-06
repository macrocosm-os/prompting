import asyncio
import time
from validator_api.deep_research.orchestrator import deep_research_agent
from loguru import logger

query = "Write a script that uses bootstrapping to estimate a range on parameters for a polynomial regression model and then plots the data as well as the model with bounds"
# query = "I want to send a 2.5kg package from Germany to the UK. What is the cheapest way to do this and what documentation do I need to provide?"
logger.info(f"Starting research process for query: {query}")
print(f"\nResearching: {query}\n")
print("This may take some time depending on the complexity of the question...\n")

start_time = time.time()
state = asyncio.run(deep_research_agent(query))
end_time = time.time()

print("\n==== Final Answer ====")
print(state.current_context)
execution_time = end_time - start_time
print(f"\nResearch completed in {execution_time:.2f} seconds")
logger.info(f"Research completed in {execution_time:.2f} seconds")