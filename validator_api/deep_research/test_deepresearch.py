import asyncio
import time
from validator_api.deep_research.orchestrator import deep_research_agent
from loguru import logger
from datetime import datetime
# Configure loguru to write logs to a file
# query = "Write a script that uses bootstrapping to estimate a range on parameters for a polynomial regression model and then plots the data as well as the model with bounds"
# query = "What is the best way to send a 2.5kg package from Germany to the UK? The package is being sent from my parents to me. Do I need to pay any taxes or customs fees?"
# query = """Write a simple neural network that can predict whether the user will click the "x" or the "z" button. The user will try to click in a random pattern, and the neuron net should try and predict the next click. Make it able to learn as the user is clicking"""
# query="""Using pythons turtle graphics, draw a infinity symbol"""
# query="Assess the top 15 subnets on Bittensor today and which 3 are the most undervalued."
query="Train a stock market prediction model and forecast NVIDIA prices for the next 7 days"
logger.add(f"./dr_logs/{datetime.now().strftime('%Y-%m-%d_%H-%M')}_deep_research.log", rotation="1 MB", retention="10 days", level="DEBUG")
# query = "I want to send a 2.5kg package from Germany to the UK. What is the cheapest way to do this and what documentation do I need to provide?"
logger.info(f"Starting research process for query: {query}")
logger.info(f"\nResearching: {query}\n")
logger.info("This may take some time depending on the complexity of the question...\n")

start_time = time.time()
state = asyncio.run(deep_research_agent(query))
end_time = time.time()

logger.info("\n==== Final Answer ====")
logger.info(state.current_context)
execution_time = end_time - start_time
logger.info(f"\nResearch completed in {execution_time:.2f} seconds")
logger.info(f"Research completed in {execution_time:.2f} seconds")