from validator_api.deep_research.shared_resources import client
from loguru import logger

async def reasoning_tool(query: str, context: str = "") -> str:
    """
    Use OpenAI for chain-of-thought reasoning
    """
    # Create prompt for reasoning
    prompt = f"""
QUERY: {query}

CONTEXT (if available):
{context}

Please think through this problem step by step to reach a well-reasoned conclusion.
First, break down the query into its key components.
Then analyze each component systematically.
Consider multiple perspectives and potential approaches.
Evaluate the evidence and reasoning for each approach.
Finally, provide your conclusion based on this analysis.

Each thinking step should be distinct with a title (What are you thinking about?) and the content (The actual thoughts). You should go through at least 5 steps and consider all possible angles to come to your final conclusion.
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a reasoning assistant with strong analytical capabilities. Think through problems step by step."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        return f"Error during reasoning: {str(e)}"
