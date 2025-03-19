from pydantic import BaseModel
from validator_api.deep_research.utils import convert_to_gemma_messages, with_retries, parse_llm_json
from validator_api.gpt_endpoints import web_retrieval, WebRetrievalRequest
from shared.settings import shared_settings
from mistralai import Mistral
from loguru import logger
import json
from typing import Any


class LLMQuery(BaseModel):
    """Records a single LLM API call with its inputs and outputs"""
    messages: list[dict]  # The input messages
    raw_response: str     # The raw response from the LLM
    parsed_response: Any | None = None  # The parsed response (if applicable)
    step_name: str       # Name of the step that made this query
    timestamp: float     # When the query was made
    model: str          # Which model was used

async def search_web(query: str, n_results: int = 5) -> list[str]:
    """Searches the web for the query using the web_retrieval endpoint"""
    response = await web_retrieval(WebRetrievalRequest(search_query=query, n_results=n_results))
    return response.results

def make_mistral_request(messages: list[dict], step_name: str) -> tuple[str, LLMQuery]:
    """Makes a request to Mistral API and records the query"""
    import time
    
    logger.debug(f"Making Mistral API request with messages:\n{json.dumps(messages, indent=2)}")
    model = "mistral-small-latest"

    client = Mistral(api_key=shared_settings.GEMMA_API_KEY)

    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    response_content = chat_response.choices[0].message.content
    logger.debug(f"Raw Mistral API response:\n{response_content}")
    
    # Record the query
    query_record = LLMQuery(
        messages=messages,
        raw_response=response_content,
        step_name=step_name,
        timestamp=time.time(),
        model=model
    )
    
    return response_content, query_record

class Step(BaseModel):
    title: str
    content: str
    next_step: str | None = None
    summary: str | None = None

    def __str__(self):
        return f"Title: {self.title}\nContent: {self.content}\nNext Step: {self.next_step}\nSummary: {self.summary}"

class StepManager(BaseModel):
    steps: list[Step]

    def __str__(self):
        output = "Here is the list of steps that were already completed:\n\n"
        for i, step in enumerate(self.steps):
            output += f"Step {i+1}:\n{step}\n\n"
        return output


class Orchestrator(BaseModel):
    todo_list: str | None = None
    current_step: int | None = None
    user_messages: str | None = None
    max_steps: int = 10
    completed_steps: StepManager = StepManager(steps=[])
    query_history: list[LLMQuery] = []
    
    def run(self, messages):
        logger.info(f"Starting orchestration run with {len(messages)} messages")
        self.user_messages = messages
        logger.debug(f"Generating initial todo list")
        self.generate_todo_list()
        for step in range(self.max_steps):
            logger.info(f"Processing step {step + 1}/{self.max_steps}")
            self.current_step = step + 1
            thinking_result = self.do_thinking()
            logger.debug(f"Thinking result for step {step + 1}: {thinking_result}")
            
            if thinking_result.next_step == "generate_final_answer":
                logger.info("Reached final step, generating final answer")
                final_answer = self.generate_final_answer()
                return {
                    "final_answer": final_answer,
                    "query_history": self.query_history
                }
            
            logger.debug("Updating todo list based on latest thinking")
            self.update_todo_list()
            
        final_answer = self.generate_final_answer()
        return {
            "final_answer": final_answer,
            "query_history": self.query_history
        }

    @with_retries(max_retries=3)
    def generate_todo_list(self):
        """Uses mistral LLM to generate a todo list for the Chain of Thought process"""
        logger.info("Generating todo list")
        prompt = """Based on the conversation history provided, create a detailed, step-by-step todo list that outlines the complete thought process needed to address the user's request. For each step:

1. Break down complex tasks into smaller, manageable sub-tasks
2. Consider potential dependencies between tasks
3. Include validation and verification steps where necessary
4. Account for edge cases and potential issues
5. Add specific research or information gathering tasks when needed

Format your response as a numbered list where each item follows this structure:
1. [Task Name]: Brief description of what needs to be done
   - Key considerations: What specific aspects need attention
   - Success criteria: How to verify this step is complete
   - Dependencies: Any prerequisites or related tasks

Your todo list should be comprehensive enough to serve as a reliable roadmap for solving the problem systematically.
"""
        logger.debug(f"User messages for todo list generation:\n{self.user_messages}")
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Here is the conversation history to base the todo list on:\n{self.user_messages}"}
        ]
        
        response, query_record = make_mistral_request(messages, "generate_todo_list")
        self.query_history.append(query_record)
        self.todo_list = response
        logger.info("Todo list generated successfully")
        logger.debug(f"Generated todo list:\n{self.todo_list}")
        return self.todo_list

    @with_retries(max_retries=3)
    def do_thinking(self):
        """Uses mistral LLM to generate thinking/reasoning tokens in line with the todo list"""
        logger.info(f"Starting thinking process for step {self.current_step}")
        logger.debug(f"Current todo list state:\n{self.todo_list}")
        
        prompt = f"""You are a systematic problem solver working through a complex task step by step. You have a todo list to follow, and you're currently on step {self.current_step}. Your goal is to think deeply about this step and provide clear, logical reasoning.

Here is your todo list (✓ marks completed steps):
{self.todo_list}

Find the first unchecked item in the todo list (items without a ✓) and analyze that step. Provide your response in the following JSON format:
{{
    "thinking_step_title": "Title of the current todo list step being analyzed",
    "thoughts": "Your detailed analysis and reasoning about this step, including:
                - Step-by-step reasoning process
                - Consideration of edge cases and potential issues
                - References to previous steps if relevant
                - Validation of your approach
                - Summary of the process that clearly states the answer to the todo list step",
    "summary": "A concise summary of your conclusions and key takeaways from this step",
    "next_action": "Either 'continue_thinking' if there are more unchecked todo steps to process, or 'generate_final_answer' if all steps are checked"
}}"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Here is the conversation history to base your thinking on:\n{self.user_messages}"}
        ]
        
        thinking_output, query_record = make_mistral_request(messages, f"thinking_step_{self.current_step}")
        logger.debug(f"Raw thinking output:\n{thinking_output}")
        
        try:
            thinking_dict = parse_llm_json(thinking_output)
            query_record.parsed_response = thinking_dict
            self.query_history.append(query_record)
            
            step = Step(
                title=thinking_dict["thinking_step_title"],
                content=thinking_dict["thoughts"],
                next_step=thinking_dict["next_action"],
                summary=thinking_dict["summary"]
            )
            logger.info(f"Created thinking step: {step.title} with next_action: {step.next_step}")
            self.completed_steps.steps.append(step)
            return step
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse thinking output as JSON: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing required key in thinking output: {e}")
            raise

    @with_retries(max_retries=3)
    def generate_final_answer(self):
        """Uses mistral LLM to generate a final answer to the user's request"""
        logger.info("Generating final answer")
        logger.debug(f"Completed steps for final answer:\n{self.completed_steps}")
        
        prompt = f"""You are tasked with generating a comprehensive final answer that thoroughly analyzes all aspects of the problem and its solution. Your goal is to provide a complete understanding that covers all angles and perspectives.

Here is the original todo list (✓ marks completed steps) and all completed thinking steps:

TODO list:
{self.todo_list}

COMPLETED THINKING STEPS:
{self.completed_steps}

Verify that all items in the todo list are marked with a checkmark (✓). If any items are not marked as complete, consider if they were actually addressed in the thinking steps or if they can be skipped as no longer relevant.

To ensure a comprehensive analysis, consider the following aspects:
1. Core Solution:
   - Main findings and conclusions
   - Key decisions made and their rationale
   - Critical insights that shaped the solution

2. Technical Perspective:
   - Implementation details and considerations
   - Technical trade-offs and their implications
   - Performance and scalability aspects
   - Security considerations if applicable

3. Business/User Perspective:
   - Impact on end users or stakeholders
   - Business value and benefits
   - Usability and user experience considerations
   - Cost-benefit analysis if applicable

4. Risk Analysis:
   - Potential failure modes
   - Edge cases and corner cases
   - Known limitations and constraints
   - Mitigation strategies

5. Alternative Approaches:
   - Other solutions considered
   - Trade-offs between different approaches
   - Why the chosen solution is optimal

6. Future Considerations:
   - Potential improvements
   - Maintenance considerations
   - Scalability for future needs
   - Areas for future exploration

Format your response in the following JSON structure:
{{
    "executive_summary": "A concise summary of the overall solution and key findings",
    "detailed_answer": "A detailed answer to the user's request, including all relevant information and insights"
    "key_insights": [
        "list of 3-5 most important insights from the analysis"
    ],
}}"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Here is the conversation history for context:\n{self.user_messages}"}
        ]
        
        final_answer, query_record = make_mistral_request(messages, "generate_final_answer")
        logger.debug(f"Generated final answer:\n{final_answer}")
        
        try:
            final_answer_dict = parse_llm_json(final_answer)
            query_record.parsed_response = final_answer_dict
            self.query_history.append(query_record)
            
            return final_answer_dict
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse final answer as JSON: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing required key in final answer: {e}")
            raise

    @with_retries(max_retries=3)
    def update_todo_list(self):
        """Uses mistral LLM to update the todo list based on the steps taken"""
        logger.info("Updating todo list")
        logger.debug(f"Current todo list before update:\n{self.todo_list}")
        logger.debug(f"Latest completed step:\n{self.completed_steps.steps[-1]}")
        
        prompt = f"""You are responsible for reviewing and updating the todo list based on the latest thinking step. 

Current todo list:
{self.todo_list}

Latest completed thinking step:
{self.completed_steps.steps[-1]}

Previous completed steps:
{self.completed_steps.steps[:-1]}

Your task is to:
1. Review the current todo list and completed steps
2. Mark completed items with a checkmark (✓) at the start of the line
3. Determine if any new tasks have emerged from the latest analysis
4. Assess if any existing tasks need to be modified based on new insights
5. Check if any tasks are now redundant or can be removed
6. Ensure task dependencies are still accurate

When marking items as complete:
- Add a "✓ " at the start of any numbered item that has been fully addressed in the completed steps
- The checkmark should be added before the number, like this: "✓ 1. [Task Name]"
- If a task was partially completed, do not add a checkmark
- Keep the original numbering intact, just add the checkmark before the number
- Maintain any existing checkmarks from previous updates

Format your response in the following JSON structure:
{{
    "updated_todo_list": "The complete, updated todo list with checkmarks for completed items",
    "changes_made": [
        "list of specific changes made to the todo list and why"
    ],
    "next_step_number": number,
    "rationale": "Brief explanation of why these updates were necessary"
}}"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Here is the conversation history for context:\n{self.user_messages}"}
        ]
        
        updated_todo, query_record = make_mistral_request(messages, f"update_todo_list_step_{self.current_step}")
        logger.debug(f"Current todo list:\n{self.todo_list}")
        logger.debug(f"Raw updated todo list response:\n{updated_todo}")
        
        try:
            updated_todo_dict = parse_llm_json(updated_todo)
            query_record.parsed_response = updated_todo_dict
            self.query_history.append(query_record)
            
            self.todo_list = updated_todo_dict["updated_todo_list"]
            logger.info(f"Todo list updated successfully with {len(updated_todo_dict['changes_made'])} changes")
            logger.debug(f"Changes made to todo list:\n{json.dumps(updated_todo_dict['changes_made'], indent=2)}")
            return updated_todo_dict
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse updated todo list as JSON: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing required key in updated todo list: {e}")
            raise

# def make_gemma_request(messages):
#     """Makes a request to the gemma LLM"""
#     import requests
#     import os
#     import json

#     url = "https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent"
#     headers = {
#         "Content-Type": "application/json"
#     }

#     # Get API key from environment
#     # Construct request payload
#     payload = {
#         "contents": [{
#             "parts": [{"text": message["content"]} for message in messages]
#         }]
#     }

#     # Make request
#     response = requests.post(
#         f"{url}?key={shared_settings.GEMMA_API_KEY}",
#         headers=headers,
#         json=payload
#     )

#     output = response.json()
#     return output["candidates"][0]["content"]["parts"][0]["text"]

