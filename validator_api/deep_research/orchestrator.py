from validator_api.deep_research.shared_resources import client, OrchestratorState, ToolRequest, ToolResponse
from loguru import logger
import json
import datetime
import asyncio
from validator_api.deep_research.web_research_tool import web_search_tool, summarize_search_results
from validator_api.deep_research.reasoning_tool import reasoning_tool
from validator_api.deep_research.programming_tool import generate_and_execute_code

# ================== Main Agent Loop ==================

async def execute_tool(tool_request: ToolRequest, query: str, state_context: str) -> ToolResponse:
    """
    Execute a single tool request and return its response
    """
    if tool_request.tool_name == "web_search":
        search_query = tool_request.tool_input.get("query", query)
        num_results = tool_request.tool_input.get("num_results", 3)
        
        # Perform web search
        search_results = await web_search_tool(search_query, num_results)
        
        # Summarize results
        summary = await summarize_search_results(search_query, search_results)
        
        return ToolResponse(tool_name="web_search", tool_output=summary)
    
    elif tool_request.tool_name == "reasoning":
        reasoning_query = tool_request.tool_input.get("query", query)
        context = tool_request.tool_input.get("context", state_context)
        
        # Perform reasoning
        reasoning_result = await reasoning_tool(reasoning_query, context)
        
        return ToolResponse(tool_name="reasoning", tool_output=reasoning_result)
    
    elif tool_request.tool_name == "code_writer":
        code = tool_request.tool_input.get("code", "")
        
        # Execute the code
        execution_result = await generate_and_execute_code(code)
        
        return ToolResponse(tool_name="code_writer", tool_output=execution_result)
    
    else:
        return ToolResponse(
            tool_name=tool_request.tool_name, 
            tool_output=f"Unknown tool: {tool_request.tool_name}"
        )

async def plan_research(state: OrchestratorState) -> dict:
    """
    First step: Plan research by creating or updating the todo list based on past work
    """
    # Create a history of steps taken so far
    history = ""
    for i, step in enumerate(state.steps_taken):
        history += f"\nStep {i+1}: Used {step['tool']} tool with input: {json.dumps(step['input'])}\n"
        history += f"Output: {step['output'][:300]}..." if len(step['output']) > 300 else f"Output: {step['output']}\n"
    
    # Create todo list status
    todo_status = "\nTODO list:\n"
    if state.todo_list:
        for i, task in enumerate(state.todo_list):
            todo_status += f"{i+1}. [ ] {task}\n"
    else:
        todo_status += "No tasks in todo list.\n"
    
    completed_status = "\nCOMPLETED TASKS:\n"
    if state.completed_tasks:
        for i, task in enumerate(state.completed_tasks):
            completed_status += f"{i+1}. [✓] {task}\n"
    else:
        completed_status += "No completed tasks yet.\n"
    
    # Get current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create prompt for planning
    prompt = f"""
Current Date/Time: {current_datetime}

You are a research planner that creates and updates todo lists for answering a research question.
You have access to the following tools:

1. web_search - Searches the web using DuckDuckGo and extracts content with Trafilatura
   Input: {{query: str, num_results?: int}}
   
2. reasoning - Uses deep reasoning to analyze information and draw conclusions
   Input: {{query: str, context?: str}}

3. code_writer - This is an expert at writing python code which will go through a process of generating code, executing it, and fixing it if it doesn't work. It will return the final code and it's corresponding inputs/outputs. You can simply send it a natural language query asking it to write a specific function/set of functions.
   Input: {{code: str}}

USER QUERY: {state.query}

CURRENT CONTEXT:
{state.current_context}

STEPS TAKEN SO FAR:
{history if state.steps_taken else "No steps taken yet."}

{todo_status}
{completed_status}

RESEARCH PLAN:
First, I'll create a comprehensive step-by-step plan to answer this query with exceptional depth and thoroughness:
1. Break down the main question into multiple detailed sub-questions that need to be answered
2. Identify what information is already known vs. what needs to be researched
3. Determine which tools would be most effective for each information gap
4. Plan the sequence of tool calls to maximize efficiency
5. Establish criteria for when I have sufficient information to provide a final answer
6. Identify potential conflicting viewpoints or information that should be investigated
7. Plan for verification of information from multiple sources
8. Consider edge cases and exceptions that might apply to this query
9. Identify deeper, less obvious aspects of the query that require investigation
10. Plan for synthesizing all information into a cohesive, comprehensive answer

Now, let me think through this query carefully and exhaustively...

{'' if state.steps_taken else "INITIAL PLAN CREATION:"}
{f"PLAN REVISION (after step {len(state.steps_taken)}):" if state.steps_taken else ""}

Based on the current state of research, decide on ONE of the following:

A) Create or update the todo list with specific research tasks.

B) Provide a final answer if you have completed all tasks and have enough information to respond to the query.

If choosing option A, return your decision as JSON:
{{"decision": "update_plan", "plan_update": "Your updated research plan here...", "todo_list": ["Task 1", "Task 2", ...], "completed_tasks": ["Task X"]}}

If choosing option B, return your decision as JSON:
{{"decision": "final_answer", "plan_summary": "Summary of how the research plan was executed...", "answer": "Your comprehensive answer here..."}}

IMPORTANT: Your todo list should be extensive and ambitious. Challenge yourself to go very deep into the research by:
1. Looking for conflicting viewpoints and information
2. Exploring multiple angles and perspectives on the topic
3. Investigating edge cases and exceptions
4. Seeking verification from multiple sources
5. Exploring historical context and future implications
6. Considering regional/geographical differences
7. Examining both practical implementation and theoretical foundations
8. Identifying potential biases in the information sources
9. Exploring less obvious aspects of the query that most researchers would miss
10. Continuously expanding the todo list as new questions arise during research

If you choose option B (final_answer), your answer should be exhaustive and well-formatted using proper Markdown. Include:
1. A clear introduction explaining the problem and approach
2. Well-formatted code blocks with syntax highlighting using ```python and ``` tags
3. Explanations of key concepts and methodologies
4. References to web sources used during research (with URLs when available)
5. Visual explanations where appropriate (described in text)
6. A conclusion summarizing the findings
7. Any limitations or considerations for future work

Your final answer should be comprehensive, educational, and professionally formatted.
"""

    try:
        # Add explicit instruction to return JSON in the prompt
        json_prompt = prompt + "\n\nImportant: Your response must be a valid JSON object following the format specified above."
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a research planner that creates and updates todo lists. You use chain-of-thought reasoning to create and revise research plans. Return your response as a JSON object."},
                {"role": "user", "content": json_prompt}
            ],
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        if result.get("decision") == "update_plan":
            # Update todo list and completed tasks
            if "todo_list" in result:
                state.update_todo_list(result["todo_list"])
            
            if "completed_tasks" in result:
                for task in result["completed_tasks"]:
                    state.mark_task_completed(task)
            
            # Store the updated plan in the state
            if "plan_update" in result:
                state.current_context += f"\n\n=== UPDATED RESEARCH PLAN ===\n{result['plan_update']}\n{'='*50}\n"
        
        return result
    
    except Exception as e:
        print(f"Planning error: {e}")
        return {"decision": "error", "error": str(e)}

async def execute_research(state: OrchestratorState) -> list[ToolRequest]:
    """
    Second step: Execute research by determining which tools to use based on the todo list
    """
    # Create a history of steps taken so far
    history = ""
    for i, step in enumerate(state.steps_taken):
        history += f"\nStep {i+1}: Used {step['tool']} tool with input: {json.dumps(step['input'])}\n"
        history += f"Output: {step['output'][:300]}..." if len(step['output']) > 300 else f"Output: {step['output']}\n"
    
    # Create todo list status
    todo_status = "\nTODO list:\n"
    if state.todo_list:
        for i, task in enumerate(state.todo_list):
            todo_status += f"{i+1}. [ ] {task}\n"
    else:
        todo_status += "No tasks in todo list.\n"
    
    # Get current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create prompt for tool execution
    prompt = f"""
Current Date/Time: {current_datetime}

You are a research executor that decides which tools to use to complete tasks from a todo list.
You have access to the following tools:

1. web_search - Searches the web using DuckDuckGo and extracts content with Trafilatura
   Input: {{query: str, num_results?: int}}
   
2. reasoning - Uses deep reasoning to analyze information and draw conclusions
   Input: {{query: str, context?: str}}

3. code_writer - This is an expert at writing python code which will go through a process of generating code, executing it, and fixing it if it doesn't work. It will return the final code and it's corresponding inputs/outputs. You can simply send it a natural language query asking it to write a specific function/set of functions.
   Input: {{code: str}}

USER QUERY: {state.query}

CURRENT CONTEXT:
{state.current_context}

STEPS TAKEN SO FAR:
{history if state.steps_taken else "No steps taken yet."}

{todo_status}

Based on the current todo list, decide which tools to use to make progress on the research tasks.
You can call multiple tools simultaneously to run in parallel. Note that you can also run the same tool multiple times in parallel with different inputs.

For each tool, specify:
- tool_name: The name of the tool to use (web_search, reasoning, or code_writer)
- tool_input: The input parameters for the tool

Return your decision as JSON:
{{"tools": [
  {{"tool_name": "...", "tool_input": {{...}}}},
  {{"tool_name": "...", "tool_input": {{...}}}}
]}}

Think carefully about which tools are most appropriate for the current tasks.
Remember you can use multiple tools in parallel for more efficient research.
"""

    try:
        # Add explicit instruction to return JSON in the prompt
        json_prompt = prompt + "\n\nImportant: Your response must be a valid JSON object following the format specified above."
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a research executor that decides which tools to use to complete tasks from a todo list. Return your response as a JSON object."},
                {"role": "user", "content": json_prompt}
            ],
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return [ToolRequest(
            tool_name=tool["tool_name"],
            tool_input=tool["tool_input"]
        ) for tool in result.get("tools", [])]
    
    except Exception as e:
        print(f"Execution error: {e}")
        return []

async def deep_research_agent(query: str, max_steps: int = 7) -> OrchestratorState:  # Increased max_steps
    """
    Main agent function with a two-step process: planning and execution
    """
    logger.info(f"Starting deep research for query: {query}")
    state = OrchestratorState(query=query)
    
    for step in range(max_steps):
        logger.info(f"Starting research step {step+1}/{max_steps}")
        
        # Step 1: Plan research by creating/updating todo list
        plan_result = await plan_research(state)
        
        if plan_result.get("decision") == "final_answer":
            # Include the plan summary in the final answer
            if "plan_summary" in plan_result:
                plan_summary = f"\n\n## Research Approach\n{plan_result['plan_summary']}\n\n"
                final_answer = plan_result.get("answer", "Unable to determine final answer.")
                state.current_context = plan_summary + final_answer
            else:
                state.current_context = plan_result.get("answer", "Unable to determine final answer.")
            logger.info("Research complete - final answer reached")
            return state
        
        # Step 2: Execute research by determining which tools to use
        tool_requests = await execute_research(state)
        
        if not tool_requests:
            logger.info("No tools to execute in this step")
            continue
        
        logger.info(f"Running {len(tool_requests)} tools in parallel")
        for tool_request in tool_requests:
            logger.info(f"Tool call: {tool_request.tool_name} with input: {tool_request.tool_input}")
        
        tool_tasks = [
            execute_tool(tool_request, query, state.current_context)
            for tool_request in tool_requests
        ]
        
        tool_responses = await asyncio.gather(*tool_tasks)
        
        for i, tool_response in enumerate(tool_responses):
            tool_request = tool_requests[i]
            logger.info(f"Completed tool: {tool_response.tool_name}")
            
            state.add_step(tool_response.tool_name, tool_request.tool_input, tool_response.tool_output)
            state.current_context += f"\n\n=== {tool_response.tool_name.upper()} RESULTS ===\n"
            state.current_context += f"Query: '{tool_request.tool_input.get('query', query)}'\n"
            state.current_context += f"Results:\n{tool_response.tool_output}\n"
            state.current_context += "="*50 + "\n"
    
    logger.warning(f"Research reached maximum steps ({max_steps}) without final answer")
    return state

