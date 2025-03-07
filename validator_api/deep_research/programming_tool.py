import io
import contextlib
import re
from validator_api.deep_research.shared_resources import client
from loguru import logger

async def execute_code(code_string):
    """
    Execute a string containing Python code and return the output.
    
    Args:
        code_string (str): Python code as a string
        
    Returns:
        str: Output from executing the code
    """
    # Create string buffer to capture output
    buffer = io.StringIO()
    
    # Redirect stdout and stderr to our buffer
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            # Execute the code
            exec(code_string, globals())  # Pass globals() to make defined functions available
            output = buffer.getvalue()
            
        except Exception as e:
            # Capture any exceptions
            output = f"Error executing code: {str(e)}"
    
    return output

async def generate_and_execute_code(query, max_attempts=5):
    """
    Generate Python code using an LLM based on a query, then execute it.
    If execution fails, try to fix the code up to max_attempts times.
    
    Args:
        query (str): Description of the code to generate
        max_attempts (int): Maximum number of attempts to fix the code
        
    Returns:
        tuple: (generated_code, execution_output, success)
    """
    generated_code = ""
    execution_output = ""
    error_message = ""
    success = False
    attempts_log = []
    
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                # First attempt - generate code based on original query
                prompt = f"""
                Write Python code that does the following:
                {query}
                
                Your code should ALWAYS contain at least one test case/example at the end that shows the code working.
                Return only the executable Python code without any explanations or markdown formatting. It should NOT be wrapped in any code blocks and should be directly executable.
                """
                system_message = "You are a Python programming assistant. Generate only executable Python code without explanations."
            else:
                # Subsequent attempts - fix the code based on error
                prompt = f"""
                The following Python code has an error:
                
                {generated_code}
                
                Error message:
                {error_message}
                
                Please fix the code and return the complete corrected version.
                Return only the executable Python code without any explanations or markdown formatting. 
                Note that the user cannot make any changes to the code. If the code fails due to installs/hardware, you have to modify it to work anyways on the user's machine. If a package is not installed, you MUST AVOID USING IT AND INSTEAD USE A SIMILAR PACKAGE THAT IS INSTALLED. It should NOT be wrapped in any code blocks and should be directly executable.
                """
                system_message = "You are a Python debugging assistant. Fix the code and return only the corrected code without explanations."
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            original_code = response.choices[0].message.content.strip()
            generated_code = re.sub(r'.*?```(.*?)```.*', r'\1', original_code, flags=re.DOTALL)
            # If the word 'python' appears at the start of the code, remove it
            if generated_code.strip().startswith('python'):
                generated_code = generated_code.strip()[6:].strip()
            
            # Execute the generated code
            execution_output = await execute_code(generated_code)
            
            # Log this attempt
            attempts_log.append({
                "attempt": attempt + 1,
                "code": generated_code,
                "output": execution_output,
                "error": "Error executing code:" in execution_output
            })
            
            # Check if there was an error
            if "Error executing code:" in execution_output:
                error_message = execution_output
                logger.warning(f"Code Execution Attempt {attempt+1} failed: \n\nORIGINAL CODE:{original_code}\n\nGENERATED CODE:{generated_code}\n\nERROR:{error_message}. Trying again...")
            else:
                # No error, code executed successfully
                success = True
                logger.info(f"Code executed successfully on attempt {attempt+1}!")
                break
                
        except Exception as e:
            error_message = f"Error generating or executing code: {str(e)}"
            logger.error(f"Attempt {attempt+1} failed with exception: {str(e)}")
            attempts_log.append({
                "attempt": attempt + 1,
                "code": generated_code,
                "output": str(e),
                "error": True
            })
    
    # Generate explanation of the process and final code
    explanation = await generate_code_explanation(query, attempts_log, generated_code, success)
    
    return generated_code, execution_output, success, explanation

async def generate_code_explanation(query, attempts_log, final_code, success):
    """
    Generate an explanation of the code generation process and the final code using LLM.
    
    Args:
        query (str): Original query
        attempts_log (list): Log of all attempts
        final_code (str): Final generated code
        success (bool): Whether the code executed successfully
    
    Returns:
        str: Explanation of the process and final code
    """
    # Format the attempts log for the LLM
    attempts_summary = ""
    for attempt in attempts_log:
        attempts_summary += f"Attempt {attempt['attempt']}:\n"
        attempts_summary += f"Success: {not attempt['error']}\n"
        if attempt['error']:
            # Include only the first few lines of error for brevity
            error_lines = attempt['output'].split('\n')[:3]
            attempts_summary += f"Error: {' '.join(error_lines)}\n"
    
    prompt = f"""
    Please provide a detailed explanation of the code generation process and the final code.
    
    Original Query: {query}
    
    Process Summary:
    - Total attempts: {len(attempts_log)}
    - Final result: {'Successful execution' if success else 'Failed to execute properly'}
    
    Attempt Details:
    {attempts_summary}
    
    Final Code:
    ```python
    {final_code}
    ```
    
    Please explain:
    1. What the code does and how it addresses the original query
    2. The key components and functions in the code
    3. Any challenges encountered during generation and how they were resolved
    4. The approach taken to solve the problem
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Python expert who explains code clearly and concisely. Format your response in Markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        explanation = response.choices[0].message.content
        return explanation
    except Exception as e:
        # Fallback to a simple explanation if the LLM call fails
        return f"""
## Code Generation Process for: {query}

### Process Summary
- Total attempts: {len(attempts_log)}
- Final result: {'Successful execution' if success else 'Failed to execute properly'}

### Note
There was an error generating a detailed explanation: {str(e)}
"""

async def write_code(query):
    """
    Write Python code using an LLM based on a query, then execute it.
    If execution fails, try to fix the code up to max_attempts times.
    
    Args:
        query (str): Description of the code to generate
    """
    code, output, success, explanation = await generate_and_execute_code(query)
    return f"""
{explanation}

## Final Code
{code}

## Output
{output}

## Success
{success}
"""