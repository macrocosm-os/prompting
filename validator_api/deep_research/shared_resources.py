from openai import AsyncOpenAI
import os
from loguru import logger
from dataclasses import dataclass
from typing import Any
from shared import settings
settings.shared_settings = settings.SharedSettings.load(mode="validator")

@dataclass
class SearchResult:
    url: str
    title: str
    content: str

@dataclass
class ToolRequest:
    tool_name: str
    tool_input: dict[str, Any]

@dataclass
class ToolResponse:
    tool_name: str
    tool_output: str

@dataclass
class OrchestratorState:
    query: str
    steps_taken: list[dict[str, Any]] = None
    current_context: str = ""
    todo_list: list[str] = None
    completed_tasks: list[str] = None

    def __post_init__(self):
        if self.steps_taken is None:
            self.steps_taken = []
        if self.todo_list is None:
            self.todo_list = []
        if self.completed_tasks is None:
            self.completed_tasks = []
    
    def add_step(self, tool_name: str, tool_input: dict[str, Any], tool_output: str):
        self.steps_taken.append({
            "tool": tool_name,
            "input": tool_input,
            "output": tool_output
        })
        logger.info(f"Added step: {tool_name}")
        logger.debug(f"Step input: {tool_input}")
        logger.debug(f"Step output: {tool_output}")
    
    def update_todo_list(self, new_todo_list: list[str]):
        self.todo_list = new_todo_list
        logger.info(f"Updated todo list: {len(self.todo_list)} items remaining")
        # Log each item in the todo list
        for i, task in enumerate(self.todo_list):
            logger.info(f"Todo item {i+1}: {task}")
    
    def mark_task_completed(self, task: str):
        if task in self.todo_list:
            self.todo_list.remove(task)
        self.completed_tasks.append(task)
        logger.info(f"Marked task as completed: {task}")
        logger.info(f"{len(self.todo_list)} tasks remaining")

client = AsyncOpenAI(api_key=settings.shared_settings.OPENAI_API_KEY)




