# from prompting.base.loop_runner import AsyncLoopRunner
# from loguru import logger
# from prompting.tasks.task_registry import TaskRegistry
# from prompting.utils.timer import Timer
# from prompting.utils.uids import get_random_uids
# from prompting.base.dendrite import DendriteResponseEvent, StreamPromptingSynapse
# from prompting.settings import settings
# from prompting.rewards.scoring import scoring_manager
# from prompting.utils.logging import ValidatorLoggingEvent, ErrorLoggingEvent

# NEURON_SAMPLE_SIZE = 100


# class TaskCreation(AsyncLoopRunner):
#     interval: int = 0

#     async def run_step(self):
#         while True:
#             try:
#                 task, dataset = TaskRegistry.create_random_task_with_dataset()
#                 break
#             except Exception as ex:
#                 logger.exception(ex)

#         try:
#             logger.debug("Task chosen:", task.__class__.__name__)
#             if not (dataset_entry := dataset.random()):
#                 logger.warning(f"Dataset {dataset.__class__.__name__} returned None. Skipping step.")
#                 return None

#             # Generate the query for the task
#             if not task.query:
#                 query = task.make_query(dataset_entry=dataset_entry)
#             """query = Task.generate_query(self.llm_pipeline, dataset_entry)"""

#             # Record event start time.
#             with Timer() as timer:

#                 # Get the list of uids to query for this step.
#                 uids = get_random_uids(k=NEURON_SAMPLE_SIZE)

#                 axons = [settings.METAGRAPH.axons[uid] for uid in uids]

#                 # Directly call dendrite and process responses in parallel
#                 streams_responses = await settings.DENDRITE(
#                     axons=axons,
#                     synapse=StreamPromptingSynapse(task=task.__class__.__name__, roles=["user"], messages=[query]),
#                     timeout=settings.NEURON_TIMEOUT,
#                     deserialize=False,
#                     streaming=True,
#                 )

#                 # Prepare the task for handling stream responses
#                 stream_results = await handle_response(stream_results_dict=dict(zip(uids, streams_responses)))

#                 log_stream_results(stream_results)

#                 # Encapsulate the responses in a response event (dataclass)
#                 response_event = DendriteResponseEvent(stream_results=stream_results, uids=uids, timeout=timeout)

#                 logger.info(f"Created DendriteResponseEvent:\n {response_event}")

#                 # Reward the responses and get the reward result (dataclass)
#                 # This contains a list of RewardEvents but can be exported as a dict (column-wise) for logging etc
#                 """reward_queue.append(Task, response_event)"""

#                 # scoring_manager will score the responses as and when the correct model is loaded
#                 scoring_manager.add_to_queue(task=task, response=response_event, dataset_entry=dataset_entry)

#             # Log the step event.
#             return ValidatorLoggingEvent(
#                 block=self.block,
#                 step=self.step,
#                 step_time=timer.elapsed_time,
#                 response_event=response_event,
#                 task_id=task.task_id,
#             )
#         except Exception as ex:
#             logger.exception(ex)
#             return ErrorLoggingEvent(
#                 error=str(ex),
#             )
