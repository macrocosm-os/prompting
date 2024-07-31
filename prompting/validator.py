import bittensor as bt
from prompting.llms.vllm_llm import vLLMPipeline
from prompting.base.validator import BaseValidatorNeuron
from prompting.forward import log_stream_results, log_event, handle_response
from prompting.dendrite import DendriteResponseEvent, StreamPromptingSynapse
from prompting.task_registry import TaskRegistry
import time
import sys
from prompting.utils.uids import get_random_uids
from prompting.tasks.task import BaseTask


class Validator(BaseValidatorNeuron):
    """
    Text prompt validator neuron.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.llm_pipeline = vLLMPipeline(
            model_id=self.config.neuron.model_id,
            gpus=self.config.neuron.gpus,
            llm_max_allowed_memory_in_gb=self.config.neuron.llm_max_allowed_memory_in_gb,
            device=self.device,
            mock=self.config.mock,
        )
        # self.translation_pipeline = TranslationPipeline()

        if abs(1 - sum(self.config.neuron.task_p)) > 0.001:
            raise ValueError("Task probabilities do not sum to 1.")

        # Filter out tasks with 0 probability
        self.active_tasks = [task for task, p in zip(self.config.neuron.tasks, self.config.neuron.task_p) if p > 0]
        # Load the reward pipeline
        # self.reward_pipeline = RewardPipeline(selected_tasks=self.active_tasks, device=self.device)

    async def run_step(self, task: BaseTask, k: int, timeout: float, exclude: list = None):
        """Executes a single step of the agent, which consists of:
        - Getting a list of uids to query
        - Querying the network
        - Rewarding the network
        - Updating the scores
        - Logging the event

        Args:
            agent (HumanAgent): The agent to run the step for.
            roles (List[str]): The roles for the synapse.
            messages (List[str]): The messages for the synapse.
            k (int): The number of uids to query.
            timeout (float): The timeout for the queries.
            exclude (list, optional): The list of uids to exclude from the query. Defaults to [].
        """
        bt.logging.debug("run_step", task.__class__.__name__)

        # Generate the query and reference for the task
        task.generate_query()
        task.generate_reference()

        # Record event start time.
        start_time = time.time()

        # Get the list of uids to query for this step.
        uids = get_random_uids(self, k=k, exclude=exclude or []).to(self.device)
        uids_cpu = uids.cpu().tolist()

        axons = [self.metagraph.axons[uid] for uid in uids]

        # Directly call dendrite and process responses in parallel
        streams_responses = await self.dendrite(
            axons=axons,
            synapse=StreamPromptingSynapse(roles=["user"], messages=[task.augmented_query]),
            timeout=timeout,
            deserialize=False,
            streaming=True,
        )

        # Prepare the task for handling stream responses
        stream_results_dict = dict(zip(uids_cpu, streams_responses))
        tokenizer = self.llm_pipeline.tokenizer
        stream_results = await handle_response(stream_results_dict, tokenizer)

        log_stream_results(stream_results)

        # Encapsulate the responses in a response event (dataclass)
        response_event = DendriteResponseEvent(stream_results=stream_results, uids=uids, timeout=timeout)

        bt.logging.info(f"Created DendriteResponseEvent:\n {response_event}")

        # Reward the responses and get the reward result (dataclass)
        # This contains a list of RewardEvents but can be exported as a dict (column-wise) for logging etc
        reward_pipeline = TaskRegistry.get_task_reward(task)
        reward_event = reward_pipeline.apply(response_event, task.reference, task.augmented_query)

        bt.logging.info(f"Created RewardResult:\n {reward_event}")

        best_response = response_event.completions[reward_event.rewards.argmax()]

        self.update_scores(reward_event.rewards, uids)

        # Log the step event.
        event = {
            "best": best_response,
            "block": self.block,
            "step": self.step,
            "step_time": time.time() - start_time,
            **reward_event.__dict__(full=self.config.neuron.log_full),
            **response_event.__dict__(),
        }

        return event

    async def forward(self):
        """
        Encapsulates a full conversation between the validator and miners. Contains one or more rounds of request-response.

        """
        bt.logging.info("🚀 Starting forward loop...")
        forward_start_time = time.time()

        while True:
            bt.logging.info(
                f"📋 Selecting task... from {self.config.neuron.tasks} with distribution {self.config.neuron.task_p}"
            )
            # Create a specific task
            # task_name = np.random.choice(self.config.neuron.tasks, p=self.config.neuron.task_p)
            task_config = TaskRegistry.random()
            bt.logging.info(f"📋 Creating {task_config.task.__class__.__name__} task... ")
            try:
                task = TaskRegistry.create_random_task(llm_pipeline=self.llm_pipeline)
                break
            except Exception:
                bt.logging.error(
                    f"Failed to create {task_config.task.__class__.__name__} task. {sys.exc_info()}. Skipping to next task."
                )
                continue

        turn = 0
        exclude_uids = []

        try:
            # when run_step is called, the agent updates its progress
            event = await self.run_step(
                self,
                task=task,
                k=self.config.neuron.sample_size,
                timeout=self.config.neuron.timeout,
                exclude=exclude_uids,
            )

            # Adds forward time to event and logs it to wandb
            event["forward_time"] = time.time() - forward_start_time
            event["turn"] = turn
            log_event(self, event)
            task.complete = True

            # accepted_answer = event["best"] if random.random() < 0.5 else agent.task.reference

        except Exception as e:
            bt.logging.error(f"Error in run_step: Skipping to next round. \n {e}")
            event = {"unexpected_errors": e}
            log_event(self, event)

        del task

    def __enter__(self):
        if self.config.no_background_thread:
            bt.logging.warning("Running validator in main thread.")
            self.run()
        else:
            self.run_in_background_thread()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")
