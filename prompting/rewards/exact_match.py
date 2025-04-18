import random

import numpy as np
from loguru import logger
from openai.types.chat import ChatCompletionChunk

from prompting.llms.model_manager import ModelManager
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from prompting.tasks.base_task import BaseTextTask
from shared import settings
from shared.dendrite import DendriteResponseEvent

shared_settings = settings.shared_settings
INCORRECT_PENALTY = 1
INCOMPLETE_PENALTY = 1
VERIFICATION_RATIO = 0.1
VERIFICATION_THRESHOLD = 0.9


def normalize_timing(timing: float, timings: float) -> float:
    """
    Normalize the timing so that a lower timing (i.e. faster response) is closer to 1.
    Ensures the normalized value is between 0 and 1.
    """

    flat_values = [
        x
        for sublist in timings
        if sublist is not None
        for x in (sublist if isinstance(sublist, list) else [sublist])
        if x is not None
    ]
    last_chunk = max(flat_values) if flat_values else shared_settings.INFERENCE_TIMEOUT
    return min(1, max(0, (last_chunk - timing) / last_chunk))


def verify_single_logit(original_logits, verification_logits):
    """
    Verify logits by computing cosine similarity between original and verification logits.

    Args:
        original_logits: Original model logits
        verification_logits: Verification model logits

    Returns:
        float: Cosine similarity score
    """
    # Create aligned vectors with same token ordering
    all_tokens = set(original_logits.keys()) | set(verification_logits.keys())

    orig_vec = []
    verif_vec = []
    for token in all_tokens:
        orig_vec.append(original_logits.get(token, -100.0))
        verif_vec.append(verification_logits.get(token, -100.0))

    orig_vec = np.array(orig_vec)
    verif_vec = np.array(verif_vec)

    # Apply softmax to convert logprobs to probabilities
    orig_vec = np.exp(orig_vec) / np.sum(np.exp(orig_vec))
    verif_vec = np.exp(verif_vec) / np.sum(np.exp(verif_vec))

    # Calculate cosine similarity
    orig_vec = orig_vec / np.linalg.norm(orig_vec)
    verif_vec = verif_vec / np.linalg.norm(verif_vec)
    return np.dot(orig_vec, verif_vec)


class LogitsRewardModel(BaseRewardModel):
    async def reward(
        self,
        reference: str,
        response_event: DendriteResponseEvent,
        task: BaseTextTask,
        model_manager: ModelManager,
        **kwargs,
    ) -> BatchRewardOutput:
        """
        Calculates rewards based on the logits of the response and verifies them.
        """

        # Check that the model_manager is set
        if model_manager is None:
            raise ValueError("Model manager must be set")

        all_chunks: list[list[str]] = response_event.stream_results_all_chunks
        all_chunk_dicts_raw: list[list[ChatCompletionChunk]] = response_event.stream_results_all_chunk_dicts_raw
        uids: np.ndarray | list[float] = response_event.uids
        all_timings: list[list[float]] = response_event.stream_results_all_chunks_timings
        completions: list[str] = response_event.completions
        timeout: float = response_event.timeout
        sampling_parameters: dict = task.sampling_params
        PENALIZE_ALL = BatchRewardOutput(
            rewards=np.array([-INCORRECT_PENALTY] * len(completions)),
            timings=np.array([0.0] * len(completions)),
        )

        max_length = 0
        for chunk in all_chunks:
            if chunk and max_length < len(chunk):
                max_length = len(chunk)

        if max_length == 0:
            logger.debug("No chunks to verify, penalizing all")
            return PENALIZE_ALL

        if timeout <= 0:
            logger.error("Timeout must be greater than 0. Received timeout: {}", timeout)
            raise ValueError("Timeout must be greater than 0.")

        timing_outputs, rewards = [], []
        num_verify = max(1, int(max_length * VERIFICATION_RATIO))
        verify_indices = random.sample(
            range(max_length - 1), num_verify - 1
        )  # Sample one less to save room for last index
        verify_indices.append(max_length - 1)  # Always verify the last index
        verify_indices.sort()

        # Iterate over each response event

        for chunks, timings, chunk_dicts_raw, uid in zip(all_chunks, all_timings, all_chunk_dicts_raw, uids):
            try:
                # If no response is provided, apply full penalty
                if not chunks:
                    rewards.append(-INCORRECT_PENALTY)
                    timing_outputs.append(0.0)
                    continue

                # Verify logits for selected indices
                verification_scores = []
                completion_length = len(chunks)

                for idx in verify_indices:
                    check_idx = min(idx, completion_length - 1)
                    if not chunk_dicts_raw[check_idx].choices[0].logprobs:
                        logger.debug(f"Miner {uid} failed to provide logprobs: {chunk_dicts_raw[check_idx]}")
                        verification_scores.append(0.0)
                        continue

                    if chunk_dicts_raw[check_idx].choices[0].logprobs.content is None:
                        logger.debug(f"Miner {uid} failed to provide logprobs content: {chunk_dicts_raw[check_idx]}")
                        verification_scores.append(0.0)
                        continue

                    original_logits = {
                        info.token: info.logprob
                        for info in chunk_dicts_raw[check_idx].choices[0].logprobs.content[0].top_logprobs
                    }

                    verification_output, prompt = await model_manager.generate_logits(
                        model=task.llm_model_id,
                        messages=task.task_messages + [{"role": "assistant", "content": "".join(chunks[:check_idx])}],
                        sampling_params=sampling_parameters,
                        continue_last_message=True,
                    )

                    logit_score = verify_single_logit(original_logits, verification_output)
                    verification_scores.append(logit_score)
                    if idx >= completion_length:
                        break
                final_score = np.mean(verification_scores)

                # Compute timing reward
                valid_chunks = []
                for chunk, timing in zip(chunks, timings):
                    if chunk:
                        valid_chunks.append(normalize_timing(timing, all_timings))

                timing_reward = np.mean(valid_chunks) if valid_chunks else 0.0

                rewards.append(float(final_score > VERIFICATION_THRESHOLD) * timing_reward)
                timing_outputs.append(np.array(valid_chunks).mean())
            except Exception as e:
                logger.debug(f"Miner {uid} failed to provide logits chunk, setting reward to 0: {e}")
                rewards.append(0.0)
                timing_outputs.append(0.0)

        reward_output = BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(timing_outputs),
        )
        logger.debug(f"REWARD OUTPUT: {reward_output.model_dump()}")
        return reward_output
