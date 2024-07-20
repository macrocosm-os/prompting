from typing import List, Dict, Union

import bittensor as bt

from prompting.cleaners.all_cleaners import (
    RemoveQuotes,
    RemoveRoles,
    PruneEnding,
    PrunePostQuestionText,
    RemoveTags,
    FirstQuestion,
)

SUPPORTED_CLEANERS = {
    "remove_quotes": RemoveQuotes,
    "remove_roles": RemoveRoles,
    "prune_ending": PruneEnding,
    "remove_post_question_text": PrunePostQuestionText,
    "first_question": FirstQuestion,
    "remove_tags": RemoveTags,
}


class CleanerPipeline:
    def __init__(self, cleaning_pipeline: List[Dict]) -> None:
        """CleanerPipeline is a pipeline that can be applied to any string to
        clean it of unwanted characters, punctuation, etc.

        cleaning_pipeline (List[Dict]): List of Dicts that define the cleaning pipeline.
            Dictionaries MUST have the keyword "name" to be valid.
            Example: [{"name": "remove_quotes", "kwargs": {}}, {"name": "prune_ending", "kwargs": {}}]

        """
        self.cleaning_pipeline: list[dict] = cleaning_pipeline

    def apply(self, generation: str) -> str:
        """Apply cleaning steps to generation listed in cleaning_pipeline.

        Args:
            generation (str): string generated from LLM or otherwise.
        Returns:
            str: Clean generated string.
        """
        try:
            for cleaner in self.cleaning_pipeline:
                if "name" not in cleaner or cleaner["name"] not in SUPPORTED_CLEANERS:
                    raise ValueError(
                        f"Cleaning pipeline step {cleaner} must have a name, or must be in SUPPORTED_CLEANERS."
                    )

                func: Union[
                    RemoveQuotes,
                    RemoveRoles,
                    PruneEnding,
                    PrunePostQuestionText,
                    FirstQuestion,
                    RemoveTags,
                ] = SUPPORTED_CLEANERS[cleaner["name"]]

                kwargs = cleaner.get("kwargs", {})
                func = func(**kwargs)  # instantiate the cleaner with the kwargs

                # apply all the filters for the specific task.
                generation: str = func.apply(generation=generation)

            return generation

        except Exception as E:
            bt.logging.error(
                f"Failed to apply cleaning pipeline {cleaner['name']}. {E},"
            )
            return generation
