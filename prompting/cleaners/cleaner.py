from typing import List, Dict

import sentry_sdk
import bittensor as bt

from prompting.cleaners.all_cleaners import RemoveQuotes, RemoveRoles, PruneEnding

SUPPORTED_CLEANERS = {
    "remove_quotes": RemoveQuotes,
    "remove_roles": RemoveRoles,
    "prune_ending": PruneEnding,
}


class CleanerPipeline:
    def __init__(self, cleaning_pipeline: List[Dict]) -> None:
        """CleanerPipeline is a pipeline that can be applied to any string to
        clean it of unwanted characters, punctuation, etc.

        cleaning_pipeline (List[Dict]): List of Dicts that define the cleaning pipeline.
            Dictionaries MUST have the keyword "name" to be valid.
            Example: [{"name": "remove_quotes", "kwargs": {}}, {"name": "prune_ending", "kwargs": {}}]

        """
        self.cleaning_pipeline = cleaning_pipeline

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

                func = SUPPORTED_CLEANERS[cleaner["name"]]

                kwargs = cleaner.get("kwargs", {})
                func = func(**kwargs)  # instantiate the cleaner with the kwargs

                # apply all the filters for the specific task.
                generation = func.apply(generation=generation)

            return generation

        except Exception as E:
            sentry_sdk.capture_exception()
            bt.logging.error(
                f"Failed to apply cleaning pipeline {cleaner['name']}. {E},"
            )
            return generation
