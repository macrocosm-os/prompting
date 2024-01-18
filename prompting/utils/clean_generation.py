import bittensor as bt


class GenerationCleaner:
    def __init__(self) -> None:
        # TODO: could create these with params?
        self.cleaning_pipelines = {
            "question-answering": [
                self.remove_quotes,
                self.prune_ending,
                self.remove_roles,
            ],
            "summarization": [self.remove_quotes, self.prune_ending, self.remove_roles],
            "date-based question answering": [
                self.remove_quotes,
                self.prune_ending,
                self.remove_roles,
            ],
            "math": [self.remove_roles],
            "generic_instruction": [],
            "debugging": [],
        }

    def remove_roles(self, generation: str):
        """Remove roles from the generation."""
        roles = [
            "User: ",
            "System: ",
            "Assistant: ",
            "Assistant, ",
            "Dear AI, ",
            "Dear AI ",
            "#Question: ",
        ]
        for role in roles:
            if role in generation:
                generation = generation.replace(role, "")

        return (
            generation.capitalize()
        )  # LLMs are good at being formal. Do the same if we remove a prefix.

    def prune_ending(self, generation: str):
        """Prune unfinished sentences to the most recent period, and replaces it with a default ending punctuation."""
        bt.logging.debug("Pruning unfinished sentence.")
        punctuation_chars = [".", "?", "!"]
        if (
            not generation.endswith(".")
            and not generation.endswith("?")
            and not generation.endswith("!")
        ):
            index = max(generation.rfind(char) for char in punctuation_chars)
            return generation[: index + 1] #Go to the index of where the punctuation is, and include it (+1)
        else:
            return generation
        
    def remove_quotes(self, generation: str):
        """Remove quotes and spaces from the generation"""
        return generation.strip("\"'")

    def apply(self, generation: str, task_name: str):
        """Apply the entire task specific pipeline to the generation."""
        try:
            for func in self.cleaning_pipelines.get(task_name, []):
                # apply all the filters for the specific task.
                generation = func(generation=generation)

            return generation

        except Exception as E:
            bt.logging.error(f"Failed to apply cleaning pipeline. {E}")
            return generation
