from prompting.cleaners.cleaner import CleanerPipeline
from prompting.cleaners.all_cleaners import RemoveQuotes, PruneEnding, RemoveRoles


def test_cleaning_pipeline():
    cleaning_pipeline = [RemoveQuotes, PruneEnding, RemoveRoles]

    generation = '"I am a quote. User: I know you are. I am asking a question. What is th"'
    answer = "I am a quote. I know you are. I am asking a question."

    clean_generation = CleanerPipeline(cleaning_pipeline=cleaning_pipeline).apply(generation=generation)

    assert clean_generation == answer


def test_phrase_without_any_punctuation():
    # arrange
    cleaning_pipeline = [PruneEnding]

    generation = "Austin is the capital of texas"

    # act
    clean_generation = CleanerPipeline(cleaning_pipeline=cleaning_pipeline).apply(generation=generation)

    # assert
    assert clean_generation == generation
