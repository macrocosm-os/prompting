from prompting.cleaners.cleaner import CleanerPipeline


def test_cleaning_pipeline():
    cleaning_pipeline = [
        dict(name="remove_quotes"),
        dict(name="prune_ending"),
        dict(name="remove_roles"),
    ]

    generation = (
        '"I am a quote. User: I know you are. I am asking a question. What is th"'
    )
    answer = "I am a quote. I know you are. I am asking a question."

    clean_generation = CleanerPipeline(cleaning_pipeline=cleaning_pipeline).apply(
        generation=generation
    )

    assert clean_generation == answer


def test_phrase_without_any_punctuation():
    # arrange
    cleaning_pipeline = [
        dict(name="prune_ending"),
    ]

    generation = "Austin is the capital of texas"

    # act
    clean_generation = CleanerPipeline(cleaning_pipeline=cleaning_pipeline).apply(
        generation=generation
    )

    # assert
    assert clean_generation == generation
