from prompting.cleaners import CleanerPipeline

DEFAULT_CLEANER_PIPELINE = CleanerPipeline([
    dict(name="remove_quotes"),
    dict(name="prune_ending"),
    dict(name="remove_roles"),
])
