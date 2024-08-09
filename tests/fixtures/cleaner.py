from prompting.utils.cleaners import RemoveQuotes, PruneEnding, RemoveRoles, CleanerPipeline

DEFAULT_CLEANER_PIPELINE = CleanerPipeline(cleaning_pipeline=[RemoveQuotes, PruneEnding, RemoveRoles])
