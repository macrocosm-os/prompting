from prompting.cleaners import CleanerPipeline
from prompting.cleaners.all_cleaners import RemoveQuotes, PruneEnding, RemoveRoles

DEFAULT_CLEANER_PIPELINE = CleanerPipeline([RemoveQuotes, PruneEnding, RemoveRoles])
