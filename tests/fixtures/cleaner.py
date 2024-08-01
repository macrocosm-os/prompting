from prompting.cleaners import CleanerPipeline
from prompting.utils.cleaners import RemoveQuotes, PruneEnding, RemoveRoles

DEFAULT_CLEANER_PIPELINE = CleanerPipeline([RemoveQuotes, PruneEnding, RemoveRoles])
