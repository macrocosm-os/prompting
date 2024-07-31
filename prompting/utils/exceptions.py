class MaxRetryError(Exception):
    """Exception raised when the maximum number of retries is exceeded."""

    def __init__(self, message="Maximum number of retries exceeded"):
        self.message = message
        super().__init__(self.message)

class BittensorError(Exception):
    """Exception raised when an error is raised from the bittensor package"""

    def __init__(self, message = "An error from the Bittensor package occured"):
        self.message = message 
        super().__init__(self.message)
