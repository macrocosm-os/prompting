class MaxRetryError(Exception):
    """Exception raised when the maximum number of retries is exceeded."""

    def __init__(self, message: str = "Maximum number of retries exceeded"):
        self.message: str = message
        super().__init__(self.message)
