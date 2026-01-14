"""RaySurfer SDK exceptions"""


class RaySurferError(Exception):
    """Base exception for RaySurfer SDK"""

    pass


class APIError(RaySurferError):
    """API returned an error response"""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(RaySurferError):
    """Authentication failed"""

    pass


class CacheUnavailableError(RaySurferError):
    """Cache backend is unreachable or returned an unexpected error"""

    pass


class RateLimitError(RaySurferError):
    """API rate limit exceeded"""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ValidationError(RaySurferError):
    """Request validation failed"""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field
