"""Typed user-facing errors."""


class Video2WorldError(Exception):
    """Base class for expected orchestration failures."""


class ConfigurationError(Video2WorldError):
    """The run configuration or command template is invalid."""


class ArtifactError(Video2WorldError):
    """An input or output artifact failed verification."""


class StageBlockedError(Video2WorldError):
    """A stage cannot run because a dependency or command is unavailable."""


class ValidationFailure(Video2WorldError):
    """A manifest or run failed validation."""
