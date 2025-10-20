from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    GENERIC_FAILURE = 1
    INVALID_ARGS = 2
    LLM_FAILURE = 3
    NO_PASSING_SOLUTION = 4
    CANCELED_BY_SIGNAL = 5
    PACKAGING_FAILURE = 6
