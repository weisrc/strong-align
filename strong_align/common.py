from dataclasses import dataclass

ALIGN_SR = 16000


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Range:
    start: float = 0.0
    end: float = 0.0


@dataclass
class Alignment(Range):
    start_time_index: int = 0
    end_time_index: int = 0
    start_token_index: int = 0
    end_token_index: int = 0
    score: float = 0.0
    text: str = ""
