from dataclasses import dataclass

SAMPLE_RATE = 16000

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