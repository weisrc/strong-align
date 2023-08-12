import warnings
from typing import List
from functools import lru_cache

import torch

from .common import Range

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad', verbose=False)

_get_speech_timestamps, *_ = utils


@lru_cache(maxsize=1)
def get_vad_model(device: torch.device) -> torch.nn.Module:
    return vad_model.to(device)

def get_speech_ranges(waveform: torch.Tensor) -> List[Range]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        timestamps = _get_speech_timestamps(waveform, get_vad_model(waveform.device))
    return [Range(timestamp['start'], timestamp['end']) for timestamp in timestamps]
