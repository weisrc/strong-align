from typing import List

from .common import Range
import torch

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad', verbose=False)

_get_speech_timestamps, *_ = utils


def get_speech_ranges(wav: torch.Tensor) -> List[Range]:
    timestamps = _get_speech_timestamps(wav, vad_model)
    return [Range(timestamp['start'], timestamp['end']) for timestamp in timestamps]
