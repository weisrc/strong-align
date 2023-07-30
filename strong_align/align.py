from typing import List

import torch

from .align_models import get_bundle
from .align_utils import (Alignment, backtrack, get_trellis, merge_repeats,
                          merge_words)
from .common import SAMPLE_RATE, Range
from .preprocess import LANGUAGES_WITHOUT_SPACES, tokenize
from .vad import get_speech_ranges


def align(text: str, waveform: torch.Tensor,
          language_code: str, device="cpu", letter_wise=False) -> List[Alignment]:

    model, labels = get_bundle(language_code, device)
    waveform = waveform.to(device)
    tokens, mappings = tokenize(text, language_code, labels)
    time_ranges = get_speech_ranges(waveform)
    time_mappings: List[Range] = []
    time_mapping_offset = 0
    emission_list: List[torch.Tensor] = []
    
    for time_range in time_ranges:
        sub_waveform = waveform[time_range.start:time_range.end]
        with torch.inference_mode():
            emissions = model(sub_waveform.unsqueeze(0))
            emissions = torch.log_softmax(emissions, dim=-1)
            emission = emissions[0].cpu().detach()
            emission_list.append(emission)
            time_mappings.append(Range(time_mapping_offset, 
                                         time_mapping_offset + len(emission)))
            time_mapping_offset += len(emission)

    emission = torch.concat(emission_list, dim=0)
    trellis = get_trellis(emission, tokens)
    points = backtrack(trellis, emission, tokens)
    if points is None:
        return None
    alignments = merge_repeats(points)
    alignments = map_alignments(alignments, len(text), mappings)
    alignments = interpolate_alignments(alignments, len(emission))

    if language_code not in LANGUAGES_WITHOUT_SPACES and not letter_wise:
        alignments = merge_words(alignments, text)

    for alignment in alignments:
        start, end = map_time_range(alignment, time_ranges, time_mappings)
        alignment.start = start
        alignment.end = end
        alignment.text = text[alignment.start_token_index:alignment.end_token_index]

    return alignments



def map_time_range(alignment: Alignment, time_ranges: List[range], 
                          time_mappings: List[range]) -> Alignment:
    time_index = alignment.end_time_index
    for time_range, time_mapping in zip(time_ranges, time_mappings):
        if time_mapping.start <= time_index and time_index < time_mapping.end:
            time_mapping_duration = time_mapping.end - time_mapping.start
            time_range_duration = time_range.end - time_range.start
            ratio = time_range_duration / time_mapping_duration
            start = (alignment.start_time_index - time_mapping.start) * ratio + time_range.start
            end = (alignment.end_time_index - time_mapping.start) * ratio + time_range.start
            return start / SAMPLE_RATE, end / SAMPLE_RATE
    raise ValueError("time_index not in time_mappings")


def map_alignments(alignments: List[Alignment], size: int, mappings: List[int]):
    mapped: List[Alignment] = [None] * size

    for s in alignments:
        i = mappings[s.start_token_index]
        s.start_token_index = i
        s.end_token_index = i + 1
        mapped[i] = s

    return mapped


def interpolate_alignments(alignments: List[Alignment], max_end_time):
    for i, s in enumerate(alignments):
        if s:
            continue

        range_start_time = alignments[i -
                                      1].end_time_index if i > 0 else 0
        j = i + 1
        while True:
            if j >= len(alignments):
                range_end_time = max_end_time
                break
            next_alignment = alignments[j]
            if next_alignment:
                range_end_time = next_alignment.start_time_index
                break
            j += 1

        range_time = range_end_time - range_start_time
        size = j - i

        for offset in range(size):
            k = i + offset
            start_time = int(
                range_start_time + offset * range_time / size)
            end_time = int(
                range_start_time + (offset + 1) * range_time / size)

            alignments[k] = Alignment(
                start_token_index=i, end_token_index=i + 1,
                start_time_index=start_time, end_time_index=end_time,
                score=0.0)

    return alignments