"""
This is slightly modified version of the code from the link below.
https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html
"""

import re
from typing import Iterable, List

import torch

from .common import Alignment, Point
from .preprocess import SPACE_RE


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0) -> Iterable[Point]:
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1]
                        if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return None
    return path[::-1]


def merge_repeats(path) -> List[Alignment]:
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        token_index = path[i1].token_index
        segments.append(
            Alignment(
                start_token_index=token_index,
                end_token_index=token_index + 1,
                start_time_index=path[i1].time_index,
                end_time_index=path[i2 - 1].time_index + 1,
                score=score,
            )
        )
        i1 = i2
    return segments


def merge_words(segments: List[Alignment], text: str) -> List[Alignment]:
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or re.match(SPACE_RE, text[segments[i2].start_token_index]):
            if i1 != i2:
                word_segments = segments[i1:i2]
                score_sum = sum(seg.score for seg in word_segments)
                word_length = sum(seg.end_time_index -
                                  seg.start_time_index for seg in word_segments)
                words.append(Alignment(
                    start_token_index=segments[i1].start_token_index,
                    end_token_index=segments[i2 - 1].end_token_index,
                    start_time_index=segments[i1].start_time_index,
                    end_time_index=segments[i2 - 1].end_time_index,
                    score=score_sum / word_length))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words