from strong_align.align import align
import torchaudio
import torch
import json
import dataclasses

waveform, sample_rate = torchaudio.load("data/doctor.mp3")

# resample to 16kHz

resampler = torchaudio.transforms.Resample(sample_rate, 16000)
waveform = resampler(waveform)

TIMES = 10

text = """
Hello Doctor George. What is your name?
Hello Mr. George. What is your name?
Hello Professor George. What is your name?
1 versus 2, which is larger?
Hello miss Jones, what is your name?
Thank you!
""" * TIMES

waveform = waveform.squeeze(0)
waveform = torch.concat([waveform] * TIMES, dim=0)

alignments = align(text, waveform, "en", on_progress=print)

with open("data/doctor.json", "w") as f:
    f.write(json.dumps(list(map(dataclasses.asdict, alignments))))

torchaudio.save("data/out.wav", waveform.unsqueeze(0), 16000)