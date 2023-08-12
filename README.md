# 🦾 Strong Align 🎯

Forced alignment using Wav2Vec2

## Installation

```bash
pip install git+https://github.com/weisrc/strong-align.git 
```

## Usage

```python
import torchaudio
from strong_align import align

text = "Hello world! This is a test."
audio, sr = torchaudio.load("test.wav")
audio = audio[0]
audio = torchaudio.transforms.Resample(sr, 16000)(audio)
audio = audio.to("cuda") # or keep it on the CPU
out = align(text, audio, "en", on_progress=print)
print(out)
```

> :warning: **Warning**: This package is still in development. The API may change in the future.

## License

MIT. Wei (weisrc)
