# 🦾 Strong Align 🎯

Forced alignment using Wav2Vec2

## Installation

```bash
pip install git+https://github.com/weisrc/strong-align.git 
```

> :warning: **Warning**: This package is still in development. The API may change in the future.

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

## Custom normalization

You can use your own normalization function by passing it to the `align` function.

```python
from strong_align.preprocess import NORMALIZE_FUNCS

def my_normalize_normalize(text, mappings, language, labels):
    return text, mappings

out = align(text, audio, "en", 
      normalize_func=NORMALIZE_FUNCS+[my_normalize_normalize])
```

> Please refer to the `normalize.py` file for examples of normalization functions.

## License

MIT. Wei (weisrc)
