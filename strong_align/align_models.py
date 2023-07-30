from functools import lru_cache
from typing import Tuple
import torchaudio
from torchaudio.pipelines import Wav2Vec2ASRBundle
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

ALIGN_MODELS = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
}

@lru_cache
def get_bundle(language_code: str, device: str) -> Tuple[lambda x: x, dict]:
    name = ALIGN_MODELS.get(language_code, "WAV2VEC2_ASR_BASE_960H")
    if name in torchaudio.pipelines.__all__:
        bundle: Wav2Vec2ASRBundle = torchaudio.pipelines.__dict__[name]
        model = bundle.get_model().to(device)
        labels = enumerate(bundle.get_labels())
        return lambda x: model(x)[0], {v.lower(): k for k, v in labels}
    else:
        processor = Wav2Vec2Processor.from_pretrained(name)
        raw_model = Wav2Vec2ForCTC.from_pretrained(name).to(device)
        model = lambda x: raw_model(x).logits
        labels = processor.tokenizer.get_vocab()
        return model, {k.lower(): v for k, v in labels.items()}
