from typing import Tuple, List
from . import normalize

NUM_RE = r"\$?(\d+,)*\d+\.?\d*%?"
SPACE_RE = r"\s"
ABBR = {
    "en": {'dr.': "doctor", 'mr.': "mister", 'mrs.': "missus", 'prof.': "professor"},
    "es": {'dr.': "doctor", 'sr.': "señor", 'sra.': "señora", 'prof.': "profesor"},
    "fr": {'dr.': "docteur", 'm.': "monsieur", 'mme.': "madame", 'prof.': "professeur"},
    "de": {'dr.': "doktor", 'prof.': "professor"},
    "it": {'dr.': "dottore", 'prof.': "professore"},
    "pt": {'dr.': "doutor", 'prof.': "professor"},
}

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

NORMALIZE_FUNCS = [
    normalize.normalize_spaces,
    normalize.normalize_accents,
    normalize.normalize_case,
    normalize.normalize_abbr,
    normalize.normalize_numerals,
]

def tokenize(text: str, mappings: List[int], labels: dict) -> Tuple[List[int], str, List[int]]:
    tokens = []
    next_mappings = []
    for i, char in enumerate(text):
        if char in labels:
            tokens.append(labels[char])
            next_mappings.append(mappings[i])
    return tokens, next_mappings


def create_mappings(text: str) -> List[int]:
    return list(range(len(text)))
    

def normalize(text: str, language: str, labels: dict, 
              funcs=NORMALIZE_FUNCS) -> Tuple[str, List[int]]:
    mappings = create_mappings(text)
    for func in funcs:
        text, mappings = func(text, mappings, language, labels)
    return text, mappings
