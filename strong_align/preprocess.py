from typing import Tuple, List
import unicodedata
from num2words import num2words
import re

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


def tokenize(text: str, language_code: str, labels: dict) -> Tuple[List[int], str, List[int]]:
    text, mappings = normalize(text, language_code)
    tokens = []
    next_mappings = []
    for i, char in enumerate(text):
        if char not in labels:
            char = strip_accents(char)
        if re.match(SPACE_RE, char) and language_code not in LANGUAGES_WITHOUT_SPACES:
            char = "|"
        if char in labels:
            tokens.append(labels[char])
            next_mappings.append(mappings[i])
    return tokens, next_mappings


def normalize(text: str, language_code: str) -> Tuple[str, List[int]]:
    mappings = list(range(len(text)))
    text = text.lower()
    text, mappings = normalize_numerals(text, mappings, language_code)
    text, mappings = normalize_abbr(text, mappings, language_code)
    return text, mappings


def normalize_abbr(text: str, mappings: List[int], language_code: str) -> Tuple[str, List[int]]:
    if language_code not in ABBR:
        return text, mappings
    for abbr, value in ABBR[language_code].items():
        while True:
            start = text.find(abbr)
            if start == -1:
                break
            end = start + len(abbr)
            text, mappings = augment(text, mappings, start, end, value)
    return text, mappings

def normalize_numerals(text: str, mappings: List[int], language_code: str) -> Tuple[str, List[int]]:
    while True:
        match = re.search(NUM_RE, text)
        if not match:
            break
        start, end = match.span()
        num_text = match.group().replace(",", "")

        num_words_suffix = ""
        if num_text.startswith("$"):
            num_text = num_text[1:]
            num_words_suffix = " dollars"
        elif num_text.endswith("%"):
            num_text = num_text[:-1]
            num_words_suffix = " percent"

        num_words = try_num2words(
            num_text, language_code) + num_words_suffix

        text, mappings = augment(text, mappings, start, end, num_words)
    return text, mappings


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def try_num2words(text: str, language_code: str) -> str:
    try:
        return num2words(text, lang=language_code)
    except:
        return text


def augment(text: str, mappings: List[int], start: int, end: int, value: str):
    mapped_start = mappings[start]
    mapped_end = mappings[end-1]
    insertion = []
    for i in range(mapped_start, mapped_start + len(value)):
        insertion.append(min(i, mapped_end))
    return (text[:start] + value + text[end:],
            mappings[:start] + insertion + mappings[end:])