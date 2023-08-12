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

def normalize_spaces(text: str, mappings: List[int], *_) -> Tuple[str, List[int]]:
    return re.sub(SPACE_RE, ' ', text), mappings

def normalize_accents(text: str, mappings: List[int], language: str, labels: dict) -> Tuple[str, List[int]]:
    text = "".join(char if char in labels else strip_accents(char) for char in text)
    return text, mappings

def normalize_case(text: str, mappings: List[int], *_) -> Tuple[str, List[int]]:
    return text.lower(), mappings

def normalize_abbr(text: str, mappings: List[int], language: str, *_) -> Tuple[str, List[int]]:
    if language not in ABBR:
        return text, mappings
    for abbr, value in ABBR[language].items():
        while True:
            start = text.find(abbr)
            if start == -1:
                break
            end = start + len(abbr)
            text, mappings = insert_between(text, mappings, start, end, value)
    return text, mappings

def normalize_numerals(text: str, mappings: List[int], language: str, *_) -> Tuple[str, List[int]]:
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
            num_text, language) + num_words_suffix

        text, mappings = insert_between(text, mappings, start, end, num_words)
    return text, mappings


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def try_num2words(text: str, language: str) -> str:
    try:
        return num2words(text, lang=language)
    except:
        return text


def insert_between(text: str, mappings: List[int], start: int, end: int, value: str):
    mapped_start = mappings[start]
    mapped_end = mappings[end-1]
    insertion = []
    for i in range(mapped_start, mapped_start + len(value)):
        insertion.append(min(i, mapped_end))
    return (text[:start] + value + text[end:],
            mappings[:start] + insertion + mappings[end:])