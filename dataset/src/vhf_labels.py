from __future__ import annotations

import re


def phoneme_key(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", " ", value.upper()).strip()
    words = []
    for word in text.split():
        word = re.sub(r"PH", "F", word)
        word = re.sub(r"GH", "G", word)
        word = re.sub(r"KN", "N", word)
        word = re.sub(r"WR", "R", word)
        word = re.sub(r"QU", "K", word)
        word = re.sub(r"CK", "K", word)
        word = re.sub(r"CH", "X", word)
        word = re.sub(r"SH", "X", word)
        word = re.sub(r"TH", "T", word)
        word = word.translate(str.maketrans({"C": "K", "Q": "K", "Z": "S", "V": "F"}))
        head = word[:1]
        tail = re.sub(r"[AEIOUYWH]", "", word[1:])
        words.append(re.sub(r"(.)\1+", r"\1", head + tail))
    return " ".join(words)


def phoneme_similarity(left: str, right: str) -> int:
    a = set(phoneme_key(left).replace(" ", ""))
    b = set(phoneme_key(right).replace(" ", ""))
    return len(a & b)
