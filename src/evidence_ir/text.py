"""Lightweight, dependency-free text normalisation for lexical retrieval."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

# a compact English stop-word list (function words only)
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may might more most no not of on or our should so such than that the their them
    then there these they this those to was we were what when where which while who why will with
    would you your""".split()
)

_SUFFIXES = (("sses", 6, "ss"), ("ies", 5, "y"), ("ing", 6, ""), ("ed", 5, ""), ("es", 5, ""))


def stem(token: str) -> str:
    """A tiny suffix stripper (plurals, -ing, -ed) that keeps BM25 dependency-free."""
    for suffix, min_len, repl in _SUFFIXES:
        if token.endswith(suffix) and len(token) >= min_len:
            return token[: -len(suffix)] + repl
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) >= 4:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
