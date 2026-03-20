from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParseResult:
    label: str | None
    status: str
    matched_text: str | None = None


def parse_mcq_answer(raw_text: str | None, valid_labels: list[str]) -> ParseResult:
    """
    Minimal parser for MCQ labels like A/B/C/D.

    Case 1:
        exact short answers only, e.g. "A", "B.", "(C)", "D)"
    Case 2:
        longer outputs with explicit answer markers, e.g.
        "Odpověď: A", "Odpověď je B", "Správná odpověď je C", "Volba D"
    """
    if raw_text is None:
        return ParseResult(label=None, status="empty", matched_text=None)

    text = raw_text.strip()
    if not text:
        return ParseResult(label=None, status="empty", matched_text=None)

    valid = {label.upper() for label in valid_labels}

    # case 1: exact short answers, only label-like outputs
    exact_patterns = [
        r"^\s*\(?([A-Za-z])\)?[\.\:\)]?\s*$",
    ]

    for pattern in exact_patterns:
        m = re.match(pattern, text, flags=re.IGNORECASE)
        if m:
            label = m.group(1).upper()
            if label in valid:
                return ParseResult(label=label, status="ok", matched_text=m.group(0))

    # case 2: longer outputs with explicit answer markers
    marker_patterns = [
        # "odpověď A", "odpověď je A", "správná odpověď: B"
        r"(?:správná\s+odpověď|spravna\s+odpoved|odpověď|odpoved|answer)(?:\s+je|\s+is)?\s*[:\-]?\s*\(?([A-Za-z])\)?\b",

        # "volba C", "možnost: D", "varianta B", "option A"
        r"(?:správná\s+volba|spravna\s+volba|volba|možnost|moznost|varianta|option)\s*[:\-]?\s*\(?([A-Za-z])\)?\b",

        # "písmeno A", "pismeno: B"
        r"(?:písmeno|pismeno)\s*[:\-]?\s*\(?([A-Za-z])\)?\b",

        # final standalone letter at end of longer output
        r"\b([A-Za-z])\b(?=\s*$)",
    ]

    candidates: list[str] = []
    for pattern in marker_patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            label = m.group(1).upper()
            if label in valid:
                candidates.append(label)

    candidates = list(dict.fromkeys(candidates))

    if len(candidates) == 1:
        return ParseResult(label=candidates[0], status="ok", matched_text=candidates[0])

    if len(candidates) > 1:
        return ParseResult(label=None, status="ambiguous", matched_text=", ".join(candidates))

    return ParseResult(label=None, status="invalid", matched_text=None)